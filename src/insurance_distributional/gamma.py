"""
GammaGBM: distributional GBM for Gamma-distributed severity.

Gamma is the workhorse severity distribution in UK personal lines. The
compound Poisson-Gamma (Tweedie) pure premium model uses Gamma for individual
claim sizes. GammaGBM is the right choice when you're modelling severity in
isolation — e.g., fitting a severity model that feeds into a frequency-severity
product, or when you need the conditional severity distribution for XL pricing.

Model:
  Y | x ~ Gamma(shape(x), scale(x))
  E[Y|x] = mu(x) = shape(x) * scale(x)
  Var[Y|x] = phi(x) * mu(x)^2 where phi = 1/shape

The Smyth-Jørgensen double GLM applies here too. For Gamma,
V(mu) = mu^2, so the dispersion response is:
    d_i = (y_i - mu_hat_i)^2 / mu_hat_i^2

This is the squared relative error. Under the Gamma model, E[d_i | x_i] = phi(x_i).

Baseline handling:
  CatBoost Tweedie loss uses a log link internally. When training with a
  Pool(baseline=b), CatBoost minimises Tweedie(y, exp(b + f(x))). The model
  stores only f(x) (the tree contributions); at predict time, model.predict(X)
  returns exp(f(x)) NOT exp(b + f(x)). The caller must re-apply the baseline.

  For GammaGBM:
  - mu model: baseline = log(mu_init), stored as _log_mu_init
  - phi model: baseline = log(phi_init), stored as _log_phi_init
  Both must be re-applied at predict time via _predict_catboost.

Phi model (v0.1.3 fix — cross-fitting):
  The root cause of the ~3-5x phi underestimation was that the phi model was
  trained on IN-SAMPLE residuals from the mu model. Because CatBoost with n=300
  and 300 trees/depth 6 significantly overfits on training data, in-sample
  mu_hat ≈ y, making d_i = (y - mu_hat)^2/mu_hat^2 ≈ 0 on training data.
  The phi model then learned to predict very small phi.

  The fix: cross-fitting (double ML / Neyman orthogonal style). We split the
  training data into K=3 folds. For each fold, we fit a mu model on the OTHER
  K-1 folds and predict on the held-out fold. This gives out-of-fold (OOF) mu
  predictions that are unbiased (not overfitted to the held-out obs). The phi
  model is then trained on OOF d_i values which have E[d_i] = phi_i as intended.

  The final mu model is still fitted on all n observations for prediction.
  This adds ~3x compute for the phi training step but is essential for correctness.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from .base import DistributionalGBM
from .prediction import DistributionalPrediction

logger = logging.getLogger(__name__)


def _gamma_log_likelihood(
    y: np.ndarray, mu: np.ndarray, phi: np.ndarray
) -> np.ndarray:
    """
    Gamma log-likelihood per observation.

    Y ~ Gamma(shape=1/phi, scale=mu*phi)
    log p(y; mu, phi) = (1/phi-1)*log(y) - y/(mu*phi) - (1/phi)*log(mu*phi) - log Gamma(1/phi)
    """
    shape = 1.0 / phi  # shape = 1/phi
    scale = mu * phi    # scale = mu*phi
    ll = (
        (shape - 1) * np.log(y + 1e-12)
        - y / (scale + 1e-12)
        - shape * np.log(scale + 1e-12)
        - gammaln(shape)
    )
    return ll


def _estimate_phi_gamma_mle(y: np.ndarray, mu: np.ndarray) -> float:
    """
    Estimate scalar phi (= 1/shape) by MLE given fixed mu.

    P1-5 fix: emits a warning if the optimum lands at either search boundary.
    """
    lo, hi = -5.0, 3.0

    def neg_ll(log_phi: float) -> float:
        phi_val = np.exp(log_phi)
        phi_arr = np.full(len(y), phi_val)
        return -np.sum(_gamma_log_likelihood(y, mu, phi_arr))

    result = minimize_scalar(neg_ll, bounds=(lo, hi), method="bounded")
    if abs(result.x - lo) < 1e-4:
        warnings.warn(
            f"Gamma phi MLE hit lower bound (log_phi={lo}); "
            f"true phi may be < {np.exp(lo):.4f}. "
            "Very low dispersion — consider inspecting the data.",
            UserWarning,
            stacklevel=2,
        )
    elif abs(result.x - hi) < 1e-4:
        warnings.warn(
            f"Gamma phi MLE hit upper bound (log_phi={hi}); "
            f"true phi may be > {np.exp(hi):.3f}. "
            "High dispersion — consider inspecting the data.",
            UserWarning,
            stacklevel=2,
        )
    return float(np.exp(result.x))


class GammaGBM(DistributionalGBM):
    """
    Distributional GBM for Gamma severity distribution.

    Models both the conditional mean mu(x) and conditional dispersion phi(x).
    phi(x) = 1/shape(x); CoV(Y|x) = sqrt(phi(x)).

    Parameters
    ----------
    n_cycles : int
        Coordinate descent cycles. Default 1.
    model_dispersion : bool
        If True (default), fit a dispersion model. If False, use scalar phi.
    phi_cv_folds : int
        Number of cross-validation folds for computing out-of-fold mu residuals
        used to train the phi model. Default 3. Higher values give less biased
        phi estimates at the cost of more compute. Set to 1 to disable cross-
        fitting (reverts to in-sample residuals, which are biased when the mu
        model overfits).
    cat_features : list, optional
    catboost_params_mu : dict, optional
    catboost_params_phi : dict, optional
    random_state : int

    Examples
    --------
    >>> from insurance_distributional import GammaGBM
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((500, 4))
    >>> y = rng.gamma(shape=2.0, scale=500.0, size=500)
    >>> model = GammaGBM()
    >>> model.fit(X, y)
    GammaGBM(...)
    >>> pred = model.predict(X)
    >>> pred.cov.shape  # CoV per risk
    (500,)
    """

    def __init__(
        self,
        n_cycles: int = 1,
        model_dispersion: bool = True,
        phi_cv_folds: int = 3,
        cat_features: Optional[List[Union[int, str]]] = None,
        catboost_params_mu: Optional[Dict[str, Any]] = None,
        catboost_params_phi: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            n_cycles=n_cycles,
            cat_features=cat_features,
            catboost_params_mu=catboost_params_mu,
            catboost_params_phi=catboost_params_phi,
            random_state=random_state,
        )
        self.model_dispersion = model_dispersion
        self.phi_cv_folds = phi_cv_folds
        self._model_mu = None
        self._model_phi = None
        self._phi_scalar: float = 1.0
        # P0-4: store log baselines for predict time re-application.
        self._log_mu_init: float = 0.0
        self._log_phi_init: float = 0.0

    def _init_params(self, y: np.ndarray, exposure: np.ndarray) -> Dict[str, Any]:
        """Unconditional MLE for Gamma parameters."""
        mu_init = float(np.mean(y))
        mu_arr = np.full(len(y), mu_init)
        phi_init = _estimate_phi_gamma_mle(y, mu_arr)
        logger.debug("GammaGBM init: mu_init=%.4f, phi_init=%.4f", mu_init, phi_init)
        return {
            "mu_init": mu_init,
            "phi_init": phi_init,
            "mu": mu_arr,
            "phi": np.full(len(y), phi_init),
        }

    def _fit_cycle(
        self,
        X: np.ndarray,
        y: np.ndarray,
        exposure: np.ndarray,
        params: Dict[str, Any],
        cycle: int,
    ) -> Dict[str, Any]:
        """
        Step 1: Fit mu via CatBoost Gamma deviance (log link) on all data.
        Step 2: Compute out-of-fold (OOF) mu residuals via K-fold cross-fitting.
        Step 3: Fit phi via Gamma deviance on OOF squared relative errors.

        P0-4 fix: mu model uses a log-scale baseline; must be re-applied at
        predict time via _predict_catboost.

        v0.1.3 fix (cross-fitting): phi residuals computed from OOF mu predictions
        to prevent overfitting bias. In-sample residuals from an overfitting mu
        model are near-zero, causing phi_hat << phi_true. OOF residuals are
        unbiased: E[d_i^{OOF}] = phi_i(x_i) by construction.
        """
        mu_params = self._merge_catboost_params(
            self._default_catboost_params("Tweedie:variance_power=1.99", iterations=300),
            self.catboost_params_mu,
        )

        if cycle == 0:
            baseline_mu = np.full(len(y), np.log(params["mu_init"]))
            self._log_mu_init = float(np.log(params["mu_init"]))
        else:
            baseline_mu = np.log(np.clip(params["mu"], 1e-6, None))
            self._baseline_mu_train = baseline_mu.copy()

        # Fit full mu model on all data (for prediction at inference time)
        self._model_mu = self._fit_catboost(
            X, y, mu_params, baseline=baseline_mu
        )

        # Full-data in-sample mu_hat (for updating params["mu"] in next cycle)
        mu_hat_full = self._predict_catboost(self._model_mu, X, baseline=baseline_mu)
        mu_hat_full = np.clip(mu_hat_full, 1e-6, None)
        params["mu"] = mu_hat_full

        if not self.model_dispersion:
            phi_scalar = _estimate_phi_gamma_mle(y, mu_hat_full)
            self._phi_scalar = phi_scalar
            params["phi"] = np.full(len(y), phi_scalar)
            return params

        # --- Cross-fitting for unbiased phi residuals ---
        # Compute out-of-fold mu predictions to avoid overfitting bias.
        # For each fold, fit mu on the training folds and predict on the held-out.
        if self.phi_cv_folds > 1:
            mu_hat_oof = self._compute_oof_mu(
                X, y, exposure, params, cycle, baseline_mu, mu_params
            )
        else:
            # phi_cv_folds=1: skip cross-fitting (in-sample residuals)
            mu_hat_oof = mu_hat_full

        # Squared relative errors from OOF mu: E[d_i] = phi_i unbiasedly
        d = (y - mu_hat_oof) ** 2 / (mu_hat_oof ** 2)
        d = np.clip(d, 1e-8, None)

        # Fit phi model with Gamma deviance and log(phi_init) baseline.
        phi_init = float(params.get("phi_init", float(np.mean(d))))
        phi_init = max(phi_init, 1e-4)
        self._log_phi_init = float(np.log(phi_init))
        baseline_phi = np.full(len(y), self._log_phi_init)

        phi_params = self._merge_catboost_params(
            self._default_catboost_params("Tweedie:variance_power=1.99", iterations=200),
            self.catboost_params_phi,
        )
        self._model_phi = self._fit_catboost(
            X, d, phi_params, baseline=baseline_phi, sample_weight=exposure
        )
        phi_hat = self._predict_catboost(
            self._model_phi, X, baseline=baseline_phi
        )
        phi_hat = np.clip(phi_hat, 1e-6, None)
        params["phi"] = phi_hat

        return params

    def _compute_oof_mu(
        self,
        X: np.ndarray,
        y: np.ndarray,
        exposure: np.ndarray,
        params: Dict[str, Any],
        cycle: int,
        baseline_mu: np.ndarray,
        mu_params: Dict[str, Any],
    ) -> np.ndarray:
        """
        Compute out-of-fold mu predictions via K-fold cross-fitting.

        For each fold k: fit mu model on the OTHER K-1 folds, predict on fold k.
        Returns mu_hat[i] computed from a model that did NOT see observation i.

        This gives E[d_i^{OOF}] = phi_i(x_i) unbiasedly, even when the mu model
        overfits on training data.
        """
        n = len(y)
        K = self.phi_cv_folds
        mu_hat_oof = np.zeros(n)

        # Reproducible fold assignment
        rng = np.random.default_rng(self.random_state + cycle * 31337)
        fold_idx = rng.permutation(n) % K
        # fold_idx[i] = fold assignment for observation i

        for k in range(K):
            train_mask = fold_idx != k
            val_mask = ~train_mask

            if val_mask.sum() == 0:
                continue

            X_tr = X[train_mask]
            y_tr = y[train_mask]
            exp_tr = exposure[train_mask]
            base_tr = baseline_mu[train_mask]

            X_val = X[val_mask]
            base_val = baseline_mu[val_mask]

            # Fit mu model on K-1 folds
            mu_model_k = self._fit_catboost(
                X_tr, y_tr, mu_params, baseline=base_tr
            )
            # Predict on held-out fold with the same baseline
            mu_hat_k = self._predict_catboost(mu_model_k, X_val, baseline=base_val)
            mu_hat_oof[val_mask] = np.clip(mu_hat_k, 1e-6, None)

            logger.debug(
                "OOF fold %d/%d: n_train=%d, n_val=%d, mu_hat_oof[val] mean=%.4f",
                k + 1, K, train_mask.sum(), val_mask.sum(),
                float(np.mean(mu_hat_oof[val_mask]))
            )

        return mu_hat_oof

    def _predict_params(
        self, X: np.ndarray, exposure: np.ndarray
    ) -> Dict[str, np.ndarray]:
        n = len(X)
        baseline = np.full(n, self._log_mu_init)
        mu_hat = np.clip(
            self._predict_catboost(self._model_mu, X, baseline=baseline),
            1e-6, None
        )

        if self.model_dispersion and self._model_phi is not None:
            baseline_phi = np.full(n, self._log_phi_init)
            phi_hat = self._predict_catboost(
                self._model_phi, X, baseline=baseline_phi
            )
        else:
            phi_hat = np.full(n, self._phi_scalar)

        return {"mu": mu_hat, "phi": np.clip(phi_hat, 1e-6, None)}

    def _make_prediction(self, params: Dict[str, np.ndarray]) -> DistributionalPrediction:
        return DistributionalPrediction(
            distribution="gamma",
            mu=params["mu"],
            phi=params["phi"],
        )

    def _neg_log_likelihood(self, y: np.ndarray, params: Dict[str, np.ndarray]) -> float:
        ll = _gamma_log_likelihood(y, params["mu"], params["phi"])
        return float(-np.mean(ll))

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"GammaGBM("
            f"model_dispersion={self.model_dispersion}, "
            f"phi_cv_folds={self.phi_cv_folds}, "
            f"n_cycles={self.n_cycles}, "
            f"status={status!r})"
        )
