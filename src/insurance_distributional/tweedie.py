"""
TweedieGBM: distributional gradient boosting for Tweedie compound Poisson-Gamma.

This is the core model for UK motor pure premium pricing. It jointly models:
- mu(x): the conditional mean, via CatBoost's built-in Tweedie loss
- phi(x): the conditional dispersion, via Smyth-Jørgensen double GLM

The Smyth-Jørgensen double GLM formulation converts dispersion estimation into a
standard regression problem. The response variable for the phi model is the
squared Pearson residual:

    d_i = (y_i - mu_hat_i)^2 / V(mu_hat_i)

where V(mu) = mu^p is the Tweedie variance function. Under the assumed model,
E[d_i] = phi_i. We fit a Gamma regression with log link on d_i to estimate phi(x).

Key implementation detail: the mu model uses CatBoost's native Tweedie loss
(loss_function="Tweedie:variance_power=p"). CatBoost handles the gradient/Hessian
computation for mu internally. We only need the dispersion residuals for phi.

Baseline handling:
  CatBoost Tweedie loss uses a log link internally. When training with a Pool
  that has a baseline=b, CatBoost minimises Tweedie(y, exp(b + f(x))). The model
  stores only f(x) (the tree contributions); model.predict(X) returns exp(f(x))
  not exp(b + f(x)). The baseline must be re-applied at inference time.

  For TweedieGBM:
  - mu model: baseline = log(exposure) + log(mu_init). At inference we pass
    baseline = log(exposure) via _predict_catboost.
  - phi model: baseline = log(phi_init). At inference we pass log(phi_init).

Phi model (v0.1.3 fix — cross-fitting):
  The root cause of the ~3-5x phi underestimation was in-sample residuals from
  an overfitting mu model. When CatBoost (depth=6, 300 trees) overfits on
  training data, mu_hat ≈ y in-sample, making d_i ≈ 0. The phi model then
  learns phi_hat ≈ 0 on training data and generalises that small value.

  Fix: K-fold cross-fitting. For each fold, fit mu on the OTHER K-1 folds and
  predict on the held-out fold. This gives out-of-fold (OOF) mu predictions that
  are not overfitted to the held-out observations. OOF d_i values have
  E[d_i^{OOF}] = phi_i(x_i) as intended by the Smyth-Jørgensen framework.

  The final mu model is fitted on all n observations for prediction.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from .base import DistributionalGBM, _clip_hessians, _to_1d
from .prediction import DistributionalPrediction

logger = logging.getLogger(__name__)


def _tweedie_log_likelihood(
    y: np.ndarray, mu: np.ndarray, phi: np.ndarray, p: float
) -> np.ndarray:
    """
    Tweedie log-likelihood per observation.

    For compound Poisson-Gamma (1 < p < 2), saddle-point approximation.
    For y=0: log p(0) = -mu^(2-p) / (phi*(2-p))
    For y>0: deviance formulation.
    """
    ll = np.zeros(len(y))

    mask0 = y == 0
    if mask0.any():
        ll[mask0] = -mu[mask0] ** (2 - p) / (phi[mask0] * (2 - p))

    mask_pos = ~mask0
    if mask_pos.any():
        y_p = y[mask_pos]
        mu_p = mu[mask_pos]
        phi_p = phi[mask_pos]
        dev = (
            y_p ** (2 - p) / ((1 - p) * (2 - p))
            - y_p * mu_p ** (1 - p) / (1 - p)
            + mu_p ** (2 - p) / (2 - p)
        )
        ll[mask_pos] = -dev / phi_p - 0.5 * np.log(2 * np.pi * phi_p * y_p ** p)

    return ll


def _estimate_phi_mle(y: np.ndarray, mu: np.ndarray, p: float) -> float:
    """
    Estimate scalar phi by MLE given fixed mu.

    P1-5 fix: emits a warning if the optimum lands at either search boundary.
    """
    lo, hi = -3.0, 3.0

    def neg_ll(log_phi: float) -> float:
        phi_val = np.exp(log_phi)
        phi_arr = np.full(len(y), phi_val)
        return -np.sum(_tweedie_log_likelihood(y, mu, phi_arr, p))

    result = minimize_scalar(neg_ll, bounds=(lo, hi), method="bounded")
    if abs(result.x - lo) < 1e-4:
        warnings.warn(
            f"Tweedie phi MLE hit lower bound (log_phi={lo}); "
            f"true phi may be < {np.exp(lo):.3f}. "
            "Consider narrowing the search bounds or inspecting the data.",
            UserWarning,
            stacklevel=2,
        )
    elif abs(result.x - hi) < 1e-4:
        warnings.warn(
            f"Tweedie phi MLE hit upper bound (log_phi={hi}); "
            f"true phi may be > {np.exp(hi):.3f}. "
            "Consider narrowing the search bounds or inspecting the data.",
            UserWarning,
            stacklevel=2,
        )
    return float(np.exp(result.x))


class TweedieGBM(DistributionalGBM):
    """
    Distributional GBM for Tweedie compound Poisson-Gamma.

    Models both the mean mu(x) and dispersion phi(x) as functions of
    covariates. The mean is fitted via CatBoost's native Tweedie loss.
    Dispersion is modelled via Smyth-Jørgensen double GLM: a Gamma
    regression on squared Pearson residuals computed via cross-fitting.

    Parameters
    ----------
    power : float
        Tweedie variance power p. Must be in (1, 2) for compound Poisson-Gamma.
        Default 1.5 (standard for UK motor). Not estimated from data.
    n_cycles : int
        Coordinate descent cycles. Default 1.
    model_dispersion : bool
        If True (default), fit a GBM model for phi(x). If False, use scalar phi.
    phi_cv_folds : int
        Number of cross-validation folds for OOF mu residuals used to train the
        phi model. Default 3. Set to 1 to disable cross-fitting.
    cat_features : list, optional
    catboost_params_mu : dict, optional
    catboost_params_phi : dict, optional
    random_state : int

    Examples
    --------
    >>> from insurance_distributional import TweedieGBM
    >>> import numpy as np
    >>> X = np.random.randn(1000, 5)
    >>> y = np.abs(np.random.randn(1000)) * 500
    >>> model = TweedieGBM(power=1.5)
    >>> model.fit(X, y)
    TweedieGBM(power=1.5)
    >>> pred = model.predict(X)
    >>> pred.mean.shape
    (1000,)
    """

    def __init__(
        self,
        power: float = 1.5,
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
        if not (1.0 < power < 2.0):
            raise ValueError(
                f"Tweedie power must be in (1, 2) for compound Poisson-Gamma, got {power}"
            )
        self.power = power
        self.model_dispersion = model_dispersion
        self.phi_cv_folds = phi_cv_folds
        self._model_mu = None
        self._model_phi = None
        self._phi_scalar: float = 1.0
        self._log_phi_init: float = 0.0

    # -------------------------------------------------------------------------
    # DistributionalGBM interface
    # -------------------------------------------------------------------------

    def _init_params(self, y: np.ndarray, exposure: np.ndarray) -> Dict[str, Any]:
        """Unconditional MLE initialisation."""
        mu_init = np.sum(y) / np.sum(exposure)
        mu_arr = np.full(len(y), mu_init * exposure)
        phi_init = _estimate_phi_mle(y, mu_arr, self.power)
        logger.debug("Init: mu_init=%.4f, phi_init=%.4f", mu_init, phi_init)
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
        Step 1 — Fit mu model on all data (CatBoost Tweedie, log link).
        Step 2 — Compute OOF mu residuals via K-fold cross-fitting.
        Step 3 — Fit phi model (Gamma deviance) on OOF squared Pearson residuals.

        P0-4 fix: re-apply the training baseline at predict time.

        v0.1.3 fix: OOF cross-fitting prevents overfitting bias in phi residuals.
        In-sample mu_hat ≈ y when CatBoost overfits, making d ≈ 0. OOF mu_hat
        is not fitted to observation i, so E[d_i^{OOF}] = phi_i as intended.
        """
        p = self.power

        mu_params = self._merge_catboost_params(
            self._default_catboost_params(
                f"Tweedie:variance_power={p}", iterations=300
            ),
            self.catboost_params_mu,
        )

        if cycle == 0:
            baseline_mu = np.log(exposure) + np.log(params["mu_init"])
        else:
            baseline_mu = np.log(params["mu"]) + np.log(exposure)

        # Fit full mu model on all data
        self._model_mu = self._fit_catboost(
            X, y, mu_params, baseline=baseline_mu
        )

        # Full in-sample mu_hat for updating params["mu"]
        mu_hat_full = self._predict_catboost(self._model_mu, X, baseline=baseline_mu)
        mu_hat_full = np.clip(mu_hat_full, 1e-6, None)
        params["mu"] = mu_hat_full

        if not self.model_dispersion:
            phi_scalar = _estimate_phi_mle(y, mu_hat_full, p)
            self._phi_scalar = phi_scalar
            params["phi"] = np.full(len(y), phi_scalar)
            return params

        # --- Cross-fitting for unbiased phi residuals ---
        if self.phi_cv_folds > 1:
            mu_hat_oof = self._compute_oof_mu(
                X, y, exposure, params, cycle, baseline_mu, mu_params
            )
        else:
            mu_hat_oof = mu_hat_full

        # Squared Pearson residuals from OOF mu
        d = (y - mu_hat_oof) ** 2 / np.power(mu_hat_oof, p)
        d = np.clip(d, 1e-8, None)

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

        For each fold k: fit mu on the OTHER K-1 folds, predict on fold k.
        Returns per-observation mu_hat that was NOT fitted to that observation.
        """
        n = len(y)
        K = self.phi_cv_folds
        mu_hat_oof = np.zeros(n)

        rng = np.random.default_rng(self.random_state + cycle * 31337)
        fold_idx = rng.permutation(n) % K

        for k in range(K):
            train_mask = fold_idx != k
            val_mask = ~train_mask

            if val_mask.sum() == 0:
                continue

            X_tr = X[train_mask]
            y_tr = y[train_mask]
            base_tr = baseline_mu[train_mask]
            X_val = X[val_mask]
            base_val = baseline_mu[val_mask]

            mu_model_k = self._fit_catboost(
                X_tr, y_tr, mu_params, baseline=base_tr
            )
            mu_hat_k = self._predict_catboost(mu_model_k, X_val, baseline=base_val)
            mu_hat_oof[val_mask] = np.clip(mu_hat_k, 1e-6, None)

            logger.debug(
                "OOF fold %d/%d: n_train=%d, n_val=%d",
                k + 1, K, train_mask.sum(), val_mask.sum()
            )

        return mu_hat_oof

    def _predict_params(
        self, X: np.ndarray, exposure: np.ndarray
    ) -> Dict[str, np.ndarray]:
        mu_hat = self._predict_catboost(
            self._model_mu, X, baseline=np.log(exposure)
        )
        mu_hat = np.clip(mu_hat, 1e-6, None)

        if self.model_dispersion and self._model_phi is not None:
            n = len(X)
            baseline_phi = np.full(n, self._log_phi_init)
            phi_hat = self._predict_catboost(
                self._model_phi, X, baseline=baseline_phi
            )
        else:
            phi_hat = np.full(len(X), self._phi_scalar)

        phi_hat = np.clip(phi_hat, 1e-6, None)
        return {"mu": mu_hat, "phi": phi_hat}

    def _make_prediction(self, params: Dict[str, np.ndarray]) -> DistributionalPrediction:
        return DistributionalPrediction(
            distribution="tweedie",
            mu=params["mu"],
            phi=params["phi"],
            power=self.power,
        )

    def _neg_log_likelihood(self, y: np.ndarray, params: Dict[str, np.ndarray]) -> float:
        ll = _tweedie_log_likelihood(y, params["mu"], params["phi"], self.power)
        return float(-np.mean(ll))

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"TweedieGBM(power={self.power}, "
            f"model_dispersion={self.model_dispersion}, "
            f"phi_cv_folds={self.phi_cv_folds}, "
            f"n_cycles={self.n_cycles}, "
            f"status={status!r})"
        )
