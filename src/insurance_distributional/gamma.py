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

This is the squared relative error — proportional to the coefficient of
variation squared.

Baseline handling:
  CatBoost Tweedie loss uses a log link internally. When training with a
  Pool(baseline=b), CatBoost minimises Tweedie(y, exp(b + f(x))). The model
  stores only f(x) (the tree contributions); at predict time, model.predict(X)
  returns exp(f(x)) NOT exp(b + f(x)). The caller must re-apply the baseline.

  For GammaGBM (no exposure offset): we train with baseline = log(mu_init),
  store _log_mu_init, and at predict time call _predict_catboost with that
  baseline so predictions come back on the correct scale.
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
    # log p = (shape-1)*log(y) - y/scale - shape*log(scale) - lgamma(shape)
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
    phi = 1/(shape) is the Gamma dispersion parameter.

    P1-5 fix: emits a warning if the optimum lands at either search boundary
    (log_phi in {-5, 3}), indicating the true phi may be outside the range.
    """
    lo, hi = -5.0, 3.0

    def neg_ll(log_phi: float) -> float:
        phi_val = np.exp(log_phi)
        phi_arr = np.full(len(y), phi_val)
        return -np.sum(_gamma_log_likelihood(y, mu, phi_arr))

    result = minimize_scalar(neg_ll, bounds=(lo, hi), method="bounded")
    # P1-5: warn if optimum is at a boundary
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
        self._model_mu = None
        self._model_phi = None
        self._phi_scalar: float = 1.0
        # P0-4 fix: store the log-scale baseline used when training the mu model.
        # CatBoost Tweedie with a training baseline does NOT include the baseline
        # in model.predict() output — it must be re-applied at inference time.
        self._log_mu_init: float = 0.0

    def _init_params(self, y: np.ndarray, exposure: np.ndarray) -> Dict[str, Any]:
        """
        Unconditional MLE for Gamma parameters.

        mu_init = mean(y) (exposure-weighted; Gamma severity typically ignores
        exposure or uses weight)
        phi_init = Var(y)/mean(y)^2 (method-of-moments starting point)
        """
        # For severity, exposure is typically the claim count, used as weight
        # Simple initialisation: use all-ones exposure
        mu_init = float(np.mean(y))
        # Method of moments for phi: CoV^2 = Var/mu^2 = phi
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
        Step 1: Fit mu via CatBoost Gamma deviance (log link).
        Step 2: Fit phi via RMSE on log(squared relative error).

        P0-4 fix: mu model is fitted with a log-scale baseline (log(mu_init)
        on cycle 0, or log of previous-cycle mu on later cycles). CatBoost
        stores only the tree increments f(x), not the baseline, so predict()
        without a baseline returns exp(f(x)) on the wrong scale. We must call
        _predict_catboost with the same baseline to recover the correct mu.
        """
        # CatBoost uses RMSE by default; for Gamma we approximate log-link
        # via LogLinQuantile or by fitting RMSE on log(y) with offset.
        # More accurate: use Poisson loss as approximation to Gamma log-link.
        # Best available: Tweedie with p approaching 2 approximates Gamma deviance.

        mu_params = self._merge_catboost_params(
            self._default_catboost_params("Tweedie:variance_power=1.99", iterations=300),
            self.catboost_params_mu,
        )

        if cycle == 0:
            baseline_mu = np.full(len(y), np.log(params["mu_init"]))
            # Store the scalar log-baseline for use in _predict_params.
            # On cycle 0 the baseline is constant (scalar), so one value suffices.
            self._log_mu_init = float(np.log(params["mu_init"]))
        else:
            # P1-4 fix: use log of the previous cycle's mu estimate so that
            # coordinate descent refines rather than restarting from zero.
            # On later cycles the baseline is per-observation, so we store
            # the per-obs array on self for _predict_params to use.
            baseline_mu = np.log(np.clip(params["mu"], 1e-6, None))
            # After the last cycle, _predict_params uses this stored baseline.
            self._baseline_mu_train = baseline_mu.copy()

        self._model_mu = self._fit_catboost(
            X, y, mu_params, baseline=baseline_mu
        )

        # P0-4 fix: apply the training baseline when extracting mu_hat.
        # Without this, model.predict(X) returns exp(f(x)) ≈ 1.0 instead of
        # exp(baseline + f(x)) ≈ mu_true. The corrupted mu_hat then feeds into
        # phi residuals d = (y - mu_hat)^2 / mu_hat^2, which are ~10^6 too large,
        # causing log_d ≈ 14 and phi_hat = exp(14) ≈ 10^6.
        mu_hat = self._predict_catboost(self._model_mu, X, baseline=baseline_mu)
        mu_hat = np.clip(mu_hat, 1e-6, None)
        params["mu"] = mu_hat

        if self.model_dispersion:
            # Squared relative errors (Gamma variance function V(mu) = mu^2)
            d = (y - mu_hat) ** 2 / (mu_hat ** 2)
            d = np.clip(d, 1e-8, None)
            log_d = np.log(d)

            phi_params = self._merge_catboost_params(
                self._default_catboost_params("RMSE", iterations=200),
                self.catboost_params_phi,
            )
            self._model_phi = self._fit_catboost(
                X, log_d, phi_params, sample_weight=exposure
            )
            log_phi_hat = self._model_phi.predict(X)
            phi_hat = np.exp(log_phi_hat)
            phi_hat = np.clip(phi_hat, 1e-6, None)
            params["phi"] = phi_hat
        else:
            phi_scalar = _estimate_phi_gamma_mle(y, mu_hat)
            self._phi_scalar = phi_scalar
            params["phi"] = np.full(len(y), phi_scalar)

        return params

    def _predict_params(
        self, X: np.ndarray, exposure: np.ndarray
    ) -> Dict[str, np.ndarray]:
        # P0-4 fix: apply the stored log-baseline so mu predictions are on the
        # correct scale. The mu model was trained with baseline = log(mu_init),
        # so at predict time we must pass the same scalar offset as a broadcast
        # array. For GammaGBM there is no exposure offset (unlike TweedieGBM).
        n = len(X)
        baseline = np.full(n, self._log_mu_init)
        mu_hat = np.clip(
            self._predict_catboost(self._model_mu, X, baseline=baseline),
            1e-6, None
        )

        if self.model_dispersion and self._model_phi is not None:
            phi_hat = np.exp(self._model_phi.predict(X))
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
            f"n_cycles={self.n_cycles}, "
            f"status={status!r})"
        )
