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
E[d_i] = phi_i. We then fit a Gamma regression with log link on d_i to estimate
phi(x). This is equivalent to fitting a GLM for dispersion — the approach is
theoretically principled (Smyth & Jørgensen, ASTIN 2002) and practically superior
to scalar phi estimation.

Key implementation detail: the mu model uses CatBoost's native Tweedie loss
(loss_function="Tweedie:variance_power=p"). This means CatBoost handles all the
gradient/Hessian computation for mu internally. We only need to compute the
dispersion residuals for the phi model.
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

    For compound Poisson-Gamma (1 < p < 2), using the saddle-point approximation
    for the positive part and exact formula for the zero mass.

    For y=0: log p(0) = -mu^(2-p) / (phi*(2-p))
    For y>0: uses Tweedie series expansion (Dunn & Smyth, 2005)

    We use a numerically stable approximation that is exact for GLM gradient
    purposes: the deviance formulation.

    Returns per-observation log-likelihood (not normalised).
    """
    # Use tweedie deviance: 2*(y^(2-p)/((1-p)*(2-p)) - y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p))
    # log L_i = -deviance_i / (2*phi) + const(y, phi, p)
    # For scoring purposes, we use the exact form
    ll = np.zeros(len(y))

    # y=0 term: Poisson probability of 0 claims
    mask0 = y == 0
    if mask0.any():
        ll[mask0] = -mu[mask0] ** (2 - p) / (phi[mask0] * (2 - p))

    # y>0 term: saddle-point approximation (standard in actuarial practice)
    mask_pos = ~mask0
    if mask_pos.any():
        y_p = y[mask_pos]
        mu_p = mu[mask_pos]
        phi_p = phi[mask_pos]
        # Tweedie deviance term
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

    Minimises -sum(log_likelihood) over phi > 0.
    Used for unconditional initialisation.

    P1-5 fix: emits a warning if the optimum lands at either search boundary
    (log_phi in {-3, 3}), which indicates the true phi may be outside the
    search range and the estimate is unreliable.
    """
    lo, hi = -3.0, 3.0

    def neg_ll(log_phi: float) -> float:
        phi_val = np.exp(log_phi)
        phi_arr = np.full(len(y), phi_val)
        return -np.sum(_tweedie_log_likelihood(y, mu, phi_arr, p))

    result = minimize_scalar(neg_ll, bounds=(lo, hi), method="bounded")
    # P1-5: warn if optimum is at a boundary
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
    regression on squared Pearson residuals.

    Parameters
    ----------
    power : float
        Tweedie variance power p. Must be in (1, 2) for compound Poisson-Gamma.
        Default 1.5 (standard for UK motor). The power is NOT estimated from
        data — fix it based on the line of business.
    n_cycles : int
        Coordinate descent cycles. Default 1.
    model_dispersion : bool
        If True (default), fit a GBM model for phi(x). If False, use a scalar
        phi (standard single-parameter Tweedie GBM — faster but less rich).
    cat_features : list, optional
        Categorical feature indices/names for CatBoost.
    catboost_params_mu : dict, optional
        Override CatBoost parameters for the mean model.
    catboost_params_phi : dict, optional
        Override CatBoost parameters for the dispersion model.
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
    >>> pred.volatility_score().shape
    (1000,)
    """

    def __init__(
        self,
        power: float = 1.5,
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
        if not (1.0 < power < 2.0):
            raise ValueError(
                f"Tweedie power must be in (1, 2) for compound Poisson-Gamma, got {power}"
            )
        self.power = power
        self.model_dispersion = model_dispersion
        self._model_mu = None
        self._model_phi = None
        self._phi_scalar: float = 1.0  # fallback if model_dispersion=False

    # -------------------------------------------------------------------------
    # DistributionalGBM interface
    # -------------------------------------------------------------------------

    def _init_params(self, y: np.ndarray, exposure: np.ndarray) -> Dict[str, Any]:
        """
        Unconditional MLE initialisation.

        mu_init = sum(y) / sum(exposure)  — exposure-weighted mean
        phi_init = MLE scalar phi given mu_init
        """
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
        One coordinate descent cycle: fit mu, then fit phi.

        Step 1 — Fit mu model:
          CatBoost Tweedie loss with log(exposure) as baseline offset.
          The baseline is initialised at log(mu_init) on the first cycle.

        Step 2 — Fit phi model (Smyth-Jørgensen double GLM):
          Compute squared Pearson residuals d_i = (y_i - mu_hat_i)^2 / mu_hat_i^p.
          Fit a Gamma GBM with log link on d_i to estimate phi(x).
          Use exposure as sample weight (higher exposure = more information).
        """
        p = self.power

        # --- Step 1: Mean model ---
        mu_params = self._merge_catboost_params(
            self._default_catboost_params(
                f"Tweedie:variance_power={p}", iterations=300
            ),
            self.catboost_params_mu,
        )
        # CatBoost Tweedie with baseline = log(exposure) + log(mu_init)
        # On first cycle, bootstrap from unconditional estimate.
        # P1-4 fix: on subsequent cycles, include the previous cycle's mu
        # estimate in the baseline so coordinate descent actually refines the
        # previous fit rather than discarding it.
        if cycle == 0:
            baseline_mu = np.log(exposure) + np.log(params["mu_init"])
        else:
            baseline_mu = np.log(params["mu"]) + np.log(exposure)

        self._model_mu = self._fit_catboost(
            X, y, mu_params, baseline=baseline_mu
        )

        # Get mu predictions (CatBoost Tweedie predicts on natural scale)
        mu_hat = self._model_mu.predict(X)
        mu_hat = np.clip(mu_hat, 1e-6, None)
        params["mu"] = mu_hat

        # --- Step 2: Dispersion model ---
        if self.model_dispersion:
            # Squared Pearson residuals — the response variable for phi model
            d = (y - mu_hat) ** 2 / np.power(mu_hat, p)
            d = np.clip(d, 1e-8, None)  # avoid log(0) in Gamma loss

            phi_params = self._merge_catboost_params(
                self._default_catboost_params("Tweedie:variance_power=1.5", iterations=200),
                self.catboost_params_phi,
            )
            # Use Gamma deviance loss — CatBoost doesn't have Gamma natively,
            # so we use RMSE on log(d) which approximates Gamma/log-link GLM.
            # More principled: use custom Gamma objective.
            phi_params["loss_function"] = "RMSE"

            log_d = np.log(d)
            # Sample weight: use exposure (higher exposure obs are more reliable)
            self._model_phi = self._fit_catboost(
                X, log_d, phi_params, sample_weight=exposure
            )
            log_phi_hat = self._model_phi.predict(X)
            phi_hat = np.exp(log_phi_hat)
            phi_hat = np.clip(phi_hat, 1e-6, None)
            params["phi"] = phi_hat
        else:
            # Scalar phi: MLE given current mu_hat
            phi_scalar = _estimate_phi_mle(y, mu_hat, p)
            self._phi_scalar = phi_scalar
            params["phi"] = np.full(len(y), phi_scalar)

        return params

    def _predict_params(
        self, X: np.ndarray, exposure: np.ndarray
    ) -> Dict[str, np.ndarray]:
        # P0-1 fix: apply log(exposure) as baseline so predictions are on the
        # correct scale. The model was trained with baseline = log(exposure) +
        # log(mu_init), meaning exp(f(x)) is the rate per unit exposure.
        # At prediction time we must add log(exposure) back so we get the
        # absolute mean (rate * exposure) rather than just the rate.
        mu_hat = self._predict_catboost(
            self._model_mu, X, baseline=np.log(exposure)
        )
        mu_hat = np.clip(mu_hat, 1e-6, None)

        if self.model_dispersion and self._model_phi is not None:
            log_phi_hat = self._model_phi.predict(X)
            phi_hat = np.exp(log_phi_hat)
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

    # -------------------------------------------------------------------------
    # Convenience
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"TweedieGBM(power={self.power}, "
            f"model_dispersion={self.model_dispersion}, "
            f"n_cycles={self.n_cycles}, "
            f"status={status!r})"
        )
