"""
NegBinomialGBM: distributional GBM for Negative Binomial count data.

The Negative Binomial is a Poisson-Gamma mixture: Y|Lambda ~ Poisson(Lambda)
where Lambda ~ Gamma(r, r/mu). This gives:
  P(Y=k) = C(k+r-1, k) * (r/(r+mu))^r * (mu/(r+mu))^k

With E[Y] = mu and Var[Y] = mu + mu^2/r.

The overdispersion parameter r (also called 'size') captures extra-Poisson
variability. Small r means high overdispersion; as r -> inf the distribution
approaches Poisson.

In UK insurance:
- Fleet motor frequency: vehicles have latent propensity heterogeneity
  (some drivers are systematically higher risk than observed features capture)
- Negative Binomial models this latent heterogeneity
- Equivalently, NB is a Poisson with random effects integrated out

Fitting strategy:
1. Fit mu via CatBoost Poisson loss (CatBoost doesn't have NB natively)
2. Fit r by profiling the negative log-likelihood given mu_hat:
   for each obs, compute the MLE of log r via scipy.optimize

Alternative: model log r as a GBM (heterogeneous overdispersion).
We implement both: r_model=True fits a GBM on log(r), r_model=False uses
a scalar r (standard NB with heterogeneous mean only).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from .base import DistributionalGBM
from .prediction import DistributionalPrediction

logger = logging.getLogger(__name__)


def _negbinom_log_likelihood(
    y: np.ndarray, mu: np.ndarray, r: np.ndarray
) -> np.ndarray:
    """
    Negative Binomial log-likelihood per observation.

    log p(y; mu, r) = lgamma(y+r) - lgamma(r) - lgamma(y+1)
                    + r*log(r/(r+mu)) + y*log(mu/(r+mu))
    """
    ll = (
        gammaln(y + r)
        - gammaln(r)
        - gammaln(y + 1)
        + r * np.log(r / (r + mu + 1e-12))
        + y * np.log(mu / (r + mu + 1e-12) + 1e-12)
    )
    return ll


def _estimate_r_mle(y: np.ndarray, mu: np.ndarray) -> float:
    """
    Estimate scalar r by MLE given fixed mu.

    Optimises over log(r) for numerical stability.
    """
    def neg_ll(log_r: float) -> float:
        r_val = np.exp(log_r)
        r_arr = np.full(len(y), r_val)
        return -np.sum(_negbinom_log_likelihood(y, mu, r_arr))

    result = minimize_scalar(neg_ll, bounds=(-2, 8), method="bounded")
    return float(np.exp(result.x))


class NegBinomialGBM(DistributionalGBM):
    """
    Distributional GBM for Negative Binomial count data.

    Models the conditional mean mu(x) via CatBoost Poisson loss.
    The overdispersion parameter r can be:
    - A scalar (r_model=False): standard NB with heterogeneous mean
    - A GBM model of log r (r_model=True): heterogeneous overdispersion

    Parameters
    ----------
    n_cycles : int
        Coordinate descent cycles. Default 1.
    model_r : bool
        If True, fit a GBM for log r(x). If False (default), use scalar r.
        Scalar r is faster and sufficient when overdispersion is uniform.
    cat_features : list, optional
    catboost_params_mu : dict, optional
    catboost_params_phi : dict, optional
        Used for the r model if model_r=True.
    random_state : int

    Examples
    --------
    >>> from insurance_distributional import NegBinomialGBM
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((500, 3))
    >>> mu_true = np.exp(0.3 * X[:, 0])
    >>> y = rng.negative_binomial(n=5, p=5/(5+mu_true))
    >>> model = NegBinomialGBM()
    >>> model.fit(X, y)
    NegBinomialGBM(...)
    >>> pred = model.predict(X)
    >>> pred.variance.shape  # mu + mu^2/r
    (500,)
    """

    def __init__(
        self,
        n_cycles: int = 1,
        model_r: bool = False,
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
        self.model_r = model_r
        self._model_mu = None
        self._model_r = None
        self._r_scalar: float = 5.0
        self._mu_init: float = 0.1

    def _init_params(self, y: np.ndarray, exposure: np.ndarray) -> Dict[str, Any]:
        """
        Unconditional NB initialisation.

        mu_init: exposure-weighted mean
        r_init: MLE scalar r given mu_init
        """
        mu_init = float(np.sum(y) / np.sum(exposure))
        mu_arr = np.full(len(y), mu_init * exposure)
        r_init = _estimate_r_mle(y, mu_arr)
        logger.debug(
            "NegBinomialGBM init: mu_init=%.4f, r_init=%.4f",
            mu_init, r_init
        )
        self._mu_init = mu_init
        self._r_scalar = r_init
        return {
            "mu_init": mu_init,
            "r_init": r_init,
            "mu": mu_arr,
            "r": np.full(len(y), r_init),
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
        Step 1: Fit mu via Poisson GBM (best available proxy for NB mean).
        Step 2: Estimate r (scalar MLE or GBM on log-normalised deviances).
        """
        # --- Step 1: Mean model ---
        mu_params = self._merge_catboost_params(
            self._default_catboost_params("Poisson", iterations=300),
            self.catboost_params_mu,
        )
        if cycle == 0:
            baseline_mu = np.log(exposure) + np.log(self._mu_init)
        else:
            baseline_mu = np.log(exposure)

        self._model_mu = self._fit_catboost(
            X, y, mu_params, baseline=baseline_mu
        )
        mu_hat = np.clip(self._model_mu.predict(X), 1e-6, None)
        params["mu"] = mu_hat

        # --- Step 2: Overdispersion r ---
        if self.model_r:
            # Fit a GBM for log r using deviance residuals
            # NB deviance residual: d_i = 2*(y*log(y/mu) - (y+r)*log((y+r)/(mu+r)))
            # We use a simpler approach: log of NB variance minus Poisson variance
            # = log(mu^2/r), so log(r) = 2*log(mu) - log(Var_excess)
            # Response: log(sample variance approximation)
            r_cur = params["r"]
            # Score residual for r: gradient of NB log-likelihood wrt log(r)
            # dLL/d(log r) = r * (psi(y+r) - psi(r) - log(1 + mu/r))
            from scipy.special import digamma
            r_arr = r_cur
            score_r = r_arr * (
                digamma(y + r_arr) - digamma(r_arr) - np.log(1.0 + mu_hat / r_arr + 1e-12)
            )
            # Pseudo-response for log(r) model: Newton step
            # Information: r^2 * (trigamma(r) - trigamma(y+r))
            from scipy.special import polygamma
            info_r = r_arr ** 2 * (polygamma(1, r_arr) - polygamma(1, y + r_arr + 1e-12))
            info_r = np.clip(np.abs(info_r), 1e-4, None)
            pseudo_r = np.log(r_arr) + score_r / info_r  # Newton step in log space
            pseudo_r = np.clip(pseudo_r, -5, 8)

            r_gparams = self._merge_catboost_params(
                self._default_catboost_params("RMSE", iterations=200),
                self.catboost_params_phi,
            )
            self._model_r = self._fit_catboost(X, pseudo_r, r_gparams)
            log_r_hat = self._model_r.predict(X)
            r_hat = np.exp(np.clip(log_r_hat, -5, 8))
            params["r"] = r_hat
        else:
            r_scalar = _estimate_r_mle(y, mu_hat)
            self._r_scalar = r_scalar
            params["r"] = np.full(len(y), r_scalar)

        return params

    def _predict_params(
        self, X: np.ndarray, exposure: np.ndarray
    ) -> Dict[str, np.ndarray]:
        mu_hat = np.clip(self._model_mu.predict(X), 1e-6, None)

        if self.model_r and self._model_r is not None:
            log_r_hat = self._model_r.predict(X)
            r_hat = np.exp(np.clip(log_r_hat, -5, 8))
        else:
            r_hat = np.full(len(X), self._r_scalar)

        return {"mu": mu_hat, "r": r_hat}

    def _make_prediction(self, params: Dict[str, np.ndarray]) -> DistributionalPrediction:
        return DistributionalPrediction(
            distribution="negbinom",
            mu=params["mu"],
            r=params["r"],
        )

    def _neg_log_likelihood(self, y: np.ndarray, params: Dict[str, np.ndarray]) -> float:
        ll = _negbinom_log_likelihood(y, params["mu"], params["r"])
        return float(-np.mean(ll))

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"NegBinomialGBM("
            f"model_r={self.model_r}, "
            f"n_cycles={self.n_cycles}, "
            f"status={status!r})"
        )
