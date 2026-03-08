"""
ZIPGBM: distributional GBM for Zero-Inflated Poisson.

Zero-inflated models are the right choice for lines with structural zeros:
- Pet insurance: many pets never claim in a year (structural zeros)
- Travel insurance: most trips are uneventful
- Breakdown cover: many vehicles never break down

The ZIP model has two components:
  P(Y=0 | x) = pi(x) + (1-pi(x)) * exp(-lambda(x))
  P(Y=k | x) = (1-pi(x)) * Poisson(k; lambda(x))   for k > 0

where pi(x) is the zero-inflation probability and lambda(x) is the Poisson
rate for the non-inflated component.

E[Y|x] = (1-pi(x)) * lambda(x)   [the reported mean]
Var[Y|x] = (1-pi(x))*lambda(x) + pi(x)*(1-pi(x))*lambda(x)^2

Implementation follows the two-stage approach from So (2023, arXiv 2307.07771),
specifically Scenario 2 (ZIPB2 — functionally unrelated mu and pi):
1. Fit a Poisson GBM on non-zero observations to estimate lambda(x)
2. Fit a logistic classifier on all observations to estimate pi(x)

The two-stage approach is simpler to implement correctly than the joint
coordinate descent and gives comparable performance for most insurance datasets.
For the joint estimation approach, see So & Valdez (2024, arXiv 2406.16206).

Note on lambda vs mu: internally we track lambda (the Poisson rate). The
predicted mean is mu = (1-pi)*lambda. The DistributionalPrediction.mu property
returns the observable mean (1-pi)*lambda, not lambda itself.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson, bernoulli

from .base import DistributionalGBM, _to_1d
from .prediction import DistributionalPrediction

logger = logging.getLogger(__name__)


def _zip_log_likelihood(
    y: np.ndarray, lam: np.ndarray, pi: np.ndarray
) -> np.ndarray:
    """
    ZIP log-likelihood per observation.

    y=0: log(pi + (1-pi)*exp(-lam))
    y>0: log(1-pi) + y*log(lam) - lam - lgamma(y+1)
    """
    from scipy.special import gammaln
    ll = np.zeros(len(y))

    mask0 = y == 0
    if mask0.any():
        # log(pi + (1-pi)*exp(-lam)) — numerically stable
        pois_zero = np.exp(-lam[mask0])
        ll[mask0] = np.log(pi[mask0] + (1.0 - pi[mask0]) * pois_zero + 1e-12)

    mask_pos = ~mask0
    if mask_pos.any():
        y_p = y[mask_pos]
        lam_p = lam[mask_pos]
        pi_p = pi[mask_pos]
        ll[mask_pos] = (
            np.log(1.0 - pi_p + 1e-12)
            + y_p * np.log(lam_p + 1e-12)
            - lam_p
            - gammaln(y_p + 1)
        )

    return ll


class ZIPGBM(DistributionalGBM):
    """
    Distributional GBM for Zero-Inflated Poisson counts.

    Jointly estimates:
    - lambda(x): Poisson rate via CatBoost Poisson loss
    - pi(x): zero-inflation probability via CatBoost Logloss (logistic)

    The predicted mean is mu = (1-pi)*lambda.

    Parameters
    ----------
    n_cycles : int
        Coordinate descent cycles. Default 1.
    cat_features : list, optional
    catboost_params_mu : dict, optional
        Parameters for the lambda (Poisson) model.
    catboost_params_phi : dict, optional
        Parameters for the pi (logistic) model.
    random_state : int

    Examples
    --------
    >>> from insurance_distributional import ZIPGBM
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal((800, 4))
    >>> # Simulate ZIP: 40% structural zeros, rest Poisson(0.3)
    >>> pi_true = 0.4
    >>> lam_true = 0.3
    >>> y = np.where(rng.random(800) < pi_true, 0, rng.poisson(lam_true, 800))
    >>> model = ZIPGBM()
    >>> model.fit(X, y)
    ZIPGBM(...)
    >>> pred = model.predict(X)
    >>> pred.pi.mean()  # should be close to 0.4
    """

    def __init__(
        self,
        n_cycles: int = 1,
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
        self._model_lambda = None
        self._model_pi = None
        self._lam_init: float = 0.1
        self._pi_init: float = 0.1

    def _init_params(self, y: np.ndarray, exposure: np.ndarray) -> Dict[str, Any]:
        """
        Unconditional initialisation.

        pi_init: method-of-moments from excess zeros.
          Observed zero rate = pi + (1-pi)*exp(-lam)
          Standard Poisson rate from non-zeros = mean(y[y>0]).

        lam_init: mean of non-zero observations (rate per unit exposure).
        """
        n = len(y)
        n_zeros = int(np.sum(y == 0))
        obs_zero_rate = n_zeros / n

        lam_nonzero = float(np.mean(y[y > 0])) if np.any(y > 0) else 0.1
        lam_nonzero = max(lam_nonzero, 1e-4)

        # Poisson zero probability at lam_nonzero
        poisson_zero = np.exp(-lam_nonzero)
        # Solve: obs_zero_rate = pi + (1-pi)*poisson_zero
        # => pi = (obs_zero_rate - poisson_zero) / (1 - poisson_zero)
        if obs_zero_rate > poisson_zero:
            pi_init = (obs_zero_rate - poisson_zero) / (1.0 - poisson_zero + 1e-12)
            pi_init = float(np.clip(pi_init, 1e-4, 0.99))
        else:
            pi_init = 1e-4  # no excess zeros, standard Poisson

        logger.debug(
            "ZIPGBM init: lam_init=%.4f, pi_init=%.4f (obs_zero_rate=%.3f)",
            lam_nonzero, pi_init, obs_zero_rate
        )
        self._lam_init = lam_nonzero
        self._pi_init = pi_init

        return {
            "lam_init": lam_nonzero,
            "pi_init": pi_init,
            "lam": np.full(n, lam_nonzero),
            "pi": np.full(n, pi_init),
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
        Two-stage coordinate descent for ZIP.

        Stage 1 — Lambda model:
          Fit CatBoost Poisson regression on all observations.
          The zero-inflation is handled implicitly: the Poisson model
          will underfit the zero-heavy data, and the pi model captures
          the residual zero probability.

        Stage 2 — Pi model:
          Create binary labels: y_binary = 1 if the zero is "excess"
          (i.e., the Poisson model assigns very low probability to observing
          this many zeros), 0 otherwise.
          Simpler alternative we use: fit logistic regression on whether
          observation is zero. The pi model learns to separate structural
          zeros from Poisson-distributed zeros.
        """
        n = len(y)
        pi_cur = params["pi"]
        lam_cur = params["lam"]

        # --- Stage 1: Lambda model (Poisson GBM on all obs) ---
        lam_params = self._merge_catboost_params(
            self._default_catboost_params("Poisson", iterations=300),
            self.catboost_params_mu,
        )
        if cycle == 0:
            baseline_lam = np.log(exposure) + np.log(self._lam_init)
        else:
            baseline_lam = np.log(exposure)

        # Use (1-pi_cur) as sample weight: observations likely to be structural
        # zeros contribute less to lambda estimation.
        # This is the E-step of the EM approach (Gu 2024).
        poisson_zero_prob = np.exp(-lam_cur)
        # Probability that y_i=0 is from Poisson (not structural zero)
        # P(not structural | y=0) = (1-pi)*pois_zero / (pi + (1-pi)*pois_zero)
        w_lambda = np.ones(n)
        mask0 = y == 0
        if mask0.any():
            denom = pi_cur[mask0] + (1.0 - pi_cur[mask0]) * poisson_zero_prob[mask0] + 1e-12
            w_lambda[mask0] = (1.0 - pi_cur[mask0]) * poisson_zero_prob[mask0] / denom

        self._model_lambda = self._fit_catboost(
            X, y, lam_params, baseline=baseline_lam, sample_weight=w_lambda
        )
        lam_hat = self._model_lambda.predict(X)
        lam_hat = np.clip(lam_hat, 1e-6, None)
        params["lam"] = lam_hat

        # --- Stage 2: Pi model (logistic on binary zero indicator) ---
        # We model P(Y=0 is structural | x) via logistic regression.
        # The "soft" label approach: for y=0, posterior probability of being
        # structural zero. For y>0, label is 0 (definitely not structural).
        pi_labels = np.zeros(n)
        if mask0.any():
            pois_zero_new = np.exp(-lam_hat[mask0])
            denom2 = pi_cur[mask0] + (1.0 - pi_cur[mask0]) * pois_zero_new + 1e-12
            pi_labels[mask0] = pi_cur[mask0] / denom2
        # Clip labels to valid range for Logloss
        pi_labels = np.clip(pi_labels, 1e-6, 1 - 1e-6)

        pi_params = self._merge_catboost_params(
            self._default_catboost_params("Logloss", iterations=200),
            self.catboost_params_phi,
        )
        if cycle == 0:
            init_logit = np.log(self._pi_init / (1.0 - self._pi_init + 1e-12))
            baseline_pi = np.full(n, init_logit)
        else:
            baseline_pi = None

        self._model_pi = self._fit_catboost(
            X, pi_labels, pi_params, baseline=baseline_pi
        )
        # CatBoost Logloss outputs probabilities
        pi_hat = self._model_pi.predict(X)
        pi_hat = np.clip(pi_hat, 1e-6, 1 - 1e-6)
        params["pi"] = pi_hat

        return params

    def _predict_params(
        self, X: np.ndarray, exposure: np.ndarray
    ) -> Dict[str, np.ndarray]:
        lam_hat = np.clip(self._model_lambda.predict(X), 1e-6, None)
        pi_hat = np.clip(self._model_pi.predict(X), 1e-6, 1 - 1e-6)
        # Observable mean
        mu_hat = (1.0 - pi_hat) * lam_hat
        return {"mu": mu_hat, "lam": lam_hat, "pi": pi_hat}

    def _make_prediction(self, params: Dict[str, np.ndarray]) -> DistributionalPrediction:
        return DistributionalPrediction(
            distribution="zip",
            mu=params["mu"],
            pi=params["pi"],
        )

    def _neg_log_likelihood(self, y: np.ndarray, params: Dict[str, np.ndarray]) -> float:
        lam = params.get("lam", params["mu"] / (1.0 - params["pi"] + 1e-12))
        ll = _zip_log_likelihood(y, lam, params["pi"])
        return float(-np.mean(ll))

    def predict_lambda(
        self,
        X: Union[np.ndarray, "pl.DataFrame"],
        exposure: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict the underlying Poisson rate lambda(x).

        This is the rate for the non-inflated component. The observed mean
        is (1-pi)*lambda. Use lambda when comparing to standard Poisson
        frequency models.
        """
        from .base import _to_numpy
        self._check_is_fitted()
        X_np = _to_numpy(X)
        lam = np.clip(self._model_lambda.predict(X_np), 1e-6, None)
        return lam

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"ZIPGBM(n_cycles={self.n_cycles}, status={status!r})"
