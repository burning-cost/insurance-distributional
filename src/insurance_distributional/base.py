"""
DistributionalGBM: base class for all distributional gradient boosting models.

The architecture follows the Smyth-Jørgensen double GLM approach extended to
gradient boosting: separate CatBoost models are fitted for each distribution
parameter via coordinate descent. Each subclass implements the distribution-
specific logic (loss functions, gradient computation, parameter initialisation,
variance formulas).

Design choices:
- Separate CatBoostRegressors per parameter, not a joint multi-output tree.
  Rationale: CatBoost's MultiRMSE custom objective has sparse documentation
  and known GPU incompatibilities. Separate models are simpler, more debuggable,
  and compositionally extensible. The statistical cost (no shared tree structure)
  is negligible in practice (XGBoostLSS uses the same approach).

- Coordinate descent, not simultaneous estimation.
  Rationale: Follows the GAMLSS EM algorithm tradition. The So & Valdez ASTIN
  2024 paper validates this for CatBoost. One cycle (default) is sufficient for
  most insurance datasets.

- SciPy for gradient/Hessian computation, not PyTorch.
  Rationale: All four v1 distributions (Tweedie, Gamma, ZIP, NegBinom) have
  well-understood log-likelihoods that SciPy can differentiate numerically with
  high reliability. PyTorch adds ~500MB to the install and Pyro adds more still.
  We avoid both. If a user needs exotic distributions with complex likelihoods,
  they can subclass and override _compute_gradients().

- Unconditional MLE initialisation.
  Rationale: Critical for convergence stability. Both XGBoostLSS and the ASTIN
  paper initialise from the unconditional estimate. We implement this via
  CatBoost's baseline parameter.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def _to_numpy(
    X: Union[np.ndarray, pl.DataFrame, "pd.DataFrame"],  # type: ignore[name-defined]
) -> np.ndarray:
    """Convert Polars DataFrame, pandas DataFrame, or numpy array to numpy."""
    if isinstance(X, pl.DataFrame):
        return X.to_numpy()
    # pandas support without hard dependency
    try:
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            return X.values
    except ImportError:
        pass
    if isinstance(X, np.ndarray):
        return X
    raise TypeError(
        f"Expected np.ndarray or polars.DataFrame, got {type(X).__name__}"
    )


def _to_1d(y: Union[np.ndarray, pl.Series, List]) -> np.ndarray:
    """Convert y to a 1D float64 numpy array."""
    if isinstance(y, pl.Series):
        return y.to_numpy().astype(np.float64)
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"y must be 1-dimensional, got shape {arr.shape}")
    return arr


def _clip_hessians(h: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    Clip Hessians to [eps, inf].

    Near-zero Hessians cause gradient explosion in boosting. This is the
    documented fix from XGBoostLSS and LightGBMLSS — clip at 1e-4 rather
    than 1e-5 to be conservative (insurance data is often heterogeneous).
    """
    return np.clip(h, eps, None)


def _normalize_gradients(g: np.ndarray, h: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalise gradients by K*n per the XGBoostLSS convention.

    This prevents learning-rate sensitivity when the number of parameters K
    or sample size n varies. g and h are divided by (K * n).
    """
    n = len(g)
    scale = K * n
    return g / scale, h / scale


class DistributionalGBM(ABC):
    """
    Abstract base class for distributional gradient boosting models.

    Subclasses must implement:
    - _get_n_params(): number of distribution parameters
    - _init_params(y, exposure): unconditional MLE initialisation
    - _fit_step(X, y, exposure, current_params, step_idx): one CD step
    - _predict_params(X, exposure): return dict of parameter arrays
    - _make_prediction(params): return DistributionalPrediction

    Parameters
    ----------
    n_cycles : int
        Number of coordinate descent cycles. Default 1. More cycles rarely
        help for well-specified models — the residual fitting step captures
        most of the dispersion signal in the first pass.
    cat_features : list of int or str, optional
        Categorical feature indices or names. Passed through to CatBoost.
    catboost_params_mu : dict, optional
        CatBoost parameters for the mean model. Merged with sensible defaults.
    catboost_params_phi : dict, optional
        CatBoost parameters for the dispersion/secondary model.
    random_state : int
        Random seed passed to CatBoost.
    """

    def __init__(
        self,
        n_cycles: int = 1,
        cat_features: Optional[List[Union[int, str]]] = None,
        catboost_params_mu: Optional[Dict[str, Any]] = None,
        catboost_params_phi: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ) -> None:
        self.n_cycles = n_cycles
        self.cat_features = cat_features or []
        self.catboost_params_mu = catboost_params_mu or {}
        self.catboost_params_phi = catboost_params_phi or {}
        self.random_state = random_state
        self._is_fitted = False

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def fit(
        self,
        X: Union[np.ndarray, pl.DataFrame],
        y: Union[np.ndarray, pl.Series, List],
        exposure: Optional[Union[np.ndarray, pl.Series]] = None,
    ) -> "DistributionalGBM":
        """
        Fit the distributional model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix. Polars DataFrame or numpy array.
        y : array-like of shape (n_samples,)
            Target variable (non-negative for insurance losses).
        exposure : array-like of shape (n_samples,), optional
            Exposure measure (years, vehicle-years, etc.). Used as offset in
            the mean model: log(E[Y|x]) = log(exposure) + f(x).
            If None, all observations treated as unit exposure.

        Returns
        -------
        self
        """
        X_np = _to_numpy(X)
        y_np = _to_1d(y)
        n = len(y_np)

        if exposure is not None:
            exp_np = _to_1d(exposure)
            if len(exp_np) != n:
                raise ValueError(
                    f"exposure length {len(exp_np)} != y length {n}"
                )
            if np.any(exp_np <= 0):
                raise ValueError("All exposure values must be positive")
        else:
            exp_np = np.ones(n, dtype=np.float64)

        if np.any(y_np < 0):
            raise ValueError(
                "Negative y values found. Insurance loss models require y >= 0."
            )

        logger.info(
            "Fitting %s on n=%d samples, n_cycles=%d",
            self.__class__.__name__, n, self.n_cycles
        )

        # Initialise parameters from unconditional MLE
        params = self._init_params(y_np, exp_np)

        # Coordinate descent over cycles
        for cycle in range(self.n_cycles):
            logger.debug("Coordinate descent cycle %d/%d", cycle + 1, self.n_cycles)
            params = self._fit_cycle(X_np, y_np, exp_np, params, cycle)

        self._is_fitted = True
        return self

    def predict(
        self,
        X: Union[np.ndarray, pl.DataFrame],
        exposure: Optional[Union[np.ndarray, pl.Series]] = None,
    ) -> "DistributionalPrediction":  # noqa: F821
        """
        Predict the full conditional distribution for new observations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        exposure : array-like of shape (n_samples,), optional

        Returns
        -------
        DistributionalPrediction
        """
        self._check_is_fitted()
        X_np = _to_numpy(X)
        n = len(X_np)

        if exposure is not None:
            exp_np = _to_1d(exposure)
        else:
            exp_np = np.ones(n, dtype=np.float64)

        params = self._predict_params(X_np, exp_np)
        return self._make_prediction(params)

    def log_score(
        self,
        X: Union[np.ndarray, pl.DataFrame],
        y: Union[np.ndarray, pl.Series],
        exposure: Optional[Union[np.ndarray, pl.Series]] = None,
    ) -> float:
        """
        Mean negative log-likelihood score (lower is better).

        Parameters
        ----------
        X, y, exposure : as for fit()

        Returns
        -------
        float
            Mean negative log-likelihood.
        """
        self._check_is_fitted()
        X_np = _to_numpy(X)
        y_np = _to_1d(y)
        n = len(y_np)
        exp_np = _to_1d(exposure) if exposure is not None else np.ones(n)
        params = self._predict_params(X_np, exp_np)
        return self._neg_log_likelihood(y_np, params)

    def crps(
        self,
        X: Union[np.ndarray, pl.DataFrame],
        y: Union[np.ndarray, pl.Series],
        exposure: Optional[Union[np.ndarray, pl.Series]] = None,
        n_samples: int = 2000,
        seed: int = 42,
    ) -> float:
        """
        Mean CRPS (Continuous Ranked Probability Score) via Monte Carlo.

        CRPS is a proper scoring rule in the same units as y. It rewards
        forecasts that assign high probability to the observed outcome while
        penalising overconfident predictions. Strictly proper — minimised by
        the true distribution.

        This implementation uses the MC estimator:
          CRPS(F, y) ≈ E_F[|X - y|] - 0.5 * E_F[|X - X'|]
        where X, X' are independent draws from F. Equivalent to the energy
        score for univariate distributions.

        Parameters
        ----------
        X, y, exposure : as for fit()
        n_samples : int
            MC samples per observation. Default 2000 gives <2% relative error.
        seed : int

        Returns
        -------
        float
            Mean CRPS across all observations.
        """
        self._check_is_fitted()
        pred = self.predict(X, exposure)
        y_np = _to_1d(y)
        rng = np.random.default_rng(seed)
        samples = pred._sample(n_samples=n_samples, rng=rng)  # (n, n_samples)
        # E_F[|X - y|]
        term1 = np.abs(samples - y_np[:, None]).mean(axis=1)
        # 0.5 * E_F[|X - X'|] — split samples in two
        half = n_samples // 2
        term2 = 0.5 * np.abs(samples[:, :half] - samples[:, half:2*half]).mean(axis=1)
        return float(np.mean(term1 - term2))

    # -------------------------------------------------------------------------
    # Abstract methods — subclasses must implement
    # -------------------------------------------------------------------------

    @abstractmethod
    def _init_params(
        self, y: np.ndarray, exposure: np.ndarray
    ) -> Dict[str, Any]:
        """
        Initialise distribution parameters from unconditional MLE.

        Returns a dict with at minimum 'mu_init' (scalar or array).
        Used to set CatBoost baselines for stable boosting start.
        """
        ...

    @abstractmethod
    def _fit_cycle(
        self,
        X: np.ndarray,
        y: np.ndarray,
        exposure: np.ndarray,
        params: Dict[str, Any],
        cycle: int,
    ) -> Dict[str, Any]:
        """
        Execute one coordinate descent cycle. Returns updated params dict."""
        ...

    @abstractmethod
    def _predict_params(
        self, X: np.ndarray, exposure: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Return dict of predicted parameter arrays for each obs."""
        ...

    @abstractmethod
    def _make_prediction(self, params: Dict[str, np.ndarray]) -> "DistributionalPrediction":  # noqa: F821
        """Construct DistributionalPrediction from parameter dict."""
        ...

    @abstractmethod
    def _neg_log_likelihood(self, y: np.ndarray, params: Dict[str, np.ndarray]) -> float:
        """Mean negative log-likelihood."""
        ...

    # -------------------------------------------------------------------------
    # Helpers for subclasses
    # -------------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted. Call fit() first."
            )

    def _default_catboost_params(
        self, loss: str, iterations: int = 300
    ) -> Dict[str, Any]:
        """Return sensible default CatBoost parameters for a given loss."""
        return {
            "loss_function": loss,
            "iterations": iterations,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_seed": self.random_state,
            "verbose": False,
            "allow_writing_files": False,
        }

    def _merge_catboost_params(
        self, defaults: Dict[str, Any], overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge user-supplied CatBoost params over defaults."""
        merged = dict(defaults)
        merged.update(overrides)
        return merged

    def _fit_catboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict[str, Any],
        baseline: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ):
        """
        Fit a CatBoostRegressor with the given parameters.

        baseline sets the initial approximation (log scale for Tweedie/Gamma,
        used for unconditional MLE initialisation).
        """
        from catboost import CatBoostRegressor, Pool

        pool_kwargs: Dict[str, Any] = {
            "data": X,
            "label": y,
        }
        if self.cat_features:
            pool_kwargs["cat_features"] = self.cat_features
        if baseline is not None:
            pool_kwargs["baseline"] = baseline
        if sample_weight is not None:
            pool_kwargs["weight"] = sample_weight

        pool = Pool(**pool_kwargs)
        model = CatBoostRegressor(**params)
        model.fit(pool)
        return model

    def _predict_catboost(
        self,
        model,
        X: np.ndarray,
        baseline: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict with CatBoost, applying baseline offset."""
        from catboost import Pool

        pool_kwargs: Dict[str, Any] = {"data": X}
        if self.cat_features:
            pool_kwargs["cat_features"] = self.cat_features
        if baseline is not None:
            pool_kwargs["baseline"] = baseline

        pool = Pool(**pool_kwargs)
        return model.predict(pool)
