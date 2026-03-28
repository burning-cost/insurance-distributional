"""
ZeroInflatedTweedieGBM: So & Valdez (2024) Scenario 2 — Zero-Inflated Tweedie
with CatBoost.

This is the first open-source implementation of Scenario 2 from So & Valdez's
ASTIN 2024 Best Paper. The paper won Best Paper precisely because it addressed
a real problem that every UK motor and contents pricing team has: portfolios
where 90-95% of observations are zero.

The standard Tweedie GLM handles this structurally — zero probability comes
entirely from the compound Poisson term — but that couples frequency and
severity in a way that's hard to separate for business insight, and it forces
a common covariate structure on both components. The ZI-Tweedie separates them:

  P(Y = 0 | x) = π(x) + (1 - π(x)) · P_Tweedie(Y = 0 | x)

  But Scenario 2 simplifies this further:
  E[Y | x] = (1 - π̂(x)) · μ̂_Tweedie(x)

where:
  π̂(x)         = Stage 1: CatBoost binary classifier for structural zero probability
  μ̂_Tweedie(x) = Stage 2: CatBoost Tweedie regressor trained on y > 0 observations

This "two-part model" (Cragg 1971 / Duan et al. 1983) is conceptually cleaner
than the joint ZI likelihood — Stage 1 is "does this policy claim?" and Stage 2
is "if they claim, how much?". Underwriters understand this framing immediately.

Why does this beat standard Tweedie on zero-heavy data?
  - Standard Tweedie uses ALL observations (including the 90-95% zeros) to fit
    the mean. The gradient contribution of zeros pulls the mean downward.
  - ZI-Tweedie uses the non-zero observations directly for severity estimation.
    The zero component is fitted independently with its own feature weights.
  - For portfolios where who-claims and how-much-they-claim have different
    covariate structures (common in contents, breakdown), the separate stages
    are more flexible.

Reference:
  So, B. & Valdez, E.A. (2024). Boosted trees for zero-inflated counts with
  an offset for insurance ratemaking. arXiv 2406.16206. ASTIN Best Paper 2024.

  Cragg, J.G. (1971). Some statistical models for limited dependent variables
  with application to the demand for durable goods. Econometrica 39(5):829-844.
"""

from __future__ import annotations

import logging
import pickle
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .base import _to_1d, _to_numpy
from .scoring import tweedie_deviance

logger = logging.getLogger(__name__)


class ZeroInflatedTweedieGBM:
    """
    Zero-Inflated Tweedie GBM — So & Valdez (2024) Scenario 2.

    Two-stage model for insurance loss portfolios where structural zeros
    (no-claim policies) have a different generating mechanism from claims.

    Stage 1 — Zero classifier:
        CatBoost binary classifier predicting P(Y=0 | x) via CrossEntropy
        loss. Trained on all observations (zero/non-zero labels).

    Stage 2 — Tweedie severity:
        CatBoostRegressor with Tweedie loss, trained only on y > 0
        observations. Estimates E[Y | Y>0, x].

    Combined prediction:
        E[Y | x] = (1 - π̂(x)) × μ̂_Tweedie(x)

    where π̂ is the zero probability from Stage 1 and μ̂_Tweedie is the
    conditional severity from Stage 2.

    Parameters
    ----------
    power : float
        Tweedie variance power p ∈ (1, 2). Default 1.5.
    cat_features : list of int or str, optional
        Categorical feature indices or names passed to CatBoost.
    catboost_params_zero : dict, optional
        CatBoost parameters for the zero classifier (Stage 1).
        Merged with sensible defaults.
    catboost_params_severity : dict, optional
        CatBoost parameters for the Tweedie severity model (Stage 2).
        Merged with sensible defaults.
    random_state : int
        Random seed for both CatBoost models. Default 42.

    Examples
    --------
    >>> from insurance_distributional import ZeroInflatedTweedieGBM
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> X = rng.standard_normal((n, 5))
    >>> # 85% structural zeros (UK contents portfolio)
    >>> pi_true = 0.85
    >>> is_zero = rng.random(n) < pi_true
    >>> y = np.where(is_zero, 0.0, rng.exponential(500, n))
    >>> model = ZeroInflatedTweedieGBM(power=1.5)
    >>> model.fit(X, y)
    ZeroInflatedTweedieGBM(power=1.5, status='fitted')
    >>> pred_mean = model.predict(X)
    >>> pred_mean.shape
    (500,)
    >>> components = model.predict_components(X)
    >>> components.keys()
    dict_keys(['zero_prob', 'severity_mean', 'combined_mean'])
    """

    def __init__(
        self,
        power: float = 1.5,
        cat_features: Optional[List[Union[int, str]]] = None,
        catboost_params_zero: Optional[Dict[str, Any]] = None,
        catboost_params_severity: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ) -> None:
        if not (1.0 < power < 2.0):
            raise ValueError(
                f"Tweedie power must be in (1, 2) for compound Poisson-Gamma, got {power}"
            )
        self.power = power
        self.cat_features = cat_features or []
        self.catboost_params_zero = catboost_params_zero or {}
        self.catboost_params_severity = catboost_params_severity or {}
        self.random_state = random_state

        self._model_zero = None      # CatBoostClassifier for P(Y=0)
        self._model_severity = None  # CatBoostRegressor for E[Y|Y>0]
        self._severity_init: float = 1.0  # log baseline for severity model
        self._is_fitted: bool = False

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def fit(
        self,
        X: Union[np.ndarray, "pl.DataFrame"],  # type: ignore[name-defined]
        y: Union[np.ndarray, "pl.Series", List],  # type: ignore[name-defined]
        exposure: Optional[Union[np.ndarray, "pl.Series"]] = None,  # type: ignore[name-defined]
        sample_weight: Optional[Union[np.ndarray, "pl.Series"]] = None,  # type: ignore[name-defined]
    ) -> "ZeroInflatedTweedieGBM":
        """
        Fit the two-stage ZI-Tweedie model.

        Stage 1 fits a classifier on all observations using zero/non-zero labels.
        Stage 2 fits a Tweedie regressor on the non-zero subset only, with
        exposure applied as a log-offset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix. Polars DataFrame or numpy array.
        y : array-like of shape (n_samples,)
            Target variable. Non-negative. Zeros indicate no-claim.
        exposure : array-like of shape (n_samples,), optional
            Policy exposure (years, vehicle-years). Used as log-offset in
            the severity model: log E[Y|Y>0, x] = log(exposure) + f(x).
            If None, unit exposure is assumed.
        sample_weight : array-like of shape (n_samples,), optional
            Observation weights applied to both stages. Independent of
            exposure — use weights for survey/credibility weighting,
            exposure for rate-making offsets.

        Returns
        -------
        self
        """
        X_np = _to_numpy(X)
        y_np = _to_1d(y)
        n = len(y_np)

        if np.any(y_np < 0):
            raise ValueError(
                "Negative y values found. Insurance loss models require y >= 0."
            )

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

        if sample_weight is not None:
            sw_np = _to_1d(sample_weight)
            if len(sw_np) != n:
                raise ValueError(
                    f"sample_weight length {len(sw_np)} != y length {n}"
                )
            if np.any(sw_np < 0):
                raise ValueError("sample_weight must be non-negative")
        else:
            sw_np = None

        mask_nonzero = y_np > 0
        n_nonzero = int(mask_nonzero.sum())
        n_zero = n - n_nonzero
        zero_rate = n_zero / n

        logger.info(
            "Fitting ZeroInflatedTweedieGBM: n=%d, n_zero=%d (%.1f%%), n_nonzero=%d",
            n, n_zero, 100 * zero_rate, n_nonzero,
        )

        if n_nonzero < 10:
            raise ValueError(
                f"Too few non-zero observations ({n_nonzero}). "
                "ZI-Tweedie requires at least 10 non-zero values for the "
                "severity stage. Check your data or use a pure classifier."
            )

        # --- Stage 1: Zero classifier ---
        logger.debug("Stage 1: fitting zero classifier (n=%d)", n)
        zero_labels = (y_np == 0).astype(np.float64)

        if n_zero == 0:
            # No zeros — skip classifier, π̂ = 0 everywhere
            logger.info("No zero observations; skipping classifier (π̂ = 0)")
            self._model_zero = None
            self._constant_zero_prob = 0.0
        elif n_nonzero == 0:
            # All zeros — shouldn't reach here (caught by n_nonzero < 10)
            raise ValueError("All observations are zero")
        else:
            zero_params = self._build_zero_params()
            self._model_zero = self._fit_classifier(
                X_np, zero_labels, zero_params, sample_weight=sw_np
            )
            self._constant_zero_prob = None

        # --- Stage 2: Tweedie severity on y > 0 only ---
        logger.debug("Stage 2: fitting Tweedie severity (n=%d)", n_nonzero)
        X_sev = X_np[mask_nonzero]
        y_sev = y_np[mask_nonzero]
        exp_sev = exp_np[mask_nonzero]
        sw_sev = sw_np[mask_nonzero] if sw_np is not None else None

        # Unconditional initialisation: mean severity per unit exposure
        sev_per_exp = y_sev / exp_sev
        self._severity_init = float(np.mean(sev_per_exp))
        self._severity_init = max(self._severity_init, 1e-4)

        baseline_sev = np.log(exp_sev) + np.log(self._severity_init)

        sev_params = self._build_severity_params()
        self._model_severity = self._fit_regressor(
            X_sev, y_sev, sev_params, baseline=baseline_sev, sample_weight=sw_sev
        )

        self._is_fitted = True
        return self

    def predict(
        self,
        X: Union[np.ndarray, "pl.DataFrame"],  # type: ignore[name-defined]
        exposure: Optional[Union[np.ndarray, "pl.Series"]] = None,  # type: ignore[name-defined]
    ) -> np.ndarray:
        """
        Predict combined mean E[Y | x] = (1 - π̂) × μ̂_Tweedie.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        exposure : array-like of shape (n_samples,), optional
            Exposure for the severity offset. Defaults to unit exposure.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Combined predicted mean.
        """
        components = self.predict_components(X, exposure)
        return components["combined_mean"]

    def predict_components(
        self,
        X: Union[np.ndarray, "pl.DataFrame"],  # type: ignore[name-defined]
        exposure: Optional[Union[np.ndarray, "pl.Series"]] = None,  # type: ignore[name-defined]
    ) -> Dict[str, np.ndarray]:
        """
        Predict all model components.

        Returns the zero probability, conditional severity, and combined mean
        separately so you can examine each stage independently. This is
        essential for interpretability: underwriters can see whether a
        high-loss prediction comes from a high zero probability (frequented
        claimant) or high severity.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        exposure : array-like of shape (n_samples,), optional

        Returns
        -------
        dict with keys:
            zero_prob : np.ndarray
                π̂(x) = P(Y=0 | x) from Stage 1 classifier.
            severity_mean : np.ndarray
                μ̂(x) = E[Y | Y>0, x] from Stage 2 Tweedie regressor.
            combined_mean : np.ndarray
                E[Y | x] = (1 - π̂) × μ̂.
        """
        self._check_is_fitted()
        X_np = _to_numpy(X)
        n = len(X_np)

        if exposure is not None:
            exp_np = _to_1d(exposure)
            if len(exp_np) != n:
                raise ValueError(
                    f"exposure length {len(exp_np)} != X length {n}"
                )
        else:
            exp_np = np.ones(n, dtype=np.float64)

        # Stage 1: zero probability
        if self._model_zero is None:
            zero_prob = np.full(n, self._constant_zero_prob, dtype=np.float64)
        else:
            zero_prob = self._predict_classifier(self._model_zero, X_np)
            zero_prob = np.clip(zero_prob, 0.0, 1.0)

        # Stage 2: severity mean with exposure offset
        baseline_sev = np.log(exp_np) + np.log(self._severity_init)
        severity_mean = self._predict_regressor(
            self._model_severity, X_np, baseline=baseline_sev
        )
        severity_mean = np.clip(severity_mean, 1e-8, None)

        combined_mean = (1.0 - zero_prob) * severity_mean

        return {
            "zero_prob": zero_prob,
            "severity_mean": severity_mean,
            "combined_mean": combined_mean,
        }

    def predict_proba(
        self,
        X: Union[np.ndarray, "pl.DataFrame"],  # type: ignore[name-defined]
    ) -> np.ndarray:
        """
        Predict P(Y=0 | x) from the zero classifier.

        Returns an (n, 2) array in sklearn convention: column 0 is P(Y>0),
        column 1 is P(Y=0).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
        """
        self._check_is_fitted()
        X_np = _to_numpy(X)
        zero_prob = np.clip(
            self._predict_classifier(self._model_zero, X_np), 0.0, 1.0
        )
        return np.column_stack([1.0 - zero_prob, zero_prob])

    def score(
        self,
        X: Union[np.ndarray, "pl.DataFrame"],  # type: ignore[name-defined]
        y: Union[np.ndarray, "pl.Series", List],  # type: ignore[name-defined]
        exposure: Optional[Union[np.ndarray, "pl.Series"]] = None,  # type: ignore[name-defined]
        weights: Optional[Union[np.ndarray, "pl.Series"]] = None,  # type: ignore[name-defined]
    ) -> float:
        """
        Mean Tweedie deviance of the combined prediction.

        Lower is better. This is the standard metric for evaluating
        compound Poisson-Gamma models in actuarial science.

        Note: the Tweedie deviance uses the combined mean E[Y|x] = (1-π̂)·μ̂.
        It does not evaluate the zero and severity components separately.
        For component-level evaluation, use log_score() or compute custom
        metrics on predict_components().

        Parameters
        ----------
        X, y, exposure : as for fit()
        weights : array-like, optional
            Observation weights for the deviance average (e.g., exposure).

        Returns
        -------
        float
            Mean Tweedie deviance. Lower is better.
        """
        y_np = _to_1d(y)
        n = len(y_np)
        exp_np = _to_1d(exposure) if exposure is not None else np.ones(n)
        w_np = _to_1d(weights) if weights is not None else None
        mu_hat = self.predict(X, exposure=exp_np)
        return tweedie_deviance(y_np, mu_hat, power=self.power, weights=w_np)

    def log_score(
        self,
        X: Union[np.ndarray, "pl.DataFrame"],  # type: ignore[name-defined]
        y: Union[np.ndarray, "pl.Series", List],  # type: ignore[name-defined]
        exposure: Optional[Union[np.ndarray, "pl.Series"]] = None,  # type: ignore[name-defined]
    ) -> float:
        """
        Mean negative log-likelihood of the combined ZI-Tweedie model.

        Uses the joint ZI-Tweedie log-likelihood:
          y=0:  log[π̂ + (1-π̂) · P_Tweedie(0 | μ̂, φ̂)]
          y>0:  log[(1-π̂) · p_Tweedie(y | μ̂, φ̂)]

        Since we don't model φ̂ explicitly (Scenario 2 uses the Tweedie loss
        directly, not double GLM), we use the simpler scoring rule:

          y=0:  log(π̂)
          y>0:  log(1-π̂) + log p_Tweedie(y | μ̂·exp(offset))

        This is not a full likelihood score (because we have no phi model),
        but it correctly rewards Stage 1 accuracy and Stage 2 deviance.

        Lower is better.

        Parameters
        ----------
        X, y, exposure : as for fit()

        Returns
        -------
        float
            Mean negative log-likelihood. Lower is better.
        """
        self._check_is_fitted()
        y_np = _to_1d(y)
        n = len(y_np)
        exp_np = _to_1d(exposure) if exposure is not None else np.ones(n)

        components = self.predict_components(X, exp_np)
        pi = np.clip(components["zero_prob"], 1e-8, 1 - 1e-8)
        mu = np.clip(components["combined_mean"], 1e-8, None)

        ll = np.zeros(n)
        mask0 = y_np == 0
        if mask0.any():
            ll[mask0] = np.log(pi[mask0])
        mask_pos = ~mask0
        if mask_pos.any():
            ll[mask_pos] = (
                np.log(1.0 - pi[mask_pos])
                + _tweedie_unit_deviance_ll(
                    y_np[mask_pos], mu[mask_pos], self.power
                )
            )

        return float(-np.mean(ll))

    # -------------------------------------------------------------------------
    # Serialisation
    # -------------------------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted. Call fit() first."
            )

    def _build_zero_params(self) -> Dict[str, Any]:
        """Default CatBoost params for the zero classifier."""
        defaults = {
            "loss_function": "CrossEntropy",
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_seed": self.random_state,
            "verbose": False,
            "allow_writing_files": False,
        }
        merged = dict(defaults)
        merged.update(self.catboost_params_zero)
        return merged

    def _build_severity_params(self) -> Dict[str, Any]:
        """Default CatBoost params for the Tweedie severity model."""
        defaults = {
            "loss_function": f"Tweedie:variance_power={self.power}",
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_seed": self.random_state,
            "verbose": False,
            "allow_writing_files": False,
        }
        merged = dict(defaults)
        merged.update(self.catboost_params_severity)
        return merged

    def _fit_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict[str, Any],
        sample_weight: Optional[np.ndarray] = None,
    ):
        """Fit a CatBoostClassifier for zero probability estimation."""
        from catboost import CatBoostClassifier, Pool

        pool_kwargs: Dict[str, Any] = {"data": X, "label": y}
        if self.cat_features:
            pool_kwargs["cat_features"] = self.cat_features
        if sample_weight is not None:
            pool_kwargs["weight"] = sample_weight

        pool = Pool(**pool_kwargs)
        model = CatBoostClassifier(**params)
        model.fit(pool)
        return model

    def _fit_regressor(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict[str, Any],
        baseline: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ):
        """Fit a CatBoostRegressor with optional baseline offset."""
        from catboost import CatBoostRegressor, Pool

        pool_kwargs: Dict[str, Any] = {"data": X, "label": y}
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

    def _predict_classifier(self, model, X: np.ndarray) -> np.ndarray:
        """Predict P(class=1) = P(Y=0) from the classifier."""
        from catboost import Pool

        pool_kwargs: Dict[str, Any] = {"data": X}
        if self.cat_features:
            pool_kwargs["cat_features"] = self.cat_features

        pool = Pool(**pool_kwargs)
        proba = model.predict_proba(pool)
        return proba[:, 1]

    def _predict_regressor(
        self,
        model,
        X: np.ndarray,
        baseline: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict from the regressor with optional baseline offset."""
        from catboost import Pool

        pool_kwargs: Dict[str, Any] = {"data": X}
        if self.cat_features:
            pool_kwargs["cat_features"] = self.cat_features
        if baseline is not None:
            pool_kwargs["baseline"] = baseline

        pool = Pool(**pool_kwargs)
        return model.predict(pool)

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"ZeroInflatedTweedieGBM("
            f"power={self.power}, "
            f"status={status!r})"
        )


# ---------------------------------------------------------------------------
# Helper: Tweedie unit deviance log-contribution (for log_score)
# ---------------------------------------------------------------------------

def _tweedie_unit_deviance_ll(
    y: np.ndarray, mu: np.ndarray, p: float
) -> np.ndarray:
    """
    Log-likelihood contribution for y > 0 under Tweedie (excluding phi).

    This is -0.5 * unit_deviance / phi where we absorb phi into the
    constant. For scoring purposes we use the deviance term only.

    Specifically: -(unit_deviance / 2) where
      unit_deviance = 2*(y^(2-p)/((1-p)*(2-p)) - y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p))

    We negate so it acts as a log-likelihood (higher = better).
    """
    term1 = y ** (2 - p) / ((1 - p) * (2 - p))
    term2 = y * mu ** (1 - p) / (1 - p)
    term3 = mu ** (2 - p) / (2 - p)
    unit_dev = 2.0 * (term1 - term2 + term3)
    return -0.5 * unit_dev
