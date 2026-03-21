"""
FlexCodeDensity: nonparametric conditional density estimation for insurance losses.

Implements FlexCode (Izbicki & Lee 2017, EJS 11:2800-2831): converts a regression
method into a conditional density estimator by expanding f(y|x) in an orthonormal
cosine basis over y-space and estimating the basis coefficients via CatBoost
MultiRMSE regression.

Primary use case: per-risk XL layer pricing via price_layer(). When parametric
families (Tweedie, Gamma) impose shape assumptions that are wrong — e.g., bimodal
motor BI, heavy-tailed commercial property — FlexCodeDensity gives you the shape
the data shows, not the shape you assumed.

Design decisions:
- CatBoost MultiRMSE for coefficient regression: fits one model for all I basis
  function targets simultaneously. ~5-10x faster than I separate models.
- First basis function excluded from regression: phi_1(z) = 1/sqrt(width) is
  constant, so E[phi_1(Z)|X=x] = 1/sqrt(width) for all x regardless of X. We
  hard-code this coefficient analytically and fit only basis functions 2..I.
  This avoids CatBoost's "all targets are equal" error on the constant column.
- Log-transform by default: insurance severity is right-skewed with heavy tails.
  Fitting in Z = log(y + epsilon) space requires far fewer basis functions to
  represent the density accurately. Without it, Gibbs phenomenon dominates near
  the maximum observed loss.
- basis.py is a self-contained MIT reimplementation: no GPL dependency.
- No new required dependencies: CatBoost already in the dep list.

What FlexCodeDensity does NOT do:
- Exposure offset. Apply to severity (positive losses only), not to pure premiums
  with zeros. For frequency-severity decomposition, fit FlexCodeDensity on
  severity and a separate frequency model.
- Structural zeros. If y has zeros (e.g., total loss with deductible), filter
  them out before fitting. log_transform=True will error on y=0 otherwise.

Reference:
    Izbicki, R. & Lee, A.B. (2017). Converting high-dimensional regression to
    high-dimensional conditional density estimation. Electronic Journal of
    Statistics, 11(2):2800-2831. arXiv:1704.08095.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

# numpy<2.0 compat: trapezoid was added in 2.0, trapz deprecated in 2.0
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]

from .basis import cosine_basis, evaluate_density, postprocess_density
from .base import _to_1d, _to_numpy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FlexCodePrediction result container
# ---------------------------------------------------------------------------


@dataclass
class FlexCodePrediction:
    """
    Container for FlexCodeDensity.predict_density() output.

    Analogous to DistributionalPrediction but for nonparametric densities:
    the full conditional density f(y|x) is returned on a grid rather than
    parameterised by a single distribution family.

    Attributes
    ----------
    cdes : np.ndarray, shape (n_obs, n_grid)
        Conditional density values f(y | x_i) evaluated on y_grid.
        Non-negative, integrates to ~1 over y_grid for each row.
    y_grid : np.ndarray, shape (n_grid,)
        Evaluation points in y-space (original scale, even when log_transform=True).
    n_basis_used : int
        Number of basis functions used. May be < max_basis if tune() was called.
    """

    cdes: np.ndarray
    y_grid: np.ndarray
    n_basis_used: int

    # -------------------------------------------------------------------------
    # Moment summaries via numerical integration
    # -------------------------------------------------------------------------

    @property
    def mean(self) -> np.ndarray:
        """
        E[Y|X] = integral y * f(y|x) dy, computed via trapz over y_grid.

        Returns
        -------
        np.ndarray, shape (n_obs,)
        """
        return np.trapezoid(self.cdes * self.y_grid[None, :], self.y_grid, axis=1)

    @property
    def variance(self) -> np.ndarray:
        """
        Var[Y|X] = E[Y^2|X] - E[Y|X]^2, computed via trapz.

        Returns
        -------
        np.ndarray, shape (n_obs,)
        """
        e_y = self.mean
        e_y2 = np.trapezoid(self.cdes * self.y_grid[None, :] ** 2, self.y_grid, axis=1)
        return np.clip(e_y2 - e_y ** 2, 0.0, None)

    @property
    def std(self) -> np.ndarray:
        """Conditional standard deviation sqrt(Var[Y|X])."""
        return np.sqrt(self.variance)

    def volatility_score(self) -> np.ndarray:
        """
        Coefficient of variation: std / mean.

        Dimensionless per-risk measure of relative uncertainty. Mirrors the
        volatility_score() on DistributionalPrediction so FlexCodeDensity and
        parametric models can be compared on the same scale.

        Returns
        -------
        np.ndarray, shape (n_obs,)
        """
        return self.std / (self.mean + 1e-12)

    def quantile(self, q: Union[float, List[float]]) -> np.ndarray:
        """
        Quantile(s) via empirical CDF inversion.

        Integrates the density over y_grid to get the CDF, then linearly
        interpolates to find the y value where CDF = q.

        Parameters
        ----------
        q : float or list of float
            Quantile level(s) in (0, 1).

        Returns
        -------
        np.ndarray of shape (n_obs,) if q is float, (n_obs, len(q)) if list.
        """
        scalar = isinstance(q, float)
        q_arr = np.atleast_1d(np.asarray(q, dtype=np.float64))

        # CDF via cumulative trapz
        delta_y = np.diff(self.y_grid)
        f_mid = 0.5 * (self.cdes[:, :-1] + self.cdes[:, 1:])  # (n_obs, n_grid-1)
        cdf_increments = f_mid * delta_y[None, :]  # (n_obs, n_grid-1)
        cdf = np.zeros((self.cdes.shape[0], len(self.y_grid)))
        cdf[:, 1:] = np.cumsum(cdf_increments, axis=1)
        cdf = np.clip(cdf, 0.0, 1.0)

        n_obs = self.cdes.shape[0]
        n_grid = len(self.y_grid)
        result = np.empty((n_obs, len(q_arr)))

        # Vectorise over quantile levels (5-10 iterations), not observations
        # (potentially 10k+ iterations). For each quantile level qi, we find
        # the grid index where cdf[i, :] first reaches qi for every row i.
        # Note: np.searchsorted only works on 1D arrays in numpy<2.0, so we
        # use np.argmax on the boolean matrix (cdf >= qi) instead.
        for j, qi in enumerate(q_arr):
            above = cdf >= qi  # (n_obs, n_grid), True where CDF has reached qi
            # argmax returns first True index per row; 0 if no True (all below)
            idx = np.argmax(above, axis=1)  # (n_obs,)
            # If CDF never reaches qi (beyond grid), hold at last grid point
            all_below = ~above.any(axis=1)
            idx = np.where(all_below, n_grid - 1, idx)
            idx = np.clip(idx, 1, n_grid - 1)  # ensure idx-1 is valid
            y_lo = self.y_grid[idx - 1]
            y_hi = self.y_grid[idx]
            c_lo = cdf[np.arange(n_obs), idx - 1]
            c_hi = cdf[np.arange(n_obs), idx]
            denom = c_hi - c_lo
            safe_denom = np.where(denom > 0, denom, 1.0)
            w = np.clip((qi - c_lo) / safe_denom, 0.0, 1.0)
            result[:, j] = y_lo + w * (y_hi - y_lo)

        if scalar:
            return result[:, 0]
        return result

    def price_layer(self, attachment: float, limit: float) -> np.ndarray:
        """
        Price a per-risk excess-of-loss reinsurance layer.

        E[min(max(Y - a, 0), l) | X=x] = integral_a^{a+l} S_Y(t) dt

        where S_Y(t) = 1 - F_Y(t) is the survival function, computed via
        numerical integration over the density grid.

        Parameters
        ----------
        attachment : float
            Layer attachment point a.
        limit : float
            Layer limit l.

        Returns
        -------
        np.ndarray, shape (n_obs,)
            Expected layer loss per risk.

        Notes
        -----
        If attachment >= y_grid.max(), a warning is issued and 0 is returned.
        If attachment + limit > y_grid.max(), integration is clipped — the
        result underestimates the true layer price.
        """
        y_max = self.y_grid[-1]

        if attachment >= y_max:
            warnings.warn(
                f"attachment {attachment:.2f} exceeds density grid maximum {y_max:.2f}. "
                "Layer price will be zero or unreliable. Increase n_grid or use a "
                "parametric tail model above the training data range.",
                UserWarning,
                stacklevel=2,
            )
            return np.zeros(self.cdes.shape[0])

        upper = attachment + limit
        if upper > y_max:
            warnings.warn(
                f"Layer upper bound {upper:.2f} exceeds density grid maximum {y_max:.2f}. "
                "Integration is clipped — layer price is underestimated. "
                "Consider setting z_max_override at fit time.",
                UserWarning,
                stacklevel=2,
            )
            upper = y_max

        # Build CDF over the full y_grid
        delta_y = np.diff(self.y_grid)
        f_mid = 0.5 * (self.cdes[:, :-1] + self.cdes[:, 1:])
        cdf_increments = f_mid * delta_y[None, :]
        cdf = np.zeros((self.cdes.shape[0], len(self.y_grid)))
        cdf[:, 1:] = np.cumsum(cdf_increments, axis=1)
        cdf = np.clip(cdf, 0.0, 1.0)

        # Survival function S(t) = 1 - CDF(t)
        survival = 1.0 - cdf  # (n_obs, n_grid)

        # Mask to [attachment, upper]
        in_layer = (self.y_grid >= attachment) & (self.y_grid <= upper)
        y_layer = self.y_grid[in_layer]
        s_layer = survival[:, in_layer]

        if len(y_layer) < 2:
            return np.zeros(self.cdes.shape[0])

        layer_ev = np.trapezoid(s_layer, y_layer, axis=1)
        return np.clip(layer_ev, 0.0, limit)

    def pit_values(self, y_obs: np.ndarray) -> np.ndarray:
        """
        Probability Integral Transform values F_hat(y_obs_i | x_i).

        A well-calibrated density produces PIT values that are uniform on [0,1].
        Use these for calibration diagnostics: plot a histogram, run KS test.

        Parameters
        ----------
        y_obs : np.ndarray, shape (n_obs,)
            Observed outcomes, one per row of cdes.

        Returns
        -------
        np.ndarray, shape (n_obs,)
            Estimated CDF values, in [0, 1].
        """
        y_obs = np.asarray(y_obs, dtype=np.float64)
        if len(y_obs) != self.cdes.shape[0]:
            raise ValueError(
                f"y_obs length {len(y_obs)} != n_obs {self.cdes.shape[0]}"
            )

        # Build CDF
        delta_y = np.diff(self.y_grid)
        f_mid = 0.5 * (self.cdes[:, :-1] + self.cdes[:, 1:])
        cdf_increments = f_mid * delta_y[None, :]
        cdf = np.zeros((self.cdes.shape[0], len(self.y_grid)))
        cdf[:, 1:] = np.cumsum(cdf_increments, axis=1)
        cdf = np.clip(cdf, 0.0, 1.0)

        n_obs = self.cdes.shape[0]
        pit = np.empty(n_obs)
        # Vectorised: for each observation i, find where y_obs[i] falls in y_grid
        # and interpolate the CDF value there. Avoids n_obs separate np.interp calls.
        idx = np.searchsorted(self.y_grid, y_obs, side="left")  # (n_obs,)
        idx = np.clip(idx, 1, len(self.y_grid) - 1)
        y_lo = self.y_grid[idx - 1]
        y_hi = self.y_grid[idx]
        c_lo = cdf[np.arange(n_obs), idx - 1]
        c_hi = cdf[np.arange(n_obs), idx]
        denom = y_hi - y_lo
        safe_denom = np.where(denom > 0, denom, 1.0)
        w = np.clip((y_obs - y_lo) / safe_denom, 0.0, 1.0)
        pit = np.clip(c_lo + w * (c_hi - c_lo), 0.0, 1.0)
        return pit

    def cde_loss(self, y_obs: np.ndarray) -> float:
        """
        CDE loss (Gneiting & Raftery 2007) on observed outcomes.

        L = E[integral f_hat(z|X)^2 dz] - 2 * E[f_hat(Z|X)]

        Strictly proper: minimised uniquely by the true conditional density.
        Lower is better.

        Parameters
        ----------
        y_obs : np.ndarray, shape (n_obs,)
            Observed outcomes.

        Returns
        -------
        float
        """
        y_obs = np.asarray(y_obs, dtype=np.float64)
        n = self.cdes.shape[0]

        term1 = float(np.mean(np.trapezoid(self.cdes ** 2, self.y_grid, axis=1)))

        # Vectorised density interpolation: same approach as pit_values()
        delta_y2 = np.diff(self.y_grid)
        f_mid2 = 0.5 * (self.cdes[:, :-1] + self.cdes[:, 1:])
        idx2 = np.searchsorted(self.y_grid, y_obs, side="left")
        idx2 = np.clip(idx2, 1, len(self.y_grid) - 1)
        y_lo2 = self.y_grid[idx2 - 1]
        y_hi2 = self.y_grid[idx2]
        d_lo2 = self.cdes[np.arange(n), idx2 - 1]
        d_hi2 = self.cdes[np.arange(n), idx2]
        denom2 = y_hi2 - y_lo2
        safe_denom2 = np.where(denom2 > 0, denom2, 1.0)
        w2 = np.clip((y_obs - y_lo2) / safe_denom2, 0.0, 1.0)
        term2_vals = d_lo2 + w2 * (d_hi2 - d_lo2)
        term2 = float(np.mean(term2_vals))

        return term1 - 2.0 * term2

    def __repr__(self) -> str:
        n_obs, n_grid = self.cdes.shape
        return (
            f"FlexCodePrediction("
            f"n_obs={n_obs}, n_grid={n_grid}, "
            f"y_grid=[{self.y_grid[0]:.2f}, {self.y_grid[-1]:.2f}], "
            f"n_basis_used={self.n_basis_used})"
        )


# ---------------------------------------------------------------------------
# FlexCodeDensity model
# ---------------------------------------------------------------------------


class FlexCodeDensity:
    """
    Nonparametric conditional density estimator for insurance losses.

    Implements FlexCode (Izbicki & Lee 2017): estimates f(y|x) by expanding
    the conditional density in an orthonormal cosine basis over y-space and
    estimating the basis coefficients via CatBoost MultiRMSE regression.

    Unlike the parametric distributional GBM classes (TweedieGBM, GammaGBM),
    FlexCodeDensity makes no assumption about the shape of the distribution.
    This matters when the loss distribution changes shape across risk profiles
    (e.g., bimodal motor BI, heavy-tailed commercial property, XL layers
    where the tail shape determines the price).

    Primary use case: per-risk XL layer pricing via price_layer().

    Parameters
    ----------
    max_basis : int
        Maximum number of basis functions I. Default 30.
        Higher values capture more distributional detail but risk overfitting.
        Call tune() after fit() to select the optimal value automatically.
    basis_system : str
        Orthonormal basis system. Currently only 'cosine'. Default 'cosine'.
    log_transform : bool
        If True (default), fit in log(y + log_epsilon) space and transform back.
        Essential for right-skewed insurance severity. Set False for frequency
        or count data, or when y can be negative.
    log_epsilon : float
        Continuity correction for log transform. Default 1.0 (suitable for
        losses in £/$ units). For normalised losses in [0, 1], use 0.01.
    n_grid : int
        Number of grid points for density evaluation. Default 200.
        Increase to 500+ for reliable layer pricing above the 95th percentile.
    z_max_override : float, optional
        If set, overrides the automatic z_max derived from training data.
        Use when pricing layers that extend beyond the observed maximum.
        In log-space for log_transform=True.
    cat_features : list of int or str, optional
        Categorical feature indices/names. Passed to CatBoost.
    catboost_params : dict, optional
        CatBoost parameters merged with defaults (iterations=300, depth=6,
        learning_rate=0.05). Set verbose=True to see training progress.
    random_state : int
        Random seed. Default 42.

    Examples
    --------
    >>> from insurance_distributional import FlexCodeDensity
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((500, 3))
    >>> y = rng.gamma(shape=2.0, scale=500.0, size=500)
    >>> model = FlexCodeDensity(max_basis=20)
    >>> model.fit(X, y)
    FlexCodeDensity(...)
    >>> pred = model.predict_density(X[:10])
    >>> pred.cdes.shape
    (10, 200)
    >>> # Price a £500 xs £500 layer
    >>> ev = model.price_layer(X[:10], attachment=500.0, limit=500.0)
    """

    def __init__(
        self,
        max_basis: int = 30,
        basis_system: str = "cosine",
        log_transform: bool = True,
        log_epsilon: float = 1.0,
        n_grid: int = 200,
        z_max_override: Optional[float] = None,
        cat_features: Optional[List[Union[int, str]]] = None,
        catboost_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ) -> None:
        if basis_system != "cosine":
            raise ValueError(
                f"basis_system must be 'cosine', got {basis_system!r}. "
                "Other basis systems are not implemented in v1."
            )
        self.max_basis = max_basis
        self.basis_system = basis_system
        self.log_transform = log_transform
        self.log_epsilon = log_epsilon
        self.n_grid = n_grid
        self.z_max_override = z_max_override
        self.cat_features = cat_features or []
        self.catboost_params = catboost_params or {}
        self.random_state = random_state

        # Set after fit()
        self._is_fitted: bool = False
        self._model = None          # CatBoostRegressor with MultiRMSE (for basis 2..I)
        self._z_min: float = 0.0
        self._z_max: float = 1.0
        self._coef0: float = 0.0    # beta_1 = 1/sqrt(width), constant for all obs
        self._z_grid: np.ndarray = np.array([])   # in z-space (transformed)
        self._y_grid: np.ndarray = np.array([])   # in y-space (original)
        self.best_basis_: Optional[int] = None    # set by tune()

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def fit(
        self,
        X: Union[np.ndarray, "pl.DataFrame"],
        y: Union[np.ndarray, "pl.Series", List],
    ) -> "FlexCodeDensity":
        """
        Fit FlexCode conditional density estimator.

        Steps:
        1. Apply log transform if log_transform=True.
        2. Determine z_min, z_max from transformed targets with margin.
        3. Evaluate cosine basis on transformed targets -> Z_train (n x I).
        4. Set beta_1 = 1/sqrt(width) analytically (constant for all x).
        5. Fit CatBoost MultiRMSE on (X_train, Z_train[:, 1:]) for basis 2..I.
        6. Store z_min, z_max, z_grid, and fitted model.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
        y : array-like (n_samples,)
            Insurance losses. Must be > 0 if log_transform=True.

        Returns
        -------
        self
        """
        X_np = _to_numpy(X)
        y_np = _to_1d(y)
        n = len(y_np)

        if self.log_transform and np.any(y_np <= 0):
            raise ValueError(
                "FlexCodeDensity with log_transform=True requires y > 0. "
                f"Found {(y_np <= 0).sum()} non-positive values. "
                "Filter zeros before fitting (apply to severity only, not Tweedie pure premium)."
            )

        # Guard against silently wrong log-transform when log_epsilon is too large
        # for the data scale. For claims in pence (sub-unit), log_epsilon=1.0 (default)
        # compresses most of the loss distribution near zero and produces unreliable densities.
        if self.log_transform and float(y_np.min()) < self.log_epsilon:
            _y_min = float(y_np.min())
            _suggested = max(1e-6, _y_min * 1e-3)
            warnings.warn(
                f"log_epsilon={self.log_epsilon} is larger than the minimum observed "
                f"y ({_y_min:.4g}). The log-likelihood will be silently wrong because "
                "log(y + log_epsilon) no longer approximates log(y) for these values. "
                f"Suggested fix: set log_epsilon={_suggested:.2g} (or log_epsilon=1e-6 "
                "for very small losses). Using a log_epsilon larger than your smallest "
                "loss compresses most of the distribution near zero and produces "
                "inaccurate densities.",
                UserWarning,
                stacklevel=2,
            )

        logger.info(
            "FlexCodeDensity.fit: n=%d, max_basis=%d, log_transform=%s",
            n, self.max_basis, self.log_transform,
        )

        # --- Step 1: transform targets ---
        z_train = self._to_z(y_np)

        # --- Step 2: determine z domain ---
        z_range = z_train.max() - z_train.min()
        margin = 0.1 * z_range
        self._z_min = float(z_train.min() - margin)
        self._z_max = float(z_train.max() + margin)

        if self.z_max_override is not None:
            self._z_max = float(self.z_max_override)
            logger.info("z_max overridden to %.4f", self._z_max)

        width = self._z_max - self._z_min

        # --- Step 3: analytic first coefficient ---
        # phi_1(z) = 1/sqrt(width) is constant, so beta_1(x) = E[phi_1(Y)|X=x] = 1/sqrt(width)
        # for all x. Hard-code this rather than fitting it (CatBoost rejects constant columns).
        self._coef0 = 1.0 / np.sqrt(width)

        # --- Step 4: evaluate basis functions 2..I on training targets ---
        if self.max_basis > 1:
            Z_train_full = cosine_basis(z_train, self._z_min, self._z_max, self.max_basis)
            Z_train_nonconstant = Z_train_full[:, 1:]  # (n, max_basis - 1)
            self._model = self._fit_multirmse(X_np, Z_train_nonconstant)
        else:
            self._model = None

        # --- Step 5: build grids ---
        self._z_grid = np.linspace(self._z_min, self._z_max, self.n_grid)
        self._y_grid = self._to_y(self._z_grid)

        self._is_fitted = True
        logger.info("FlexCodeDensity fitted successfully.")
        return self

    def tune(
        self,
        X_val: Union[np.ndarray, "pl.DataFrame"],
        y_val: Union[np.ndarray, "pl.Series"],
        basis_candidates: Optional[List[int]] = None,
    ) -> "FlexCodeDensity":
        """
        Select optimal max_basis by minimising CDE loss on validation data.

        No refit required: because all max_basis coefficients are already
        fitted, we evaluate using only the first I' basis functions for each
        candidate I'. The best I' is set as self.best_basis_.

        Parameters
        ----------
        X_val : array-like (n_val, n_features)
            Validation features (separate from training data).
        y_val : array-like (n_val,)
            Validation targets.
        basis_candidates : list of int, optional
            Basis counts to evaluate. Default: range(step, max_basis+1, step).
            The value max_basis is always included.

        Returns
        -------
        self (with best_basis_ set)
        """
        self._check_is_fitted()
        X_np = _to_numpy(X_val)
        y_np = _to_1d(y_val)

        if basis_candidates is None:
            step = max(1, self.max_basis // 6)
            candidates = list(range(step, self.max_basis, step))
            if self.max_basis not in candidates:
                candidates.append(self.max_basis)
        else:
            candidates = sorted(basis_candidates)

        # Clamp candidates to [1, max_basis]
        candidates = [c for c in candidates if 1 <= c <= self.max_basis]
        if not candidates:
            candidates = [self.max_basis]

        best_loss = float("inf")
        best_basis = self.max_basis

        logger.info("Tuning FlexCode: evaluating %d basis candidates", len(candidates))

        for n_b in candidates:
            pred = self._predict_with_basis_count(X_np, n_b)
            loss = pred.cde_loss(y_np)
            logger.debug("  n_basis=%d -> CDE loss=%.6f", n_b, loss)
            if loss < best_loss:
                best_loss = loss
                best_basis = n_b

        self.best_basis_ = best_basis
        logger.info("tune(): best_basis_=%d (CDE loss=%.6f)", best_basis, best_loss)
        return self

    def predict_density(
        self,
        X_new: Union[np.ndarray, "pl.DataFrame"],
        n_grid: Optional[int] = None,
    ) -> FlexCodePrediction:
        """
        Predict conditional density f(y | x) for new observations.

        Parameters
        ----------
        X_new : array-like (n_test, n_features)
        n_grid : int, optional
            Grid resolution. Defaults to self.n_grid (set at fit time).
            Increase for higher accuracy in layer pricing.

        Returns
        -------
        FlexCodePrediction
            Contains cdes (n_test, n_grid) and y_grid (n_grid,) in y-space.
        """
        self._check_is_fitted()
        X_np = _to_numpy(X_new)

        n_basis = self.best_basis_ if self.best_basis_ is not None else self.max_basis
        return self._predict_with_basis_count(X_np, n_basis, n_grid=n_grid)

    def predict_quantile(
        self,
        X_new: Union[np.ndarray, "pl.DataFrame"],
        q: Union[float, List[float]],
    ) -> np.ndarray:
        """
        Predict the q-th quantile of f(y|x) for each new observation.

        Parameters
        ----------
        X_new : array-like (n_test, n_features)
        q : float or list of float
            Quantile level(s) in (0, 1).

        Returns
        -------
        np.ndarray of shape (n_test,) if q is float, (n_test, len(q)) if list.
        """
        pred = self.predict_density(X_new)
        return pred.quantile(q)

    def price_layer(
        self,
        X_new: Union[np.ndarray, "pl.DataFrame"],
        attachment: float,
        limit: float,
    ) -> np.ndarray:
        """
        Price a per-risk excess-of-loss reinsurance layer.

        E[min(max(Y - attachment, 0), limit) | X=x] for each risk, via
        numerical integration of the survival function over the density grid.

        Parameters
        ----------
        X_new : array-like (n_test, n_features)
        attachment : float
            Layer attachment point a.
        limit : float
            Layer limit l.

        Returns
        -------
        np.ndarray, shape (n_test,)
            Expected layer loss per risk.

        Notes
        -----
        If attachment + limit > the density grid maximum, results are clipped
        and a warning is issued. For reliable pricing above the training data
        range, use a parametric tail splice with GammaGBM above z_max.
        """
        pred = self.predict_density(X_new)
        return pred.price_layer(attachment, limit)

    def log_score(
        self,
        X_test: Union[np.ndarray, "pl.DataFrame"],
        y_test: Union[np.ndarray, "pl.Series"],
    ) -> float:
        """
        Mean negative log-likelihood (lower is better).

        Evaluates log f_hat(y_i | x_i) for each test observation by
        interpolating the density at y_test[i] on the y_grid.

        Parameters
        ----------
        X_test, y_test : test data

        Returns
        -------
        float
        """
        self._check_is_fitted()
        pred = self.predict_density(X_test)
        y_np = _to_1d(y_test)
        n = len(y_np)

        log_densities = np.empty(n)
        for i in range(n):
            f_i = float(np.interp(y_np[i], pred.y_grid, pred.cdes[i]))
            log_densities[i] = np.log(max(f_i, 1e-300))

        return float(-np.mean(log_densities))

    def crps(
        self,
        X_test: Union[np.ndarray, "pl.DataFrame"],
        y_test: Union[np.ndarray, "pl.Series"],
    ) -> float:
        """
        Mean CRPS via numerical integration of the CDF.

        CRPS(F, y) = integral_{-inf}^{inf} (F(t) - 1{t >= y})^2 dt

        Computed via trapz over the density grid. Strictly proper. Lower is better.
        Units match the target y.

        Parameters
        ----------
        X_test, y_test : test data

        Returns
        -------
        float
        """
        self._check_is_fitted()
        pred = self.predict_density(X_test)
        y_np = _to_1d(y_test)
        n = len(y_np)

        delta_y = np.diff(pred.y_grid)
        f_mid = 0.5 * (pred.cdes[:, :-1] + pred.cdes[:, 1:])
        cdf_increments = f_mid * delta_y[None, :]
        cdf = np.zeros((n, len(pred.y_grid)))
        cdf[:, 1:] = np.cumsum(cdf_increments, axis=1)
        cdf = np.clip(cdf, 0.0, 1.0)

        crps_vals = np.empty(n)
        for i in range(n):
            heaviside = (pred.y_grid >= y_np[i]).astype(float)
            crps_vals[i] = float(np.trapezoid((cdf[i] - heaviside) ** 2, pred.y_grid))

        return float(np.mean(crps_vals))

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _to_z(self, y: np.ndarray) -> np.ndarray:
        """Transform y -> z (log space if log_transform=True)."""
        if self.log_transform:
            return np.log(y + self.log_epsilon)
        return y.copy()

    def _to_y(self, z: np.ndarray) -> np.ndarray:
        """Transform z -> y (inverse of _to_z)."""
        if self.log_transform:
            return np.exp(z) - self.log_epsilon
        return z.copy()

    def _fit_multirmse(
        self,
        X_train: np.ndarray,
        Z_train: np.ndarray,
    ):
        """
        Fit CatBoost MultiRMSE to predict Z_train (n x (I-1)) from X_train.

        This fits basis functions 2..I only. The first basis function's
        coefficient (beta_1 = 1/sqrt(width)) is constant for all x and is
        stored analytically in self._coef0.

        Parameters
        ----------
        X_train : (n, D) feature matrix
        Z_train : (n, I-1) basis evaluations for basis functions 2..I

        Returns
        -------
        Fitted CatBoostRegressor
        """
        from catboost import CatBoostRegressor, Pool

        defaults: Dict[str, Any] = {
            "loss_function": "MultiRMSE",
            "iterations": 300,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "random_seed": self.random_state,
            "verbose": False,
            "allow_writing_files": False,
        }
        params = {**defaults, **self.catboost_params}

        pool_kwargs: Dict[str, Any] = {
            "data": X_train,
            "label": Z_train,
        }
        if self.cat_features:
            pool_kwargs["cat_features"] = self.cat_features

        pool = Pool(**pool_kwargs)
        model = CatBoostRegressor(**params)
        model.fit(pool)
        return model

    def _predict_coefs(self, X: np.ndarray, n_basis: int) -> np.ndarray:
        """
        Predict basis coefficients beta_hat_i(x) for each observation.

        Returns full coefficient matrix (n_obs, n_basis), where:
        - Column 0 is the analytic constant: 1/sqrt(width)
        - Columns 1..n_basis-1 are from the fitted CatBoost model

        Parameters
        ----------
        X : (n_obs, D) feature matrix
        n_basis : int
            Number of basis functions to return coefficients for.

        Returns
        -------
        np.ndarray, shape (n_obs, n_basis)
        """
        n_obs = X.shape[0]

        # First coefficient: analytic constant
        coefs = np.empty((n_obs, n_basis), dtype=np.float64)
        coefs[:, 0] = self._coef0

        if n_basis > 1 and self._model is not None:
            from catboost import Pool
            pool_kwargs: Dict[str, Any] = {"data": X}
            if self.cat_features:
                pool_kwargs["cat_features"] = self.cat_features
            pool = Pool(**pool_kwargs)
            model_preds = self._model.predict(pool)  # (n_obs, max_basis - 1) or (n_obs,)
            if model_preds.ndim == 1:
                model_preds = model_preds[:, None]
            # Take only the first n_basis-1 columns
            n_nonconstant = min(n_basis - 1, model_preds.shape[1])
            coefs[:, 1:1 + n_nonconstant] = model_preds[:, :n_nonconstant]
            # If we request fewer than max_basis, remaining are zero-filled above

        return coefs

    def _predict_with_basis_count(
        self,
        X: np.ndarray,
        n_basis: int,
        n_grid: Optional[int] = None,
    ) -> FlexCodePrediction:
        """
        Evaluate density using only the first n_basis basis functions.

        This is the key trick for cheap tuning: no refit, just truncate the sum.
        """
        grid_size = n_grid if n_grid is not None else self.n_grid
        z_grid = np.linspace(self._z_min, self._z_max, grid_size)

        coefs = self._predict_coefs(X, n_basis)  # (n_obs, n_basis)

        # Evaluate basis on z_grid
        B = cosine_basis(z_grid, self._z_min, self._z_max, n_basis)  # (grid_size, n_basis)
        cdes_z = coefs @ B.T  # (n_obs, grid_size)

        # Post-process in z-space (clip negative, renormalise)
        cdes_z = postprocess_density(cdes_z, z_grid)

        if self.log_transform:
            # Change of variables: f_Y(y) = f_Z(z) * |dz/dy|
            # where z = log(y + epsilon), dz/dy = 1/(y+epsilon)
            y_grid = np.exp(z_grid) - self.log_epsilon
            jacobian = 1.0 / (y_grid + self.log_epsilon)  # = exp(-z)
            cdes_y = cdes_z * jacobian[None, :]
            # Renormalise in y-space
            cdes_y = postprocess_density(cdes_y, y_grid)
        else:
            y_grid = z_grid
            cdes_y = cdes_z

        return FlexCodePrediction(
            cdes=cdes_y,
            y_grid=y_grid,
            n_basis_used=n_basis,
        )

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "FlexCodeDensity is not fitted. Call fit() first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        best = f", best_basis_={self.best_basis_}" if self.best_basis_ is not None else ""
        return (
            f"FlexCodeDensity("
            f"max_basis={self.max_basis}, "
            f"log_transform={self.log_transform}, "
            f"n_grid={self.n_grid}, "
            f"status={status!r}"
            f"{best})"
        )
