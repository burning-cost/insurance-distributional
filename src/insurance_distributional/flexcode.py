"""
FlexCodeDensity: nonparametric conditional density estimation via FlexCode.

FlexCode (Izbicki & Lee, 2017) estimates the full conditional density f(y|x)
as a series expansion in an orthonormal basis, where each coefficient is a
separate regression problem. The result is a function-valued prediction:
for each new x, you get a density estimate over all y, not just a point
estimate.

Insurance applications
----------------------
The primary use case in insurance is XL layer pricing without GPD assumptions.

A standard GPD-based extreme quantile model (EQRN, insurance_quantile.eqrn)
requires:
  1. A parametric tail assumption (GPD).
  2. A threshold selection procedure (mean excess plot, etc.).
  3. An intermediate quantile estimator to condition the GPD tail.

FlexCode bypasses all three. It estimates f(y|x) nonparametrically and
derives any functional of the conditional distribution — quantiles, layer
expected values, tail probabilities — by numerically integrating the
estimated density.

Trade-off: FlexCode is more flexible but requires a larger calibration set.
GPD models extrapolate beyond the training data range; FlexCode does not.
For lines where claims routinely exceed the historical maximum (e.g. liability),
use EQRN. For lines where the historical range covers the pricing layer
(motor own damage, property), FlexCode is simpler and does not require
threshold selection.

Algorithm
---------
The FlexCode series expansion is:

    f(y|x) = sum_{k=1}^{K} beta_k(x) * phi_k(y)

where {phi_k} is an orthonormal basis for L2 (Fourier or cosine basis on
a compact interval), and beta_k(x) = E[phi_k(Y) | X = x] is estimated
by a separate regression for each basis function. Any regression method
can be used; we use CatBoost for consistency with the rest of the
insurance_distributional package.

Reference
---------
Izbicki, R. and Lee, A.B. (2017). Converting high-dimensional regression
to high-dimensional conditional density estimation. Electronic Journal of
Statistics 11(2), 2800-2831.

Usage::

    from insurance_distributional.flexcode import FlexCodeDensity

    model = FlexCodeDensity(n_basis=30, y_grid_size=200)
    model.fit(X_train, y_train)

    density = model.predict_density(X_test)   # shape (n_test, y_grid_size)
    y_grid = model.y_grid_                    # the y-axis grid

    # Any layer: E[Y | a < Y < b | X]
    layer_ev = model.layer_expected_value(X_test, attachment=100_000, limit=500_000)

    # Quantiles without GPD assumption
    q95 = model.quantile(X_test, alpha=0.95)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

__all__ = ["FlexCodeDensity"]


def _to_numpy(X: Any) -> np.ndarray:
    if isinstance(X, pl.DataFrame):
        return X.to_numpy().astype(np.float64)
    if isinstance(X, pl.Series):
        return X.to_numpy().astype(np.float64)
    return np.asarray(X, dtype=np.float64)


def _cosine_basis(y: np.ndarray, n_basis: int, y_min: float, y_max: float) -> np.ndarray:
    """
    Evaluate the first n_basis terms of the cosine orthonormal basis on [y_min, y_max].

    phi_0(y) = 1 / sqrt(y_max - y_min)   (constant term)
    phi_k(y) = sqrt(2 / (y_max - y_min)) * cos(k * pi * (y - y_min) / (y_max - y_min))

    Returns array of shape (len(y), n_basis).
    """
    L = y_max - y_min
    if L <= 0:
        raise ValueError(f"y_max ({y_max}) must be greater than y_min ({y_min}).")

    u = (y - y_min) / L   # normalised to [0, 1]
    basis = np.empty((len(y), n_basis), dtype=np.float64)
    basis[:, 0] = 1.0 / np.sqrt(L)
    for k in range(1, n_basis):
        basis[:, k] = np.sqrt(2.0 / L) * np.cos(k * np.pi * u)
    return basis


def _project_onto_basis(y: np.ndarray, n_basis: int, y_min: float, y_max: float) -> np.ndarray:
    """
    Compute the basis function values phi_k(y_i) for all i and k.
    Returns shape (n_obs, n_basis).
    """
    return _cosine_basis(y, n_basis, y_min, y_max)


class FlexCodeDensity:
    """
    Nonparametric conditional density estimator via FlexCode series expansion.

    Estimates f(y|x) as a finite cosine-basis expansion where each coefficient
    beta_k(x) is learned by CatBoost regression. Provides quantile estimation,
    layer expected values, and tail probabilities without distributional
    assumptions.

    Parameters
    ----------
    n_basis:
        Number of orthonormal basis functions. More basis functions = more
        flexible density estimates but more regression models to fit and more
        risk of overfitting. Typically 20-50 for insurance severity data.
    y_grid_size:
        Number of grid points for density evaluation and numerical integration.
        200 is sufficient for smooth densities; increase to 500+ for accurate
        tail probabilities when the density has sharp features.
    y_min:
        Lower bound of the density support. Defaults to 0.0 (non-negative losses).
        Set to a small positive value (e.g. 1.0) to exclude structural zeros.
    y_max:
        Upper bound of the density support. If None, set to 1.05 * max(y_train).
        Claims beyond y_max are truncated in density estimates — check this
        against the actual tail of your training data.
    catboost_params:
        Dict of CatBoost parameters passed to each basis regression.
        Defaults: iterations=300, learning_rate=0.05, depth=6, verbose=0.
    normalise:
        If True (default), normalise predicted densities to integrate to 1.
        Normalisation corrects for basis truncation bias in the tails.
    random_state:
        Random seed for CatBoost.

    Attributes set after fit()
    --------------------------
    y_grid_:
        The y-axis grid used for density evaluation.
    basis_models_:
        List of n_basis fitted CatBoost models, one per basis function.
    y_min_:
        Effective y_min used in fitting.
    y_max_:
        Effective y_max used in fitting.
    """

    def __init__(
        self,
        n_basis: int = 30,
        y_grid_size: int = 200,
        y_min: float = 0.0,
        y_max: float | None = None,
        catboost_params: dict | None = None,
        normalise: bool = True,
        random_state: int = 42,
    ) -> None:
        self.n_basis = n_basis
        self.y_grid_size = y_grid_size
        self.y_min = y_min
        self.y_max = y_max
        self.catboost_params = catboost_params or {}
        self.normalise = normalise
        self.random_state = random_state

        # Set after fit
        self.y_min_: float | None = None
        self.y_max_: float | None = None
        self.y_grid_: np.ndarray | None = None
        self.basis_models_: list | None = None
        self._basis_on_grid: np.ndarray | None = None

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any | None = None,
    ) -> "FlexCodeDensity":
        """
        Fit the conditional density model.

        Parameters
        ----------
        X:
            Feature matrix. Polars DataFrame, numpy array, or pandas DataFrame.
        y:
            Observed outcomes. Must be positive (severity, not pure premium).
            Polars Series or numpy array.
        sample_weight:
            Optional exposure weights. Passed to each CatBoost model as
            sample_weight.

        Returns
        -------
        self
        """
        from catboost import CatBoostRegressor  # noqa: PLC0415

        X_arr = _to_numpy(X)
        y_arr = _to_numpy(y).ravel()

        if np.any(y_arr < 0):
            raise ValueError(
                "y contains negative values. FlexCodeDensity expects non-negative "
                "losses (severity). Filter to non-zero claims before fitting."
            )

        self.y_min_ = self.y_min
        self.y_max_ = float(self.y_max) if self.y_max is not None else float(y_arr.max() * 1.05)

        # Build training targets: phi_k(y_i) for each basis function
        phi_train = _project_onto_basis(y_arr, self.n_basis, self.y_min_, self.y_max_)

        # Default CatBoost params
        cb_defaults: dict = {
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 6,
            "verbose": 0,
            "random_seed": self.random_state,
            "allow_writing_files": False,
        }
        cb_defaults.update(self.catboost_params)

        # Fit one regression per basis function
        self.basis_models_ = []
        sw = _to_numpy(sample_weight).ravel() if sample_weight is not None else None

        for k in range(self.n_basis):
            model_k = CatBoostRegressor(**cb_defaults)
            if sw is not None:
                model_k.fit(X_arr, phi_train[:, k], sample_weight=sw)
            else:
                model_k.fit(X_arr, phi_train[:, k])
            self.basis_models_.append(model_k)

        # Precompute basis on the evaluation grid
        self.y_grid_ = np.linspace(self.y_min_, self.y_max_, self.y_grid_size)
        self._basis_on_grid = _cosine_basis(
            self.y_grid_, self.n_basis, self.y_min_, self.y_max_
        )  # shape (y_grid_size, n_basis)

        return self

    def _predict_coefficients(self, X: Any) -> np.ndarray:
        """
        Predict basis coefficients beta_k(x) for all test points.

        Returns array of shape (n_test, n_basis).
        """
        if self.basis_models_ is None:
            raise RuntimeError("Call fit() before predict_density().")
        X_arr = _to_numpy(X)
        n_test = X_arr.shape[0]
        beta = np.empty((n_test, self.n_basis), dtype=np.float64)
        for k, model_k in enumerate(self.basis_models_):
            beta[:, k] = model_k.predict(X_arr)
        return beta

    def predict_density(self, X: Any) -> np.ndarray:
        """
        Predict the conditional density f(y|x) on the y_grid_ for each x.

        Returns array of shape (n_test, y_grid_size). Each row is a density
        estimate evaluated at self.y_grid_ points.

        The density may be negative in regions where the basis expansion
        undershoots zero (Gibbs phenomenon). We clip to zero and renormalise
        if normalise=True (default).

        Parameters
        ----------
        X:
            Test feature matrix.

        Returns
        -------
        density:
            Array of shape (n_test, y_grid_size).
        """
        beta = self._predict_coefficients(X)  # (n_test, n_basis)
        # f(y|x) = sum_k beta_k(x) * phi_k(y)
        # density[i, j] = sum_k beta[i, k] * basis_grid[j, k]
        density = beta @ self._basis_on_grid.T  # (n_test, y_grid_size)

        # Clip negative values (basis truncation artefacts)
        density = np.maximum(density, 0.0)

        if self.normalise:
            # Normalise each row to integrate to 1 over y_grid_
            dy = self.y_grid_[1] - self.y_grid_[0]
            row_integrals = density.sum(axis=1, keepdims=True) * dy
            row_integrals = np.where(row_integrals > 0, row_integrals, 1.0)
            density = density / row_integrals

        return density

    def quantile(self, X: Any, alpha: float) -> np.ndarray:
        """
        Estimate the alpha-quantile of Y | X for each test point.

        Parameters
        ----------
        X:
            Test feature matrix.
        alpha:
            Quantile level in (0, 1). E.g. 0.95 for the 95th percentile.

        Returns
        -------
        numpy array of shape (n_test,) with per-risk quantile estimates.
        """
        density = self.predict_density(X)
        dy = self.y_grid_[1] - self.y_grid_[0]
        # CDF by cumulative trapezoidal integration
        cdf = np.cumsum(density * dy, axis=1)
        # Clamp to [0, 1]
        cdf = np.clip(cdf, 0.0, 1.0)

        n_test = density.shape[0]
        q_vals = np.empty(n_test, dtype=np.float64)
        for i in range(n_test):
            q_vals[i] = np.interp(alpha, cdf[i], self.y_grid_)
        return q_vals

    def layer_expected_value(
        self, X: Any, attachment: float, limit: float
    ) -> np.ndarray:
        """
        Estimate the expected value of losses in an XL layer (attachment, attachment + limit].

        E[min(max(Y - attachment, 0), limit) | X]

        This is the actuarial expected value of losses in the excess-of-loss layer
        with attachment point `attachment` and limit `limit`. It is computed by
        numerically integrating the conditional density over the layer interval.

        Parameters
        ----------
        X:
            Test feature matrix.
        attachment:
            Layer attachment point in the same units as y_train.
        limit:
            Layer limit (width), not the upper bound. The layer covers losses
            between `attachment` and `attachment + limit`.

        Returns
        -------
        numpy array of shape (n_test,) with per-risk layer expected values.
        """
        upper = attachment + limit
        density = self.predict_density(X)  # (n_test, y_grid_size)

        # Integrate over [attachment, attachment+limit]
        y = self.y_grid_
        dy = y[1] - y[0]

        # Per-cell contribution to layer expected value
        # = integral_{attachment}^{upper} (y - attachment) * f(y|x) dy
        #   + limit * integral_{upper}^{inf} f(y|x) dy
        # We discretise both terms.
        capped_y = np.clip(y - attachment, 0.0, limit)  # (y_grid_size,)
        # Broadcast: (n_test, y_grid_size) * (y_grid_size,)
        lev = (density * capped_y).sum(axis=1) * dy
        return lev

    def tail_probability(self, X: Any, threshold: float) -> np.ndarray:
        """
        Estimate P(Y > threshold | X) for each test point.

        Parameters
        ----------
        X:
            Test feature matrix.
        threshold:
            The exceedance threshold.

        Returns
        -------
        numpy array of shape (n_test,) with per-risk tail probabilities in [0, 1].
        """
        density = self.predict_density(X)
        dy = self.y_grid_[1] - self.y_grid_[0]
        mask = self.y_grid_ > threshold
        return (density[:, mask] * dy).sum(axis=1)
