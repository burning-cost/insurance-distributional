"""
Scoring and calibration utilities for distributional GBM models.

Proper scoring rules let you compare distributional forecasts in a principled
way. The key property — strict properness — means that if the true distribution
is F, the score is minimised only when you report F. This prevents gaming via
over-dispersed or under-dispersed forecasts.

Functions here:
- tweedie_deviance: standard actuarial loss metric for pure premium models
- poisson_deviance: frequency model loss
- gamma_deviance: severity model loss
- pit_histogram: probability integral transform calibration check
- dispersion_calibration_plot: predicted phi vs squared Pearson residuals
- gini_index: discrimination power for pricing models

The CRPS is on the DistributionalPrediction class itself (it needs samples
from the distribution, so it belongs there).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Deviance metrics (standard actuarial)
# ---------------------------------------------------------------------------


def tweedie_deviance(
    y: np.ndarray,
    mu: np.ndarray,
    power: float,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Mean Tweedie deviance.

    d_i = 2 * (y_i^(2-p)/((1-p)*(2-p)) - y_i*mu_i^(1-p)/(1-p) + mu_i^(2-p)/(2-p))

    This is the standard metric for evaluating compound Poisson-Gamma models.
    Lower is better.

    Parameters
    ----------
    y : np.ndarray
        Observed values.
    mu : np.ndarray
        Predicted means.
    power : float
        Tweedie power p in (1, 2).
    weights : np.ndarray, optional
        Observation weights (e.g., exposure).

    Returns
    -------
    float
        Weighted mean Tweedie deviance.
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    mu = np.clip(mu, 1e-10, None)
    p = power

    dev = 2.0 * (
        y ** (2 - p) / ((1 - p) * (2 - p))
        - y * mu ** (1 - p) / (1 - p)
        + mu ** (2 - p) / (2 - p)
    )

    # Handle y=0 separately (y^(2-p) is 0 for y=0, formula still works)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        return float(np.average(dev, weights=w))
    return float(np.mean(dev))


def poisson_deviance(
    y: np.ndarray,
    mu: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Mean Poisson deviance.

    d_i = 2 * (y_i * log(y_i/mu_i) - (y_i - mu_i))

    Standard metric for frequency (claim count) models.
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    mu = np.clip(mu, 1e-10, None)

    # Handle y=0 (0*log(0) = 0 by convention)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(y > 0, y * np.log(y / mu), 0.0)
    dev = 2.0 * (log_ratio - (y - mu))

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        return float(np.average(dev, weights=w))
    return float(np.mean(dev))


def gamma_deviance(
    y: np.ndarray,
    mu: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Mean Gamma deviance.

    d_i = 2 * ((y_i - mu_i)/mu_i - log(y_i/mu_i))

    Standard metric for severity (claim size) models.
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    mu = np.clip(mu, 1e-10, None)
    y = np.clip(y, 1e-10, None)  # log requires y > 0

    dev = 2.0 * ((y - mu) / mu - np.log(y / mu))

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        return float(np.average(dev, weights=w))
    return float(np.mean(dev))


def negbinom_deviance(
    y: np.ndarray,
    mu: np.ndarray,
    r: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Mean Negative Binomial deviance.

    d_i = 2 * (y*log(y/mu) - (y+r)*log((y+r)/(mu+r)))

    where r is the size parameter.
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    mu = np.clip(mu, 1e-10, None)
    r = np.clip(r, 1e-10, None)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_y_mu = np.where(y > 0, y * np.log(y / mu), 0.0)
    dev = 2.0 * (log_y_mu - (y + r) * np.log((y + r) / (mu + r)))

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        return float(np.average(dev, weights=w))
    return float(np.mean(dev))


# ---------------------------------------------------------------------------
# Calibration diagnostics
# ---------------------------------------------------------------------------


def pit_values(
    y: np.ndarray,
    pred,  # DistributionalPrediction
    n_samples: int = 5000,
    seed: int = 42,
) -> np.ndarray:
    """
    Compute Probability Integral Transform (PIT) values.

    PIT(y_i) = F(y_i | x_i), where F is the predicted CDF.

    For a well-calibrated model, PIT values should be uniform on [0, 1].
    Computed via Monte Carlo: PIT_i ≈ P_F(X <= y_i) = mean(samples <= y_i).

    Parameters
    ----------
    y : array-like
        Observed values.
    pred : DistributionalPrediction
        Fitted distributional prediction.
    n_samples : int
        MC samples per observation.
    seed : int

    Returns
    -------
    np.ndarray
        PIT values in [0, 1], one per observation.
    """
    rng = np.random.default_rng(seed)
    samples = pred._sample(n_samples=n_samples, rng=rng)  # (n, n_samples)
    y_arr = np.asarray(y, dtype=np.float64)
    # PIT_i = fraction of samples <= y_i
    pit = (samples <= y_arr[:, None]).mean(axis=1)
    return pit.astype(np.float64)


def coverage(
    y: np.ndarray,
    pred,  # DistributionalPrediction
    levels: Tuple[float, ...] = (0.80, 0.90, 0.95),
    n_samples: int = 5000,
    seed: int = 42,
) -> dict:
    """
    Empirical coverage at specified prediction interval levels.

    For each level alpha, computes the fraction of observations falling
    within the central (1-alpha)/2 to (1+alpha)/2 prediction interval.
    For a well-calibrated model, empirical coverage should match nominal.

    Parameters
    ----------
    y : array-like
        Observed values.
    pred : DistributionalPrediction
    levels : tuple of float
        Interval levels to check, e.g. (0.80, 0.90, 0.95).
    n_samples : int
    seed : int

    Returns
    -------
    dict
        {level: empirical_coverage} for each requested level.
    """
    rng = np.random.default_rng(seed)
    samples = pred._sample(n_samples=n_samples, rng=rng)
    y_arr = np.asarray(y, dtype=np.float64)

    result = {}
    for alpha in levels:
        lo = (1.0 - alpha) / 2.0
        hi = 1.0 - lo
        lower = np.quantile(samples, lo, axis=1)
        upper = np.quantile(samples, hi, axis=1)
        in_interval = (y_arr >= lower) & (y_arr <= upper)
        result[alpha] = float(in_interval.mean())

    return result


def pearson_residuals(
    y: np.ndarray,
    pred,  # DistributionalPrediction
) -> np.ndarray:
    """
    Standardised Pearson residuals.

    r_i = (y_i - mu_i) / sqrt(Var[Y_i|x_i])

    Should be approximately standard normal for a well-specified model.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    return (y_arr - pred.mu) / (pred.std + 1e-10)


def gini_index(
    y: np.ndarray,
    score: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Normalised Gini index (aka 2*AUC - 1 for binary targets).

    For insurance pricing, computed on the Lorenz curve of actual losses
    ranked by predicted score. Measures discrimination power.

    Parameters
    ----------
    y : array-like
        Actual outcomes (losses, claims).
    score : array-like
        Predicted scores (predicted mean, CoV, etc.).
    weights : array-like, optional
        Observation weights (exposure).

    Returns
    -------
    float
        Normalised Gini in [-1, 1]. 1.0 = perfect discrimination.
    """
    y = np.asarray(y, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    if weights is None:
        weights = np.ones(len(y))
    weights = np.asarray(weights, dtype=np.float64)

    # Sort by predicted score ascending (low-risk first).
    # For a useful model, high losses are concentrated at the high end, so the
    # Lorenz curve lies below the diagonal (AUC < 0.5).
    # Gini = 1 - 2*AUC: perfect discrimination -> AUC near 0 -> Gini near 1.
    # Random model: AUC = 0.5 -> Gini = 0. Anti-rank: AUC > 0.5 -> Gini < 0.
    order = np.argsort(score)
    y_sorted = y[order]
    w_sorted = weights[order]

    # Lorenz curve coordinates
    cum_w = np.cumsum(w_sorted)
    cum_y = np.cumsum(y_sorted * w_sorted)
    total_w = cum_w[-1]
    total_y = cum_y[-1]

    if total_y == 0:
        return 0.0

    # Normalise to [0, 1]
    x_lorenz = cum_w / total_w
    y_lorenz = cum_y / total_y

    # Area under Lorenz curve via trapezoid rule
    auc = float(np.trapezoid(y_lorenz, x_lorenz))
    # Gini = 1 - 2*AUC: good model concentrates losses at high scores (end of
    # ascending sort), Lorenz curve below diagonal, AUC < 0.5, Gini > 0.
    return 1.0 - 2.0 * auc


# ---------------------------------------------------------------------------
# CDE scoring utilities (for FlexCodeDensity)
# ---------------------------------------------------------------------------


def cde_loss(
    cdes: np.ndarray,
    z_grid: np.ndarray,
    z_test: np.ndarray,
) -> float:
    """
    CDE loss (conditional density estimation loss, Gneiting & Raftery 2007).

    L = E[integral f_hat(z|X)^2 dz] - 2 * E[f_hat(Z|X)]

    Strictly proper: minimised uniquely by the true conditional density.
    Used to tune the number of basis functions in FlexCodeDensity.

    Parameters
    ----------
    cdes : np.ndarray, shape (n_test, n_grid)
        Predicted conditional density values on z_grid for each test obs.
    z_grid : np.ndarray, shape (n_grid,)
        Grid of z values at which cdes are evaluated.
    z_test : np.ndarray, shape (n_test,)
        True z-values (transformed targets) for test observations.

    Returns
    -------
    float
        CDE loss. Lower is better.

    Notes
    -----
    z_test should be in the same space as z_grid. If the model uses
    log_transform=True, pass z_test = log(y_test + epsilon).
    """
    cdes = np.asarray(cdes, dtype=np.float64)
    z_grid = np.asarray(z_grid, dtype=np.float64)
    z_test = np.asarray(z_test, dtype=np.float64)

    if cdes.ndim != 2:
        raise ValueError(f"cdes must be 2D (n_test, n_grid), got shape {cdes.shape}")
    if cdes.shape[1] != len(z_grid):
        raise ValueError(
            f"cdes.shape[1]={cdes.shape[1]} must match len(z_grid)={len(z_grid)}"
        )

    # term1: E[integral f_hat(z|X)^2 dz]
    term1 = float(np.mean(np.trapezoid(cdes ** 2, z_grid, axis=1)))

    # term2: E[f_hat(z_true | X)] — density evaluated at the true test point
    n_test = len(z_test)
    term2_vals = np.empty(n_test)
    for i in range(n_test):
        term2_vals[i] = float(np.interp(z_test[i], z_grid, cdes[i]))
    term2 = float(np.mean(term2_vals))

    return term1 - 2.0 * term2
