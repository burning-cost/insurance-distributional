"""
Cosine orthonormal basis for FlexCode conditional density estimation.

The cosine system is complete and orthonormal on [z_min, z_max]:

  phi_1(z) = 1 / sqrt(z_max - z_min)
  phi_i(z) = sqrt(2 / (z_max - z_min)) * cos(pi * (i-1) * (z - z_min) / (z_max - z_min))
             for i = 2, ..., n_basis

This is a direct implementation of the basis from Izbicki & Lee (2017),
Electronic Journal of Statistics 11:2800-2831.

Why reimplement rather than import upstream flexcode?
The upstream `flexcode` PyPI package is GPL-2.0, which is incompatible with
our MIT licence. The cosine basis itself is simple mathematics — no copyright
can attach to the formula. This 80-line reimplementation is entirely our own
code, written from the published paper.
"""

from __future__ import annotations

import numpy as np


def cosine_basis(
    z: np.ndarray,
    z_min: float,
    z_max: float,
    n_basis: int,
) -> np.ndarray:
    """
    Evaluate cosine orthonormal basis functions on z.

    phi_1(z) = 1 / sqrt(width)
    phi_i(z) = sqrt(2/width) * cos(pi * (i-1) * (z - z_min) / width)
               for i = 2, ..., n_basis

    These are orthonormal on [z_min, z_max]: integral_{z_min}^{z_max} phi_i * phi_j dz = delta_{ij}.

    Parameters
    ----------
    z : np.ndarray, shape (n,)
        Points at which to evaluate the basis. Should lie in [z_min, z_max].
        Values outside this range are extrapolated (cosine is well-defined
        everywhere but the orthonormality guarantee only holds on [z_min, z_max]).
    z_min : float
        Lower domain bound.
    z_max : float
        Upper domain bound.
    n_basis : int
        Number of basis functions to evaluate.

    Returns
    -------
    np.ndarray of shape (len(z), n_basis)
        Column i is phi_{i+1}(z) (0-indexed: column 0 is phi_1, column 1 is phi_2, ...).

    Notes
    -----
    The Gram matrix integral_{z_min}^{z_max} B.T @ B dz should be close to
    the identity matrix (up to grid approximation error). Verified in tests.
    """
    z = np.asarray(z, dtype=np.float64)
    width = float(z_max - z_min)
    if width <= 0:
        raise ValueError(f"z_max must be greater than z_min, got z_min={z_min}, z_max={z_max}")
    if n_basis < 1:
        raise ValueError(f"n_basis must be >= 1, got {n_basis}")

    n = len(z)
    basis = np.empty((n, n_basis), dtype=np.float64)

    # phi_1: constant function
    basis[:, 0] = 1.0 / np.sqrt(width)

    if n_basis > 1:
        # Scaled argument: (z - z_min) / width in [0, 1]
        z_scaled = (z - z_min) / width
        scale = np.sqrt(2.0 / width)
        for i in range(1, n_basis):
            basis[:, i] = scale * np.cos(np.pi * i * z_scaled)

    return basis


def evaluate_density(
    coefs: np.ndarray,
    z_grid: np.ndarray,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    """
    Evaluate density estimates from basis coefficients on a grid.

    Computes f_hat(z | x) = sum_i coefs[i] * phi_i(z) for each observation.

    Parameters
    ----------
    coefs : np.ndarray, shape (n_obs, n_basis)
        Predicted basis coefficients beta_hat_i(x) for each observation.
    z_grid : np.ndarray, shape (n_grid,)
        Grid of z values at which to evaluate the density.
    z_min : float
        Domain lower bound (used to construct the basis).
    z_max : float
        Domain upper bound.

    Returns
    -------
    np.ndarray of shape (n_obs, n_grid)
        Raw density estimates. May contain negative values — caller must
        clip and renormalise.
    """
    n_basis = coefs.shape[1]
    B = cosine_basis(z_grid, z_min, z_max, n_basis)  # (n_grid, n_basis)
    # (n_obs, n_basis) @ (n_basis, n_grid) -> (n_obs, n_grid)
    cdes = coefs @ B.T
    return cdes


def postprocess_density(
    cdes: np.ndarray,
    z_grid: np.ndarray,
) -> np.ndarray:
    """
    Clip negative density values and renormalise to integrate to 1.

    Basis truncation can produce small negative values, especially in the tails.
    This post-processing step is mandatory for valid probability density estimates.

    Parameters
    ----------
    cdes : np.ndarray, shape (n_obs, n_grid)
        Raw density values (may contain negatives).
    z_grid : np.ndarray, shape (n_grid,)
        Grid points corresponding to cdes columns.

    Returns
    -------
    np.ndarray of shape (n_obs, n_grid)
        Non-negative densities that integrate (approximately) to 1 over z_grid.
    """
    cdes = np.clip(cdes, 0.0, None)

    # Renormalise each row: integral f(z) dz = 1
    norms = np.trapezoid(cdes, z_grid, axis=1)  # (n_obs,)
    # Avoid division by zero for observations where all density was clipped
    norms = np.where(norms > 0, norms, 1.0)
    cdes = cdes / norms[:, None]

    return cdes
