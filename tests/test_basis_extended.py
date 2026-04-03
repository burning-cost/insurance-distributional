"""
Extended tests for basis.py: cosine_basis, evaluate_density, postprocess_density.

The basis.py module has minimal existing test coverage. These tests focus on:
- Mathematical properties of the cosine orthonormal basis
- Input validation edge cases
- evaluate_density and postprocess_density correctness
- Numerical stability at boundary points
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_distributional.basis import (
    cosine_basis,
    evaluate_density,
    postprocess_density,
)


# ---------------------------------------------------------------------------
# cosine_basis: shape and structure
# ---------------------------------------------------------------------------


class TestCosineBasisShape:
    def test_output_shape(self):
        """Returns (len(z), n_basis) array."""
        z = np.linspace(0.0, 1.0, 50)
        result = cosine_basis(z, z_min=0.0, z_max=1.0, n_basis=5)
        assert result.shape == (50, 5)

    def test_single_point(self):
        """Works on a single z value."""
        result = cosine_basis(np.array([0.5]), z_min=0.0, z_max=1.0, n_basis=3)
        assert result.shape == (1, 3)

    def test_n_basis_one(self):
        """n_basis=1 returns the constant function only."""
        z = np.array([0.2, 0.5, 0.8])
        result = cosine_basis(z, z_min=0.0, z_max=1.0, n_basis=1)
        assert result.shape == (3, 1)
        # All constant: phi_1 = 1/sqrt(width) = 1/sqrt(1) = 1.0
        np.testing.assert_allclose(result[:, 0], 1.0)


class TestCosineBasisValues:
    def test_constant_basis_function(self):
        """phi_1 = 1/sqrt(width) for all z."""
        z = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        width = 1.0
        result = cosine_basis(z, z_min=0.0, z_max=1.0, n_basis=3)
        expected_phi1 = 1.0 / np.sqrt(width)
        np.testing.assert_allclose(result[:, 0], expected_phi1, rtol=1e-12)

    def test_cosine_basis_at_zmin(self):
        """cos(0) = 1, so phi_2 at z=z_min should be sqrt(2/width)."""
        z = np.array([0.0])
        width = 2.0
        result = cosine_basis(z, z_min=0.0, z_max=2.0, n_basis=2)
        expected_phi2 = np.sqrt(2.0 / width) * np.cos(0.0)  # = sqrt(2/2) = 1
        np.testing.assert_allclose(result[0, 1], expected_phi2, rtol=1e-12)

    def test_non_unit_width(self):
        """Basis norms correctly for non-unit-width domain."""
        z_min, z_max = 500.0, 3000.0
        z = np.array([500.0, 1750.0, 3000.0])
        width = z_max - z_min
        result = cosine_basis(z, z_min=z_min, z_max=z_max, n_basis=1)
        expected = 1.0 / np.sqrt(width)
        np.testing.assert_allclose(result[:, 0], expected, rtol=1e-12)

    def test_orthonormality_approx(self):
        """
        The Gram matrix ~= I: integral phi_i * phi_j dz ≈ delta_{ij}.

        Verified via trapezoidal rule on a fine grid.
        """
        z_min, z_max = 0.0, 1.0
        n_basis = 5
        z_grid = np.linspace(z_min, z_max, 2000)
        B = cosine_basis(z_grid, z_min, z_max, n_basis)  # (2000, 5)
        # Gram matrix: each entry = integral phi_i * phi_j dz
        gram = np.trapezoid(B[:, :, None] * B[:, None, :], z_grid, axis=0)
        np.testing.assert_allclose(gram, np.eye(n_basis), atol=0.01)


class TestCosineBasisValidation:
    def test_zmax_leq_zmin_raises(self):
        """z_max <= z_min should raise ValueError."""
        z = np.array([0.5])
        with pytest.raises(ValueError, match="z_max must be greater than z_min"):
            cosine_basis(z, z_min=1.0, z_max=0.0, n_basis=3)

    def test_zmax_equals_zmin_raises(self):
        """z_max == z_min is degenerate."""
        z = np.array([0.5])
        with pytest.raises(ValueError, match="z_max must be greater than z_min"):
            cosine_basis(z, z_min=1.0, z_max=1.0, n_basis=3)

    def test_n_basis_zero_raises(self):
        """n_basis < 1 should raise ValueError."""
        z = np.array([0.5])
        with pytest.raises(ValueError, match="n_basis must be >= 1"):
            cosine_basis(z, z_min=0.0, z_max=1.0, n_basis=0)

    def test_n_basis_negative_raises(self):
        z = np.array([0.5])
        with pytest.raises(ValueError, match="n_basis must be >= 1"):
            cosine_basis(z, z_min=0.0, z_max=1.0, n_basis=-5)


# ---------------------------------------------------------------------------
# evaluate_density
# ---------------------------------------------------------------------------


class TestEvaluateDensity:
    def test_output_shape(self):
        """Returns (n_obs, n_grid)."""
        n_obs = 10
        n_basis = 6
        n_grid = 50
        coefs = np.random.default_rng(0).standard_normal((n_obs, n_basis))
        z_grid = np.linspace(0.0, 1.0, n_grid)
        result = evaluate_density(coefs, z_grid, z_min=0.0, z_max=1.0)
        assert result.shape == (n_obs, n_grid)

    def test_constant_coefficient_gives_constant_density(self):
        """
        If coefs = [c, 0, 0, ...], density = c * phi_1(z) = c / sqrt(width).
        Constant across z.
        """
        n_obs = 3
        n_basis = 4
        width = 2.0
        z_grid = np.linspace(0.0, 2.0, 100)
        c = 5.0
        coefs = np.zeros((n_obs, n_basis))
        coefs[:, 0] = c  # only the constant basis

        result = evaluate_density(coefs, z_grid, z_min=0.0, z_max=2.0)
        expected = c / np.sqrt(width)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_zero_coefficients_gives_zero_density(self):
        """All-zero coefficients -> zero density everywhere."""
        coefs = np.zeros((5, 4))
        z_grid = np.linspace(0.0, 1.0, 30)
        result = evaluate_density(coefs, z_grid, z_min=0.0, z_max=1.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# postprocess_density
# ---------------------------------------------------------------------------


class TestPostprocessDensity:
    def test_output_nonneg(self):
        """postprocess_density should clip all values to >= 0."""
        rng = np.random.default_rng(1)
        cdes = rng.standard_normal((10, 50))  # contains negatives
        z_grid = np.linspace(0.0, 1.0, 50)
        result = postprocess_density(cdes, z_grid)
        assert np.all(result >= 0.0)

    def test_integrates_to_one(self):
        """
        Each row should integrate to approximately 1 after postprocessing.
        """
        rng = np.random.default_rng(2)
        n_obs = 20
        n_grid = 200
        # Use mostly positive densities
        cdes = np.abs(rng.standard_normal((n_obs, n_grid))) + 0.5
        z_grid = np.linspace(0.0, 5.0, n_grid)
        result = postprocess_density(cdes, z_grid)
        integrals = np.trapezoid(result, z_grid, axis=1)
        np.testing.assert_allclose(integrals, 1.0, atol=0.01)

    def test_all_negative_row_handled(self):
        """
        When an entire row is negative (clipped to 0), postprocessing should
        not divide by zero or produce NaN. The row stays zero.
        """
        n_grid = 50
        cdes = np.ones((3, n_grid))
        cdes[1, :] = -1.0  # all negative row
        z_grid = np.linspace(0.0, 1.0, n_grid)
        result = postprocess_density(cdes, z_grid)
        assert np.all(np.isfinite(result))
        # All-zero row stays zero after no-op normalisation
        np.testing.assert_allclose(result[1, :], 0.0, atol=1e-15)

    def test_preserves_shape(self):
        """Output shape matches input shape."""
        cdes = np.ones((7, 30))
        z_grid = np.linspace(0.0, 1.0, 30)
        result = postprocess_density(cdes, z_grid)
        assert result.shape == (7, 30)


# ---------------------------------------------------------------------------
# cde_loss (from scoring.py — tests the basis integration path)
# ---------------------------------------------------------------------------


class TestCdeLoss:
    def test_cde_loss_basic(self):
        """
        cde_loss returns a finite float for well-formed inputs.
        """
        from insurance_distributional.scoring import cde_loss

        rng = np.random.default_rng(0)
        n_test = 30
        n_grid = 80
        z_grid = np.linspace(0.0, 3.0, n_grid)
        # Uniform density: cdes = 1/width everywhere
        width = 3.0
        cdes = np.full((n_test, n_grid), 1.0 / width)
        z_test = rng.uniform(0.0, 3.0, n_test)
        loss = cde_loss(cdes, z_grid, z_test)
        assert np.isfinite(loss)
        assert isinstance(loss, float)

    def test_cde_loss_shape_mismatch_raises(self):
        """
        cdes.shape[1] != len(z_grid) should raise ValueError.
        """
        from insurance_distributional.scoring import cde_loss

        cdes = np.ones((10, 50))
        z_grid = np.linspace(0.0, 1.0, 40)  # length mismatch
        z_test = np.ones(10) * 0.5
        with pytest.raises(ValueError, match="z_grid"):
            cde_loss(cdes, z_grid, z_test)

    def test_cde_loss_1d_cdes_raises(self):
        """1D cdes should raise ValueError."""
        from insurance_distributional.scoring import cde_loss

        cdes = np.ones(50)  # 1D, not 2D
        z_grid = np.linspace(0.0, 1.0, 50)
        z_test = np.array([0.5])
        with pytest.raises(ValueError, match="2D"):
            cde_loss(cdes, z_grid, z_test)

    def test_perfect_density_lower_than_flat(self):
        """
        A density concentrated at the true z_test values scores better (lower)
        than a flat uniform density.
        """
        from insurance_distributional.scoring import cde_loss

        n_test = 50
        n_grid = 100
        z_min, z_max = 0.0, 5.0
        z_grid = np.linspace(z_min, z_max, n_grid)
        width = z_max - z_min

        # All test points at z=2.5
        z_test = np.full(n_test, 2.5)

        # Flat density: 1/width everywhere
        cdes_flat = np.full((n_test, n_grid), 1.0 / width)

        # "Good" density: high mass near z=2.5
        cdes_good = np.ones((n_test, n_grid)) * 0.1
        idx = np.argmin(np.abs(z_grid - 2.5))
        cdes_good[:, max(0, idx - 3): idx + 4] = 5.0

        loss_flat = cde_loss(cdes_flat, z_grid, z_test)
        loss_good = cde_loss(cdes_good, z_grid, z_test)

        assert loss_good < loss_flat, (
            f"Concentrated density ({loss_good:.4f}) should score better "
            f"than flat ({loss_flat:.4f})"
        )
