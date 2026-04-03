"""
Extended scoring tests covering gaps in test_scoring.py.

Targets:
- negbinom_deviance: perfect prediction, weighted, zeros
- coverage: calibration (empirical coverage close to nominal for well-specified model)
- gini_index: all-same y edge case (zero Gini)
- pearson_residuals: sign and scale
- poisson_deviance: weighted variant
- cde_loss: additional correctness checks (covered in test_basis_extended, some overlap ok)
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_distributional.prediction import DistributionalPrediction
from insurance_distributional.scoring import (
    coverage,
    gamma_deviance,
    gini_index,
    negbinom_deviance,
    pearson_residuals,
    pit_values,
    poisson_deviance,
    tweedie_deviance,
)


# ---------------------------------------------------------------------------
# negbinom_deviance: extended tests
# ---------------------------------------------------------------------------


class TestNegBinomDevianceExtended:
    def test_perfect_predictions_near_zero(self):
        """NegBinom deviance at perfect prediction should be near zero."""
        rng = np.random.default_rng(10)
        y = rng.negative_binomial(n=5, p=0.5, size=30).astype(float)
        mu = y.copy()
        r = np.full(30, 5.0)
        dev = negbinom_deviance(y, mu, r)
        assert dev < 1e-8, f"Expected near-zero deviance at y=mu, got {dev}"

    def test_zeros_handled(self):
        """Zero counts should not cause NaN or inf."""
        y = np.array([0.0, 1.0, 0.0, 3.0])
        mu = np.array([0.5, 1.0, 0.3, 3.0])
        r = np.full(4, 3.0)
        dev = negbinom_deviance(y, mu, r)
        assert np.isfinite(dev)

    def test_weighted_deviance(self):
        """Weighted NegBinom deviance differs from unweighted with unequal weights."""
        y = np.array([2.0, 5.0, 1.0])
        mu = np.array([2.5, 5.5, 1.5])
        r = np.full(3, 3.0)
        w = np.array([1.0, 10.0, 1.0])
        dev_uw = negbinom_deviance(y, mu, r)
        dev_w = negbinom_deviance(y, mu, r, weights=w)
        assert dev_uw != dev_w

    def test_worse_predictions_higher_deviance(self):
        """Better predictions should give lower deviance."""
        y = np.array([2.0, 3.0, 5.0])
        r = np.full(3, 4.0)
        mu_close = y * 1.05
        mu_far = y * 3.0
        assert negbinom_deviance(y, mu_close, r) < negbinom_deviance(y, mu_far, r)

    def test_nonneg(self):
        """NegBinom deviance should always be >= 0."""
        rng = np.random.default_rng(20)
        y = rng.negative_binomial(3, 0.5, 50).astype(float)
        mu = rng.uniform(1.0, 8.0, 50)
        r = np.full(50, 3.0)
        assert negbinom_deviance(y, mu, r) >= 0


# ---------------------------------------------------------------------------
# poisson_deviance: weighted variant
# ---------------------------------------------------------------------------


class TestPoissonDevianceWeighted:
    def test_weighted_deviance_differs(self):
        """Weighted and unweighted deviance differ with non-uniform weights."""
        y = np.array([1.0, 4.0, 2.0])
        mu = np.array([1.2, 4.5, 2.3])
        w = np.array([1.0, 20.0, 1.0])
        dev_uw = poisson_deviance(y, mu)
        dev_w = poisson_deviance(y, mu, weights=w)
        assert dev_uw != dev_w

    def test_equal_weights_same_as_unweighted(self):
        """Uniform weights give same result as unweighted."""
        y = np.array([2.0, 3.0, 5.0])
        mu = np.array([2.1, 3.1, 5.1])
        w = np.ones(3)
        dev_uw = poisson_deviance(y, mu)
        dev_w = poisson_deviance(y, mu, weights=w)
        assert abs(dev_uw - dev_w) < 1e-12


# ---------------------------------------------------------------------------
# gamma_deviance: additional
# ---------------------------------------------------------------------------


class TestGammaDevianceExtended:
    def test_weighted_gamma_deviance(self):
        """Weighted gamma deviance differs from unweighted when weights vary."""
        y = np.array([500.0, 2000.0, 800.0])
        mu = np.array([520.0, 2100.0, 850.0])
        w = np.array([1.0, 10.0, 1.0])
        dev_uw = gamma_deviance(y, mu)
        dev_w = gamma_deviance(y, mu, weights=w)
        assert abs(dev_uw - dev_w) > 1e-8  # should differ


# ---------------------------------------------------------------------------
# coverage: calibration check
# ---------------------------------------------------------------------------


class TestCoverageCalibration:
    def test_well_calibrated_coverage_near_nominal(self):
        """
        For a well-calibrated Gamma model (same DGP as prediction), empirical
        coverage should be close to the nominal level at 80% and 95%.

        Uses n=500 to get reasonable MC stability. Tolerance is ±15 percentage
        points — loose enough to avoid flakiness but tight enough to catch a
        badly mis-calibrated model.
        """
        rng = np.random.default_rng(55)
        n = 500
        shape = 4.0        # phi = 1/shape = 0.25
        mu_val = 1000.0
        mu = np.full(n, mu_val)
        phi = np.full(n, 1.0 / shape)
        pred = DistributionalPrediction("gamma", mu=mu, phi=phi)
        # Sample from the same distribution
        y = rng.gamma(shape=shape, scale=mu_val / shape, size=n)

        result = coverage(y, pred, levels=(0.80, 0.95), n_samples=2000, seed=7)
        assert abs(result[0.80] - 0.80) < 0.15, (
            f"80% coverage: expected 0.80 ± 0.15, got {result[0.80]:.3f}"
        )
        assert abs(result[0.95] - 0.95) < 0.15, (
            f"95% coverage: expected 0.95 ± 0.15, got {result[0.95]:.3f}"
        )

    def test_coverage_values_in_0_1(self):
        """Coverage values must be between 0 and 1."""
        rng = np.random.default_rng(10)
        n = 50
        mu = rng.uniform(100, 500, n)
        phi = np.full(n, 0.5)
        pred = DistributionalPrediction("gamma", mu=mu, phi=phi)
        y = rng.gamma(2.0, mu / 2.0, n)
        result = coverage(y, pred, levels=(0.50, 0.75, 0.90), n_samples=300)
        for level, val in result.items():
            assert 0.0 <= val <= 1.0, f"Coverage at {level}: {val} not in [0, 1]"

    def test_higher_level_higher_coverage(self):
        """
        For a well-calibrated model, 95% coverage should generally be >=
        80% coverage (monotone in level).
        """
        rng = np.random.default_rng(42)
        n = 200
        mu = np.full(n, 500.0)
        phi = np.full(n, 0.25)
        pred = DistributionalPrediction("gamma", mu=mu, phi=phi)
        y = rng.gamma(shape=4.0, scale=125.0, size=n)
        result = coverage(y, pred, levels=(0.80, 0.95), n_samples=2000, seed=0)
        assert result[0.95] >= result[0.80] - 0.05  # allow small MC fluctuation


# ---------------------------------------------------------------------------
# pearson_residuals: sign and scale checks
# ---------------------------------------------------------------------------


class TestPearsonResidualsExtended:
    def test_positive_residual_when_y_above_mu(self):
        """When y > mu, Pearson residual should be positive."""
        mu = np.array([100.0, 200.0])
        phi = np.array([0.5, 0.5])
        pred = DistributionalPrediction("gamma", mu=mu, phi=phi)
        y = np.array([150.0, 250.0])  # both above mu
        resid = pearson_residuals(y, pred)
        assert np.all(resid > 0)

    def test_negative_residual_when_y_below_mu(self):
        """When y < mu, Pearson residual should be negative."""
        mu = np.array([300.0, 400.0])
        phi = np.array([0.4, 0.4])
        pred = DistributionalPrediction("gamma", mu=mu, phi=phi)
        y = np.array([200.0, 300.0])  # both below mu
        resid = pearson_residuals(y, pred)
        assert np.all(resid < 0)

    def test_residual_scale(self):
        """
        Pearson residual = (y - mu) / std. Verify the magnitude is approximately
        right for a known Gamma parameterisation.

        Gamma with mu=500, phi=0.25: std = sqrt(phi) * mu = 0.5 * 500 = 250.
        If y=750: residual should be ~(750-500)/250 = 1.0.
        """
        mu = np.array([500.0])
        phi = np.array([0.25])  # CoV = sqrt(0.25) = 0.5, std = 250
        pred = DistributionalPrediction("gamma", mu=mu, phi=phi)
        y = np.array([750.0])
        resid = pearson_residuals(y, pred)
        np.testing.assert_allclose(resid, [1.0], atol=1e-6)


# ---------------------------------------------------------------------------
# gini_index: edge cases
# ---------------------------------------------------------------------------


class TestGiniIndexEdgeCases:
    def test_all_y_equal_gini_zero(self):
        """
        When all y are identical, the Lorenz curve is the diagonal and Gini = 0.
        """
        y = np.array([100.0, 100.0, 100.0, 100.0])
        score = np.array([1.0, 2.0, 3.0, 4.0])
        g = gini_index(y, score)
        assert abs(g) < 1e-10, f"Expected Gini=0 for uniform y, got {g}"

    def test_all_y_zero_gini_zero(self):
        """If total_y = 0 (all losses zero), Gini is defined as 0."""
        y = np.zeros(5)
        score = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        g = gini_index(y, score)
        assert g == 0.0

    def test_gini_is_minus_one_for_perfect_inverse(self):
        """
        With two observations where all loss is in the first (rank-ascending) position:
        sorted ascending -> all loss at the start -> Lorenz curve above diagonal -> Gini < 0.
        """
        y = np.array([1000.0, 1.0])
        score = np.array([1.0, 2.0])  # ascending: 1.0 first (1000 loss), 2.0 second (1 loss)
        g = gini_index(y, score)
        # Large-loss observation sorted first (lowest score) means Lorenz curve above diagonal
        assert g < 0

    def test_single_observation(self):
        """Single observation: Gini is trivially 0 (no discrimination)."""
        g = gini_index(np.array([500.0]), np.array([1.0]))
        # With 1 obs, cumulative shares both = 1 at the only point, AUC = 0.5, Gini = 0
        assert abs(g) < 1e-10


# ---------------------------------------------------------------------------
# tweedie_deviance: additional power values
# ---------------------------------------------------------------------------


class TestTweedieDeviancePower:
    def test_different_powers_give_different_deviances(self):
        """Two different Tweedie powers should give different deviances."""
        y = np.array([100.0, 200.0, 300.0])
        mu = y * 1.1
        dev_15 = tweedie_deviance(y, mu, power=1.5)
        dev_18 = tweedie_deviance(y, mu, power=1.8)
        assert abs(dev_15 - dev_18) > 1e-8

    def test_mu_clipped_to_positive(self):
        """Even if mu contains very small values, deviance should be finite."""
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1e-15, 2.0, 3.0])  # first is essentially zero
        dev = tweedie_deviance(y, mu, power=1.5)
        assert np.isfinite(dev)
