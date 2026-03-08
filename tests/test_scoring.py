"""
Tests for scoring utilities.

Tests both the mathematical correctness of deviance formulas and the
calibration diagnostics.
"""

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
# Deviance metrics
# ---------------------------------------------------------------------------


class TestTweedieDeviance:
    def test_perfect_predictions_near_zero(self):
        """Perfect predictions should give near-zero deviance."""
        y = np.array([100.0, 200.0, 300.0])
        mu = y.copy()
        dev = tweedie_deviance(y, mu, power=1.5)
        assert dev < 1e-8

    def test_worse_predictions_higher_deviance(self):
        y = np.array([100.0, 200.0, 300.0])
        mu_good = y * 1.01
        mu_bad = y * 2.0
        assert tweedie_deviance(y, mu_good, 1.5) < tweedie_deviance(y, mu_bad, 1.5)

    def test_zeros_handled(self):
        """y=0 values should not cause errors."""
        y = np.array([0.0, 100.0, 0.0, 200.0])
        mu = np.array([10.0, 100.0, 5.0, 200.0])
        dev = tweedie_deviance(y, mu, power=1.5)
        assert np.isfinite(dev)
        assert dev >= 0

    def test_weighted_deviance(self):
        y = np.array([100.0, 200.0])
        mu = np.array([110.0, 210.0])
        w = np.array([1.0, 10.0])
        dev_unweighted = tweedie_deviance(y, mu, 1.5)
        dev_weighted = tweedie_deviance(y, mu, 1.5, weights=w)
        # Should differ because of unequal weights
        assert dev_weighted != dev_unweighted

    def test_nonneg(self):
        rng = np.random.default_rng(0)
        y = np.abs(rng.standard_normal(50)) * 100
        mu = y * rng.uniform(0.8, 1.2, 50)
        assert tweedie_deviance(y, mu, 1.5) >= 0


class TestPoissonDeviance:
    def test_perfect_predictions(self):
        y = np.array([1.0, 3.0, 5.0])
        dev = poisson_deviance(y, y.copy())
        assert dev < 1e-8

    def test_zeros_handled(self):
        y = np.array([0.0, 1.0, 2.0])
        mu = np.array([0.5, 1.0, 2.0])
        dev = poisson_deviance(y, mu)
        assert np.isfinite(dev)

    def test_worse_worse(self):
        y = np.array([2.0, 3.0, 5.0])
        assert poisson_deviance(y, y * 1.1) < poisson_deviance(y, y * 2.0)


class TestGammaDeviance:
    def test_perfect_predictions(self):
        y = np.array([500.0, 1000.0, 2000.0])
        dev = gamma_deviance(y, y.copy())
        assert dev < 1e-8

    def test_nonneg(self):
        rng = np.random.default_rng(1)
        y = rng.gamma(2.0, 500.0, 100)
        mu = y * rng.uniform(0.5, 2.0, 100)
        assert gamma_deviance(y, mu) >= 0


class TestNegBinomDeviance:
    def test_finite(self):
        rng = np.random.default_rng(2)
        y = rng.negative_binomial(n=3, p=0.5, size=50).astype(float)
        mu = np.full(50, 3.0)
        r = np.full(50, 3.0)
        dev = negbinom_deviance(y, mu, r)
        assert np.isfinite(dev)


# ---------------------------------------------------------------------------
# Calibration diagnostics
# ---------------------------------------------------------------------------


class TestPITValues:
    def test_shape(self):
        """PIT should return one value per observation."""
        rng = np.random.default_rng(42)
        n = 20
        mu = rng.uniform(100, 500, n)
        phi = np.full(n, 0.4)
        pred = DistributionalPrediction("tweedie", mu=mu, phi=phi, power=1.5)
        y = np.abs(rng.standard_normal(n)) * 200
        pit = pit_values(y, pred, n_samples=500, seed=0)
        assert pit.shape == (n,)

    def test_range(self):
        """PIT values must be in [0, 1]."""
        rng = np.random.default_rng(42)
        n = 30
        mu = rng.uniform(1.0, 5.0, n)
        phi = np.full(n, 0.5)
        pred = DistributionalPrediction("tweedie", mu=mu, phi=phi, power=1.5)
        y = np.abs(rng.standard_normal(n)) * 3
        pit = pit_values(y, pred, n_samples=500)
        assert np.all(pit >= 0) and np.all(pit <= 1)

    def test_calibration_uniform_ish(self):
        """
        For well-calibrated Gamma predictions, PIT values should be roughly
        uniform. Test that the distribution isn't heavily skewed.
        """
        rng = np.random.default_rng(99)
        n = 200
        mu = np.full(n, 500.0)
        phi = np.full(n, 0.5)  # shape = 2
        # Sample y from the SAME distribution as predicted
        y = rng.gamma(shape=2.0, scale=250.0, size=n)  # mu=500, phi=0.5
        pred = DistributionalPrediction("gamma", mu=mu, phi=phi)
        pit = pit_values(y, pred, n_samples=1000, seed=42)
        # For uniform [0,1], mean should be ~0.5
        assert 0.35 < pit.mean() < 0.65


class TestCoverage:
    def test_returns_dict(self):
        rng = np.random.default_rng(42)
        n = 30
        mu = rng.uniform(100, 500, n)
        phi = np.full(n, 0.4)
        pred = DistributionalPrediction("gamma", mu=mu, phi=phi)
        y = rng.gamma(2.5, 200.0, n)
        result = coverage(y, pred, levels=(0.80, 0.95), n_samples=500)
        assert set(result.keys()) == {0.80, 0.95}
        for val in result.values():
            assert 0.0 <= val <= 1.0


class TestPearsonResiduals:
    def test_shape(self):
        n = 50
        rng = np.random.default_rng(0)
        mu = rng.uniform(100, 500, n)
        phi = np.full(n, 0.5)
        pred = DistributionalPrediction("gamma", mu=mu, phi=phi)
        y = rng.gamma(2.0, 250.0, n)
        resid = pearson_residuals(y, pred)
        assert resid.shape == (n,)

    def test_perfect_zero(self):
        mu = np.array([100.0, 200.0])
        phi = np.array([0.5, 0.5])
        pred = DistributionalPrediction("gamma", mu=mu, phi=phi)
        # y = mu -> residual = 0
        resid = pearson_residuals(mu, pred)
        np.testing.assert_allclose(resid, 0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# Gini index
# ---------------------------------------------------------------------------


class TestGiniIndex:
    def test_random_near_zero(self):
        """Random predictions should give near-zero Gini."""
        rng = np.random.default_rng(42)
        y = rng.exponential(100, 500)
        score = rng.exponential(100, 500)  # independent of y
        g = gini_index(y, score)
        assert abs(g) < 0.2  # loose bound for random

    def test_perfect_discrimination(self):
        """Perfect ranking gives Gini close to 1."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        score = y.copy()
        g = gini_index(y, score)
        assert g > 0.8

    def test_anti_rank(self):
        """Reversed ranking gives negative Gini."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        score = -y.copy()
        g = gini_index(y, score)
        assert g < 0

    def test_weighted(self):
        y = np.array([1.0, 2.0, 3.0])
        score = y.copy()
        w = np.array([1.0, 1.0, 1.0])
        g1 = gini_index(y, score)
        g2 = gini_index(y, score, weights=w)
        assert abs(g1 - g2) < 1e-6  # uniform weights should match

    def test_range(self):
        rng = np.random.default_rng(7)
        y = rng.exponential(1, 100)
        score = rng.exponential(1, 100)
        g = gini_index(y, score)
        assert -1.0 <= g <= 1.0
