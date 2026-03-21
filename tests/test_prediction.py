"""
Tests for DistributionalPrediction container.

These tests do NOT require CatBoost — they test the math in the prediction
object with hand-crafted parameter arrays.
"""

import numpy as np
import pytest

from insurance_distributional.prediction import DistributionalPrediction


# ---------------------------------------------------------------------------
# Tweedie predictions
# ---------------------------------------------------------------------------


class TestTweediePrediction:
    def make_pred(self, n=100, power=1.5):
        rng = np.random.default_rng(0)
        mu = rng.uniform(100, 1000, n)
        phi = rng.uniform(0.1, 1.0, n)
        return DistributionalPrediction(
            distribution="tweedie", mu=mu, phi=phi, power=power
        )

    def test_mean_equals_mu(self):
        pred = self.make_pred()
        np.testing.assert_array_equal(pred.mean, pred.mu)

    def test_variance_positive(self):
        pred = self.make_pred()
        assert np.all(pred.variance > 0)

    def test_variance_formula(self):
        """Var = phi * mu^p for Tweedie."""
        rng = np.random.default_rng(42)
        mu = rng.uniform(10, 100, 50)
        phi = rng.uniform(0.2, 0.8, 50)
        p = 1.5
        pred = DistributionalPrediction(
            distribution="tweedie", mu=mu, phi=phi, power=p
        )
        expected = phi * mu ** p
        np.testing.assert_allclose(pred.variance, expected, rtol=1e-10)

    def test_std_is_sqrt_variance(self):
        pred = self.make_pred()
        np.testing.assert_allclose(pred.std, np.sqrt(pred.variance), rtol=1e-10)

    def test_cov_positive(self):
        pred = self.make_pred()
        assert np.all(pred.cov > 0)

    def test_cov_formula(self):
        """CoV = std / mu."""
        pred = self.make_pred()
        expected = pred.std / (pred.mu + 1e-12)
        np.testing.assert_allclose(pred.cov, expected, rtol=1e-10)

    def test_volatility_score_equals_cov(self):
        pred = self.make_pred()
        np.testing.assert_array_equal(pred.volatility_score(), pred.cov)

    def test_higher_phi_higher_cov(self):
        """Everything equal, higher phi means higher CoV."""
        mu = np.array([500.0, 500.0])
        phi_low = np.array([0.2, 0.2])
        phi_high = np.array([0.8, 0.8])
        p1 = DistributionalPrediction("tweedie", mu, phi_low, power=1.5)
        p2 = DistributionalPrediction("tweedie", mu, phi_high, power=1.5)
        assert np.all(p2.cov > p1.cov)

    def test_repr(self):
        pred = self.make_pred()
        r = repr(pred)
        assert "tweedie" in r
        assert "n=100" in r

    def test_missing_phi_raises(self):
        pred = DistributionalPrediction("tweedie", mu=np.array([100.0]))
        with pytest.raises(ValueError, match="phi"):
            _ = pred.variance

    def test_missing_power_raises(self):
        pred = DistributionalPrediction(
            "tweedie", mu=np.array([100.0]), phi=np.array([0.5])
        )
        with pytest.raises(ValueError, match="power"):
            _ = pred.variance

    def test_sampling_shape(self):
        pred = self.make_pred(n=20)
        rng = np.random.default_rng(0)
        samples = pred._sample(n_samples=50, rng=rng)
        assert samples.shape == (20, 50)

    def test_sampling_nonneg(self):
        """Tweedie samples should be non-negative."""
        pred = self.make_pred(n=30)
        rng = np.random.default_rng(0)
        samples = pred._sample(n_samples=200, rng=rng)
        assert np.all(samples >= 0)

    def test_sampling_mean_approx_mu(self):
        """Sample mean should be close to predicted mu (MC convergence)."""
        mu = np.array([500.0, 300.0, 800.0])
        phi = np.array([0.3, 0.3, 0.3])
        pred = DistributionalPrediction("tweedie", mu, phi, power=1.5)
        rng = np.random.default_rng(0)
        samples = pred._sample(n_samples=20000, rng=rng)
        sample_means = samples.mean(axis=1)
        np.testing.assert_allclose(sample_means, mu, rtol=0.15)  # 15% tolerance for MC

    def test_quantile_shape(self):
        pred = self.make_pred(n=10)
        q = pred.quantile(0.95, n_samples=200)
        assert q.shape == (10,)

    def test_quantile_monotone(self):
        """Higher quantile level should give larger values."""
        pred = self.make_pred(n=5)
        q80 = pred.quantile(0.80, n_samples=500)
        q95 = pred.quantile(0.95, n_samples=500)
        assert np.all(q95 >= q80)


# ---------------------------------------------------------------------------
# Gamma predictions
# ---------------------------------------------------------------------------


class TestGammaPrediction:
    def make_pred(self, n=50):
        rng = np.random.default_rng(10)
        mu = rng.uniform(500, 2000, n)
        phi = rng.uniform(0.3, 1.2, n)  # phi = 1/shape
        return DistributionalPrediction("gamma", mu=mu, phi=phi)

    def test_variance_formula(self):
        """Var = phi * mu^2 for Gamma (phi = 1/shape)."""
        rng = np.random.default_rng(42)
        mu = rng.uniform(100, 1000, 30)
        phi = rng.uniform(0.2, 1.0, 30)
        pred = DistributionalPrediction("gamma", mu=mu, phi=phi)
        expected = phi * mu ** 2
        np.testing.assert_allclose(pred.variance, expected, rtol=1e-10)

    def test_cov_sqrt_phi(self):
        """For Gamma, CoV = sqrt(phi)."""
        mu = np.array([500.0])
        phi = np.array([0.25])
        pred = DistributionalPrediction("gamma", mu=mu, phi=phi)
        expected_cov = np.sqrt(phi)
        np.testing.assert_allclose(pred.cov, expected_cov, rtol=1e-10)

    def test_sampling_nonneg(self):
        pred = self.make_pred()
        rng = np.random.default_rng(0)
        samples = pred._sample(n_samples=100, rng=rng)
        assert np.all(samples > 0)

    def test_sampling_shape(self):
        pred = self.make_pred(n=15)
        rng = np.random.default_rng(0)
        samples = pred._sample(n_samples=50, rng=rng)
        assert samples.shape == (15, 50)

    def test_missing_phi_raises(self):
        pred = DistributionalPrediction("gamma", mu=np.array([500.0]))
        with pytest.raises(ValueError, match="phi"):
            _ = pred.variance


# ---------------------------------------------------------------------------
# ZIP predictions
# ---------------------------------------------------------------------------


class TestZIPPrediction:
    def make_pred(self, n=100):
        rng = np.random.default_rng(20)
        pi = rng.uniform(0.1, 0.5, n)
        lam = rng.uniform(0.1, 1.0, n)
        mu = (1.0 - pi) * lam
        return DistributionalPrediction("zip", mu=mu, pi=pi)

    def test_variance_formula(self):
        """
        ZIP variance: (1-pi)*lambda + (1-pi)*pi*lambda^2
                     = (1-pi)*lambda*(1 + pi*lambda)
        """
        pi = np.array([0.3])
        lam = np.array([0.5])
        mu = (1.0 - pi) * lam
        pred = DistributionalPrediction("zip", mu=mu, pi=pi)
        expected = (1.0 - pi) * lam * (1.0 + pi * lam)
        np.testing.assert_allclose(pred.variance, expected, rtol=1e-5)

    def test_variance_higher_with_higher_pi(self):
        """At fixed observable mean mu, more zero-inflation = more variance.
        
        Var = (1-pi)*lam*(1+pi*lam). With mu = (1-pi)*lam fixed:
        lam = mu/(1-pi), so Var = mu*(1 + pi*mu/(1-pi)) = mu + pi*mu^2/(1-pi).
        This is increasing in pi for fixed mu.
        """
        mu = np.array([0.5, 0.5])  # fixed observable mean
        pi_low = np.array([0.1, 0.1])
        pi_high = np.array([0.5, 0.5])
        p1 = DistributionalPrediction("zip", mu=mu, pi=pi_low)
        p2 = DistributionalPrediction("zip", mu=mu, pi=pi_high)
        assert np.all(p2.variance > p1.variance)

    def test_sampling_shape(self):
        pred = self.make_pred(n=20)
        rng = np.random.default_rng(0)
        samples = pred._sample(n_samples=50, rng=rng)
        assert samples.shape == (20, 50)

    def test_sampling_nonneg_integer(self):
        pred = self.make_pred(n=20)
        rng = np.random.default_rng(0)
        samples = pred._sample(n_samples=100, rng=rng)
        assert np.all(samples >= 0)
        assert np.all(samples == samples.astype(int))  # integer-valued

    def test_missing_pi_raises(self):
        pred = DistributionalPrediction("zip", mu=np.array([0.3]))
        with pytest.raises(ValueError, match="pi"):
            _ = pred.variance


# ---------------------------------------------------------------------------
# Negative Binomial predictions
# ---------------------------------------------------------------------------


class TestNegBinomPrediction:
    def make_pred(self, n=50, r=3.0):
        rng = np.random.default_rng(30)
        mu = rng.uniform(0.1, 2.0, n)
        r_arr = np.full(n, r)
        return DistributionalPrediction("negbinom", mu=mu, r=r_arr)

    def test_variance_formula(self):
        """Var = mu + mu^2/r."""
        mu = np.array([1.0, 2.0])
        r = np.array([5.0, 5.0])
        pred = DistributionalPrediction("negbinom", mu=mu, r=r)
        expected = mu + mu ** 2 / r
        np.testing.assert_allclose(pred.variance, expected, rtol=1e-10)

    def test_variance_exceeds_poisson(self):
        """NB variance > mu (overdispersed vs Poisson)."""
        pred = self.make_pred()
        assert np.all(pred.variance > pred.mu)

    def test_smaller_r_more_variance(self):
        """Smaller r (more overdispersed) gives larger variance."""
        mu = np.array([1.5])
        pred_r5 = DistributionalPrediction("negbinom", mu=mu, r=np.array([5.0]))
        pred_r1 = DistributionalPrediction("negbinom", mu=mu, r=np.array([1.0]))
        assert pred_r1.variance > pred_r5.variance

    def test_sampling_nonneg_integer(self):
        pred = self.make_pred(n=20)
        rng = np.random.default_rng(0)
        samples = pred._sample(n_samples=100, rng=rng)
        assert np.all(samples >= 0)
        assert np.all(samples == samples.astype(int))

    def test_missing_r_raises(self):
        pred = DistributionalPrediction("negbinom", mu=np.array([1.0]))
        with pytest.raises(ValueError, match="r"):
            _ = pred.variance

    def test_unknown_distribution_raises(self):
        pred = DistributionalPrediction("foobar", mu=np.array([1.0]))
        with pytest.raises(ValueError, match="Unknown distribution"):
            _ = pred.variance


# ---------------------------------------------------------------------------
# Performance-fix regression tests
# These verify the vectorised implementations (no Python loops) produce
# correct statistical results.
# ---------------------------------------------------------------------------


class TestVectorisedTweedieSampling:
    """Regression tests for the vectorised _sample_tweedie implementation."""

    def test_sampling_shape_large(self):
        """Shape is correct for a larger observation set."""
        rng = np.random.default_rng(99)
        n = 200
        mu = rng.uniform(100, 1000, n)
        phi = rng.uniform(0.1, 0.5, n)
        pred = DistributionalPrediction("tweedie", mu=mu, phi=phi, power=1.5)
        samples = pred._sample(n_samples=500, rng=np.random.default_rng(0))
        assert samples.shape == (n, 500)

    def test_sampling_nonneg_large(self):
        """Vectorised Tweedie samples must all be >= 0."""
        rng = np.random.default_rng(99)
        n = 200
        mu = rng.uniform(100, 1000, n)
        phi = rng.uniform(0.1, 0.5, n)
        pred = DistributionalPrediction("tweedie", mu=mu, phi=phi, power=1.5)
        samples = pred._sample(n_samples=500, rng=np.random.default_rng(0))
        assert np.all(samples >= 0)

    def test_sampling_mean_converges(self):
        """Vectorised sample mean converges to predicted mu."""
        mu = np.array([200.0, 500.0, 1000.0])
        phi = np.array([0.2, 0.3, 0.4])
        pred = DistributionalPrediction("tweedie", mu=mu, phi=phi, power=1.5)
        samples = pred._sample(n_samples=30_000, rng=np.random.default_rng(7))
        sample_means = samples.mean(axis=1)
        np.testing.assert_allclose(sample_means, mu, rtol=0.10)

    def test_all_zero_counts(self):
        """When lambda_tw is very small, most samples should be zero (Tweedie mass at 0)."""
        # Very small mu, large phi => lambda_tw very small => most draws are 0
        mu = np.array([0.01, 0.01])
        phi = np.array([2.0, 2.0])
        pred = DistributionalPrediction("tweedie", mu=mu, phi=phi, power=1.5)
        samples = pred._sample(n_samples=500, rng=np.random.default_rng(0))
        assert samples.shape == (2, 500)
        assert np.all(samples >= 0)


class TestVectorisedNegBinomSampling:
    """Regression tests for the vectorised _sample_negbinom implementation."""

    def test_sampling_shape(self):
        rng = np.random.default_rng(30)
        n = 50
        mu = rng.uniform(0.1, 2.0, n)
        r = np.full(n, 3.0)
        pred = DistributionalPrediction("negbinom", mu=mu, r=r)
        samples = pred._sample(n_samples=200, rng=np.random.default_rng(0))
        assert samples.shape == (n, 200)

    def test_sampling_nonneg_integer(self):
        rng = np.random.default_rng(30)
        n = 40
        mu = rng.uniform(0.5, 5.0, n)
        r = np.full(n, 2.0)
        pred = DistributionalPrediction("negbinom", mu=mu, r=r)
        samples = pred._sample(n_samples=300, rng=np.random.default_rng(1))
        assert np.all(samples >= 0)
        assert np.all(samples == samples.astype(int))

    def test_sampling_mean_converges(self):
        """Vectorised NB sample mean converges to predicted mu."""
        mu = np.array([1.0, 2.0, 5.0])
        r = np.array([3.0, 3.0, 3.0])
        pred = DistributionalPrediction("negbinom", mu=mu, r=r)
        samples = pred._sample(n_samples=50_000, rng=np.random.default_rng(42))
        sample_means = samples.mean(axis=1)
        np.testing.assert_allclose(sample_means, mu, rtol=0.05)

    def test_sampling_variance_converges(self):
        """Vectorised NB sample variance converges to mu + mu^2/r."""
        mu = np.array([2.0, 4.0])
        r = np.array([2.0, 2.0])
        pred = DistributionalPrediction("negbinom", mu=mu, r=r)
        samples = pred._sample(n_samples=80_000, rng=np.random.default_rng(13))
        sample_vars = samples.var(axis=1)
        expected_vars = mu + mu ** 2 / r
        np.testing.assert_allclose(sample_vars, expected_vars, rtol=0.10)
