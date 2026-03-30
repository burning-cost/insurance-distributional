"""
Tests for threshold-weighted CRPS (twCRPS).

Four properties we care about:
1. Limit: tw_crps at threshold=0 should equal standard CRPS for non-negative data
2. Tail discrimination: twCRPS at 90th percentile should favour a correctly
   specified Gamma model over a misspecified one
3. Profile monotonicity: twCRPS should generally decrease as threshold increases
4. Determinism: same seed -> same result
"""

import numpy as np
import pytest

from insurance_distributional.prediction import DistributionalPrediction
from insurance_distributional.scoring import tw_crps, tw_crps_profile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gamma_pred(mu: np.ndarray, phi: np.ndarray) -> DistributionalPrediction:
    """Build a Gamma DistributionalPrediction directly (no model fitting needed)."""
    return DistributionalPrediction("gamma", mu=mu, phi=phi)


# ---------------------------------------------------------------------------
# Test 1: Limit — tw_crps(threshold=0) ≈ standard CRPS
# ---------------------------------------------------------------------------


class TestTwCRPSLimit:
    def test_limit_equals_crps_at_zero_threshold(self):
        """
        At threshold=0, the chaining function phi(z)=max(z,0) is the identity
        for non-negative insurance losses. twCRPS should match the standard
        CRPS within MC sampling noise.

        Tolerance is 2% of the CRPS value — consistent with the MC error
        guarantee for n_samples=2000.
        """
        rng = np.random.default_rng(0)
        n = 200
        mu = rng.uniform(500, 2000, n)
        phi = np.full(n, 0.5)  # shape=2 Gamma
        pred = _gamma_pred(mu, phi)
        y = rng.gamma(shape=2.0, scale=mu / 2.0, size=n)

        # Standard CRPS via energy score
        rng2 = np.random.default_rng(42)
        samples = pred._sample(n_samples=2000, rng=rng2)
        term1 = np.abs(samples - y[:, None]).mean(axis=1)
        half = 2000 // 2
        term2 = 0.5 * np.abs(samples[:, :half] - samples[:, half:2 * half]).mean(axis=1)
        standard_crps = float(np.mean(term1 - term2))

        tw = tw_crps(y, pred, threshold=0.0, n_samples=2000, seed=42)

        rel_diff = abs(tw - standard_crps) / (abs(standard_crps) + 1e-10)
        assert rel_diff < 0.02, (
            f"tw_crps at threshold=0 ({tw:.4f}) differs from standard CRPS "
            f"({standard_crps:.4f}) by {rel_diff:.2%}, expected < 2%"
        )

    def test_threshold_below_minimum_y_equals_crps(self):
        """
        When threshold is strictly below all y and sample values, the chaining
        function clips nothing and twCRPS must equal standard CRPS exactly
        (same seed, same samples).
        """
        rng = np.random.default_rng(1)
        n = 100
        # Positive losses well above threshold
        mu = rng.uniform(1000, 5000, n)
        phi = np.full(n, 0.3)
        pred = _gamma_pred(mu, phi)
        y = rng.gamma(shape=1.0 / 0.3, scale=mu * 0.3, size=n)

        # With threshold = -1 (below all values), should match standard CRPS
        tw = tw_crps(y, pred, threshold=-1.0, n_samples=2000, seed=42)

        rng2 = np.random.default_rng(42)
        samples = pred._sample(n_samples=2000, rng=rng2)
        term1 = np.abs(samples - y[:, None]).mean(axis=1)
        half = 2000 // 2
        term2 = 0.5 * np.abs(samples[:, :half] - samples[:, half:2 * half]).mean(axis=1)
        standard_crps = float(np.mean(term1 - term2))

        np.testing.assert_allclose(tw, standard_crps, rtol=1e-10)


# ---------------------------------------------------------------------------
# Test 2: Tail discrimination
# ---------------------------------------------------------------------------


class TestTailDiscrimination:
    def test_twcrps_favours_correct_model_in_tail(self):
        """
        Fit two Gamma predictions with the same mean but different dispersion:
        - correct model: phi matches the data-generating process
        - wrong model: phi is substantially different (over- or under-dispersed)

        twCRPS at the 90th percentile of y_true should be lower for the correct
        model, demonstrating tail discrimination.
        """
        rng = np.random.default_rng(99)
        n = 500

        # Data-generating parameters
        mu_true = np.full(n, 1000.0)
        phi_true = 0.25  # shape = 4

        y = rng.gamma(shape=4.0, scale=250.0, size=n)

        # Correct model
        pred_correct = _gamma_pred(mu_true, np.full(n, phi_true))

        # Wrong model — dispersion is 4x too large (over-dispersed)
        pred_wrong = _gamma_pred(mu_true, np.full(n, phi_true * 4.0))

        threshold = float(np.quantile(y, 0.90))

        tw_correct = tw_crps(y, pred_correct, threshold=threshold, n_samples=3000, seed=7)
        tw_wrong = tw_crps(y, pred_wrong, threshold=threshold, n_samples=3000, seed=7)

        assert tw_correct < tw_wrong, (
            f"Correct model twCRPS ({tw_correct:.4f}) should be lower than "
            f"wrong model ({tw_wrong:.4f}) at threshold={threshold:.1f}"
        )

    def test_twcrps_is_sensitive_at_high_threshold(self):
        """
        At a very high threshold, only observations in the extreme tail
        contribute. A model with correct tail shape should score better than
        one with an artificially thin tail (very high shape parameter).
        """
        rng = np.random.default_rng(42)
        n = 400

        # Heavy-tailed Gamma (shape=1 ~ Exponential)
        y = rng.gamma(shape=1.0, scale=1000.0, size=n)

        pred_correct = _gamma_pred(
            np.full(n, 1000.0),
            np.full(n, 1.0),  # phi=1 -> shape=1, matches DGP
        )
        # Wrong model: thin tail (shape=10, much less dispersed)
        pred_wrong = _gamma_pred(
            np.full(n, 1000.0),
            np.full(n, 0.1),  # phi=0.1 -> shape=10
        )

        threshold = float(np.quantile(y, 0.95))
        tw_correct = tw_crps(y, pred_correct, threshold=threshold, n_samples=3000, seed=11)
        tw_wrong = tw_crps(y, pred_wrong, threshold=threshold, n_samples=3000, seed=11)

        assert tw_correct < tw_wrong, (
            f"Heavy-tailed model ({tw_correct:.4f}) should beat thin-tailed "
            f"model ({tw_wrong:.4f}) at 95th percentile threshold"
        )


# ---------------------------------------------------------------------------
# Test 3: Profile monotonicity
# ---------------------------------------------------------------------------


class TestProfileMonotonicity:
    def test_profile_returns_dict_with_correct_keys(self):
        """tw_crps_profile should return a dict keyed by threshold values."""
        rng = np.random.default_rng(0)
        n = 50
        mu = rng.uniform(500, 2000, n)
        phi = np.full(n, 0.5)
        pred = _gamma_pred(mu, phi)
        y = rng.gamma(2.0, mu / 2.0, n)

        thresholds = np.array([0.0, 500.0, 1000.0, 2000.0])
        profile = tw_crps_profile(y, pred, thresholds=thresholds, n_samples=500, seed=0)

        assert isinstance(profile, dict)
        assert len(profile) == len(thresholds)
        for t in thresholds:
            assert float(t) in profile

    def test_profile_values_are_finite(self):
        """All profile values should be finite floats."""
        rng = np.random.default_rng(3)
        n = 80
        mu = rng.uniform(300, 1500, n)
        phi = np.full(n, 0.4)
        pred = _gamma_pred(mu, phi)
        y = rng.gamma(2.5, mu / 2.5, n)

        thresholds = np.linspace(0, float(np.quantile(y, 0.95)), 10)
        profile = tw_crps_profile(y, pred, thresholds=thresholds, n_samples=500, seed=1)

        for t, v in profile.items():
            assert np.isfinite(v), f"Non-finite twCRPS at threshold={t}"

    def test_profile_generally_decreasing(self):
        """
        As threshold increases, fewer observations contribute scores above
        baseline, so the mean twCRPS should generally decrease.

        We check this via linear regression slope rather than strict
        monotonicity (MC noise can cause local reversals).
        """
        rng = np.random.default_rng(5)
        n = 300
        mu = np.full(n, 1000.0)
        phi = np.full(n, 0.5)
        pred = _gamma_pred(mu, phi)
        y = rng.gamma(2.0, 500.0, n)

        thresholds = np.linspace(0.0, float(np.quantile(y, 0.95)), 20)
        profile = tw_crps_profile(y, pred, thresholds=thresholds, n_samples=2000, seed=2)

        sorted_t = sorted(profile.keys())
        values = np.array([profile[t] for t in sorted_t])

        # OLS slope should be negative (values decrease as threshold increases)
        t_arr = np.array(sorted_t)
        slope = np.polyfit(t_arr, values, 1)[0]
        assert slope < 0, (
            f"Expected negative slope in twCRPS profile (got {slope:.6f}). "
            "twCRPS should decrease as threshold increases."
        )

    def test_profile_matches_individual_calls(self):
        """
        Values from tw_crps_profile must match individual tw_crps() calls
        (same seed, same samples).
        """
        rng = np.random.default_rng(8)
        n = 60
        mu = rng.uniform(200, 800, n)
        phi = np.full(n, 0.6)
        pred = _gamma_pred(mu, phi)
        y = rng.gamma(1.0 / 0.6, mu * 0.6, n)

        thresholds = np.array([0.0, 300.0, 700.0])
        profile = tw_crps_profile(y, pred, thresholds=thresholds, n_samples=1000, seed=99)

        for t in thresholds:
            individual = tw_crps(y, pred, threshold=float(t), n_samples=1000, seed=99)
            np.testing.assert_allclose(
                profile[float(t)],
                individual,
                rtol=1e-10,
                err_msg=f"Profile and individual tw_crps disagree at threshold={t}",
            )


# ---------------------------------------------------------------------------
# Test 4: Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_result(self):
        """Identical inputs and seed must produce identical results."""
        rng = np.random.default_rng(0)
        n = 100
        mu = rng.uniform(100, 500, n)
        phi = np.full(n, 0.5)
        pred = _gamma_pred(mu, phi)
        y = rng.gamma(2.0, mu / 2.0, n)

        result1 = tw_crps(y, pred, threshold=200.0, n_samples=500, seed=42)
        result2 = tw_crps(y, pred, threshold=200.0, n_samples=500, seed=42)

        assert result1 == result2

    def test_different_seeds_different_results(self):
        """Different seeds should (almost certainly) produce different results."""
        rng = np.random.default_rng(0)
        n = 100
        mu = rng.uniform(100, 500, n)
        phi = np.full(n, 0.5)
        pred = _gamma_pred(mu, phi)
        y = rng.gamma(2.0, mu / 2.0, n)

        result1 = tw_crps(y, pred, threshold=200.0, n_samples=500, seed=1)
        result2 = tw_crps(y, pred, threshold=200.0, n_samples=500, seed=2)

        assert result1 != result2

    def test_profile_determinism(self):
        """tw_crps_profile is also deterministic given the same seed."""
        rng = np.random.default_rng(0)
        n = 80
        mu = rng.uniform(200, 1000, n)
        phi = np.full(n, 0.4)
        pred = _gamma_pred(mu, phi)
        y = rng.gamma(2.5, mu / 2.5, n)

        thresholds = np.array([0.0, 500.0, 1000.0])
        p1 = tw_crps_profile(y, pred, thresholds=thresholds, n_samples=500, seed=77)
        p2 = tw_crps_profile(y, pred, thresholds=thresholds, n_samples=500, seed=77)

        assert p1 == p2


# ---------------------------------------------------------------------------
# Test 5: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_high_threshold_clips_all_values(self):
        """
        When threshold is above all y and all samples, phi(z) = threshold
        everywhere. term1 and term2 both become 0, so twCRPS = 0.
        """
        n = 20
        mu = np.full(n, 100.0)
        phi = np.full(n, 0.5)
        pred = _gamma_pred(mu, phi)
        # y values all at or below threshold
        y = np.full(n, 50.0)

        # Threshold way above everything — Gamma(shape=2, scale=50) samples
        # will be in [0, ~500]; threshold=1e9 clips everything to 1e9
        tw = tw_crps(y, pred, threshold=1e9, n_samples=200, seed=0)
        assert tw == pytest.approx(0.0, abs=1e-8)

    def test_single_observation(self):
        """tw_crps should work with a single observation."""
        pred = _gamma_pred(
            mu=np.array([500.0]),
            phi=np.array([0.5]),
        )
        y = np.array([480.0])
        tw = tw_crps(y, pred, threshold=100.0, n_samples=500, seed=0)
        assert np.isfinite(tw)
        assert tw >= 0.0

    def test_returns_float(self):
        """tw_crps must return a Python float, not np.float64."""
        rng = np.random.default_rng(0)
        n = 30
        mu = rng.uniform(100, 500, n)
        phi = np.full(n, 0.5)
        pred = _gamma_pred(mu, phi)
        y = rng.gamma(2.0, mu / 2.0, n)
        result = tw_crps(y, pred, threshold=0.0, n_samples=200, seed=0)
        assert isinstance(result, float)
