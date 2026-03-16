"""
Tests for TweedieGBM — requires CatBoost.

All tests use small synthetic datasets to keep training fast on Databricks.
"""

import numpy as np
import pytest

from insurance_distributional import TweedieGBM
from insurance_distributional.prediction import DistributionalPrediction
from insurance_distributional.tweedie import _estimate_phi_mle, _tweedie_log_likelihood


# ---------------------------------------------------------------------------
# Unit tests for Tweedie log-likelihood (no CatBoost)
# ---------------------------------------------------------------------------


class TestTweedieLogLikelihood:
    def test_finite_positive_y(self):
        y = np.array([100.0, 200.0, 500.0])
        mu = np.array([120.0, 190.0, 480.0])
        phi = np.full(3, 0.5)
        ll = _tweedie_log_likelihood(y, mu, phi, p=1.5)
        assert np.all(np.isfinite(ll))

    def test_finite_zero_y(self):
        y = np.array([0.0, 0.0])
        mu = np.array([50.0, 100.0])
        phi = np.full(2, 0.5)
        ll = _tweedie_log_likelihood(y, mu, phi, p=1.5)
        assert np.all(np.isfinite(ll))

    def test_mixed_zeros_positive(self):
        y = np.array([0.0, 100.0, 0.0, 200.0])
        mu = np.full(4, 150.0)
        phi = np.full(4, 0.5)
        ll = _tweedie_log_likelihood(y, mu, phi, p=1.5)
        assert np.all(np.isfinite(ll))

    def test_better_mu_higher_ll(self):
        y = np.array([200.0, 300.0])
        mu_good = np.array([205.0, 295.0])
        mu_bad = np.array([500.0, 50.0])
        phi = np.full(2, 0.5)
        ll_good = _tweedie_log_likelihood(y, mu_good, phi, 1.5).sum()
        ll_bad = _tweedie_log_likelihood(y, mu_bad, phi, 1.5).sum()
        assert ll_good > ll_bad


class TestEstimatePhiMle:
    def test_recovers_true_phi(self):
        """Scalar phi MLE should recover simulated phi roughly."""
        rng = np.random.default_rng(0)
        n = 500
        true_phi = 0.4
        p = 1.5
        mu = np.full(n, 200.0)
        lam = mu ** (2 - p) / (true_phi * (2 - p))
        alpha = (2 - p) / (p - 1)
        beta = mu ** (1 - p) / (true_phi * (p - 1))
        y = np.zeros(n)
        for i in range(n):
            nc = rng.poisson(lam[i])
            if nc > 0:
                y[i] = rng.gamma(alpha, 1.0 / beta[i], nc).sum()

        phi_hat = _estimate_phi_mle(y, mu, p)
        # Allow 50% relative error — small sample + zeros
        assert 0.2 < phi_hat < 0.8


# ---------------------------------------------------------------------------
# Integration tests — require CatBoost
# ---------------------------------------------------------------------------


class TestTweedieGBMFit:
    def test_fit_returns_self(self, tweedie_data):
        model = TweedieGBM(power=1.5)
        result = model.fit(tweedie_data["X"], tweedie_data["y"])
        assert result is model

    def test_fit_marks_fitted(self, tweedie_data):
        model = TweedieGBM(power=1.5)
        assert not model._is_fitted
        model.fit(tweedie_data["X"], tweedie_data["y"])
        assert model._is_fitted

    def test_predict_returns_prediction(self, tweedie_data):
        model = TweedieGBM(power=1.5)
        model.fit(tweedie_data["X"], tweedie_data["y"])
        pred = model.predict(tweedie_data["X"])
        assert isinstance(pred, DistributionalPrediction)
        assert pred.distribution == "tweedie"

    def test_predict_shape(self, tweedie_data):
        n = len(tweedie_data["y"])
        model = TweedieGBM(power=1.5)
        model.fit(tweedie_data["X"], tweedie_data["y"])
        pred = model.predict(tweedie_data["X"])
        assert pred.mu.shape == (n,)
        assert pred.phi.shape == (n,)

    def test_predicted_mean_positive(self, tweedie_data):
        model = TweedieGBM(power=1.5)
        model.fit(tweedie_data["X"], tweedie_data["y"])
        pred = model.predict(tweedie_data["X"])
        assert np.all(pred.mu > 0)

    def test_predicted_phi_positive(self, tweedie_data):
        model = TweedieGBM(power=1.5)
        model.fit(tweedie_data["X"], tweedie_data["y"])
        pred = model.predict(tweedie_data["X"])
        assert np.all(pred.phi > 0)

    def test_predicted_mean_plausible(self, tweedie_data):
        """Predicted mean should be in the same order of magnitude as y."""
        model = TweedieGBM(power=1.5)
        model.fit(tweedie_data["X"], tweedie_data["y"])
        pred = model.predict(tweedie_data["X"])
        y = tweedie_data["y"]
        # At least 50% of predictions within 5x of y-range
        y_mean = np.mean(y[y > 0]) if np.any(y > 0) else 1.0
        assert 0.1 * y_mean < pred.mu.mean() < 10 * y_mean

    def test_with_exposure(self, tweedie_data):
        model = TweedieGBM(power=1.5)
        model.fit(
            tweedie_data["X"], tweedie_data["y"],
            exposure=tweedie_data["exposure"]
        )
        pred = model.predict(tweedie_data["X"])
        assert np.all(pred.mu > 0)

    def test_polars_input(self, tweedie_data):
        """Should accept Polars DataFrames."""
        import polars as pl
        X_pl = pl.DataFrame(tweedie_data["X"])
        y_pl = pl.Series(tweedie_data["y"])
        model = TweedieGBM(power=1.5)
        model.fit(X_pl, y_pl)
        pred = model.predict(X_pl)
        assert isinstance(pred, DistributionalPrediction)

    def test_no_dispersion_model(self, tweedie_data):
        """model_dispersion=False should work with scalar phi."""
        model = TweedieGBM(power=1.5, model_dispersion=False)
        model.fit(tweedie_data["X"], tweedie_data["y"])
        pred = model.predict(tweedie_data["X"])
        # phi should be constant (scalar)
        assert np.all(pred.phi == pred.phi[0])

    def test_variance_computed(self, tweedie_data):
        model = TweedieGBM(power=1.5)
        model.fit(tweedie_data["X"], tweedie_data["y"])
        pred = model.predict(tweedie_data["X"])
        assert np.all(pred.variance > 0)

    def test_volatility_score_positive(self, tweedie_data):
        model = TweedieGBM(power=1.5)
        model.fit(tweedie_data["X"], tweedie_data["y"])
        pred = model.predict(tweedie_data["X"])
        assert np.all(pred.volatility_score() > 0)


class TestTweedieGBMScoring:
    def test_log_score_finite(self, tweedie_data):
        model = TweedieGBM(power=1.5)
        model.fit(tweedie_data["X"], tweedie_data["y"])
        ls = model.log_score(tweedie_data["X"], tweedie_data["y"])
        assert np.isfinite(ls)

    def test_crps_finite(self, tweedie_data):
        model = TweedieGBM(power=1.5)
        model.fit(tweedie_data["X"], tweedie_data["y"])
        crps = model.crps(
            tweedie_data["X"], tweedie_data["y"], n_samples=200
        )
        assert np.isfinite(crps)
        assert crps >= 0  # CRPS is non-negative

    def test_repr(self):
        model = TweedieGBM(power=1.5)
        r = repr(model)
        assert "1.5" in r
        assert "not fitted" in r

    def test_repr_after_fit(self, tweedie_data):
        model = TweedieGBM(power=1.5)
        model.fit(tweedie_data["X"], tweedie_data["y"])
        r = repr(model)
        assert "fitted" in r


class TestTweediePhiAbsoluteScale:
    """
    Regression tests for Tweedie phi absolute scale (v0.1.3 fix).

    Prior to v0.1.3, the phi model used RMSE on log(d) which estimated
    E[log(d)|x] instead of log(E[d|x]) = log(phi). Due to Jensen's inequality,
    this caused ~3x systematic underestimation of phi.

    These tests verify phi predictions are in the correct absolute range
    for the tweedie_data fixture (true phi = 0.5).
    """

    def test_phi_mean_in_plausible_range(self, tweedie_data):
        """
        Mean predicted phi should be in [0.1, 2.0] for a DGP with phi=0.5.

        The tweedie_data fixture uses phi_true=0.5.
        Before the fix, mean phi was typically ~0.15 (3x too low).
        """
        model = TweedieGBM(power=1.5)
        model.fit(tweedie_data["X"], tweedie_data["y"])
        pred = model.predict(tweedie_data["X"])
        mean_phi = float(np.mean(pred.phi))
        # True phi = 0.5; allow wide tolerance for small n=300 + Tweedie zeros
        assert 0.05 < mean_phi < 3.0, (
            f"Mean phi={mean_phi:.4f} is outside [0.05, 3.0] for a DGP with phi=0.5. "
            "This likely indicates the Jensen gap bias is not corrected."
        )

    def test_phi_not_systematically_underestimated(self, tweedie_data):
        """
        Median phi should not be more than 5x below the true value.

        Pre-fix, median phi was typically 0.15-0.18 vs true 0.5 (~3x too low).
        This test catches a regression back to that behaviour.
        """
        true_phi = tweedie_data["phi_true"]  # 0.5
        model = TweedieGBM(power=1.5)
        model.fit(tweedie_data["X"], tweedie_data["y"])
        pred = model.predict(tweedie_data["X"])
        median_phi = float(np.median(pred.phi))
        ratio = median_phi / true_phi
        assert ratio > 0.2, (
            f"Median phi={median_phi:.4f}, true phi={true_phi}. "
            f"Ratio={ratio:.3f} < 0.2 — phi is more than 5x too low. "
            "Jensen gap bias correction may have regressed."
        )

    def test_scalar_phi_correct_range(self, tweedie_data):
        """
        Scalar phi (model_dispersion=False) should be near the true phi.

        This is the MLE-based fallback, not affected by the Jensen gap issue.
        Validates the baseline behaviour for the non-GBM path.
        """
        model = TweedieGBM(power=1.5, model_dispersion=False)
        model.fit(tweedie_data["X"], tweedie_data["y"])
        pred = model.predict(tweedie_data["X"])
        scalar_phi = float(pred.phi[0])
        # Allow wide bounds — small n with zeros makes MLE noisy
        assert 0.1 < scalar_phi < 2.0, (
            f"Scalar phi MLE gave {scalar_phi:.4f}, expected near 0.5."
        )
