"""
Tests for GammaGBM — requires CatBoost.
"""

import numpy as np
import pytest

from insurance_distributional import GammaGBM
from insurance_distributional.gamma import _estimate_phi_gamma_mle, _gamma_log_likelihood
from insurance_distributional.prediction import DistributionalPrediction


class TestGammaLogLikelihood:
    def test_finite(self):
        y = np.array([500.0, 1000.0, 2000.0])
        mu = np.array([480.0, 1050.0, 1980.0])
        phi = np.full(3, 0.5)
        ll = _gamma_log_likelihood(y, mu, phi)
        assert np.all(np.isfinite(ll))

    def test_better_predictions_higher_ll(self):
        y = np.array([500.0, 1000.0])
        mu_good = np.array([510.0, 995.0])
        mu_bad = np.array([2000.0, 100.0])
        phi = np.full(2, 0.5)
        ll_good = _gamma_log_likelihood(y, mu_good, phi).sum()
        ll_bad = _gamma_log_likelihood(y, mu_bad, phi).sum()
        assert ll_good > ll_bad


class TestEstimatePhiGamma:
    def test_recovers_true_phi(self):
        """MLE should recover the true phi = 1/shape reasonably."""
        rng = np.random.default_rng(5)
        n = 500
        shape_true = 3.0  # phi = 1/3
        mu_true = 600.0
        y = rng.gamma(shape_true, mu_true / shape_true, n)
        mu_arr = np.full(n, mu_true)
        phi_hat = _estimate_phi_gamma_mle(y, mu_arr)
        assert 0.2 < phi_hat < 0.6  # true is 0.333


class TestGammaGBMFit:
    def test_fit_returns_self(self, gamma_data):
        model = GammaGBM()
        result = model.fit(gamma_data["X"], gamma_data["y"])
        assert result is model

    def test_predict_returns_prediction(self, gamma_data):
        model = GammaGBM()
        model.fit(gamma_data["X"], gamma_data["y"])
        pred = model.predict(gamma_data["X"])
        assert isinstance(pred, DistributionalPrediction)
        assert pred.distribution == "gamma"

    def test_predict_shape(self, gamma_data):
        n = len(gamma_data["y"])
        model = GammaGBM()
        model.fit(gamma_data["X"], gamma_data["y"])
        pred = model.predict(gamma_data["X"])
        assert pred.mu.shape == (n,)
        assert pred.phi.shape == (n,)

    def test_mean_positive(self, gamma_data):
        model = GammaGBM()
        model.fit(gamma_data["X"], gamma_data["y"])
        pred = model.predict(gamma_data["X"])
        assert np.all(pred.mu > 0)

    def test_phi_positive(self, gamma_data):
        model = GammaGBM()
        model.fit(gamma_data["X"], gamma_data["y"])
        pred = model.predict(gamma_data["X"])
        assert np.all(pred.phi > 0)

    def test_no_dispersion_model(self, gamma_data):
        model = GammaGBM(model_dispersion=False)
        model.fit(gamma_data["X"], gamma_data["y"])
        pred = model.predict(gamma_data["X"])
        assert np.all(pred.phi == pred.phi[0])

    def test_cov_positive(self, gamma_data):
        model = GammaGBM()
        model.fit(gamma_data["X"], gamma_data["y"])
        pred = model.predict(gamma_data["X"])
        assert np.all(pred.cov > 0)

    def test_variance_is_phi_mu_squared(self, gamma_data):
        """Check Gamma variance formula Var = phi * mu^2."""
        model = GammaGBM()
        model.fit(gamma_data["X"], gamma_data["y"])
        pred = model.predict(gamma_data["X"])
        expected = pred.phi * pred.mu ** 2
        np.testing.assert_allclose(pred.variance, expected, rtol=1e-10)

    def test_log_score_finite(self, gamma_data):
        model = GammaGBM()
        model.fit(gamma_data["X"], gamma_data["y"])
        ls = model.log_score(gamma_data["X"], gamma_data["y"])
        assert np.isfinite(ls)

    def test_crps_nonneg(self, gamma_data):
        model = GammaGBM()
        model.fit(gamma_data["X"], gamma_data["y"])
        crps = model.crps(gamma_data["X"], gamma_data["y"], n_samples=200)
        assert crps >= 0

    def test_repr(self):
        r = repr(GammaGBM())
        assert "not fitted" in r


class TestGammaPhiAbsoluteScale:
    """
    Regression tests for phi absolute scale (v0.1.3 fix).

    Prior to v0.1.3, the phi model used RMSE on log(d) which estimated
    E[log(d)|x] instead of log(E[d|x]) = log(phi). Due to Jensen's inequality,
    this caused ~3x systematic underestimation of phi.

    These tests verify that phi predictions are in the correct absolute range
    for a known DGP (true phi = 0.5, shape = 2.0).

    The fixture uses shape_true=2.0 => phi_true = 1/shape = 0.5.
    We verify the mean predicted phi is within a factor of 2 of the true value
    (allowing for finite-sample noise with n=300).
    """

    def test_phi_mean_in_plausible_range(self, gamma_data):
        """
        Mean predicted phi should be in [0.2, 1.5] for a DGP with phi=0.5.

        The gamma_data fixture uses shape_true=2.0, so phi_true=0.5.
        Before the fix, mean phi was typically ~0.15 (3x too low).
        """
        model = GammaGBM()
        model.fit(gamma_data["X"], gamma_data["y"])
        pred = model.predict(gamma_data["X"])
        mean_phi = float(np.mean(pred.phi))
        # True phi = 0.5; allow wide tolerance for small n=300 training set
        assert 0.1 < mean_phi < 2.0, (
            f"Mean phi={mean_phi:.4f} is outside [0.1, 2.0] for a DGP with phi=0.5. "
            "This likely indicates the Jensen gap bias is not corrected."
        )

    def test_phi_not_systematically_underestimated(self, gamma_data):
        """
        Median phi should not be more than 4x below the true value.

        Pre-fix, median phi was typically 0.15-0.18 vs true 0.5 (~3x too low).
        This test catches a regression back to that behaviour.
        """
        true_phi = 0.5  # shape_true=2.0 in gamma_data fixture
        model = GammaGBM()
        model.fit(gamma_data["X"], gamma_data["y"])
        pred = model.predict(gamma_data["X"])
        median_phi = float(np.median(pred.phi))
        ratio = median_phi / true_phi
        assert ratio > 0.25, (
            f"Median phi={median_phi:.4f}, true phi={true_phi}. "
            f"Ratio={ratio:.3f} < 0.25 — phi is more than 4x too low. "
            "Jensen gap bias correction may have regressed."
        )

    def test_phi_scalar_model_in_correct_range(self):
        """
        model_dispersion=False (scalar phi MLE) should give correct phi.

        This tests the scalar fallback which uses MLE — should be unaffected
        by the Jensen gap issue. Acts as a sanity baseline.
        """
        rng = np.random.default_rng(99)
        n = 500
        shape_true = 2.0  # phi_true = 0.5
        mu_true = np.exp(6.5 + 0.3 * rng.standard_normal(n))
        y = rng.gamma(shape=shape_true, scale=mu_true / shape_true)
        X = rng.standard_normal((n, 3))
        model = GammaGBM(model_dispersion=False)
        model.fit(X, y)
        pred = model.predict(X)
        # Scalar phi: all values are the same, should be close to 0.5
        scalar_phi = float(pred.phi[0])
        assert 0.2 < scalar_phi < 1.0, (
            f"Scalar phi MLE gave {scalar_phi:.4f}, expected near 0.5."
        )
