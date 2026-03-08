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
