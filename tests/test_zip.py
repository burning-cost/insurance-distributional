"""
Tests for ZIPGBM — requires CatBoost.
"""

import numpy as np
import pytest

from insurance_distributional import ZIPGBM
from insurance_distributional.prediction import DistributionalPrediction
from insurance_distributional.zip import _zip_log_likelihood


class TestZIPLogLikelihood:
    def test_finite_mixed(self):
        y = np.array([0.0, 0.0, 1.0, 3.0])
        lam = np.array([0.3, 0.5, 0.3, 0.8])
        pi = np.array([0.3, 0.4, 0.3, 0.2])
        ll = _zip_log_likelihood(y, lam, pi)
        assert np.all(np.isfinite(ll))

    def test_zeros_higher_with_higher_pi(self):
        """Higher pi -> higher likelihood for y=0."""
        y = np.array([0.0])
        lam = np.array([0.5])
        pi_lo = np.array([0.1])
        pi_hi = np.array([0.6])
        ll_lo = _zip_log_likelihood(y, lam, pi_lo)
        ll_hi = _zip_log_likelihood(y, lam, pi_hi)
        assert ll_hi > ll_lo

    def test_all_zeros_dataset(self):
        y = np.zeros(10)
        lam = np.full(10, 0.5)
        pi = np.full(10, 0.4)
        ll = _zip_log_likelihood(y, lam, pi)
        assert np.all(np.isfinite(ll))
        assert np.all(ll < 0)  # probabilities < 1

    def test_positive_counts_ignore_pi(self):
        """For y>0, only (1-pi)*Poisson(y;lambda) matters."""
        y = np.array([2.0])
        lam = np.array([1.0])
        pi_lo = np.array([0.1])
        pi_hi = np.array([0.4])
        ll_lo = _zip_log_likelihood(y, lam, pi_lo)
        ll_hi = _zip_log_likelihood(y, lam, pi_hi)
        assert ll_lo > ll_hi  # lower pi -> higher ll for y>0


class TestZIPGBMFit:
    def test_fit_returns_self(self, zip_data):
        model = ZIPGBM()
        result = model.fit(zip_data["X"], zip_data["y"])
        assert result is model

    def test_predict_returns_prediction(self, zip_data):
        model = ZIPGBM()
        model.fit(zip_data["X"], zip_data["y"])
        pred = model.predict(zip_data["X"])
        assert isinstance(pred, DistributionalPrediction)
        assert pred.distribution == "zip"

    def test_predict_shape(self, zip_data):
        n = len(zip_data["y"])
        model = ZIPGBM()
        model.fit(zip_data["X"], zip_data["y"])
        pred = model.predict(zip_data["X"])
        assert pred.mu.shape == (n,)
        assert pred.pi.shape == (n,)

    def test_pi_in_range(self, zip_data):
        model = ZIPGBM()
        model.fit(zip_data["X"], zip_data["y"])
        pred = model.predict(zip_data["X"])
        assert np.all(pred.pi > 0)
        assert np.all(pred.pi < 1)

    def test_mean_positive(self, zip_data):
        model = ZIPGBM()
        model.fit(zip_data["X"], zip_data["y"])
        pred = model.predict(zip_data["X"])
        assert np.all(pred.mu >= 0)

    def test_predicted_mean_less_than_lambda(self, zip_data):
        """mu = (1-pi)*lambda < lambda since pi > 0."""
        model = ZIPGBM()
        model.fit(zip_data["X"], zip_data["y"])
        pred = model.predict(zip_data["X"])
        lam = model.predict_lambda(zip_data["X"])
        assert np.all(pred.mu <= lam + 1e-6)

    def test_variance_positive(self, zip_data):
        model = ZIPGBM()
        model.fit(zip_data["X"], zip_data["y"])
        pred = model.predict(zip_data["X"])
        assert np.all(pred.variance >= 0)

    def test_pi_detects_zero_inflation(self, zip_data):
        """
        Average predicted pi should be in reasonable range.
        True pi varies between ~0.35 and ~0.50.
        """
        model = ZIPGBM()
        model.fit(zip_data["X"], zip_data["y"])
        pred = model.predict(zip_data["X"])
        pi_mean = pred.pi.mean()
        assert 0.1 < pi_mean < 0.9  # loose bound — just checking it's learning something

    def test_log_score_finite(self, zip_data):
        model = ZIPGBM()
        model.fit(zip_data["X"], zip_data["y"])
        ls = model.log_score(zip_data["X"], zip_data["y"])
        assert np.isfinite(ls)

    def test_crps_nonneg(self, zip_data):
        model = ZIPGBM()
        model.fit(zip_data["X"], zip_data["y"])
        crps = model.crps(zip_data["X"], zip_data["y"], n_samples=200)
        assert crps >= 0

    def test_predict_lambda(self, zip_data):
        model = ZIPGBM()
        model.fit(zip_data["X"], zip_data["y"])
        lam = model.predict_lambda(zip_data["X"])
        assert lam.shape == (len(zip_data["y"]),)
        assert np.all(lam > 0)

    def test_repr(self):
        r = repr(ZIPGBM())
        assert "not fitted" in r

    def test_near_zero_data(self):
        """Should handle data with very few non-zeros."""
        rng = np.random.default_rng(42)
        n = 200
        X = rng.standard_normal((n, 3))
        y = np.zeros(n)
        # Only 5% non-zero
        idx = rng.choice(n, size=10, replace=False)
        y[idx] = rng.poisson(0.5, size=10).astype(float)

        model = ZIPGBM()
        model.fit(X, y)
        pred = model.predict(X)
        assert np.all(pred.pi > 0)
