"""
Tests for NegBinomialGBM — requires CatBoost.
"""

import numpy as np
import pytest

from insurance_distributional import NegBinomialGBM
from insurance_distributional.negbinom import _estimate_r_mle, _negbinom_log_likelihood
from insurance_distributional.prediction import DistributionalPrediction


class TestNegBinomLogLikelihood:
    def test_finite(self):
        y = np.array([0.0, 1.0, 3.0, 5.0])
        mu = np.array([1.5, 1.5, 2.0, 4.0])
        r = np.full(4, 3.0)
        ll = _negbinom_log_likelihood(y, mu, r)
        assert np.all(np.isfinite(ll))

    def test_zeros_handled(self):
        y = np.array([0.0, 0.0])
        mu = np.array([1.0, 2.0])
        r = np.full(2, 5.0)
        ll = _negbinom_log_likelihood(y, mu, r)
        assert np.all(np.isfinite(ll))

    def test_better_mu_higher_ll(self):
        y = np.array([2.0, 3.0])
        mu_good = np.array([2.1, 2.9])
        mu_bad = np.array([0.1, 10.0])
        r = np.full(2, 5.0)
        assert (
            _negbinom_log_likelihood(y, mu_good, r).sum()
            > _negbinom_log_likelihood(y, mu_bad, r).sum()
        )


class TestEstimateRMle:
    def test_recovers_true_r(self):
        """MLE for r should recover the true value roughly."""
        rng = np.random.default_rng(10)
        n = 500
        r_true = 3.0
        mu_true = 1.5
        mu_arr = np.full(n, mu_true)
        p_nb = r_true / (r_true + mu_true)
        y = rng.negative_binomial(n=r_true, p=p_nb, size=n).astype(float)
        r_hat = _estimate_r_mle(y, mu_arr)
        # Allow 50% relative error
        assert 1.5 < r_hat < 6.0


class TestNegBinomialGBMFit:
    def test_fit_returns_self(self, negbinom_data):
        model = NegBinomialGBM()
        result = model.fit(
            negbinom_data["X"], negbinom_data["y"],
            exposure=negbinom_data["exposure"]
        )
        assert result is model

    def test_predict_returns_prediction(self, negbinom_data):
        model = NegBinomialGBM()
        model.fit(negbinom_data["X"], negbinom_data["y"])
        pred = model.predict(negbinom_data["X"])
        assert isinstance(pred, DistributionalPrediction)
        assert pred.distribution == "negbinom"

    def test_predict_shape(self, negbinom_data):
        n = len(negbinom_data["y"])
        model = NegBinomialGBM()
        model.fit(negbinom_data["X"], negbinom_data["y"])
        pred = model.predict(negbinom_data["X"])
        assert pred.mu.shape == (n,)
        assert pred.r.shape == (n,)

    def test_mu_positive(self, negbinom_data):
        model = NegBinomialGBM()
        model.fit(negbinom_data["X"], negbinom_data["y"])
        pred = model.predict(negbinom_data["X"])
        assert np.all(pred.mu > 0)

    def test_r_positive(self, negbinom_data):
        model = NegBinomialGBM()
        model.fit(negbinom_data["X"], negbinom_data["y"])
        pred = model.predict(negbinom_data["X"])
        assert np.all(pred.r > 0)

    def test_variance_exceeds_poisson(self, negbinom_data):
        """NB variance = mu + mu^2/r > mu always."""
        model = NegBinomialGBM()
        model.fit(negbinom_data["X"], negbinom_data["y"])
        pred = model.predict(negbinom_data["X"])
        assert np.all(pred.variance > pred.mu)

    def test_scalar_r_constant(self, negbinom_data):
        """model_r=False should give constant r."""
        model = NegBinomialGBM(model_r=False)
        model.fit(negbinom_data["X"], negbinom_data["y"])
        pred = model.predict(negbinom_data["X"])
        assert np.all(pred.r == pred.r[0])

    def test_model_r_varying(self, negbinom_data):
        """model_r=True should give non-constant r (at least some variation)."""
        model = NegBinomialGBM(model_r=True)
        model.fit(negbinom_data["X"], negbinom_data["y"])
        pred = model.predict(negbinom_data["X"])
        # Should have some variation if features are informative
        assert pred.r.max() > pred.r.min()

    def test_log_score_finite(self, negbinom_data):
        model = NegBinomialGBM()
        model.fit(negbinom_data["X"], negbinom_data["y"])
        ls = model.log_score(negbinom_data["X"], negbinom_data["y"])
        assert np.isfinite(ls)

    def test_crps_nonneg(self, negbinom_data):
        model = NegBinomialGBM()
        model.fit(negbinom_data["X"], negbinom_data["y"])
        crps = model.crps(negbinom_data["X"], negbinom_data["y"], n_samples=200)
        assert crps >= 0

    def test_repr(self):
        r = repr(NegBinomialGBM())
        assert "not fitted" in r
