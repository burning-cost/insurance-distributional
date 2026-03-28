"""
Tests for ZeroInflatedTweedieGBM — So & Valdez (2024) Scenario 2.

Coverage:
- Basic fit/predict on synthetic zero-inflated data
- Component predictions (zero_prob, severity_mean, combined_mean)
- Exposure handling
- Sample weight handling
- Edge cases (all zeros, no zeros, minimal non-zeros)
- Serialisation (pickle)
- Parameter passing to CatBoost
- Scoring methods (score, log_score)
- Validation errors
- predict_proba shape and bounds
- repr and fitted status
- ZI helper function (_tweedie_unit_deviance_ll)
"""

import pickle

import numpy as np
import pytest

from insurance_distributional import ZeroInflatedTweedieGBM
from insurance_distributional.zi_tweedie import _tweedie_unit_deviance_ll


# ---------------------------------------------------------------------------
# Shared synthetic dataset fixture
# ---------------------------------------------------------------------------

N_SMALL = 300


def _make_zi_tweedie_data(
    n: int = N_SMALL,
    zero_rate: float = 0.85,
    seed: int = 10,
    with_exposure: bool = False,
) -> dict:
    """
    Synthetic ZI-Tweedie dataset.

    zero_rate fraction are structural zeros; the rest are drawn from a
    Tweedie-like distribution (gamma severity as simplification for testing).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    # True zero probability varies with X[:,0]
    pi_true = np.clip(zero_rate + 0.05 * X[:, 0], 0.6, 0.97)
    # True severity varies with X[:,1]
    mu_true = np.exp(5.5 + 0.4 * X[:, 1])

    y = np.zeros(n)
    for i in range(n):
        if rng.random() >= pi_true[i]:
            # Gamma draw as proxy for Tweedie severity
            y[i] = rng.gamma(shape=2.0, scale=mu_true[i] / 2.0)

    if with_exposure:
        exposure = rng.uniform(0.5, 1.0, n)
    else:
        exposure = np.ones(n)

    return {
        "X": X,
        "y": y,
        "exposure": exposure,
        "pi_true": pi_true,
        "mu_true": mu_true,
    }


@pytest.fixture(scope="module")
def zi_data():
    return _make_zi_tweedie_data(n=N_SMALL, zero_rate=0.85)


@pytest.fixture(scope="module")
def fitted_model(zi_data):
    model = ZeroInflatedTweedieGBM(power=1.5)
    model.fit(zi_data["X"], zi_data["y"])
    return model


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestTweedieUnitDevianceLl:
    def test_returns_array(self):
        y = np.array([100.0, 200.0, 500.0])
        mu = np.array([150.0, 180.0, 520.0])
        result = _tweedie_unit_deviance_ll(y, mu, p=1.5)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_finite_values(self):
        rng = np.random.default_rng(0)
        y = rng.exponential(100, 20) + 1.0
        mu = rng.exponential(100, 20) + 1.0
        result = _tweedie_unit_deviance_ll(y, mu, p=1.5)
        assert np.all(np.isfinite(result))

    def test_perfect_fit_is_zero(self):
        """When y == mu, unit deviance is zero so ll contribution is 0."""
        y = np.array([200.0, 300.0])
        mu = np.array([200.0, 300.0])
        result = _tweedie_unit_deviance_ll(y, mu, p=1.5)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_worse_predictions_lower_ll(self):
        """A prediction far from truth should give lower LL than one close to truth."""
        y = np.array([200.0])
        mu_good = np.array([210.0])
        mu_bad = np.array([2000.0])
        ll_good = _tweedie_unit_deviance_ll(y, mu_good, p=1.5)
        ll_bad = _tweedie_unit_deviance_ll(y, mu_bad, p=1.5)
        assert ll_good > ll_bad


# ---------------------------------------------------------------------------
# Initialisation and parameter validation
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_power(self):
        model = ZeroInflatedTweedieGBM()
        assert model.power == 1.5

    def test_custom_power(self):
        model = ZeroInflatedTweedieGBM(power=1.8)
        assert model.power == 1.8

    def test_invalid_power_low(self):
        with pytest.raises(ValueError, match="power must be in"):
            ZeroInflatedTweedieGBM(power=0.9)

    def test_invalid_power_high(self):
        with pytest.raises(ValueError, match="power must be in"):
            ZeroInflatedTweedieGBM(power=2.0)

    def test_invalid_power_equals_one(self):
        with pytest.raises(ValueError, match="power must be in"):
            ZeroInflatedTweedieGBM(power=1.0)

    def test_not_fitted_initially(self):
        model = ZeroInflatedTweedieGBM()
        assert not model._is_fitted

    def test_repr_not_fitted(self):
        model = ZeroInflatedTweedieGBM(power=1.7)
        r = repr(model)
        assert "not fitted" in r
        assert "1.7" in r


# ---------------------------------------------------------------------------
# Fit tests
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_returns_self(self, zi_data):
        model = ZeroInflatedTweedieGBM()
        result = model.fit(zi_data["X"], zi_data["y"])
        assert result is model

    def test_is_fitted_after_fit(self, zi_data):
        model = ZeroInflatedTweedieGBM()
        model.fit(zi_data["X"], zi_data["y"])
        assert model._is_fitted

    def test_repr_fitted(self, fitted_model):
        r = repr(fitted_model)
        assert "fitted" in r
        assert "not fitted" not in r

    def test_fit_with_exposure(self, zi_data):
        model = ZeroInflatedTweedieGBM()
        # Should not raise
        model.fit(zi_data["X"], zi_data["y"], exposure=zi_data["exposure"])
        assert model._is_fitted

    def test_fit_with_sample_weight(self, zi_data):
        rng = np.random.default_rng(99)
        weights = rng.uniform(0.5, 2.0, len(zi_data["y"]))
        model = ZeroInflatedTweedieGBM()
        model.fit(zi_data["X"], zi_data["y"], sample_weight=weights)
        assert model._is_fitted

    def test_fit_stores_internal_models(self, fitted_model):
        assert fitted_model._model_zero is not None
        assert fitted_model._model_severity is not None

    def test_fit_severity_init_positive(self, fitted_model):
        assert fitted_model._severity_init > 0

    def test_negative_y_raises(self, zi_data):
        y_bad = zi_data["y"].copy()
        y_bad[0] = -1.0
        model = ZeroInflatedTweedieGBM()
        with pytest.raises(ValueError, match="Negative y values"):
            model.fit(zi_data["X"], y_bad)

    def test_exposure_length_mismatch_raises(self, zi_data):
        bad_exp = np.ones(len(zi_data["y"]) - 1)
        model = ZeroInflatedTweedieGBM()
        with pytest.raises(ValueError, match="exposure length"):
            model.fit(zi_data["X"], zi_data["y"], exposure=bad_exp)

    def test_zero_exposure_raises(self, zi_data):
        bad_exp = np.ones(len(zi_data["y"]))
        bad_exp[0] = 0.0
        model = ZeroInflatedTweedieGBM()
        with pytest.raises(ValueError, match="positive"):
            model.fit(zi_data["X"], zi_data["y"], exposure=bad_exp)

    def test_sample_weight_length_mismatch_raises(self, zi_data):
        bad_w = np.ones(len(zi_data["y"]) - 1)
        model = ZeroInflatedTweedieGBM()
        with pytest.raises(ValueError, match="sample_weight length"):
            model.fit(zi_data["X"], zi_data["y"], sample_weight=bad_w)

    def test_negative_sample_weight_raises(self, zi_data):
        bad_w = np.ones(len(zi_data["y"]))
        bad_w[0] = -0.1
        model = ZeroInflatedTweedieGBM()
        with pytest.raises(ValueError, match="non-negative"):
            model.fit(zi_data["X"], zi_data["y"], sample_weight=bad_w)

    def test_too_few_nonzero_raises(self):
        """Should raise if fewer than 10 non-zeros."""
        rng = np.random.default_rng(0)
        n = 200
        X = rng.standard_normal((n, 3))
        y = np.zeros(n)
        y[:5] = 100.0  # only 5 non-zeros
        model = ZeroInflatedTweedieGBM()
        with pytest.raises(ValueError, match="Too few non-zero"):
            model.fit(X, y)


# ---------------------------------------------------------------------------
# Predict tests
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_shape(self, fitted_model, zi_data):
        n = len(zi_data["y"])
        mu = fitted_model.predict(zi_data["X"])
        assert mu.shape == (n,)

    def test_predict_non_negative(self, fitted_model, zi_data):
        mu = fitted_model.predict(zi_data["X"])
        assert np.all(mu >= 0)

    def test_predict_finite(self, fitted_model, zi_data):
        mu = fitted_model.predict(zi_data["X"])
        assert np.all(np.isfinite(mu))

    def test_predict_with_exposure(self, zi_data):
        model = ZeroInflatedTweedieGBM()
        model.fit(zi_data["X"], zi_data["y"], exposure=zi_data["exposure"])
        mu = model.predict(zi_data["X"], exposure=zi_data["exposure"])
        assert mu.shape == (len(zi_data["y"]),)
        assert np.all(mu >= 0)

    def test_predict_exposure_scales_output(self, zi_data):
        """Doubling exposure should roughly double the combined mean."""
        model = ZeroInflatedTweedieGBM()
        model.fit(zi_data["X"], zi_data["y"], exposure=zi_data["exposure"])
        exp1 = np.ones(len(zi_data["y"]))
        exp2 = np.full(len(zi_data["y"]), 2.0)
        mu1 = model.predict(zi_data["X"], exposure=exp1)
        mu2 = model.predict(zi_data["X"], exposure=exp2)
        # Severity doubles, zero_prob unchanged -> combined roughly doubles
        # Allow generous tolerance due to CatBoost non-linearity
        ratio = mu2 / (mu1 + 1e-8)
        assert np.median(ratio) > 1.5, f"Median ratio {np.median(ratio):.2f} should be > 1.5"

    def test_predict_not_fitted_raises(self, zi_data):
        model = ZeroInflatedTweedieGBM()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(zi_data["X"])

    def test_predict_exposure_length_mismatch_raises(self, fitted_model, zi_data):
        bad_exp = np.ones(len(zi_data["y"]) - 1)
        with pytest.raises(ValueError, match="exposure length"):
            fitted_model.predict_components(zi_data["X"], exposure=bad_exp)


# ---------------------------------------------------------------------------
# Component prediction tests
# ---------------------------------------------------------------------------

class TestPredictComponents:
    def test_returns_dict_with_correct_keys(self, fitted_model, zi_data):
        components = fitted_model.predict_components(zi_data["X"])
        assert set(components.keys()) == {"zero_prob", "severity_mean", "combined_mean"}

    def test_zero_prob_in_range(self, fitted_model, zi_data):
        components = fitted_model.predict_components(zi_data["X"])
        pi = components["zero_prob"]
        assert np.all(pi >= 0.0)
        assert np.all(pi <= 1.0)

    def test_severity_mean_positive(self, fitted_model, zi_data):
        components = fitted_model.predict_components(zi_data["X"])
        assert np.all(components["severity_mean"] > 0)

    def test_combined_mean_less_than_severity(self, fitted_model, zi_data):
        """combined = (1-pi)*severity <= severity since pi >= 0."""
        components = fitted_model.predict_components(zi_data["X"])
        assert np.all(
            components["combined_mean"] <= components["severity_mean"] + 1e-8
        )

    def test_combined_is_one_minus_pi_times_severity(self, fitted_model, zi_data):
        """Check the algebra: combined = (1 - zero_prob) * severity_mean."""
        components = fitted_model.predict_components(zi_data["X"])
        expected = (1.0 - components["zero_prob"]) * components["severity_mean"]
        np.testing.assert_allclose(
            components["combined_mean"], expected, rtol=1e-6
        )

    def test_predict_equals_combined_mean(self, fitted_model, zi_data):
        """predict() should return the same values as combined_mean."""
        mu = fitted_model.predict(zi_data["X"])
        components = fitted_model.predict_components(zi_data["X"])
        np.testing.assert_allclose(mu, components["combined_mean"], rtol=1e-6)

    def test_zero_prob_detects_high_zero_rate(self, zi_data):
        """With 85% zero rate, mean zero_prob should be well above 0.5."""
        model = ZeroInflatedTweedieGBM()
        model.fit(zi_data["X"], zi_data["y"])
        components = model.predict_components(zi_data["X"])
        assert components["zero_prob"].mean() > 0.5


# ---------------------------------------------------------------------------
# predict_proba tests
# ---------------------------------------------------------------------------

class TestPredictProba:
    def test_shape(self, fitted_model, zi_data):
        proba = fitted_model.predict_proba(zi_data["X"])
        assert proba.shape == (len(zi_data["y"]), 2)

    def test_rows_sum_to_one(self, fitted_model, zi_data):
        proba = fitted_model.predict_proba(zi_data["X"])
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_all_in_range(self, fitted_model, zi_data):
        proba = fitted_model.predict_proba(zi_data["X"])
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_col1_matches_zero_prob(self, fitted_model, zi_data):
        proba = fitted_model.predict_proba(zi_data["X"])
        components = fitted_model.predict_components(zi_data["X"])
        np.testing.assert_allclose(proba[:, 1], components["zero_prob"], atol=1e-6)

    def test_not_fitted_raises(self, zi_data):
        model = ZeroInflatedTweedieGBM()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(zi_data["X"])


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------

class TestScoring:
    def test_score_returns_float(self, fitted_model, zi_data):
        s = fitted_model.score(zi_data["X"], zi_data["y"])
        assert isinstance(s, float)

    def test_score_finite(self, fitted_model, zi_data):
        s = fitted_model.score(zi_data["X"], zi_data["y"])
        assert np.isfinite(s)

    def test_score_positive(self, fitted_model, zi_data):
        """Tweedie deviance should be non-negative."""
        s = fitted_model.score(zi_data["X"], zi_data["y"])
        assert s >= 0.0

    def test_score_with_weights(self, fitted_model, zi_data):
        weights = np.ones(len(zi_data["y"]))
        s = fitted_model.score(zi_data["X"], zi_data["y"], weights=weights)
        assert np.isfinite(s)

    def test_log_score_returns_float(self, fitted_model, zi_data):
        ls = fitted_model.log_score(zi_data["X"], zi_data["y"])
        assert isinstance(ls, float)

    def test_log_score_finite(self, fitted_model, zi_data):
        ls = fitted_model.log_score(zi_data["X"], zi_data["y"])
        assert np.isfinite(ls)

    def test_log_score_not_fitted_raises(self, zi_data):
        model = ZeroInflatedTweedieGBM()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.log_score(zi_data["X"], zi_data["y"])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_zeros(self):
        """Dataset with no structural zeros: should still train (zero_rate -> 0)."""
        rng = np.random.default_rng(77)
        n = 200
        X = rng.standard_normal((n, 4))
        y = rng.exponential(500, n)  # all positive
        assert np.all(y > 0)

        model = ZeroInflatedTweedieGBM()
        model.fit(X, y)
        mu = model.predict(X)
        assert mu.shape == (n,)
        assert np.all(mu >= 0)
        # Zero probability should be very low
        components = model.predict_components(X)
        assert components["zero_prob"].mean() < 0.5

    def test_high_zero_rate(self):
        """95% zeros — typical UK contents portfolio."""
        rng = np.random.default_rng(88)
        n = 400
        X = rng.standard_normal((n, 4))
        y = np.zeros(n)
        nonzero_idx = rng.choice(n, size=20, replace=False)
        y[nonzero_idx] = rng.exponential(300, size=20)

        model = ZeroInflatedTweedieGBM()
        model.fit(X, y)
        mu = model.predict(X)
        assert np.all(mu >= 0)

    def test_unit_exposure(self, zi_data):
        """fit/predict with explicit unit exposure should match default."""
        model1 = ZeroInflatedTweedieGBM(random_state=0)
        model1.fit(zi_data["X"], zi_data["y"])
        mu1 = model1.predict(zi_data["X"])

        unit_exp = np.ones(len(zi_data["y"]))
        model2 = ZeroInflatedTweedieGBM(random_state=0)
        model2.fit(zi_data["X"], zi_data["y"], exposure=unit_exp)
        mu2 = model2.predict(zi_data["X"], exposure=unit_exp)

        np.testing.assert_allclose(mu1, mu2, rtol=1e-5)


# ---------------------------------------------------------------------------
# Serialisation (pickle)
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_pickle_roundtrip(self, fitted_model, zi_data):
        serialised = pickle.dumps(fitted_model)
        loaded = pickle.loads(serialised)
        mu_orig = fitted_model.predict(zi_data["X"])
        mu_loaded = loaded.predict(zi_data["X"])
        np.testing.assert_allclose(mu_orig, mu_loaded, rtol=1e-6)

    def test_pickle_preserves_components(self, fitted_model, zi_data):
        serialised = pickle.dumps(fitted_model)
        loaded = pickle.loads(serialised)
        c_orig = fitted_model.predict_components(zi_data["X"])
        c_loaded = loaded.predict_components(zi_data["X"])
        np.testing.assert_allclose(
            c_orig["zero_prob"], c_loaded["zero_prob"], rtol=1e-6
        )
        np.testing.assert_allclose(
            c_orig["severity_mean"], c_loaded["severity_mean"], rtol=1e-6
        )

    def test_pickle_is_fitted(self, fitted_model):
        serialised = pickle.dumps(fitted_model)
        loaded = pickle.loads(serialised)
        assert loaded._is_fitted

    def test_pickle_unfitted_roundtrip(self):
        model = ZeroInflatedTweedieGBM(power=1.7, random_state=99)
        serialised = pickle.dumps(model)
        loaded = pickle.loads(serialised)
        assert loaded.power == 1.7
        assert loaded.random_state == 99
        assert not loaded._is_fitted


# ---------------------------------------------------------------------------
# CatBoost parameter passing
# ---------------------------------------------------------------------------

class TestCatBoostParams:
    def test_custom_zero_classifier_params(self, zi_data):
        """Custom iterations for zero classifier should be accepted."""
        model = ZeroInflatedTweedieGBM(
            catboost_params_zero={"iterations": 50}
        )
        model.fit(zi_data["X"], zi_data["y"])
        assert model._is_fitted

    def test_custom_severity_params(self, zi_data):
        """Custom iterations for severity model should be accepted."""
        model = ZeroInflatedTweedieGBM(
            catboost_params_severity={"iterations": 50}
        )
        model.fit(zi_data["X"], zi_data["y"])
        assert model._is_fitted

    def test_custom_params_override_defaults(self, zi_data):
        """Learning rate override should be stored in the params."""
        model = ZeroInflatedTweedieGBM(
            catboost_params_zero={"learning_rate": 0.1, "iterations": 50},
            catboost_params_severity={"learning_rate": 0.1, "iterations": 50},
        )
        model.fit(zi_data["X"], zi_data["y"])
        # If we got here without error, the params were accepted by CatBoost
        assert model._is_fitted

    def test_different_random_seeds_give_different_results(self, zi_data):
        """Two models with different seeds should give at least slightly different predictions."""
        model1 = ZeroInflatedTweedieGBM(random_state=1)
        model2 = ZeroInflatedTweedieGBM(random_state=999)
        model1.fit(zi_data["X"], zi_data["y"])
        model2.fit(zi_data["X"], zi_data["y"])
        mu1 = model1.predict(zi_data["X"])
        mu2 = model2.predict(zi_data["X"])
        # Not identical (different random seeds -> different tree splits)
        assert not np.allclose(mu1, mu2)


# ---------------------------------------------------------------------------
# Export / import test
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_importable_from_package(self):
        from insurance_distributional import ZeroInflatedTweedieGBM as ZIT
        assert ZIT is ZeroInflatedTweedieGBM

    def test_in_all(self):
        import insurance_distributional as pkg
        assert "ZeroInflatedTweedieGBM" in pkg.__all__
