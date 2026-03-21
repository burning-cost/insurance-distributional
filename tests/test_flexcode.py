"""
Tests for FlexCodeDensity, FlexCodePrediction, basis.py, and cde_loss.

Structure:
- TestCosinesBasis: unit tests for basis.py (orthonormality, shape, values)
- TestCdeLoss: unit tests for scoring.cde_loss
- TestFlexCodeDensityFit: integration tests for FlexCodeDensity.fit()
- TestFlexCodePrediction: tests on the FlexCodePrediction container
- TestFlexCodeTune: tests for the tune() method
- TestLayerPricingVsClosedForm: actuarial validity check vs Gamma analytical formula

All CatBoost fits use small data (n=300-500) and few iterations to keep
Databricks runtime under 10 minutes. Tests are designed to run there, not
on the Raspberry Pi.
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_distributional import FlexCodeDensity, FlexCodePrediction, cde_loss
from insurance_distributional.basis import (
    cosine_basis,
    evaluate_density,
    postprocess_density,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gamma_severity_data():
    """
    Synthetic Gamma severity dataset with covariate-dependent shape.

    True model: Y | X ~ Gamma(shape(X), scale=500)
    where shape = exp(0.5 * X[:, 0]) + 1.0

    This exercises the model's ability to capture changing distributional shape
    across risk profiles — the primary use case.
    """
    rng = np.random.default_rng(42)
    n = 500
    X = rng.normal(size=(n, 3))
    shape = np.exp(0.5 * X[:, 0]) + 1.0
    scale = 500.0
    y = rng.gamma(shape, scale, n)
    return {"X": X, "y": y}


@pytest.fixture(scope="module")
def fitted_model(gamma_severity_data):
    """Pre-fitted FlexCodeDensity for use across tests."""
    model = FlexCodeDensity(
        max_basis=10,
        catboost_params={"iterations": 100},
    )
    model.fit(gamma_severity_data["X"], gamma_severity_data["y"])
    return model


# ---------------------------------------------------------------------------
# TestCosinesBasis
# ---------------------------------------------------------------------------


class TestCosinesBasis:
    def test_shape(self):
        z = np.linspace(0.0, 10.0, 100)
        B = cosine_basis(z, z_min=0.0, z_max=10.0, n_basis=8)
        assert B.shape == (100, 8)

    def test_first_basis_is_constant(self):
        """phi_1(z) = 1/sqrt(width) — a constant."""
        z = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
        B = cosine_basis(z, z_min=0.0, z_max=10.0, n_basis=5)
        expected = 1.0 / np.sqrt(10.0)
        np.testing.assert_allclose(B[:, 0], expected, rtol=1e-10)

    def test_orthonormality(self):
        """
        Basis functions should be orthonormal on [z_min, z_max].
        Gram matrix G_ij = integral phi_i * phi_j dz should equal identity.
        """
        z_grid = np.linspace(0.0, 10.0, 5000)
        B = cosine_basis(z_grid, z_min=0.0, z_max=10.0, n_basis=10)
        # G[i, j] = trapz(B[:,i] * B[:,j], z_grid)
        gram = np.trapezoid(B[:, :, None] * B[:, None, :], z_grid, axis=0)
        np.testing.assert_allclose(gram, np.eye(10), atol=1e-3)

    def test_dtype_float64(self):
        z = np.array([1.0, 2.0, 3.0])
        B = cosine_basis(z, z_min=0.0, z_max=5.0, n_basis=3)
        assert B.dtype == np.float64

    def test_single_basis_function(self):
        """n_basis=1 should return only the constant function."""
        z = np.linspace(0, 1, 10)
        B = cosine_basis(z, z_min=0.0, z_max=1.0, n_basis=1)
        assert B.shape == (10, 1)
        np.testing.assert_allclose(B[:, 0], 1.0, rtol=1e-10)

    def test_invalid_domain_raises(self):
        with pytest.raises(ValueError, match="z_max must be greater"):
            cosine_basis(np.array([1.0]), z_min=5.0, z_max=1.0, n_basis=3)

    def test_invalid_n_basis_raises(self):
        with pytest.raises(ValueError, match="n_basis must be >= 1"):
            cosine_basis(np.array([1.0]), z_min=0.0, z_max=1.0, n_basis=0)

    def test_evaluate_density_shape(self):
        coefs = np.random.default_rng(0).normal(size=(5, 8))
        z_grid = np.linspace(0.0, 10.0, 100)
        cdes = evaluate_density(coefs, z_grid, z_min=0.0, z_max=10.0)
        assert cdes.shape == (5, 100)

    def test_postprocess_non_negative(self):
        """After postprocessing, all values should be >= 0."""
        rng = np.random.default_rng(1)
        cdes = rng.normal(size=(10, 50))  # contains negatives
        z_grid = np.linspace(0.0, 10.0, 50)
        result = postprocess_density(cdes, z_grid)
        assert np.all(result >= 0)

    def test_postprocess_integrates_to_one(self):
        """After postprocessing, each row should integrate to ~1."""
        rng = np.random.default_rng(2)
        # Start with valid positive densities
        z_grid = np.linspace(0.0, 10.0, 500)
        from scipy.stats import norm
        cdes = np.tile(norm.pdf(z_grid, 5, 1), (10, 1))
        result = postprocess_density(cdes, z_grid)
        integrals = np.trapezoid(result, z_grid, axis=1)
        np.testing.assert_allclose(integrals, 1.0, atol=1e-3)


# ---------------------------------------------------------------------------
# TestCdeLoss
# ---------------------------------------------------------------------------


class TestCdeLoss:
    def test_perfect_density_better_than_wrong(self):
        """
        Perfect density (correct mean) should have lower CDE loss than
        density with wrong mean.
        """
        rng = np.random.default_rng(0)
        y_test = rng.normal(5.0, 1.0, 200)
        z_grid = np.linspace(0.0, 10.0, 500)

        from scipy.stats import norm
        cdes_perfect = np.tile(norm.pdf(z_grid, 5.0, 1.0), (200, 1))
        cdes_wrong = np.tile(norm.pdf(z_grid, 2.0, 1.0), (200, 1))

        loss_perfect = cde_loss(cdes_perfect, z_grid, y_test)
        loss_wrong = cde_loss(cdes_wrong, z_grid, y_test)
        assert loss_perfect < loss_wrong

    def test_returns_float(self):
        rng = np.random.default_rng(1)
        z_grid = np.linspace(0.0, 5.0, 100)
        cdes = np.abs(rng.normal(size=(20, 100)))
        z_test = rng.uniform(0.0, 5.0, 20)
        result = cde_loss(cdes, z_grid, z_test)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_shape_mismatch_raises(self):
        z_grid = np.linspace(0.0, 5.0, 100)
        cdes = np.ones((10, 50))  # wrong n_grid
        z_test = np.ones(10)
        with pytest.raises(ValueError):
            cde_loss(cdes, z_grid, z_test)

    def test_1d_cdes_raises(self):
        with pytest.raises(ValueError):
            cde_loss(np.ones(100), np.linspace(0, 1, 100), np.array([0.5]))


# ---------------------------------------------------------------------------
# TestFlexCodeDensityFit
# ---------------------------------------------------------------------------


class TestFlexCodeDensityFit:
    def test_fit_returns_self(self, gamma_severity_data):
        model = FlexCodeDensity(
            max_basis=5,
            catboost_params={"iterations": 50},
        )
        result = model.fit(gamma_severity_data["X"], gamma_severity_data["y"])
        assert result is model

    def test_is_fitted_flag(self, gamma_severity_data):
        model = FlexCodeDensity(
            max_basis=5,
            catboost_params={"iterations": 50},
        )
        assert not model._is_fitted
        model.fit(gamma_severity_data["X"], gamma_severity_data["y"])
        assert model._is_fitted

    def test_z_min_less_than_z_max(self, fitted_model):
        assert fitted_model._z_min < fitted_model._z_max

    def test_y_grid_shape(self, fitted_model):
        assert fitted_model._y_grid.shape == (fitted_model.n_grid,)

    def test_y_grid_positive_with_log_transform(self, fitted_model):
        """y_grid should be positive when log_transform=True."""
        assert np.all(fitted_model._y_grid > -fitted_model.log_epsilon)

    def test_y_grid_monotone(self, fitted_model):
        assert np.all(np.diff(fitted_model._y_grid) > 0)

    def test_log_transform_true_rejects_non_positive(self, gamma_severity_data):
        X = gamma_severity_data["X"]
        y = gamma_severity_data["y"].copy()
        y[0] = 0.0  # inject zero
        model = FlexCodeDensity(log_transform=True, catboost_params={"iterations": 10})
        with pytest.raises(ValueError, match="log_transform=True requires y > 0"):
            model.fit(X, y)

    def test_predict_before_fit_raises(self):
        model = FlexCodeDensity()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_density(np.zeros((5, 3)))

    def test_invalid_basis_system_raises(self):
        with pytest.raises(ValueError, match="basis_system must be 'cosine'"):
            FlexCodeDensity(basis_system="fourier")

    def test_repr_contains_status(self, fitted_model):
        r = repr(fitted_model)
        assert "fitted" in r
        assert "FlexCodeDensity" in r


# ---------------------------------------------------------------------------
# TestFlexCodePrediction
# ---------------------------------------------------------------------------


class TestFlexCodePrediction:
    def test_predict_density_shape(self, fitted_model, gamma_severity_data):
        pred = fitted_model.predict_density(gamma_severity_data["X"][:10])
        assert pred.cdes.shape == (10, fitted_model.n_grid)
        assert pred.y_grid.shape == (fitted_model.n_grid,)

    def test_density_non_negative(self, fitted_model, gamma_severity_data):
        pred = fitted_model.predict_density(gamma_severity_data["X"][:20])
        assert np.all(pred.cdes >= 0)

    def test_density_integrates_to_one(self, fitted_model, gamma_severity_data):
        pred = fitted_model.predict_density(gamma_severity_data["X"][:20])
        integrals = np.trapezoid(pred.cdes, pred.y_grid, axis=1)
        np.testing.assert_allclose(integrals, 1.0, atol=0.05)

    def test_mean_positive(self, fitted_model, gamma_severity_data):
        pred = fitted_model.predict_density(gamma_severity_data["X"][:10])
        assert np.all(pred.mean > 0)

    def test_variance_non_negative(self, fitted_model, gamma_severity_data):
        pred = fitted_model.predict_density(gamma_severity_data["X"][:10])
        assert np.all(pred.variance >= 0)

    def test_volatility_score_positive(self, fitted_model, gamma_severity_data):
        pred = fitted_model.predict_density(gamma_severity_data["X"][:10])
        assert np.all(pred.volatility_score() > 0)

    def test_quantile_ordering(self, fitted_model, gamma_severity_data):
        """q=0.95 quantile should exceed q=0.5 quantile for all obs."""
        X = gamma_severity_data["X"][:20]
        q50 = fitted_model.predict_quantile(X, 0.5)
        q95 = fitted_model.predict_quantile(X, 0.95)
        assert np.all(q95 >= q50)

    def test_quantile_scalar_returns_1d(self, fitted_model, gamma_severity_data):
        X = gamma_severity_data["X"][:10]
        q = fitted_model.predict_quantile(X, 0.5)
        assert q.ndim == 1
        assert len(q) == 10

    def test_quantile_list_returns_2d(self, fitted_model, gamma_severity_data):
        X = gamma_severity_data["X"][:10]
        q = fitted_model.predict_quantile(X, [0.25, 0.5, 0.75])
        assert q.shape == (10, 3)
        # Quantiles should be non-decreasing across the q axis
        assert np.all(q[:, 1] >= q[:, 0])
        assert np.all(q[:, 2] >= q[:, 1])

    def test_price_layer_between_zero_and_limit(self, fitted_model, gamma_severity_data):
        X = gamma_severity_data["X"][:20]
        limit = 1000.0
        ev = fitted_model.price_layer(X, attachment=500.0, limit=limit)
        assert np.all(ev >= 0)
        assert np.all(ev <= limit)

    def test_price_layer_increases_with_lower_attachment(self, fitted_model, gamma_severity_data):
        """Lower attachment -> higher expected layer loss."""
        X = gamma_severity_data["X"][:20]
        ev_low = fitted_model.price_layer(X, attachment=200.0, limit=1000.0)
        ev_high = fitted_model.price_layer(X, attachment=1000.0, limit=1000.0)
        # On average, lower attachment should give higher layer EV
        assert np.mean(ev_low) >= np.mean(ev_high)

    def test_price_layer_warns_on_high_attachment(self, fitted_model, gamma_severity_data):
        pred = fitted_model.predict_density(gamma_severity_data["X"][:5])
        with pytest.warns(UserWarning, match="attachment"):
            pred.price_layer(attachment=1e12, limit=1e12)

    def test_pit_values_shape_and_range(self, fitted_model, gamma_severity_data):
        X = gamma_severity_data["X"][:30]
        y = gamma_severity_data["y"][:30]
        pred = fitted_model.predict_density(X)
        pit = pred.pit_values(y)
        assert pit.shape == (30,)
        assert np.all(pit >= 0) and np.all(pit <= 1)

    def test_pit_values_length_mismatch_raises(self, fitted_model, gamma_severity_data):
        pred = fitted_model.predict_density(gamma_severity_data["X"][:10])
        with pytest.raises(ValueError, match="y_obs length"):
            pred.pit_values(np.ones(5))

    def test_cde_loss_finite(self, fitted_model, gamma_severity_data):
        X = gamma_severity_data["X"][:30]
        y = gamma_severity_data["y"][:30]
        pred = fitted_model.predict_density(X)
        loss = pred.cde_loss(y)
        assert np.isfinite(loss)
        assert isinstance(loss, float)

    def test_n_basis_used_attribute(self, fitted_model, gamma_severity_data):
        pred = fitted_model.predict_density(gamma_severity_data["X"][:5])
        assert pred.n_basis_used == fitted_model.max_basis

    def test_prediction_repr(self, fitted_model, gamma_severity_data):
        pred = fitted_model.predict_density(gamma_severity_data["X"][:5])
        r = repr(pred)
        assert "FlexCodePrediction" in r


# ---------------------------------------------------------------------------
# TestFlexCodeTune
# ---------------------------------------------------------------------------


class TestFlexCodeTune:
    def test_tune_sets_best_basis(self, gamma_severity_data):
        X, y = gamma_severity_data["X"], gamma_severity_data["y"]
        model = FlexCodeDensity(
            max_basis=20,
            catboost_params={"iterations": 80},
        )
        model.fit(X[:400], y[:400])
        model.tune(X[400:], y[400:])
        assert hasattr(model, "best_basis_")
        assert model.best_basis_ is not None

    def test_best_basis_in_range(self, gamma_severity_data):
        X, y = gamma_severity_data["X"], gamma_severity_data["y"]
        model = FlexCodeDensity(
            max_basis=15,
            catboost_params={"iterations": 80},
        )
        model.fit(X[:400], y[:400])
        model.tune(X[400:], y[400:])
        assert 1 <= model.best_basis_ <= 15

    def test_tune_uses_best_basis_in_predict(self, gamma_severity_data):
        """After tune(), predict_density should use best_basis_, not max_basis."""
        X, y = gamma_severity_data["X"], gamma_severity_data["y"]
        model = FlexCodeDensity(
            max_basis=15,
            catboost_params={"iterations": 80},
        )
        model.fit(X[:400], y[:400])
        model.tune(X[400:], y[400:])
        pred = model.predict_density(X[:5])
        assert pred.n_basis_used == model.best_basis_

    def test_tune_before_fit_raises(self, gamma_severity_data):
        model = FlexCodeDensity()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.tune(gamma_severity_data["X"], gamma_severity_data["y"])

    def test_tune_custom_candidates(self, gamma_severity_data):
        X, y = gamma_severity_data["X"], gamma_severity_data["y"]
        model = FlexCodeDensity(
            max_basis=10,
            catboost_params={"iterations": 50},
        )
        model.fit(X[:400], y[:400])
        model.tune(X[400:], y[400:], basis_candidates=[3, 5, 7, 10])
        assert model.best_basis_ in [3, 5, 7, 10]


# ---------------------------------------------------------------------------
# TestLayerPricingVsClosedForm (actuarial validity)
# ---------------------------------------------------------------------------


class TestLayerPricingVsClosedForm:
    def test_layer_pricing_vs_analytical_gamma(self):
        """
        For marginal Gamma severity (null features), FlexCode layer price
        should be within 15% of the analytical expected layer loss.

        Analytical: E[min(max(Y-a,0),l)] = integral_a^{a+l} S_Y(t) dt
        where S_Y(t) = 1 - Gamma_CDF(t; shape, scale).

        Uses n_train=800, max_basis=20, n_grid=400 for reasonable accuracy.
        The 15% tolerance accounts for finite sample variability.
        """
        from scipy.integrate import quad
        from scipy.stats import gamma as gamma_dist

        rng = np.random.default_rng(0)
        shape, scale = 3.0, 500.0
        n_train = 800
        y_train = rng.gamma(shape, scale, n_train)
        # Use tiny noise so CatBoost does not reject a constant feature column.
        # The signal is negligible vs the variance of y, so this approximates
        # a marginal density estimate.
        X_train = rng.normal(0, 0.001, size=(n_train, 1))

        model = FlexCodeDensity(
            max_basis=20,
            n_grid=400,
            log_transform=True,
            catboost_params={"iterations": 150},
        )
        model.fit(X_train, y_train)

        X_test = np.zeros((1, 1))
        a, l = 500.0, 1000.0

        # FlexCode estimate
        fc_ev = model.price_layer(X_test, attachment=a, limit=l)[0]

        # Analytical: integral_a^{a+l} (1 - Gamma_CDF(t; shape, scale)) dt
        true_ev, _ = quad(
            lambda t: 1.0 - gamma_dist.cdf(t, a=shape, scale=scale),
            a, a + l,
        )

        rel_err = abs(fc_ev - true_ev) / true_ev
        assert rel_err < 0.15, (
            f"Layer price relative error {rel_err:.3%} exceeds 15% tolerance. "
            f"FlexCode: {fc_ev:.2f}, Analytical: {true_ev:.2f}."
        )

    def test_mean_vs_gamma_mean(self):
        """
        For Gamma data, the predicted mean E[Y|X] should be within 20% of
        the true Gamma mean, given null features (marginal estimate).
        """
        rng = np.random.default_rng(1)
        shape, scale = 2.0, 600.0
        true_mean = shape * scale  # 1200.0
        n_train = 600
        y_train = rng.gamma(shape, scale, n_train)
        # Tiny noise to avoid CatBoost constant-feature rejection
        X_train = rng.normal(0, 0.001, size=(n_train, 1))

        model = FlexCodeDensity(
            max_basis=15,
            n_grid=300,
            catboost_params={"iterations": 120},
        )
        model.fit(X_train, y_train)

        pred = model.predict_density(rng.normal(0, 0.001, size=(1, 1)))
        estimated_mean = pred.mean[0]

        rel_err = abs(estimated_mean - true_mean) / true_mean
        assert rel_err < 0.20, (
            f"Mean relative error {rel_err:.3%} exceeds 20% tolerance. "
            f"FlexCode mean: {estimated_mean:.2f}, True mean: {true_mean:.2f}."
        )


# ---------------------------------------------------------------------------
# TestFlexCodeLogScore
# ---------------------------------------------------------------------------


class TestFlexCodeLogScore:
    def test_log_score_finite(self, fitted_model, gamma_severity_data):
        X = gamma_severity_data["X"][:30]
        y = gamma_severity_data["y"][:30]
        ls = fitted_model.log_score(X, y)
        assert np.isfinite(ls)
        assert isinstance(ls, float)

    def test_better_model_lower_log_score(self, gamma_severity_data):
        """
        A model trained on the same distribution should have lower log-score
        than a model that has never seen the data range.
        """
        X, y = gamma_severity_data["X"], gamma_severity_data["y"]
        rng = np.random.default_rng(99)

        # Model trained on correct Gamma data
        model_good = FlexCodeDensity(
            max_basis=10,
            catboost_params={"iterations": 80},
        )
        model_good.fit(X[:400], y[:400])

        # Model trained on data with a completely different scale
        y_wrong = rng.gamma(2, 50.0, 400)  # much smaller values
        model_wrong = FlexCodeDensity(
            max_basis=10,
            catboost_params={"iterations": 80},
        )
        model_wrong.fit(X[:400], y_wrong)

        ls_good = model_good.log_score(X[400:], y[400:])
        ls_wrong = model_wrong.log_score(X[400:], y[400:])
        assert ls_good < ls_wrong


# ---------------------------------------------------------------------------
# TestFlexCodeCrps
# ---------------------------------------------------------------------------


class TestFlexCodeCrps:
    def test_crps_finite_positive(self, fitted_model, gamma_severity_data):
        X = gamma_severity_data["X"][:20]
        y = gamma_severity_data["y"][:20]
        crps_val = fitted_model.crps(X, y)
        assert np.isfinite(crps_val)
        assert crps_val > 0  # CRPS is non-negative


# ---------------------------------------------------------------------------
# TestPolarsInput
# ---------------------------------------------------------------------------


class TestPolarsInput:
    def test_polars_dataframe_input(self, gamma_severity_data):
        """FlexCodeDensity should accept polars DataFrames."""
        import polars as pl

        X = gamma_severity_data["X"][:100]
        y = gamma_severity_data["y"][:100]
        X_pl = pl.DataFrame(X, schema=[f"f{i}" for i in range(X.shape[1])])
        y_pl = pl.Series("y", y)

        model = FlexCodeDensity(
            max_basis=5,
            catboost_params={"iterations": 30},
        )
        model.fit(X_pl, y_pl)
        pred = model.predict_density(X_pl[:10])
        assert pred.cdes.shape[0] == 10


# ---------------------------------------------------------------------------
# log_epsilon warning tests (Bug 3 fix)
# ---------------------------------------------------------------------------


class TestLogEpsilonWarning:
    """
    Verify that FlexCodeDensity emits a UserWarning when log_epsilon is
    larger than the minimum observed y. This guards against silently wrong
    log-likelihoods when losses are in sub-unit scale (pence, cents, [0,1]).
    """

    def test_warns_when_y_min_lt_log_epsilon(self):
        """Fitting on y values < log_epsilon should trigger a UserWarning."""
        rng = np.random.default_rng(0)
        # y in [0.001, 0.1] — much smaller than the default log_epsilon=1.0
        y = rng.uniform(0.001, 0.1, 100)
        X = rng.normal(size=(100, 2))
        model = FlexCodeDensity(
            max_basis=3,
            log_epsilon=1.0,  # default — will be larger than y.min()
            catboost_params={"iterations": 5},
        )
        with pytest.warns(UserWarning, match="log_epsilon"):
            model.fit(X, y)

    def test_warning_suggests_1e6(self):
        """Warning message should reference 1e-6 as a suggested value."""
        rng = np.random.default_rng(1)
        y = rng.uniform(0.001, 0.1, 80)
        X = rng.normal(size=(80, 2))
        model = FlexCodeDensity(
            max_basis=3,
            log_epsilon=1.0,
            catboost_params={"iterations": 5},
        )
        with pytest.warns(UserWarning, match="1e-6"):
            model.fit(X, y)

    def test_no_warning_when_log_epsilon_appropriate(self):
        """No warning when log_epsilon is well below y.min()."""
        rng = np.random.default_rng(2)
        # y in [100, 5000] — log_epsilon=1.0 is safely below y.min()
        y = rng.uniform(100.0, 5000.0, 80)
        X = rng.normal(size=(80, 2))
        model = FlexCodeDensity(
            max_basis=3,
            log_epsilon=1.0,
            catboost_params={"iterations": 5},
        )
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", UserWarning)
            # Should not raise — log_epsilon < y.min()
            model.fit(X, y)
