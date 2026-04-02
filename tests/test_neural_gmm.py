"""
Tests for NeuralGaussianMixture and GMMPrediction (insurance_distributional.neural_gmm).

Tests are designed to run on Databricks where torch is available. Locally they
will be skipped if torch is absent. Data sizes and epoch counts are kept small
(n=500, epochs=50) to avoid crashing the Raspberry Pi.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def simple_data():
    """Unimodal Gaussian-like data, n=500."""
    rng = np.random.default_rng(0)
    n = 500
    X = rng.standard_normal((n, 4)).astype(np.float32)
    y = 10.0 + 2.0 * X[:, 0] + rng.standard_normal(n).astype(np.float32)
    return X, y


@pytest.fixture(scope="module")
def bimodal_data():
    """
    Bimodal data: two distinct clusters at y~5 and y~20, feature X[:,0] identifies mode.

    Mode 1: X[:,0] < 0 -> y ~ N(5, 1)
    Mode 2: X[:,0] > 0 -> y ~ N(20, 1)

    A well-fitted K=2 GMM should recover means near 5 and 20.
    """
    rng = np.random.default_rng(1)
    n = 500
    X = rng.standard_normal((n, 3)).astype(np.float32)
    y = np.where(
        X[:, 0] < 0,
        rng.normal(5.0, 1.0, n),
        rng.normal(20.0, 1.0, n),
    ).astype(np.float32)
    return X, y


@pytest.fixture(scope="module")
def fitted_model(simple_data):
    """Fitted NeuralGaussianMixture on simple_data, small epochs for speed."""
    from insurance_distributional.neural_gmm import NeuralGaussianMixture
    X, y = simple_data
    model = NeuralGaussianMixture(
        n_components=3,
        hidden_size=32,
        n_layers=2,
        energy_weight=0.5,
        epochs=50,
        batch_size=128,
        random_state=42,
    )
    model.fit(X, y)
    return model, X, y


@pytest.fixture(scope="module")
def fitted_bimodal(bimodal_data):
    """Fitted K=2 model on bimodal_data."""
    from insurance_distributional.neural_gmm import NeuralGaussianMixture
    X, y = bimodal_data
    model = NeuralGaussianMixture(
        n_components=2,
        hidden_size=32,
        n_layers=2,
        energy_weight=0.5,
        epochs=50,
        batch_size=128,
        random_state=7,
    )
    model.fit(X, y)
    return model, X, y


# ---------------------------------------------------------------------------
# 1. fit() returns self
# ---------------------------------------------------------------------------

def test_fit_returns_self(simple_data):
    from insurance_distributional.neural_gmm import NeuralGaussianMixture
    X, y = simple_data
    model = NeuralGaussianMixture(n_components=2, epochs=5, batch_size=256, random_state=0)
    result = model.fit(X, y)
    assert result is model


# ---------------------------------------------------------------------------
# 2. predict before fit raises
# ---------------------------------------------------------------------------

def test_predict_before_fit_raises():
    from insurance_distributional.neural_gmm import NeuralGaussianMixture
    model = NeuralGaussianMixture()
    X = np.random.standard_normal((10, 3)).astype(np.float32)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(X)


# ---------------------------------------------------------------------------
# 3. predict output shape
# ---------------------------------------------------------------------------

def test_predict_shape(fitted_model):
    model, X, y = fitted_model
    K = model.n_components
    pred = model.predict(X)
    n = len(X)
    assert pred.weights.shape == (n, K), f"weights shape wrong: {pred.weights.shape}"
    assert pred.means.shape == (n, K), f"means shape wrong: {pred.means.shape}"
    assert pred.vars.shape == (n, K), f"vars shape wrong: {pred.vars.shape}"


# ---------------------------------------------------------------------------
# 4. mean matches weighted sum
# ---------------------------------------------------------------------------

def test_mean_matches_weighted_sum(fitted_model):
    model, X, _ = fitted_model
    pred = model.predict(X)
    expected = (pred.weights * pred.means).sum(axis=1)
    np.testing.assert_allclose(pred.mean, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# 5. variance analytic formula vs MC samples
# ---------------------------------------------------------------------------

def test_variance_analytic_formula(fitted_model):
    """Analytic Var[Y] should be close to Var computed from .sample()."""
    model, X, _ = fitted_model
    pred = model.predict(X[:50])  # small subset for speed
    analytic_var = pred.variance  # (50,)
    samples = pred.sample(n_samples=5000, seed=0)  # (50, 5000)
    sample_var = samples.var(axis=1)               # (50,)
    # Relative tolerance: MC estimate converges at O(1/sqrt(S)), allow 15%
    np.testing.assert_allclose(analytic_var, sample_var, rtol=0.15)


# ---------------------------------------------------------------------------
# 6. weights sum to one
# ---------------------------------------------------------------------------

def test_weights_sum_to_one(fitted_model):
    model, X, _ = fitted_model
    pred = model.predict(X)
    sums = pred.weights.sum(axis=1)
    np.testing.assert_allclose(sums, np.ones(len(X)), atol=1e-5)


# ---------------------------------------------------------------------------
# 7. variances are positive
# ---------------------------------------------------------------------------

def test_variances_positive(fitted_model):
    model, X, _ = fitted_model
    pred = model.predict(X)
    assert (pred.vars > 0).all(), "All component variances must be > 0"


# ---------------------------------------------------------------------------
# 8. log_score is finite
# ---------------------------------------------------------------------------

def test_log_score_is_finite(fitted_model):
    model, X, y = fitted_model
    score = model.log_score(X, y)
    assert np.isfinite(score), f"log_score is not finite: {score}"


# ---------------------------------------------------------------------------
# 9. energy_score is finite
# ---------------------------------------------------------------------------

def test_energy_score_is_finite(fitted_model):
    model, X, y = fitted_model
    es = model.energy_score(X, y)
    assert np.isfinite(es), f"energy_score is not finite: {es}"


# ---------------------------------------------------------------------------
# 10. crps is positive
# ---------------------------------------------------------------------------

def test_crps_positive(fitted_model):
    model, X, y = fitted_model
    crps_val = model.crps(X[:50], y[:50], n_samples=500, seed=42)
    assert crps_val > 0, f"CRPS should be positive, got {crps_val}"
    assert np.isfinite(crps_val), f"CRPS is not finite: {crps_val}"


# ---------------------------------------------------------------------------
# 11. bimodal data: means near both modes
# ---------------------------------------------------------------------------

def test_bimodal_captures_two_modes(fitted_bimodal):
    """K=2 model trained on bimodal data should have means spanning both modes.

    We check that the per-observation mean predictions:
      - For X[:,0] < 0 (mode-1 observations): predicted mean closer to 5
      - For X[:,0] > 0 (mode-2 observations): predicted mean closer to 20
    """
    model, X, y = fitted_bimodal
    pred = model.predict(X)
    means = pred.mean  # (n,)

    mode1_mask = X[:, 0] < 0
    mode2_mask = X[:, 0] > 0

    mean_mode1 = means[mode1_mask].mean()
    mean_mode2 = means[mode2_mask].mean()

    # Mode 1 predictions should average below 15; mode 2 above 10
    assert mean_mode1 < 15.0, (
        f"Mode 1 mean {mean_mode1:.2f} should be < 15 (true mode at 5)"
    )
    assert mean_mode2 > 10.0, (
        f"Mode 2 mean {mean_mode2:.2f} should be > 10 (true mode at 20)"
    )
    # They should be meaningfully different
    assert mean_mode2 - mean_mode1 > 5.0, (
        f"Modes not separated: mode1={mean_mode1:.2f}, mode2={mean_mode2:.2f}"
    )


# ---------------------------------------------------------------------------
# 12. analytic energy score matches MC (within 5%)
# ---------------------------------------------------------------------------

def test_analytic_energy_score_matches_mc(fitted_model):
    """Analytic ES should match Monte Carlo CRPS within 5% on held-out data.

    CRPS = ES for univariate distributions. The analytic formula and MC
    estimate should agree when n_samples is large enough.
    """
    model, X, y = fitted_model
    # Use a fixed subset for speed
    X_sub, y_sub = X[:100], y[:100]

    analytic_es = model.energy_score(X_sub, y_sub)
    mc_crps = model.crps(X_sub, y_sub, n_samples=4000, seed=0)

    rel_diff = abs(analytic_es - mc_crps) / (abs(mc_crps) + 1e-8)
    assert rel_diff < 0.05, (
        f"Analytic ES {analytic_es:.4f} vs MC CRPS {mc_crps:.4f}: "
        f"relative diff {rel_diff:.3f} exceeds 5%"
    )


# ---------------------------------------------------------------------------
# 13. energy_weight=0 (pure NLL) still trains
# ---------------------------------------------------------------------------

def test_energy_weight_zero_still_trains(simple_data):
    from insurance_distributional.neural_gmm import NeuralGaussianMixture
    X, y = simple_data
    model = NeuralGaussianMixture(
        n_components=2, energy_weight=0.0, epochs=10, batch_size=256, random_state=1
    )
    model.fit(X, y)
    assert model._is_fitted
    pred = model.predict(X)
    assert pred.weights.shape == (len(X), 2)
    assert np.isfinite(pred.means).all()


# ---------------------------------------------------------------------------
# 14. energy_weight=1 (pure ES) still trains
# ---------------------------------------------------------------------------

def test_energy_weight_one_still_trains(simple_data):
    from insurance_distributional.neural_gmm import NeuralGaussianMixture
    X, y = simple_data
    model = NeuralGaussianMixture(
        n_components=2, energy_weight=1.0, epochs=10, batch_size=256, random_state=2
    )
    model.fit(X, y)
    assert model._is_fitted
    pred = model.predict(X)
    assert pred.weights.shape == (len(X), 2)
    assert np.isfinite(pred.means).all()


# ---------------------------------------------------------------------------
# 15. Polars input accepted
# ---------------------------------------------------------------------------

def test_polars_input_accepted(simple_data):
    """NeuralGaussianMixture should accept Polars DataFrames and Series as inputs."""
    try:
        import polars as pl
    except ImportError:
        pytest.skip("polars not installed")

    from insurance_distributional.neural_gmm import NeuralGaussianMixture
    X_np, y_np = simple_data

    X_pl = pl.DataFrame(X_np, schema=[f"x{i}" for i in range(X_np.shape[1])])
    y_pl = pl.Series("y", y_np)

    model = NeuralGaussianMixture(
        n_components=2, epochs=5, batch_size=256, random_state=3
    )
    model.fit(X_pl, y_pl)
    assert model._is_fitted

    pred = model.predict(X_pl)
    assert pred.weights.shape == (len(X_np), 2)
