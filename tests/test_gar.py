"""
Tests for GARScenarioGenerator (insurance_distributional.gar).

Tests are designed to be runnable on Databricks where torch is available.
Locally this file can be imported but tests will be skipped if torch is absent.
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

@pytest.fixture
def small_data():
    """Minimal synthetic dataset for smoke tests."""
    rng = np.random.default_rng(42)
    n = 80
    context_size = 3
    n_assets = 1
    C = rng.standard_normal((n, context_size)).astype(np.float32)
    # Log-normal losses correlated with context
    log_losses = 0.5 * C[:, 0] + rng.standard_normal(n)
    Y = np.exp(log_losses)[:, None].astype(np.float32)
    return C, Y, context_size, n_assets


@pytest.fixture
def trained_gar(small_data):
    """A minimal trained GARScenarioGenerator for reuse across tests."""
    from insurance_distributional.gar import GARScenarioGenerator
    C, Y, context_size, n_assets = small_data
    gar = GARScenarioGenerator(
        n_assets=n_assets,
        context_size=context_size,
        latent_dim=8,
        hidden_size=16,
        risk_functional="var_es",
        alpha=0.1,
        n_mc=20,
        max_epochs=5,
        batch_size=32,
        random_state=42,
    )
    gar.fit(C, Y)
    return gar, C, Y


# ---------------------------------------------------------------------------
# 1. Import and construction
# ---------------------------------------------------------------------------

def test_import():
    from insurance_distributional.gar import GARScenarioGenerator
    assert GARScenarioGenerator is not None


def test_init_defaults():
    from insurance_distributional.gar import GARScenarioGenerator
    gar = GARScenarioGenerator(n_assets=1, context_size=2)
    assert gar.n_assets == 1
    assert gar.context_size == 2
    assert gar.risk_functional == "var_es"
    assert gar.alpha == 0.05
    assert gar.encoder == "linear"
    assert not gar._is_fitted


def test_init_invalid_alpha():
    from insurance_distributional.gar import GARScenarioGenerator
    with pytest.raises(ValueError, match="alpha"):
        GARScenarioGenerator(n_assets=1, context_size=2, alpha=1.5)


def test_init_invalid_functional():
    from insurance_distributional.gar import GARScenarioGenerator
    with pytest.raises(ValueError, match="risk_functional"):
        GARScenarioGenerator(n_assets=1, context_size=2, risk_functional="median")


def test_repr_unfitted():
    from insurance_distributional.gar import GARScenarioGenerator
    gar = GARScenarioGenerator(n_assets=1, context_size=2)
    r = repr(gar)
    assert "GARScenarioGenerator" in r
    assert "fitted=False" in r


# ---------------------------------------------------------------------------
# 2. Smoke tests — fit completes without error
# ---------------------------------------------------------------------------

def test_fit_smoke_var(small_data):
    from insurance_distributional.gar import GARScenarioGenerator
    C, Y, ctx, na = small_data
    gar = GARScenarioGenerator(
        n_assets=na, context_size=ctx, n_mc=10, max_epochs=2, batch_size=16,
        risk_functional="var", alpha=0.1, latent_dim=4, hidden_size=8, random_state=0,
    )
    gar.fit(C, Y)
    assert gar._is_fitted


def test_fit_smoke_expectile(small_data):
    from insurance_distributional.gar import GARScenarioGenerator
    C, Y, ctx, na = small_data
    gar = GARScenarioGenerator(
        n_assets=na, context_size=ctx, n_mc=10, max_epochs=2, batch_size=16,
        risk_functional="expectile", alpha=0.1, latent_dim=4, hidden_size=8, random_state=1,
    )
    gar.fit(C, Y)
    assert gar._is_fitted


def test_fit_smoke_var_es(small_data):
    from insurance_distributional.gar import GARScenarioGenerator
    C, Y, ctx, na = small_data
    gar = GARScenarioGenerator(
        n_assets=na, context_size=ctx, n_mc=10, max_epochs=2, batch_size=16,
        risk_functional="var_es", alpha=0.1, latent_dim=4, hidden_size=8, random_state=2,
    )
    gar.fit(C, Y)
    assert gar._is_fitted


def test_fit_smoke_lstm(small_data):
    from insurance_distributional.gar import GARScenarioGenerator
    C, Y, ctx, na = small_data
    gar = GARScenarioGenerator(
        n_assets=na, context_size=ctx, n_mc=10, max_epochs=2, batch_size=16,
        encoder="lstm", latent_dim=4, hidden_size=8, random_state=3,
    )
    gar.fit(C, Y)
    assert gar._is_fitted


def test_fit_1d_input():
    """Fit with 1D C and 1D Y inputs (should auto-reshape)."""
    from insurance_distributional.gar import GARScenarioGenerator
    rng = np.random.default_rng(0)
    C = rng.standard_normal(50).astype(np.float32)
    Y = np.abs(rng.standard_normal(50)).astype(np.float32)
    gar = GARScenarioGenerator(
        n_assets=1, context_size=1, n_mc=10, max_epochs=2, batch_size=16,
        latent_dim=4, hidden_size=8, random_state=4,
    )
    gar.fit(C, Y)
    assert gar._is_fitted


# ---------------------------------------------------------------------------
# 3. generate() output shape
# ---------------------------------------------------------------------------

def test_generate_shape(trained_gar):
    gar, C, Y = trained_gar
    n_scenarios = 50
    out = gar.generate(C, n_scenarios=n_scenarios)
    assert out.shape == (len(C), n_scenarios, gar.n_assets)


def test_generate_single_row(trained_gar):
    gar, C, Y = trained_gar
    c_row = C[0]  # 1D
    out = gar.generate(c_row, n_scenarios=100)
    assert out.shape == (1, 100, gar.n_assets)


def test_generate_multi_asset():
    """Test with n_assets=2."""
    from insurance_distributional.gar import GARScenarioGenerator
    rng = np.random.default_rng(5)
    n, ctx, na = 60, 2, 2
    C = rng.standard_normal((n, ctx)).astype(np.float32)
    Y = np.abs(rng.standard_normal((n, na))).astype(np.float32)
    gar = GARScenarioGenerator(
        n_assets=na, context_size=ctx, n_mc=10, max_epochs=3, batch_size=16,
        latent_dim=4, hidden_size=8, random_state=6, risk_functional="var",
    )
    gar.fit(C, Y)
    out = gar.generate(C[:5], n_scenarios=30)
    assert out.shape == (5, 30, na)


# ---------------------------------------------------------------------------
# 4. score() method
# ---------------------------------------------------------------------------

def test_score_returns_float(trained_gar):
    gar, C, Y = trained_gar
    s = gar.score(C, Y)
    assert isinstance(s, float)
    assert np.isfinite(s)


# ---------------------------------------------------------------------------
# 5. var() and es() methods
# ---------------------------------------------------------------------------

def test_var_shape(trained_gar):
    gar, C, Y = trained_gar
    v = gar.var(C, n_scenarios=100)
    assert v.shape == (len(C),)


def test_es_shape(trained_gar):
    gar, C, Y = trained_gar
    e = gar.es(C, n_scenarios=100)
    assert e.shape == (len(C),)


def test_es_geq_var(trained_gar):
    """ES >= VaR by definition for upper-tail risk measures."""
    gar, C, Y = trained_gar
    v = gar.var(C[:10], n_scenarios=500)
    e = gar.es(C[:10], n_scenarios=500)
    assert np.all(e >= v - 1e-4), "ES should be >= VaR for all observations"


# ---------------------------------------------------------------------------
# 6. Serialisation
# ---------------------------------------------------------------------------

def test_serialisation_state_dict(trained_gar, tmp_path):
    """After save/load, generate() gives identical output for the same seed."""
    import torch
    gar, C, Y = trained_gar
    state = gar.get_generator_state()

    from insurance_distributional.gar import GARScenarioGenerator
    gar2 = GARScenarioGenerator(
        n_assets=gar.n_assets, context_size=gar.context_size,
        latent_dim=gar.latent_dim, hidden_size=gar.hidden_size,
        encoder=gar.encoder, random_state=gar.random_state,
    )
    gar2.load_generator_state(state)

    torch.manual_seed(100)
    out1 = gar.generate(C[:5], n_scenarios=20)
    torch.manual_seed(100)
    out2 = gar2.generate(C[:5], n_scenarios=20)

    np.testing.assert_allclose(out1, out2, rtol=1e-5)


# ---------------------------------------------------------------------------
# 7. Scoring function unit tests
# ---------------------------------------------------------------------------

def test_score_var_known_values():
    """Quantile score at a=0, l=1, alpha=0.1: |0.1 - 1_{1<=0}| * |1 - 0| = 0.1."""
    import torch
    from insurance_distributional.gar import _score_var
    rho_hat = torch.tensor([0.0])
    pi_y = torch.tensor([1.0])
    s = _score_var(rho_hat, pi_y, alpha=0.1)
    # indicator 1_{pi_y <= rho_hat} = 1_{1 <= 0} = 0
    # |0.1 - 0| * |1 - 0| = 0.1
    np.testing.assert_allclose(s.item(), 0.1, rtol=1e-6)


def test_score_var_indicator_hit():
    """Quantile score when pi_y <= rho_hat: |alpha - 1| * |l - a|."""
    import torch
    from insurance_distributional.gar import _score_var
    rho_hat = torch.tensor([1.0])
    pi_y = torch.tensor([0.5])
    # indicator = 1_{0.5 <= 1.0} = 1
    # |0.1 - 1| * |0.5 - 1.0| = 0.9 * 0.5 = 0.45
    s = _score_var(rho_hat, pi_y, alpha=0.1)
    np.testing.assert_allclose(s.item(), 0.45, rtol=1e-6)


def test_risk_estimate_var_known():
    """VaR at alpha=0.5 of a 2-element uniform sample should be the median."""
    import torch
    from insurance_distributional.gar import _estimate_var
    # 4 samples: 1, 2, 3, 4. Upper-tail quantile at level 0.5 = quantile(x, 0.5) = 2.5
    pi_syn = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # (1, 4)
    var = _estimate_var(pi_syn, alpha=0.5)
    np.testing.assert_allclose(var.item(), 2.5, atol=1e-5)


def test_risk_estimate_expectile_mean_at_half():
    """At tau=0.5, expectile should equal the mean."""
    import torch
    from insurance_distributional.gar import _estimate_expectile
    vals = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # mean=2.5
    e = _estimate_expectile(vals, alpha=0.5)
    np.testing.assert_allclose(e.item(), 2.5, atol=1e-4)


def test_score_var_es_returns_finite():
    """Fissler-Ziegel joint score returns finite values for reasonable inputs."""
    import torch
    from insurance_distributional.gar import _score_var_es
    rho_hat = torch.tensor([[0.5, 0.8]])  # VaR=0.5, ES=0.8
    pi_y = torch.tensor([0.3])
    s = _score_var_es(rho_hat, pi_y, alpha=0.1)
    assert torch.isfinite(s).all()


# ---------------------------------------------------------------------------
# 8. Training loss recorded
# ---------------------------------------------------------------------------

def test_training_losses_recorded(trained_gar):
    gar, C, Y = trained_gar
    assert len(gar.training_losses_) == gar.max_epochs
    assert all(np.isfinite(l) for l in gar.training_losses_)


# ---------------------------------------------------------------------------
# 9. Context size mismatch raises
# ---------------------------------------------------------------------------

def test_context_size_mismatch_raises(small_data):
    from insurance_distributional.gar import GARScenarioGenerator
    C, Y, ctx, na = small_data
    gar = GARScenarioGenerator(n_assets=na, context_size=ctx + 1, n_mc=5, max_epochs=1)
    with pytest.raises(ValueError, match="context_size"):
        gar.fit(C, Y)


# ---------------------------------------------------------------------------
# 10. Not-fitted error
# ---------------------------------------------------------------------------

def test_not_fitted_raises():
    from insurance_distributional.gar import GARScenarioGenerator
    gar = GARScenarioGenerator(n_assets=1, context_size=2)
    c = np.array([[0.0, 1.0]])
    with pytest.raises(RuntimeError, match="not fitted"):
        gar.generate(c)


# ---------------------------------------------------------------------------
# 11. Top-level import guard
# ---------------------------------------------------------------------------

def test_top_level_import():
    """GARScenarioGenerator is accessible from package top level when torch is installed."""
    import insurance_distributional
    assert hasattr(insurance_distributional, "GARScenarioGenerator")


# ---------------------------------------------------------------------------
# 12. VaR level parameter override
# ---------------------------------------------------------------------------

def test_var_level_override(trained_gar):
    """var(C, level=0.01) at 99th pctile should exceed var(C, level=0.1) at 90th pctile."""
    gar, C, Y = trained_gar
    v_90 = gar.var(C[:5], level=0.10, n_scenarios=500)
    v_99 = gar.var(C[:5], level=0.01, n_scenarios=500)
    # For most well-behaved distributions, 99th pctile > 90th pctile
    assert np.mean(v_99 >= v_90) >= 0.7, "99th VaR should generally exceed 90th VaR"
