"""
Shared fixtures for insurance-distributional tests.

Synthetic data is designed to be minimal (fast to train CatBoost on) while
still being realistic enough to test the model logic. All random seeds are
fixed for reproducibility.
"""

import numpy as np
import pytest


# ---- Dataset sizes that keep CatBoost fast enough on the Pi (not run locally) ----
N_SMALL = 300   # used for most unit tests
N_MEDIUM = 800  # used for integration-style tests


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def tweedie_data():
    """
    Synthetic Tweedie dataset.

    Generates from compound Poisson-Gamma with known parameters so we can
    verify predictions are in the right ballpark.
    """
    rng = np.random.default_rng(0)
    n = N_SMALL
    X = rng.standard_normal((n, 5))
    X_cat = rng.integers(0, 3, size=(n, 1)).astype(float)
    X_full = np.hstack([X, X_cat])

    # True mu depends on X[:,0]
    mu_true = np.exp(1.5 + 0.8 * X[:, 0])
    phi_true = 0.5
    p = 1.5
    exposure = np.ones(n)

    # Sample from Tweedie via compound Poisson-Gamma
    lam_tw = mu_true ** (2 - p) / (phi_true * (2 - p))
    alpha = (2 - p) / (p - 1)
    beta = mu_true ** (1 - p) / (phi_true * (p - 1))

    y = np.zeros(n)
    for i in range(n):
        n_claims = rng.poisson(lam_tw[i])
        if n_claims > 0:
            y[i] = rng.gamma(shape=alpha, scale=1.0 / beta[i], size=n_claims).sum()

    return {"X": X_full, "y": y, "exposure": exposure, "mu_true": mu_true, "phi_true": phi_true}


@pytest.fixture(scope="session")
def gamma_data():
    """Synthetic Gamma severity dataset."""
    rng = np.random.default_rng(1)
    n = N_SMALL
    X = rng.standard_normal((n, 4))
    mu_true = np.exp(6.5 + 0.5 * X[:, 0])  # severity in £ range
    shape_true = 2.0  # phi = 1/shape = 0.5
    # y ~ Gamma(shape, mu/shape)
    y = rng.gamma(shape=shape_true, scale=mu_true / shape_true)
    return {"X": X, "y": y, "mu_true": mu_true}


@pytest.fixture(scope="session")
def zip_data():
    """Synthetic ZIP dataset with known pi and lambda."""
    rng = np.random.default_rng(2)
    n = N_SMALL
    X = rng.standard_normal((n, 4))
    pi_true = 0.35 + 0.15 * (X[:, 0] > 0)  # pi varies by feature
    lam_true = 0.25 + 0.15 * np.abs(X[:, 1])

    y = np.zeros(n, dtype=float)
    for i in range(n):
        if rng.random() >= pi_true[i]:  # not a structural zero
            y[i] = float(rng.poisson(lam_true[i]))

    return {"X": X, "y": y, "pi_true": pi_true, "lam_true": lam_true}


@pytest.fixture(scope="session")
def negbinom_data():
    """Synthetic Negative Binomial dataset."""
    rng = np.random.default_rng(3)
    n = N_SMALL
    X = rng.standard_normal((n, 4))
    mu_true = np.exp(0.3 + 0.5 * X[:, 0])
    r_true = 3.0
    exposure = np.ones(n)

    p_nb = r_true / (r_true + mu_true)
    y = rng.negative_binomial(n=r_true, p=p_nb).astype(float)

    return {"X": X, "y": y, "mu_true": mu_true, "r_true": r_true, "exposure": exposure}
