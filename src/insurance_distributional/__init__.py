"""
insurance-distributional: distributional GBM for insurance pricing.

Implements the ASTIN 2024 Best Paper approach (So & Valdez 2024) with no
existing open-source equivalent. Models the full conditional distribution of
insurance losses — not just the mean — using CatBoost as the boosting engine.

Why this matters:
  Two risks with the same E[Y|x] but different Var[Y|x] are priced identically
  by standard GLMs. They shouldn't be. The high-variance risk needs a larger
  safety loading, more cautious reinsurance, and more capital under Solvency II.
  Distributional GBM reveals this variation.

Quick start:
    from insurance_distributional import TweedieGBM, GammaGBM, ZIPGBM, NegBinomialGBM
    import numpy as np

    # Motor pure premium (Tweedie)
    model = TweedieGBM(power=1.5)
    model.fit(X_train, y_train, exposure=exposure_train)
    pred = model.predict(X_test, exposure=exposure_test)

    pred.mean            # E[Y|X] -- pure premium
    pred.variance        # Var[Y|X] -- conditional variance
    pred.volatility_score()  # CoV per risk -- for safety loading

    # Proper scoring
    model.crps(X_test, y_test)
    model.log_score(X_test, y_test)

    # Zero-inflated Tweedie (So & Valdez Scenario 2)
    from insurance_distributional import ZeroInflatedTweedieGBM
    zi = ZeroInflatedTweedieGBM(power=1.5)
    zi.fit(X_train, y_train, exposure=exposure_train)
    mu_hat = zi.predict(X_test)
    components = zi.predict_components(X_test)
    # components['zero_prob']     -- P(Y=0 | x) per risk
    # components['severity_mean'] -- E[Y | Y>0, x] per risk
    # components['combined_mean'] -- E[Y | x] = (1-pi) * mu_sev

    # Nonparametric per-risk XL layer pricing
    from insurance_distributional import FlexCodeDensity
    fc = FlexCodeDensity(max_basis=30)
    fc.fit(X_severity, y_severity)
    ev = fc.price_layer(X_test, attachment=500.0, limit=1000.0)

    # Solvency II SCR scenario generation (requires torch)
    from insurance_distributional import GARScenarioGenerator
    gar = GARScenarioGenerator(n_assets=1, context_size=10, risk_functional='var_es', alpha=0.005)
    gar.fit(C_train, y_aggregate)
    scenarios = gar.generate(C_current, n_scenarios=10_000)
    scr_99_5 = np.quantile(scenarios[0, :, 0], 0.995)

Distributions:
    TweedieGBM              Compound Poisson-Gamma. Motor/home pure premium. (mu, phi)
    GammaGBM                Gamma severity. Single-peril severity models. (mu, phi)
    ZIPGBM                  Zero-Inflated Poisson. Pet/travel/breakdown frequency. (lambda, pi)
    NegBinomialGBM          Negative Binomial counts. Overdispersed frequency. (mu, r)
    ZeroInflatedTweedieGBM  ZI-Tweedie (So & Valdez 2024 Scenario 2). (pi, mu_sev)
    FlexCodeDensity         Nonparametric CDE via cosine basis + CatBoost. XL pricing.
    GARScenarioGenerator    Generative Adversarial Regression. SCR scenario generation.
                            Requires PyTorch: pip install insurance-distributional[gar]

Reference:
    So & Valdez (2024). Zero-Inflated Tweedie Boosted Trees with CatBoost
    for Insurance Loss Analytics. arXiv 2406.16206. ASTIN Best Paper 2024.

    Smyth & Jorgensen (2002). Fitting Tweedie's Compound Poisson Model to
    Insurance Claims Data: Dispersion Modelling. ASTIN Bulletin 32(1):143-157.

    Izbicki & Lee (2017). Converting high-dimensional regression to
    high-dimensional conditional density estimation. Electronic Journal of
    Statistics, 11(2):2800-2831. arXiv:1704.08095.

    Asadi & Li (2026). Generative Adversarial Regression: Learning Conditional
    Risk Scenarios. arXiv:2603.08553.
"""

from __future__ import annotations

from .flexcode import FlexCodeDensity, FlexCodePrediction
from .gamma import GammaGBM
from .negbinom import NegBinomialGBM
from .prediction import DistributionalPrediction
from .scoring import (
    cde_loss,
    coverage,
    gamma_deviance,
    gini_index,
    negbinom_deviance,
    pearson_residuals,
    pit_values,
    poisson_deviance,
    tw_crps,
    tw_crps_profile,
    tweedie_deviance,
)
from .tweedie import TweedieGBM
from .zip import ZIPGBM
from .zi_tweedie import ZeroInflatedTweedieGBM

# GAR scenario generator — optional dependency (torch required).
# Install with: pip install insurance-distributional[gar]
try:
    from .gar import GARScenarioGenerator
except ImportError:
    # torch not installed; GARScenarioGenerator not available at top level.
    pass

# Neural GMM — optional dependency (torch required).
# Install with: pip install insurance-distributional[neural]
try:
    from .neural_gmm import NeuralGaussianMixture, GMMPrediction
except ImportError:
    # torch not installed; NeuralGaussianMixture not available at top level.
    pass

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-distributional")
except PackageNotFoundError:
    __version__ = "0.0.0"  # not installed

__all__ = [
    # Parametric models
    "TweedieGBM",
    "GammaGBM",
    "ZIPGBM",
    "NegBinomialGBM",
    "ZeroInflatedTweedieGBM",
    # Nonparametric CDE
    "FlexCodeDensity",
    "FlexCodePrediction",
    # Prediction containers
    "DistributionalPrediction",
    # Scoring utilities
    "tweedie_deviance",
    "poisson_deviance",
    "gamma_deviance",
    "negbinom_deviance",
    "cde_loss",
    "coverage",
    "pit_values",
    "pearson_residuals",
    "gini_index",
    "tw_crps",
    "tw_crps_profile",
    # GAR scenario generator (optional: requires torch)
    "GARScenarioGenerator",
    # Neural GMM (optional: requires torch)
    "NeuralGaussianMixture",
    "GMMPrediction",
]
