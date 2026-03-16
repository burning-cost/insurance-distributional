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

Distributions:
    TweedieGBM      Compound Poisson-Gamma. Motor/home pure premium. (mu, phi)
    GammaGBM        Gamma severity. Single-peril severity models. (mu, phi)
    ZIPGBM          Zero-Inflated Poisson. Pet/travel/breakdown frequency. (lambda, pi)
    NegBinomialGBM  Negative Binomial counts. Overdispersed frequency. (mu, r)

Reference:
    So & Valdez (2024). Zero-Inflated Tweedie Boosted Trees with CatBoost
    for Insurance Loss Analytics. arXiv 2406.16206. ASTIN Best Paper 2024.

    Smyth & Jorgensen (2002). Fitting Tweedie's Compound Poisson Model to
    Insurance Claims Data: Dispersion Modelling. ASTIN Bulletin 32(1):143-157.
"""

from __future__ import annotations

from .gamma import GammaGBM
from .negbinom import NegBinomialGBM
from .prediction import DistributionalPrediction
from .scoring import (
    coverage,
    gamma_deviance,
    gini_index,
    negbinom_deviance,
    pearson_residuals,
    pit_values,
    poisson_deviance,
    tweedie_deviance,
)
from .tweedie import TweedieGBM
from .zip import ZIPGBM

__version__ = "0.1.2"

__all__ = [
    # Models
    "TweedieGBM",
    "GammaGBM",
    "ZIPGBM",
    "NegBinomialGBM",
    # Prediction container
    "DistributionalPrediction",
    # Scoring utilities
    "tweedie_deviance",
    "poisson_deviance",
    "gamma_deviance",
    "negbinom_deviance",
    "coverage",
    "pit_values",
    "pearson_residuals",
    "gini_index",
]
