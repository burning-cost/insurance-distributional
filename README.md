# insurance-distributional
[![Tests](https://github.com/burning-cost/insurance-distributional/actions/workflows/ci.yml/badge.svg)](https://github.com/burning-cost/insurance-distributional/actions/workflows/ci.yml)

Distributional gradient boosting for insurance pricing. Models the full conditional distribution of losses — not just the mean.

Implements the ASTIN 2024 Best Paper approach (So & Valdez, arXiv 2406.16206) with CatBoost as the boosting engine. As of March 2026, no open-source implementation of this exists.

## The problem

Standard GLMs and GBMs predict E[Y|x]. This is fine for pricing the average risk. But it tells you nothing about Var[Y|x] — the uncertainty around that prediction.

Two motor policies priced at £350 pure premium can have very different risk profiles:

- Policy A: low-frequency, moderate-severity risk. Var[Y|x] is modest.
- Policy B: low-frequency, high-severity risk (sports car, young driver). Var[Y|x] is large.

The appropriate safety loading, reinsurance attachment, and capital allocation differ between A and B. A model that only outputs the mean treats them identically.

Distributional GBM gives you both. You get `pred.mean` for the price and `pred.volatility_score()` for the uncertainty — a coefficient of variation per risk.

## What is included

Four distribution classes, each following the same fit/predict interface:

| Class | Distribution | Parameters | Use case |
|-------|-------------|------------|----------|
| `TweedieGBM` | Compound Poisson-Gamma | mu, phi | Motor/home pure premium |
| `GammaGBM` | Gamma | mu, phi | Severity-only models |
| `ZIPGBM` | Zero-Inflated Poisson | lambda, pi | Pet/travel/breakdown frequency |
| `NegBinomialGBM` | Negative Binomial | mu, r | Overdispersed frequency |

## Installation

```bash
pip install insurance-distributional
```

Requires CatBoost, NumPy, Polars, SciPy.

## Quick start

```python
from insurance_distributional import TweedieGBM
import numpy as np

# Motor pure premium -- Tweedie compound Poisson-Gamma
model = TweedieGBM(power=1.5)
model.fit(X_train, y_train, exposure=exposure_train)

pred = model.predict(X_test, exposure=exposure_test)

pred.mean             # E[Y|X] -- pure premium
pred.variance         # Var[Y|X] -- conditional variance
pred.std              # sqrt(Var[Y|X])
pred.cov              # CoV = SD/mean per risk
pred.volatility_score()  # same as cov -- for safety loading

# Proper scoring rules
model.crps(X_test, y_test)       # CRPS (lower is better)
model.log_score(X_test, y_test)  # negative log-likelihood
```

```python
from insurance_distributional import ZIPGBM

# Pet insurance frequency -- Zero-Inflated Poisson
model = ZIPGBM()
model.fit(X_train, y_train)
pred = model.predict(X_test)

pred.mu              # observable mean = (1-pi)*lambda
pred.pi              # structural zero probability per risk
model.predict_lambda(X_test)  # underlying Poisson rate
```

```python
from insurance_distributional import NegBinomialGBM

# Overdispersed frequency (fleet motor)
model = NegBinomialGBM()
model.fit(X_train, y_train, exposure=exposure_train)
pred = model.predict(X_test)

pred.variance  # mu + mu^2/r -- always > mu (unlike Poisson)
pred.r         # overdispersion size parameter
```

## Distribution parameter interpretation

**Tweedie (mu, phi):**
- `mu`: pure premium, conditional mean E[Y|x]
- `phi`: dispersion. Var[Y|x] = phi * mu^p. Higher phi = more volatile risk.
- `cov = sqrt(phi) * mu^(p/2-1)`: coefficient of variation per risk

**Gamma (mu, phi):**
- `mu`: expected severity
- `phi = 1/shape`: dispersion. CoV = sqrt(phi). phi=0.25 means shape=4, CoV=0.5.

**ZIP (lambda, pi):**
- `lambda`: Poisson rate for the non-inflated component
- `pi`: probability of structural zero (never claiming)
- Observable mean: `(1-pi)*lambda`

**Negative Binomial (mu, r):**
- `mu`: expected count
- `r`: size/overdispersion parameter. Var = mu + mu^2/r. As r -> inf, NB -> Poisson.
- `r = 3` is typical for UK fleet motor.

## Dispersion modelling

By default, `TweedieGBM` and `GammaGBM` fit a GBM for the dispersion parameter using the Smyth-Jorgensen double GLM approach (ASTIN Bulletin 2002). The dispersion model fits on squared Pearson residuals:

```
d_i = (y_i - mu_hat_i)^2 / V(mu_hat_i)
```

where V(mu) = mu^p for Tweedie and V(mu) = mu^2 for Gamma. This converts dispersion estimation into a standard regression problem.

To use a scalar dispersion (faster, less flexible):

```python
model = TweedieGBM(power=1.5, model_dispersion=False)
```

## Actuarial uses of volatility_score()

The volatility score (CoV per risk) is the key output beyond the mean. UK pricing teams use it for:

1. **Safety loading**: P = mu * (1 + k * CoV) where k is the risk appetite parameter.
2. **Underwriter referrals**: Flag risks where CoV exceeds a threshold.
3. **Reinsurance**: Identify segments that drive tail exposure.
4. **IFRS 17**: Risk adjustment is proportional to uncertainty -- CoV provides the input.
5. **Capital allocation**: High-CoV risks consume more SCR per unit of premium.

No commercial pricing tool (Emblem, Radar, Guidewire) currently outputs CoV per risk. This library fills that gap.

## Scoring utilities

```python
from insurance_distributional import (
    tweedie_deviance, poisson_deviance, gamma_deviance,
    coverage, pit_values, pearson_residuals, gini_index,
)

# Standard actuarial deviance metrics
tweedie_deviance(y_test, pred.mean, power=1.5)
poisson_deviance(y_test, pred.mean)
gamma_deviance(y_test, pred.mean)

# Calibration: does 95% PI contain 95% of observations?
cov = coverage(y_test, pred, levels=(0.80, 0.90, 0.95))

# PIT histogram -- uniform for well-calibrated model
pit = pit_values(y_test, pred)

# Discrimination: Gini index on mean vs CoV
gini_mean = gini_index(y_test, pred.mean)
gini_vol = gini_index(y_test, pred.volatility_score())
```

## CatBoost parameters

Pass custom CatBoost parameters via `catboost_params_mu` and `catboost_params_phi`:

```python
model = TweedieGBM(
    power=1.5,
    catboost_params_mu={
        "iterations": 500,
        "learning_rate": 0.03,
        "depth": 8,
        "task_type": "GPU",
    },
    catboost_params_phi={
        "iterations": 200,
        "learning_rate": 0.05,
    },
    cat_features=["vehicle_group", "occupation_code", "postcode_area"],
)
```

CatBoost's native handling of high-cardinality categoricals via ordered target statistics is the main reason we use CatBoost rather than XGBoost or LightGBM for UK personal lines data.

## Design notes

**Why separate models per parameter, not joint multi-output?** CatBoost's MultiTargetCustomObjective has sparse documentation and known GPU incompatibilities. Separate models are simpler, debuggable, and extensible. The statistical cost (no shared tree structure) is negligible.

**Why coordinate descent, not simultaneous estimation?** Follows the GAMLSS EM algorithm tradition, validated for CatBoost by So and Valdez (ASTIN 2024). One cycle is sufficient for most insurance datasets.

**Why no PyTorch?** All four distributions have well-understood log-likelihoods that SciPy handles reliably. PyTorch adds 500MB+ to the install and is not needed. XGBoostLSS uses Pyro, which is overkill for actuarial use cases.

## References

- So & Valdez (2024). *Zero-Inflated Tweedie Boosted Trees with CatBoost for Insurance Loss Analytics*. Applied Soft Computing. doi:10.1016/j.asoc.2025.113226. arXiv 2406.16206. **ASTIN Best Paper 2024.**
- Smyth & Jorgensen (2002). *Fitting Tweedie's Compound Poisson Model to Insurance Claims Data: Dispersion Modelling*. ASTIN Bulletin 32(1):143-157.
- Chevalier & Cote (2025). *From point to probabilistic gradient boosting for claim frequency and severity prediction*. European Actuarial Journal. doi:10.1007/s13385-025-00428-5.
- Rigby & Stasinopoulos (2005). *Generalized Additive Models for Location, Scale and Shape*. Journal of the Royal Statistical Society Series C, 54(3):507-554.
