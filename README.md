# insurance-distributional
[![Tests](https://github.com/burning-cost/insurance-distributional/actions/workflows/ci.yml/badge.svg)](https://github.com/burning-cost/insurance-distributional/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/insurance-distributional)](https://pypi.org/project/insurance-distributional/)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Distributional gradient boosting for insurance pricing. Models the full conditional distribution of losses - not just the mean - so you get a coefficient of variation per risk, not just E[Y|x].

Implements the ASTIN 2024 Best Paper approach (So & Valdez, arXiv 2406.16206) with CatBoost as the boosting engine. As of March 2026, no other open-source implementation of this exists.

---

## The problem

Standard GLMs and GBMs predict E[Y|x]. This is fine for pricing the average risk. But it tells you nothing about Var[Y|x] - the uncertainty around that prediction.

Two motor policies priced at £350 pure premium can have very different risk profiles:

- Policy A: low-frequency, moderate-severity risk. Var[Y|x] is modest.
- Policy B: low-frequency, high-severity risk (sports car, young driver). Var[Y|x] is large.

The appropriate safety loading, reinsurance attachment, and capital allocation differ between A and B. A model that only outputs the mean treats them identically.

Distributional GBM gives you both. You get `pred.mean` for the price and `pred.volatility_score()` for the uncertainty - a coefficient of variation per risk.

---

## Blog post

[Your Technical Price Ignores Variance](https://burning-cost.github.io/2026/03/08/insurance-distributional/)

---

## What is included

Four distribution classes, each following the same fit/predict interface:

| Class | Distribution | Parameters | Use case |
|-------|-------------|------------|----------|
| `TweedieGBM` | Compound Poisson-Gamma | mu, phi | Motor/home pure premium |
| `GammaGBM` | Gamma | mu, phi | Severity-only models |
| `ZIPGBM` | Zero-Inflated Poisson | lambda, pi | Pet/travel/breakdown frequency |
| `NegBinomialGBM` | Negative Binomial | mu, r | Overdispersed frequency |

---

## Installation

```bash
pip install insurance-distributional
```

Requires CatBoost, NumPy, Polars, SciPy.

---

## Quick start

```python
import numpy as np
from insurance_distributional import TweedieGBM

rng = np.random.default_rng(42)
n = 1000

# Synthetic UK motor portfolio
vehicle_age = rng.integers(0, 15, n).astype(float)
driver_age = rng.integers(18, 75, n).astype(float)
ncd_years = rng.integers(0, 5, n).astype(float)
vehicle_group = rng.integers(1, 6, n).astype(float)  # 1=small, 5=prestige

X = np.column_stack([vehicle_age, driver_age, ncd_years, vehicle_group])

# True mu: Tweedie with covariate-dependent dispersion
log_mu = (
    5.0
    + 0.02 * np.maximum(30 - driver_age, 0)
    + 0.05 * vehicle_age
    - 0.10 * ncd_years
    + 0.08 * vehicle_group
)
mu_true = np.exp(log_mu)
phi_true = 0.5 + 0.02 * vehicle_age  # older vehicles -> more dispersed losses

# Simulate Tweedie (compound Poisson-Gamma) outcomes
# Approximate via: Poisson frequency * Gamma severity
lam = mu_true ** (2 - 1.5) / (phi_true * (2 - 1.5))  # Poisson parameter
counts = rng.poisson(lam)
gamma_scale = mu_true / lam  # severity scale
y = np.array([
    rng.gamma(c, gamma_scale[i]) if c > 0 else 0.0
    for i, c in enumerate(counts)
])
exposure = rng.uniform(0.5, 1.0, n)

# 80/20 split
n_train = int(0.8 * n)
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]
exposure_train, exposure_test = exposure[:n_train], exposure[n_train:]

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

---

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

---

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

---

## Actuarial uses of volatility_score()

The volatility score (CoV per risk) is the key output beyond the mean. UK pricing teams use it for:

1. **Safety loading**: P = mu * (1 + k * CoV) where k is the risk appetite parameter.
2. **Underwriter referrals**: Flag risks where CoV exceeds a threshold.
3. **Reinsurance**: Identify segments that drive tail exposure.
4. **IFRS 17**: Risk adjustment is proportional to uncertainty - CoV provides the input.
5. **Capital allocation**: High-CoV risks consume more SCR per unit of premium.

No commercial pricing tool (Emblem, Radar, Guidewire) currently outputs CoV per risk. This library fills that gap.

---

## Scoring utilities

```python
from insurance_distributional import (
    tweedie_deviance, poisson_deviance, gamma_deviance,
    coverage, pit_values, pearson_residuals, gini_index,
)

# pred is from model.predict(X_test, exposure=exposure_test) as shown above

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

---

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

---

## Design notes

**Why separate models per parameter, not joint multi-output?** CatBoost's MultiTargetCustomObjective has sparse documentation and known GPU incompatibilities. Separate models are simpler, debuggable, and extensible. The statistical cost (no shared tree structure) is negligible.

**Why coordinate descent, not simultaneous estimation?** Follows the GAMLSS EM algorithm tradition, validated for CatBoost by So and Valdez (ASTIN 2024). One cycle is sufficient for most insurance datasets.

**Why no PyTorch?** All four distributions have well-understood log-likelihoods that SciPy handles reliably. PyTorch adds 500MB+ to the install and is not needed. XGBoostLSS uses Pyro, which is overkill for actuarial use cases.

---

## Performance

Benchmarked on a synthetic UK motor portfolio (5,000 policies, known DGP, 80/20 train/test split). Full notebook: `notebooks/01_distributional_gbm_demo.py`.

The benchmark compares `TweedieGBM(model_dispersion=True)` against a standard CatBoost Tweedie with scalar dispersion, both using identical tree hyperparameters.

| Metric | Standard GBM | Distributional GBM | Notes |
|--------|-------------|-------------------|-------|
| Tweedie deviance | baseline | comparable | Mean prediction is similar |
| Gini (mean) | baseline | comparable | Discrimination of the mean is not the point |
| CRPS | not available | measured | Only distributional model produces this |
| Log-score | not available | measured | Same — distributional coverage |
| Coverage at 80%/90%/95% | not available | close to nominal | Calibration of full distribution |
| Safety loading spread (CoV) | low (scalar phi) | materially higher | Per-risk differentiation |

The key finding: on mean prediction metrics (Tweedie deviance, Gini), the distributional model is not materially better or worse than a standard GBM — the same trees fit the same mean. The improvement is structural, not metric-based. Distributional GBM is the only model in the comparison that outputs per-risk CoV, CRPS, and calibrated coverage intervals.

Safety loading spread: the distributional model (per-risk phi) produces a 40–60% wider distribution of safety-loaded prices than the scalar-phi baseline on the same portfolio. The scalar model assigns the same CoV to every risk; the distributional model concentrates additional loading on older vehicles and young drivers — which matches the true data generating process.

**When to use:** When the actuarial requirement is a per-risk volatility score — for safety loading, underwriter referral rules, IFRS 17 risk adjustment, or reinsurance attachment analysis. Not a replacement for the mean model; an addition to it.

**When NOT to use:** When you only need E[Y|x] and have no downstream use for Var[Y|x]. The dispersion GBM adds a second fitting step (~1.5–2x total fit time) with no improvement to mean metrics.

---

## References

- So & Valdez (2024). *Zero-Inflated Tweedie Boosted Trees with CatBoost for Insurance Loss Analytics*. Applied Soft Computing. doi:10.1016/j.asoc.2025.113226. arXiv 2406.16206. **ASTIN Best Paper 2024.**
- Smyth & Jorgensen (2002). *Fitting Tweedie's Compound Poisson Model to Insurance Claims Data: Dispersion Modelling*. ASTIN Bulletin 32(1):143-157.
- Chevalier & Cote (2025). *From point to probabilistic gradient boosting for claim frequency and severity prediction*. European Actuarial Journal. doi:10.1007/s13385-025-00428-5.
- Rigby & Stasinopoulos (2005). *Generalized Additive Models for Location, Scale and Shape*. Journal of the Royal Statistical Society Series C, 54(3):507-554.

---

## Other Burning Cost libraries

**Model building**

| Library | Description |
|---------|-------------|
| [shap-relativities](https://github.com/burning-cost/shap-relativities) | Extract rating relativities from GBMs using SHAP |
| [insurance-interactions](https://github.com/burning-cost/insurance-interactions) | Automated GLM interaction detection via CANN and NID scores |
| [insurance-cv](https://github.com/burning-cost/insurance-cv) | Walk-forward cross-validation respecting IBNR structure |

**Uncertainty quantification**

| Library | Description |
|---------|-------------|
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Distribution-free prediction intervals for Tweedie models |
| [bayesian-pricing](https://github.com/burning-cost/bayesian-pricing) | Hierarchical Bayesian models for thin-data segments |
| [credibility](https://github.com/burning-cost/credibility) | Bühlmann-Straub credibility weighting |

**Deployment and optimisation**

| Library | Description |
|---------|-------------|
| [rate-optimiser](https://github.com/burning-cost/rate-optimiser) | Constrained rate change optimisation with FCA PS21/5 compliance |
| [insurance-demand](https://github.com/burning-cost/insurance-demand) | Conversion, retention, and price elasticity modelling |

**Governance**

| Library | Description |
|---------|-------------|
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Proxy discrimination auditing for UK insurance models |
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | Double Machine Learning for causal pricing inference |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model monitoring: PSI, A/E ratios, Gini drift test |

**Spatial**

| Library | Description |
|---------|-------------|
| [insurance-spatial](https://github.com/burning-cost/insurance-spatial) | BYM2 spatial territory ratemaking for UK personal lines |

[All libraries](https://burning-cost.github.io)

---

## Licence

MIT
