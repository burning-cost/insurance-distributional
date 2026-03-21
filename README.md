# insurance-distributional
[![Tests](https://github.com/burning-cost/insurance-distributional/actions/workflows/ci.yml/badge.svg)](https://github.com/burning-cost/insurance-distributional/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/insurance-distributional)](https://pypi.org/project/insurance-distributional/)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/insurance-distributional/blob/main/notebooks/quickstart.ipynb)

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
uv add insurance-distributional
```

Requires CatBoost, NumPy, Polars, SciPy.

> 💬 Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-distributional/discussions). Found it useful? A ⭐ helps others find it.

---

## Quick start

```python
import numpy as np
from insurance_distributional import TweedieGBM

rng = np.random.default_rng(42)
n = 1000

# Synthetic covariate features
vehicle_age = rng.integers(0, 15, n).astype(float)
driver_age = rng.integers(18, 75, n).astype(float)
ncd_years = rng.integers(0, 5, n).astype(float)
vehicle_group = rng.integers(1, 6, n).astype(float)  # 1=small, 5=prestige

X = np.column_stack([vehicle_age, driver_age, ncd_years, vehicle_group])

# Statistical demonstration DGP — mu values here are large (exp(5) ≈ 150)
# to produce clearly non-zero Tweedie outcomes with n=1000. This is not
# calibrated to UK motor frequencies (~0.12 claims/year). For realistic
# motor pure premium modelling, set log_mu to produce mu ≈ 200-600 (claim costs)
# or use a separate frequency * severity structure.
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

# Simulate Tweedie (compound Poisson-Gamma) outcomes: correct compound Poisson approach.
# For Tweedie(p), individual claim severity ~ Gamma(alpha, beta) where:
#   alpha = (2-p)/(p-1)   [shape of each individual claim]
#   lambda = mu^(2-p) / (phi*(2-p))   [Poisson claim count parameter]
# Y_i = sum of `count_i` independent Gamma(alpha, beta_i) draws.
# At p=1.5: alpha = (2-1.5)/(1.5-1) = 0.5/0.5 = 1.0 (Exponential severity).
p_tweedie = 1.5
alpha_sev = (2 - p_tweedie) / (p_tweedie - 1)  # = 1.0 for p=1.5
lam = mu_true ** (2 - p_tweedie) / (phi_true * (2 - p_tweedie))  # Poisson parameter
counts = rng.poisson(lam)
# Individual severity mean = mu / lambda; scale = mean / alpha
beta_sev = mu_true / (lam * alpha_sev)
y = np.array([
    rng.gamma(alpha_sev, beta_sev[i], size=c).sum() if c > 0 else 0.0
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
import numpy as np
from insurance_distributional import ZIPGBM

# Synthetic pet insurance frequency — Zero-Inflated Poisson
# Many pets never claim in a year (structural zeros)
rng_zip = np.random.default_rng(7)
n_zip = 800

pet_age    = rng_zip.integers(0, 15, n_zip).astype(float)
breed_risk = rng_zip.integers(1, 5, n_zip).astype(float)  # 1=low, 4=high
vet_cover  = rng_zip.uniform(2_000, 10_000, n_zip)         # excess level

X_zip = np.column_stack([pet_age, breed_risk, vet_cover])

# True DGP: 55% structural zeros, rest Poisson(0.4 + 0.05*breed_risk)
pi_true  = 0.55
lam_true = 0.40 + 0.05 * breed_risk
y_zip = np.where(
    rng_zip.random(n_zip) < pi_true,
    0,
    rng_zip.poisson(lam_true),
).astype(float)

n_tr = int(0.8 * n_zip)
X_zip_train, X_zip_test = X_zip[:n_tr], X_zip[n_tr:]
y_zip_train = y_zip[:n_tr]

model = ZIPGBM()
model.fit(X_zip_train, y_zip_train)
pred = model.predict(X_zip_test)

pred.mu              # observable mean = (1-pi)*lambda
pred.pi              # structural zero probability per risk
model.predict_lambda(X_zip_test)  # underlying Poisson rate
```

```python
import numpy as np
from insurance_distributional import NegBinomialGBM

# Synthetic fleet motor frequency — Negative Binomial (overdispersed)
# Fleet vehicles have heterogeneous latent risk not captured by rating factors
rng_nb = np.random.default_rng(99)
n_nb = 600

vehicle_age_nb = rng_nb.integers(1, 10, n_nb).astype(float)
fleet_size     = rng_nb.integers(5, 50, n_nb).astype(float)   # fleet vehicles
annual_miles   = rng_nb.uniform(15_000, 80_000, n_nb)

X_nb = np.column_stack([vehicle_age_nb, fleet_size, annual_miles / 10_000])
exposure_nb = rng_nb.uniform(0.5, 1.0, n_nb)  # vehicle-years

# True DGP: NB with mu=0.12 per vehicle-year, r=3 (typical UK fleet)
mu_true_nb = 0.12 * exposure_nb
r_true     = 3.0
p_nb = r_true / (r_true + mu_true_nb)
y_nb = rng_nb.negative_binomial(r_true, p_nb).astype(float)

n_tr_nb = int(0.8 * n_nb)
X_nb_train, X_nb_test = X_nb[:n_tr_nb], X_nb[n_tr_nb:]
y_nb_train = y_nb[:n_tr_nb]
exp_nb_train = exposure_nb[:n_tr_nb]
exp_nb_test  = exposure_nb[n_tr_nb:]

# Overdispersed frequency (fleet motor)
model = NegBinomialGBM()
model.fit(X_nb_train, y_nb_train, exposure=exp_nb_train)
pred = model.predict(X_nb_test, exposure=exp_nb_test)

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
- `phi = 1/shape`: dispersion (`phi` here is the GLM canonical dispersion parameter, equal to 1/shape for the Gamma — not the shape parameter itself). CoV = sqrt(phi). phi=0.25 means shape=4, CoV=0.5.

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
4. **IFRS 17 risk adjustment (variance-based approach)**: per-risk CoV provides an input to variance-proportional RA methods. Other approaches (quantile, cost of capital, CTE) are more common in practice.
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

Benchmarked against a constant-phi Gamma GLM on 6,000 synthetic UK motor severity observations (known heteroskedastic DGP: phi varies 0.42–1.18 across vehicle age and vehicle group). Run on Databricks serverless using `insurance-distributional==0.1.3` (the v0.1.3 cross-fitting fix). Run date: 2026-03-16.

| Metric | Constant-phi Gamma GLM | GammaGBM (per-risk phi) | Notes |
|--------|----------------------|------------------------|-------|
| Gamma deviance | 1.201 | 0.959 | GBM mean prediction is better |
| Log-likelihood | -10,205.7 | -10,050.3 | GBM wins; +1.5% improvement |
| Coverage at 80% | 74.8% | 80.4% | GBM near-nominal; GLM under-covers |
| Coverage at 90% | 84.2% | 89.5% | GBM near-nominal; GLM under-covers |
| Coverage at 95% | 90.1% | 94.9% | GBM near-nominal; GLM under-covers |
| Phi correlation with true phi | 0.000 (flat) | +0.702 | GBM recovers the correct ordering |
| Safety loading spread | 0 | 0.086 | GBM differentiates risks; GLM cannot |

**CoV accuracy by vehicle age quartile (test set):**

| Quartile | True CoV | GammaGBM predicted CoV | Constant-phi CoV |
|----------|----------|------------------------|-----------------|
| va <= 4 | 0.767 | 0.809 | 1.015 |
| va <= 7 | 0.861 | 0.906 | 1.015 |
| va <= 11 | 0.935 | 1.026 | 1.015 |
| va > 11 | 1.010 | 1.117 | 1.015 |

**What the v0.1.3 cross-fitting fix changed:** In v0.1.2, the phi model was trained on in-sample residuals from the mu model. Because CatBoost with 300 trees overfits on training data, in-sample mu predictions were close to y, making the squared relative errors near-zero. The phi model learned to predict very small phi, causing 42% actual coverage at the 80% nominal level.

The v0.1.3 fix uses K=3 cross-fitting (double-ML style): for each fold, the phi residuals are computed from a mu model that did not see that fold's data. This makes E[d_i] = phi_i unbiasedly. Coverage is now calibrated.

**Honest assessment of remaining limitations:** The GBM consistently overestimates absolute CoV by ~5–12% across vehicle age quartiles (predicted 0.81–1.12 vs true 0.77–1.01). This is a systematic upward bias, likely from the phi model fitting on OOF residuals that include out-of-support predictions at fold boundaries. The ordering is correct — phi correlation +0.70 — but you should not use the raw `pred.phi` values as literal point estimates without validation against holdout data.

The "safety loading spread" ratio in the output script shows a spuriously large number (the constant-phi spread is exactly zero by construction). The meaningful number is the GBM spread itself: 0.086, meaning the loading ratio varies by roughly ±8.6% across the portfolio.

**Practical implications:**
- `pred.mean` is reliable and better than OLS on this DGP. Use it.
- `pred.phi` and `pred.volatility_score()` now produce calibrated prediction intervals. The 80%/90%/95% prediction intervals are accurate on this DGP (80.4%/89.5%/94.9% empirical coverage).
- The absolute scale of `pred.phi` has an upward bias of ~5–12%. Cross-validate on your own data before using raw phi values in safety loading formulas.
- `pred.volatility_score()` correctly ranks which risks are higher-CoV (phi correlation +0.70). Use it for relative comparisons and underwriter referral thresholds.

**When to use:** Any time you need per-risk uncertainty beyond the mean. Coverage intervals are now calibrated, so prediction intervals from `coverage()` are trustworthy for reinsurance attachment, IFRS 17 risk adjustment, and capital allocation.

**When to validate before using:** If you need `pred.phi` as a literal number in a pricing formula (e.g., P = mu * (1 + 0.5 * sqrt(phi))), cross-validate the absolute scale on a holdout set first. The directional signal is reliable; the absolute value may need recalibration.

---


---

## FlexCodeDensity

**New in v0.2.0.** Nonparametric conditional density estimation f(y|x) using the FlexCode series expansion (Izbicki & Lee, 2017). Gives you the full conditional distribution — quantiles, XL layer expected values, tail probabilities — without assuming a parametric tail (no GPD, no threshold selection).

The density is estimated as a finite cosine-basis expansion:

    f(y|x) = Σ_{k=1}^{I} β_k(x) · φ_k(y)

where each β_k(x) is learned by CatBoost MultiRMSE regression — one model for all I basis functions simultaneously. The log-transform handles right-skewed severity: fitting in Z = log(y + ε) space requires far fewer basis functions.

**When to use this instead of EQRN:** FlexCode is simpler when the training data covers the pricing layer (motor OD, property, home). EQRN (in `insurance_quantile.eqrn`) is better for lines where the worst historical claim is not a credible upper bound on future losses (motor BI, liability).

```python
import numpy as np
from insurance_distributional import FlexCodeDensity

rng = np.random.default_rng(42)
n_train, n_test = 8_000, 1_000

vehicle_age   = rng.integers(1, 15, n_train + n_test).astype(float)
driver_age    = rng.integers(21, 75, n_train + n_test).astype(float)
ncd_years     = rng.integers(0, 9, n_train + n_test).astype(float)
vehicle_group = rng.choice([1.0, 2.0, 3.0, 4.0], size=n_train + n_test)

# Heteroskedastic lognormal severity (tail weight varies by vehicle group)
log_mu    = 7.2 + 0.03 * vehicle_age - 0.008 * ncd_years + 0.15 * vehicle_group
log_sigma = 0.45 + 0.06 * vehicle_group
y_sev = np.exp(rng.normal(log_mu, log_sigma))  # severity only (positive losses)

X = np.column_stack([vehicle_age, driver_age, ncd_years, vehicle_group])
X_train, X_test = X[:n_train], X[n_train:]
y_train = y_sev[:n_train]

# Fit FlexCode conditional density
model = FlexCodeDensity(
    max_basis=30,        # number of cosine basis functions
    log_transform=True,  # essential for right-skewed severity (default)
    n_grid=250,          # density grid points for integration
)
model.fit(X_train, y_train)

# Tune: select optimal number of basis functions by CDE loss on a validation set
model.tune(X_test, y_sev[n_train:])
# model.best_basis_ is set automatically

# Full conditional density (n_test, n_grid)
pred = model.predict_density(X_test)
pred.cdes      # density values, shape (n_test, n_grid)
pred.y_grid    # y-axis in original scale

# Conditional quantiles — no GPD assumption
q95 = pred.quantile(0.95)       # shape (n_test,)
q99 = pred.quantile(0.99)

# Or use the shortcut method directly
q95_direct = model.predict_quantile(X_test, q=0.95)

# XL layer expected value: E[loss in (£100k xs £400k) | X]
# This is what a reinsurance underwriter wants for a £400k xs £100k layer
layer_ev = model.price_layer(X_test, attachment=100_000, limit=400_000)

# Scoring
model.crps(X_test, y_sev[n_train:])      # CRPS (lower is better)
model.log_score(X_test, y_sev[n_train:]) # mean negative log-likelihood

# PIT histogram for calibration diagnostics
pit = pred.pit_values(y_sev[n_train:])  # should be uniform if well-calibrated
```

```
FlexCodeDensity params:
  max_basis       — 30 for smooth unimodal severity, 50+ for bimodal or heavy tails
  log_transform   — True by default; set False only for count data or when y can be negative
  log_epsilon     — continuity correction for log transform (default 1.0, for £ claims)
                    For sub-unit losses, set to a fraction of the minimum observed loss
  n_grid          — density grid resolution (200 is fine, 500+ for sharp tails)
  z_max_override  — override the automatic upper bound; use when pricing layers
                    that extend beyond the observed maximum (in log-space)
```

**Key advantage over parametric GBMs:** FlexCodeDensity makes no shape assumption. If motor BI severity changes from roughly Gamma near the mean to something heavier-tailed at high values — and the tails differ across vehicle groups — FlexCode learns that directly from the data rather than forcing a Gamma or Tweedie shape.

**Limitation:** Does not extrapolate beyond the training data range. If your XL layer extends beyond the historical maximum loss, use `EQRNModel` from `insurance_quantile.eqrn` for a GPD tail that extrapolates. The two can be spliced: FlexCode for the body of the distribution, GPD for the extreme tail.


## References

- So & Valdez (2024). *Zero-Inflated Tweedie Boosted Trees with CatBoost for Insurance Loss Analytics*. Applied Soft Computing. doi:10.1016/j.asoc.2025.113226. arXiv 2406.16206. **ASTIN Best Paper 2024.**
- Smyth & Jorgensen (2002). *Fitting Tweedie's Compound Poisson Model to Insurance Claims Data: Dispersion Modelling*. ASTIN Bulletin 32(1):143-157.
- Chevalier & Cote (2025). *From point to probabilistic gradient boosting for claim frequency and severity prediction*. European Actuarial Journal. doi:10.1007/s13385-025-00428-5.
- Izbicki, R. and Lee, A.B. (2017). Converting high-dimensional regression to high-dimensional conditional density estimation. Electronic Journal of Statistics 11(2), 2800-2831.
- Rigby & Stasinopoulos (2005). *Generalized Additive Models for Location, Scale and Shape*. Journal of the Royal Statistical Society Series C, 54(3):507-554.

---


## Databricks Notebook

A ready-to-run Databricks notebook benchmarking this library against standard approaches is available in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_distributional_demo.py).

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
| [insurance-credibility](https://github.com/burning-cost/insurance-credibility) | Bühlmann-Straub credibility weighting |

**Deployment and optimisation**

| Library | Description |
|---------|-------------|
| [insurance-optimise](https://github.com/burning-cost/insurance-optimise) | Constrained rate change optimisation with FCA PS21/5 compliance |
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


## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-distributional-glm](https://github.com/burning-cost/insurance-distributional-glm) | GAMLSS — extends distributional modelling to let sigma and shape depend on covariates, not just the mean |
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Conformal prediction — distribution-free alternative when parametric assumptions cannot be verified |
| [insurance-quantile](https://github.com/burning-cost/insurance-quantile) | Quantile GBM for tail risk — non-parametric alternative for heteroskedastic severity |

## Licence

MIT
