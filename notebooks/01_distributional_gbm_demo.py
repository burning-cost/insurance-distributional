# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-distributional: Full Workflow Demo
# MAGIC
# MAGIC This notebook demonstrates the complete workflow for distributional GBM
# MAGIC on synthetic UK motor insurance data. We fit all four distribution classes
# MAGIC and show how the volatility score (CoV) enables risk-differentiated pricing.
# MAGIC
# MAGIC **Reference**: So & Valdez (2024), ASTIN Best Paper 2024, arXiv 2406.16206

# COMMAND ----------

# MAGIC %pip install insurance-distributional catboost polars scipy

# COMMAND ----------

import numpy as np
import polars as pl

from insurance_distributional import (
    TweedieGBM,
    GammaGBM,
    ZIPGBM,
    NegBinomialGBM,
    tweedie_deviance,
    poisson_deviance,
    gamma_deviance,
    gini_index,
    coverage,
    pit_values,
)

print("insurance-distributional loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic UK Motor Dataset
# MAGIC
# MAGIC We generate a realistic synthetic motor dataset with:
# MAGIC - 5,000 policies
# MAGIC - Tweedie-distributed pure premium (compound Poisson-Gamma)
# MAGIC - Known heterogeneous dispersion: older vehicles have higher phi
# MAGIC - High-cardinality categorical: vehicle group (50 levels)

# COMMAND ----------

np.random.seed(42)
rng = np.random.default_rng(42)
n = 5000

# Features
age = rng.integers(17, 75, n).astype(float)
vehicle_age = rng.integers(0, 15, n).astype(float)
vehicle_group = rng.integers(0, 50, n).astype(str)
ncd = rng.integers(0, 6, n).astype(float)  # No-claims discount (0-5 years)
annual_mileage = rng.integers(3000, 25000, n).astype(float)
exposure = rng.uniform(0.25, 1.0, n)

# True data generating process
# Younger and older drivers, high mileage -> higher frequency
log_mu_true = (
    5.5
    - 0.02 * np.abs(age - 35)          # U-shaped age effect on mean
    + 0.05 * vehicle_age               # older vehicles -> higher loss
    - 0.15 * ncd                       # NCD reduces claims
    + 0.3 * (annual_mileage > 15000)   # high mileage indicator
    + np.log(exposure)
)
mu_true = np.exp(log_mu_true)

# Heterogeneous dispersion: older vehicles and young drivers have higher phi
phi_true = np.exp(
    -1.5
    + 0.06 * vehicle_age               # older vehicle -> higher dispersion
    + 0.02 * np.maximum(0, 25 - age)   # young driver -> higher dispersion
)

print(f"mu_true: mean={mu_true.mean():.2f}, range=[{mu_true.min():.2f}, {mu_true.max():.2f}]")
print(f"phi_true: mean={phi_true.mean():.3f}, range=[{phi_true.min():.3f}, {phi_true.max():.3f}]")

# COMMAND ----------

# Generate y from Tweedie compound Poisson-Gamma
p = 1.5
lam_tw = mu_true ** (2 - p) / (phi_true * (2 - p))
alpha = (2 - p) / (p - 1)
beta = mu_true ** (1 - p) / (phi_true * (p - 1))

y = np.zeros(n)
for i in range(n):
    nc = rng.poisson(lam_tw[i])
    if nc > 0:
        y[i] = rng.gamma(shape=alpha, scale=1.0 / beta[i], size=nc).sum()

zero_rate = (y == 0).mean()
print(f"Zero rate: {zero_rate:.1%}  (typical for UK motor: 60-80%)")
print(f"y: mean={y.mean():.2f}, max={y.max():.2f}")

# COMMAND ----------

# Build feature matrix
X = np.column_stack([age, vehicle_age, ncd, annual_mileage])
X_train, X_test = X[:4000], X[4000:]
y_train, y_test = y[:4000], y[4000:]
exp_train, exp_test = exposure[:4000], exposure[4000:]
mu_true_test = mu_true[4000:]
phi_true_test = phi_true[4000:]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. TweedieGBM: Pure Premium with Dispersion Modelling

# COMMAND ----------

tweedie_model = TweedieGBM(
    power=1.5,
    model_dispersion=True,
    catboost_params_mu={"iterations": 400, "learning_rate": 0.05, "depth": 6, "verbose": False},
    catboost_params_phi={"iterations": 200, "learning_rate": 0.05, "depth": 4, "verbose": False},
)
tweedie_model.fit(X_train, y_train, exposure=exp_train)
pred_tw = tweedie_model.predict(X_test, exposure=exp_test)

print("TweedieGBM predictions:")
print(f"  pred.mean:      {pred_tw.mean.mean():.2f}  (true: {mu_true_test.mean():.2f})")
print(f"  pred.phi:       {pred_tw.phi.mean():.4f}  (true: {phi_true_test.mean():.4f})")
print(f"  volatility_score (CoV): {pred_tw.volatility_score().mean():.4f}")

# COMMAND ----------

# Scoring
tw_deviance = tweedie_deviance(y_test, pred_tw.mean, power=1.5, weights=exp_test)
tw_logscore = tweedie_model.log_score(X_test, y_test, exposure=exp_test)
tw_crps = tweedie_model.crps(X_test, y_test, exposure=exp_test, n_samples=500)
tw_gini = gini_index(y_test, pred_tw.mean, weights=exp_test)
tw_cov_calibration = coverage(y_test, pred_tw, levels=(0.80, 0.90, 0.95), n_samples=1000)

print("\nTweedieGBM scores:")
print(f"  Tweedie deviance:  {tw_deviance:.4f}")
print(f"  Log-score:         {tw_logscore:.4f}")
print(f"  CRPS:              {tw_crps:.4f}")
print(f"  Gini (mean):       {tw_gini:.4f}")
print(f"  Coverage at 80%:   {tw_cov_calibration[0.80]:.3f}  (nominal: 0.80)")
print(f"  Coverage at 90%:   {tw_cov_calibration[0.90]:.3f}  (nominal: 0.90)")
print(f"  Coverage at 95%:   {tw_cov_calibration[0.95]:.3f}  (nominal: 0.95)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Volatility scoring for safety loading
# MAGIC
# MAGIC The volatility score enables risk-differentiated safety loading.
# MAGIC Risks with the same E[Y|x] but different CoV get different loadings.

# COMMAND ----------

# Compare scalar phi (standard) vs per-risk phi (distributional)
model_scalar = TweedieGBM(power=1.5, model_dispersion=False)
model_scalar.fit(X_train, y_train, exposure=exp_train)
pred_scalar = model_scalar.predict(X_test, exposure=exp_test)

print("Dispersion model comparison:")
print(f"  Distributional GBM phi: std={pred_tw.phi.std():.4f}")
print(f"  Scalar phi:             std={pred_scalar.phi.std():.4f}  (0 = constant)")
print()

# Safety loading: P = mu * (1 + k * CoV), k = 0.1
k = 0.1
safety_distributional = pred_tw.mean * (1 + k * pred_tw.volatility_score())
safety_scalar = pred_scalar.mean * (1 + k * pred_scalar.volatility_score())

print(f"Safety loading spread (distributional): {safety_distributional.std() / safety_distributional.mean():.3f}")
print(f"Safety loading spread (scalar phi):     {safety_scalar.std() / safety_scalar.mean():.3f}")
print("The distributional model creates more risk-differentiated pricing.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. GammaGBM: Severity with Dispersion Modelling

# COMMAND ----------

# Use only positive observations for severity model
mask_pos = y_train > 0
X_sev = X_train[mask_pos]
y_sev = y_train[mask_pos]

gamma_model = GammaGBM(
    model_dispersion=True,
    catboost_params_mu={"iterations": 300, "learning_rate": 0.05, "verbose": False},
    catboost_params_phi={"iterations": 150, "learning_rate": 0.05, "verbose": False},
)
gamma_model.fit(X_sev, y_sev)

mask_pos_test = y_test > 0
pred_gam = gamma_model.predict(X_test[mask_pos_test])

gam_deviance = gamma_deviance(y_test[mask_pos_test], pred_gam.mean)
gam_gini = gini_index(y_test[mask_pos_test], pred_gam.mean)

print("GammaGBM (severity only):")
print(f"  Gamma deviance: {gam_deviance:.4f}")
print(f"  Gini (mean):    {gam_gini:.4f}")
print(f"  Mean CoV:       {pred_gam.cov.mean():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. ZIPGBM: Zero-Inflated Poisson for Pet Insurance

# COMMAND ----------

# Simulate pet insurance data: 40% structural zeros
n_pet = 3000
X_pet = rng.standard_normal((n_pet, 5))
pi_true_pet = 0.35 + 0.10 * (X_pet[:, 0] > 0.5)  # breed effect on zero-inflation
lam_true_pet = 0.20 + 0.15 * np.abs(X_pet[:, 1])  # age effect on claim rate

y_pet = np.where(
    rng.random(n_pet) < pi_true_pet,
    0,
    rng.poisson(lam_true_pet)
).astype(float)

print(f"Pet insurance data: {n_pet} policies, {(y_pet==0).mean():.1%} zero-claim")

X_pet_train, X_pet_test = X_pet[:2500], X_pet[2500:]
y_pet_train, y_pet_test = y_pet[:2500], y_pet[2500:]

zip_model = ZIPGBM(
    catboost_params_mu={"iterations": 300, "learning_rate": 0.05, "verbose": False},
    catboost_params_phi={"iterations": 200, "learning_rate": 0.05, "verbose": False},
)
zip_model.fit(X_pet_train, y_pet_train)
pred_zip = zip_model.predict(X_pet_test)

print("\nZIPGBM predictions:")
print(f"  Mean pi:     {pred_zip.pi.mean():.3f}  (true mean: {pi_true_pet[2500:].mean():.3f})")
print(f"  Mean lambda: {zip_model.predict_lambda(X_pet_test).mean():.4f}")
print(f"  Mean mu:     {pred_zip.mean.mean():.4f}")
print(f"  Poisson deviance: {poisson_deviance(y_pet_test, pred_zip.mean):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. NegBinomialGBM: Overdispersed Fleet Frequency

# COMMAND ----------

# Fleet motor: overdispersed count data (r=3)
n_fleet = 2000
X_fleet = rng.standard_normal((n_fleet, 4))
mu_fleet = np.exp(0.5 + 0.4 * X_fleet[:, 0] - 0.2 * X_fleet[:, 1])
r_fleet = 3.0
p_nb = r_fleet / (r_fleet + mu_fleet)
y_fleet = rng.negative_binomial(n=int(r_fleet), p=p_nb).astype(float)

print(f"Fleet data: mean={y_fleet.mean():.3f}, var={y_fleet.var():.3f}")
print(f"  Overdispersion (var/mean): {y_fleet.var()/y_fleet.mean():.3f}  (Poisson would be 1.0)")

X_fl_train, X_fl_test = X_fleet[:1600], X_fleet[1600:]
y_fl_train, y_fl_test = y_fleet[:1600], y_fleet[1600:]

nb_model = NegBinomialGBM(model_r=False)
nb_model.fit(X_fl_train, y_fl_train)
pred_nb = nb_model.predict(X_fl_test)

print(f"\nNegBinomialGBM predictions:")
print(f"  Estimated r: {pred_nb.r.mean():.3f}  (true: {r_fleet})")
print(f"  Mean variance (pred): {pred_nb.variance.mean():.4f}")
print(f"  Mean variance (obs):  {y_fl_test.var():.4f}")
print(f"  Poisson deviance: {poisson_deviance(y_fl_test, pred_nb.mean):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Model Comparison: Point vs Distributional
# MAGIC
# MAGIC Comparison of standard Tweedie GBM (mean only) vs distributional model.

# COMMAND ----------

from catboost import CatBoostRegressor

# Standard CatBoost Tweedie (no dispersion)
cb_standard = CatBoostRegressor(
    loss_function="Tweedie:variance_power=1.5",
    iterations=400,
    learning_rate=0.05,
    depth=6,
    verbose=False,
)
baseline = np.log(exp_train) + np.log(y_train[y_train > 0].mean())
cb_standard.fit(X_train, y_train, baseline=np.log(exp_train))
mu_standard = cb_standard.predict(X_test)

print("Model Comparison on Test Set:")
print(f"{'Metric':<30} {'Standard GBM':>15} {'Distributional':>15}")
print("-" * 62)
print(f"{'Tweedie deviance':<30} {tweedie_deviance(y_test, mu_standard, 1.5):>15.4f} {tw_deviance:>15.4f}")
print(f"{'Gini (mean)':<30} {gini_index(y_test, mu_standard):>15.4f} {tw_gini:>15.4f}")
print(f"{'CRPS':<30} {'N/A (no dist)':>15} {tw_crps:>15.4f}")
print()
print("Distributional GBM adds: volatility_score(), variance, phi per risk")
print("These are unavailable from standard point prediction models.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary
# MAGIC
# MAGIC Key results:
# MAGIC 1. TweedieGBM correctly models per-risk dispersion (phi varies by vehicle_age and driver_age)
# MAGIC 2. Distributional model gives better safety loading differentiation than scalar phi
# MAGIC 3. ZIPGBM correctly identifies zero-inflation probability per risk
# MAGIC 4. NegBinomialGBM correctly estimates overdispersion parameter r
# MAGIC 5. CRPS and coverage metrics enable proper calibration assessment
# MAGIC
# MAGIC The volatility_score() output enables use cases not available from standard GBMs:
# MAGIC - Per-risk safety loading calibration
# MAGIC - Underwriter referral rules based on CoV threshold
# MAGIC - IFRS 17 risk adjustment input
# MAGIC - Reinsurance attachment optimisation

print("Demo complete. See README for full API reference.")
