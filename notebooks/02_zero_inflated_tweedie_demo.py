# Databricks notebook source
# MAGIC %md
# MAGIC # ZeroInflatedTweedieGBM: So & Valdez (2024) Scenario 2 Demo
# MAGIC
# MAGIC This notebook demonstrates the first open-source implementation of the
# MAGIC So & Valdez (2024) ASTIN Best Paper "Scenario 2" — ZI-Tweedie with CatBoost.
# MAGIC
# MAGIC ## The Problem
# MAGIC
# MAGIC Standard Tweedie GLM handles zero-heavy portfolios by letting the compound
# MAGIC Poisson term account for zero probability. This works when claim frequency
# MAGIC and severity have the same covariate structure. In practice they often don't:
# MAGIC - UK motor: frequency driven by driver profile, severity by vehicle value
# MAGIC - Contents: frequency driven by area/security, severity by sum insured
# MAGIC - Breakdown: frequency by vehicle age, severity roughly constant
# MAGIC
# MAGIC ZI-Tweedie separates the two stages explicitly.
# MAGIC
# MAGIC ## Reference
# MAGIC So, B. & Valdez, E.A. (2024). Boosted trees for zero-inflated counts with
# MAGIC an offset for insurance ratemaking. arXiv 2406.16206. ASTIN Best Paper 2024.

# COMMAND ----------

# MAGIC %pip install insurance-distributional catboost polars scipy --quiet

# COMMAND ----------

import numpy as np
import polars as pl

from insurance_distributional import (
    ZeroInflatedTweedieGBM,
    TweedieGBM,
    tweedie_deviance,
    gini_index,
)

print("insurance-distributional loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic UK Contents Portfolio
# MAGIC
# MAGIC We generate a realistic contents insurance dataset:
# MAGIC - 10,000 policies with ~88% zero claims
# MAGIC - Claim probability (Stage 1) driven by: area deprivation score, property type, security features
# MAGIC - Claim severity (Stage 2) driven by: sum insured, property age, contents value estimate
# MAGIC - These covariate sets deliberately overlap but differ in importance

# COMMAND ----------

rng = np.random.default_rng(42)
n = 10_000

# Features
area_deprivation = rng.integers(1, 11, n).astype(float)      # 1=affluent, 10=deprived
property_type = rng.integers(0, 4, n).astype(float)           # 0=detached, 1=semi, 2=terrace, 3=flat
security_score = rng.integers(0, 5, n).astype(float)          # 0=none, 4=excellent
sum_insured = rng.integers(15000, 80000, n).astype(float)     # £15k–£80k
property_age = rng.integers(1900, 2020, n).astype(float)      # year built
contents_value_est = sum_insured * rng.uniform(0.3, 0.8, n)  # fraction of SI
exposure = rng.uniform(0.5, 1.0, n)                           # fraction of year

# True zero probability: high deprivation -> more claims; better security -> fewer
logit_pi = (
    1.5                                          # base
    - 0.15 * area_deprivation                   # deprived -> lower zero_prob
    + 0.25 * security_score                     # security -> higher zero_prob
    + 0.3 * (property_type == 3)                # flats slightly more claims
)
pi_true = 1.0 / (1.0 + np.exp(-logit_pi))       # sigmoid -> P(Y=0)
pi_true = np.clip(pi_true, 0.5, 0.97)

# True severity: sum insured driven, with property age effect
log_mu_sev = (
    np.log(sum_insured * 0.08)                  # ~8% of SI as base severity
    + 0.02 * (2020 - property_age) / 10         # older property -> slightly higher
    + np.log(exposure)                           # exposure offset
)
mu_sev_true = np.exp(log_mu_sev)

# Generate observed losses
y = np.zeros(n)
for i in range(n):
    if rng.random() >= pi_true[i]:  # not a structural zero
        # Gamma draw as Tweedie approximation
        y[i] = rng.gamma(shape=2.0, scale=mu_sev_true[i] / 2.0)

zero_rate = (y == 0).mean()
print(f"n={n}, zero_rate={zero_rate:.1%}")
print(f"True pi: mean={pi_true.mean():.3f}, range=[{pi_true.min():.3f}, {pi_true.max():.3f}]")
print(f"Mean non-zero loss: £{y[y>0].mean():.0f}")

# Build feature matrix
X = np.column_stack([
    area_deprivation,
    property_type,
    security_score,
    sum_insured,
    property_age,
    contents_value_est,
])

# Train/test split
train_idx = rng.choice(n, size=int(0.8 * n), replace=False)
test_idx = np.setdiff1d(np.arange(n), train_idx)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
exp_train, exp_test = exposure[train_idx], exposure[test_idx]

print(f"\nTrain: n={len(y_train)}, Test: n={len(y_test)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fit ZeroInflatedTweedieGBM

# COMMAND ----------

zi_model = ZeroInflatedTweedieGBM(
    power=1.5,
    random_state=42,
    catboost_params_zero={"iterations": 500, "learning_rate": 0.05, "depth": 6},
    catboost_params_severity={"iterations": 500, "learning_rate": 0.05, "depth": 6},
)
zi_model.fit(X_train, y_train, exposure=exp_train)
print(zi_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Inspect Components
# MAGIC
# MAGIC The key advantage of ZI-Tweedie: you can see whether a high predicted
# MAGIC loss comes from a high claim probability or a high expected severity if claimed.

# COMMAND ----------

components = zi_model.predict_components(X_test, exposure=exp_test)
pi_hat = components["zero_prob"]
sev_hat = components["severity_mean"]
mu_hat = components["combined_mean"]

print("=== Component summary (test set) ===")
print(f"P(Y=0) predicted:   mean={pi_hat.mean():.3f}, range=[{pi_hat.min():.3f}, {pi_hat.max():.3f}]")
print(f"E[Y|Y>0] predicted: mean=£{sev_hat.mean():.0f}, range=[£{sev_hat.min():.0f}, £{sev_hat.max():.0f}]")
print(f"E[Y] combined:      mean=£{mu_hat.mean():.0f}")
print(f"Actual mean loss:   £{y_test.mean():.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Benchmark: ZI-Tweedie vs Standard Tweedie
# MAGIC
# MAGIC Fit a standard TweedieGBM on the same data for comparison.

# COMMAND ----------

std_model = TweedieGBM(power=1.5, model_dispersion=False, random_state=42)
std_model.fit(X_train, y_train, exposure=exp_train)
pred_std = std_model.predict(X_test, exposure=exp_test)
mu_std = pred_std.mean

print("=== Tweedie deviance (lower = better) ===")
dev_zi = tweedie_deviance(y_test, mu_hat, power=1.5, weights=exp_test)
dev_std = tweedie_deviance(y_test, mu_std, power=1.5, weights=exp_test)
print(f"ZI-Tweedie:       {dev_zi:.4f}")
print(f"Standard Tweedie: {dev_std:.4f}")
improvement = (dev_std - dev_zi) / dev_std * 100
print(f"Improvement:      {improvement:+.1f}%")

print("\n=== Gini index (higher = better discrimination) ===")
gini_zi = gini_index(y_test, mu_hat, weights=exp_test)
gini_std = gini_index(y_test, mu_std, weights=exp_test)
print(f"ZI-Tweedie:       {gini_zi:.4f}")
print(f"Standard Tweedie: {gini_std:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Component-Level Business Insight
# MAGIC
# MAGIC The real power of ZI-Tweedie is interpretability at the component level.
# MAGIC Let's see which factors drive zero probability vs severity.

# COMMAND ----------

# Sort by area deprivation — should drive zero_prob strongly
area_dep_test = area_deprivation[test_idx]
sec_test = security_score[test_idx]
si_test = sum_insured[test_idx]

# Average by deprivation decile
print("=== P(Claim) by area deprivation (1=affluent, 10=deprived) ===")
for dep in range(1, 11):
    mask = area_dep_test == dep
    if mask.sum() > 5:
        pi_grp = 1.0 - pi_hat[mask].mean()  # P(claim) = 1 - P(zero)
        n_grp = mask.sum()
        print(f"  Deprivation {dep:2d}: P(claim)={pi_grp:.3f}  (n={n_grp})")

print("\n=== E[Y|Y>0] by sum insured quartile ===")
si_quartiles = np.percentile(si_test, [0, 25, 50, 75, 100])
for q in range(4):
    mask = (si_test >= si_quartiles[q]) & (si_test < si_quartiles[q + 1])
    if mask.sum() > 5:
        sev_grp = sev_hat[mask].mean()
        print(f"  SI quartile {q+1} (£{si_quartiles[q]:.0f}–£{si_quartiles[q+1]:.0f}): "
              f"E[sev|claim]=£{sev_grp:.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Scoring

# COMMAND ----------

score = zi_model.score(X_test, y_test, exposure=exp_test)
log_score = zi_model.log_score(X_test, y_test, exposure=exp_test)

print(f"Tweedie deviance score: {score:.4f}")
print(f"Log score (neg ll):     {log_score:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Serialisation
# MAGIC
# MAGIC Models serialise cleanly with pickle — important for production deployment.

# COMMAND ----------

import pickle

serialised = pickle.dumps(zi_model)
loaded = pickle.loads(serialised)

mu_orig = zi_model.predict(X_test, exposure=exp_test)
mu_loaded = loaded.predict(X_test, exposure=exp_test)

assert np.allclose(mu_orig, mu_loaded, rtol=1e-5), "Pickle roundtrip failed!"
print(f"Pickle roundtrip OK. Serialised size: {len(serialised) / 1024:.0f} KB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ZeroInflatedTweedieGBM implements So & Valdez (2024) Scenario 2 — the only
# MAGIC open-source CatBoost implementation of this approach. Key properties:
# MAGIC
# MAGIC - **Two-stage**: separate CatBoost models for zero probability and severity
# MAGIC - **Exposure-aware**: log(exposure) offset in the severity model
# MAGIC - **Interpretable**: `predict_components()` gives zero_prob and severity_mean separately
# MAGIC - **Standard API**: fit/predict/score like sklearn
# MAGIC - **Serialisable**: pickle roundtrip for deployment
# MAGIC
# MAGIC Best suited for: UK motor third-party property, contents, breakdown cover,
# MAGIC any line where 85%+ of policies have zero claims in the period.

print("Demo complete.")
