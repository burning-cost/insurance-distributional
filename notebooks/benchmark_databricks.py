# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmark: GammaGBM (per-risk phi) vs constant-phi Gamma GLM
# MAGIC
# MAGIC **Library:** `insurance-distributional` v0.1.2 — distributional GBM with per-risk dispersion
# MAGIC
# MAGIC **Bug context:** v0.1.1 had a phi scaling bug where pred.phi was returned ~3 orders
# MAGIC of magnitude too large (range ~939–1175 vs true 0.42–1.18). v0.1.2 fixes this.
# MAGIC This benchmark confirms the fix and produces honest performance numbers.
# MAGIC
# MAGIC **DGP:** 6,000 UK motor severity observations. phi varies 3x across portfolio
# MAGIC (vehicle age + vehicle group drive dispersion). Constant-phi GLM cannot see this.
# MAGIC
# MAGIC **Date:** 2026-03-16

# COMMAND ----------

%pip install insurance-distributional==0.1.2 catboost scikit-learn scipy numpy --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
from scipy import stats
from scipy.special import gammaln

# ---------------------------------------------------------------------------
# 1. Generate synthetic data — known heteroskedastic DGP
# ---------------------------------------------------------------------------
rng = np.random.default_rng(99)
N = 6_000

vehicle_age = rng.integers(1, 15, N).astype(float)
driver_age = rng.integers(21, 75, N).astype(float)
ncd_years = rng.integers(0, 9, N).astype(float)
vehicle_group = rng.choice([1.0, 2.0, 3.0, 4.0], size=N)

# True DGP
log_mu_true = 7.0 + 0.03 * vehicle_age - 0.02 * ncd_years + 0.08 * vehicle_group
mu_true = np.exp(log_mu_true)
phi_true = 0.30 + 0.04 * vehicle_age + 0.08 * vehicle_group  # varies per risk
shape_true = 1.0 / phi_true
scale_true = mu_true * phi_true

y = np.array([rng.gamma(shape_true[i], scale_true[i]) for i in range(N)])

X = np.column_stack([vehicle_age, driver_age, ncd_years, vehicle_group])
feature_names = ["vehicle_age", "driver_age", "ncd_years", "vehicle_group"]

# 80/20 split
n_train = int(0.8 * N)
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]
phi_test_true = phi_true[n_train:]
mu_test_true = mu_true[n_train:]

print("=" * 70)
print("BENCHMARK: GammaGBM (per-risk phi) vs constant-phi Gamma GLM")
print(f"  Training rows: {n_train}, Test rows: {N - n_train}")
print(f"  True phi range: [{phi_true.min():.3f}, {phi_true.max():.3f}]")
print(f"  True CoV range: [{np.sqrt(phi_true.min()):.3f}, {np.sqrt(phi_true.max()):.3f}]")
print("=" * 70)

# COMMAND ----------

# ---------------------------------------------------------------------------
# 2. Baseline: Gamma GLM with constant phi
# ---------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize_scalar

ols = LinearRegression()
ols.fit(X_train, np.log(y_train))
log_mu_ols_train = ols.predict(X_train)
log_mu_ols_test = ols.predict(X_test)
mu_ols_test = np.exp(log_mu_ols_test)
mu_ols_train = np.exp(log_mu_ols_train)

def gamma_mle_phi(y, mu):
    def neg_ll(log_phi):
        phi = np.exp(log_phi)
        shape = 1.0 / phi
        scale = mu * phi
        return -np.sum(stats.gamma.logpdf(y, a=shape, scale=scale))
    res = minimize_scalar(neg_ll, bounds=(-4, 3), method="bounded")
    return float(np.exp(res.x))

phi_constant = gamma_mle_phi(y_train, mu_ols_train)
print(f"Baseline (constant-phi Gamma GLM): phi = {phi_constant:.4f}, CoV = {np.sqrt(phi_constant):.4f}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# 3. GammaGBM (per-risk phi) — v0.1.2 with phi scaling fix
# ---------------------------------------------------------------------------
from insurance_distributional import GammaGBM, coverage, gamma_deviance

print("Fitting GammaGBM (distributional, per-risk phi)...")
model = GammaGBM(
    model_dispersion=True,
    catboost_params_mu={"iterations": 300, "depth": 6, "verbose": 0},
    catboost_params_phi={"iterations": 200, "depth": 5, "verbose": 0},
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(f"GammaGBM pred.phi range: [{pred.phi.min():.4f}, {pred.phi.max():.4f}]")
print(f"True phi range:          [{phi_test_true.min():.4f}, {phi_test_true.max():.4f}]")
print("(phi ranges should now be on the same scale — this confirms the v0.1.2 fix)")

# COMMAND ----------

# ---------------------------------------------------------------------------
# 4. Gamma deviance (mean prediction quality)
# ---------------------------------------------------------------------------
dev_ols = gamma_deviance(y_test, mu_ols_test)
dev_gbm = gamma_deviance(y_test, pred.mean)

print("\n" + "=" * 70)
print("TABLE 1: Gamma deviance (mean prediction quality)")
print(f"  {'Method':<30}  {'Gamma Deviance':>16}")
print("-" * 55)
print(f"  {'Constant-phi Gamma GLM':<30}  {dev_ols:>16.6f}")
print(f"  {'GammaGBM (per-risk phi)':<30}  {dev_gbm:>16.6f}")
if dev_gbm < dev_ols:
    imp = 100 * (dev_ols - dev_gbm) / dev_ols
    print(f"  GammaGBM mean improvement: {imp:.1f}%")
else:
    diff = 100 * (dev_gbm - dev_ols) / dev_ols
    print(f"  GammaGBM mean is {diff:.1f}% worse (mean not the differentiator here)")

# COMMAND ----------

# ---------------------------------------------------------------------------
# 5. Log-likelihood comparison
# ---------------------------------------------------------------------------
def gamma_loglik(y, mu, phi):
    shape = 1.0 / phi
    scale = mu * phi
    return float(np.sum(stats.gamma.logpdf(y + 1e-12, a=shape, scale=scale)))

phi_const_arr = np.full(len(y_test), phi_constant)
ll_baseline = gamma_loglik(y_test, mu_ols_test, phi_const_arr)
ll_gbm = gamma_loglik(y_test, pred.mean, pred.phi)

print("\n" + "=" * 70)
print("TABLE 2: Log-likelihood on test set (higher = better)")
print(f"  {'Method':<30}  {'Log-likelihood':>16}")
print("-" * 55)
print(f"  {'Constant-phi Gamma GLM':<30}  {ll_baseline:>16.1f}")
print(f"  {'GammaGBM (per-risk phi)':<30}  {ll_gbm:>16.1f}")
improvement = 100 * (ll_gbm - ll_baseline) / abs(ll_baseline)
print(f"  GammaGBM improvement: {improvement:+.1f}%")

# COMMAND ----------

# ---------------------------------------------------------------------------
# 6. Prediction interval coverage
# ---------------------------------------------------------------------------
def gamma_coverage_constant_phi(y, mu, phi_val, levels):
    result = {}
    for alpha in levels:
        lo_q = (1.0 - alpha) / 2.0
        hi_q = 1.0 - lo_q
        shape = 1.0 / phi_val
        scale = mu * phi_val
        lower = stats.gamma.ppf(lo_q, a=shape, scale=scale)
        upper = stats.gamma.ppf(hi_q, a=shape, scale=scale)
        result[alpha] = float(np.mean((y >= lower) & (y <= upper)))
    return result

levels = (0.80, 0.90, 0.95)
cov_baseline = gamma_coverage_constant_phi(y_test, mu_ols_test, phi_constant, levels)
cov_gbm = coverage(y_test, pred, levels=levels, n_samples=3000, seed=7)

print("\n" + "=" * 70)
print("TABLE 3: Prediction interval coverage (nominal vs empirical)")
print(f"  {'Level':>8}  {'Nominal':>10}  {'Constant-phi':>14}  {'GammaGBM':>12}  {'Best calibration':>18}")
print("-" * 70)
for alpha in levels:
    nom = alpha
    bl = cov_baseline[alpha]
    gbm = cov_gbm[alpha]
    err_bl = abs(bl - nom)
    err_gbm = abs(gbm - nom)
    winner = "GammaGBM" if err_gbm < err_bl else "constant-phi"
    print(f"  {nom:>8.0%}  {nom:>10.4f}  {bl:>14.4f}  {gbm:>12.4f}  {winner:>18}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# 7. Volatility scoring
# ---------------------------------------------------------------------------
cov_gbm_pred = pred.volatility_score()
cov_const_pred = np.sqrt(phi_constant) * np.ones(len(y_test))

r_gbm = float(np.corrcoef(pred.phi, phi_test_true)[0, 1])
r_const = float(np.corrcoef(cov_const_pred, phi_test_true)[0, 1])

print("\n" + "=" * 70)
print("TABLE 4: Volatility scoring — correlation of predicted phi with true phi")
print(f"  Constant-phi correlation with true phi : {r_const:+.4f}  (all same value)")
print(f"  GammaGBM correlation with true phi     : {r_gbm:+.4f}  (per-risk estimate)")
print("  (positive = correctly identifies high-dispersion risks; ~1.0 = perfect)")

va_test = vehicle_age[n_train:]
quartiles = np.percentile(va_test, [25, 50, 75])
print("\n  CoV (sqrt(phi)) by vehicle_age quartile in test set:")
print(f"  {'Quartile':<15}  {'True CoV':>12}  {'GBM pred CoV':>14}  {'Const CoV':>12}")
print("-" * 58)
prev = -np.inf
for q in list(quartiles) + [np.inf]:
    mask = (va_test > prev) & (va_test <= q)
    if mask.sum() > 0:
        true_cov = float(np.sqrt(phi_test_true[mask]).mean())
        gbm_cov = float(cov_gbm_pred[mask].mean())
        const_cov = float(np.sqrt(phi_constant))
        label = f"va <= {int(q)}" if q < np.inf else f"va > {int(prev)}"
        print(f"  {label:<15}  {true_cov:>12.4f}  {gbm_cov:>14.4f}  {const_cov:>12.4f}")
    prev = q

# COMMAND ----------

# ---------------------------------------------------------------------------
# 8. Safety loading spread
# ---------------------------------------------------------------------------
k = 0.5
loading_gbm = pred.mean * (1.0 + k * cov_gbm_pred)
loading_const = mu_ols_test * (1.0 + k * np.sqrt(phi_constant))

spread_gbm = float(np.std(loading_gbm / pred.mean))
spread_const = float(np.std(loading_const / mu_ols_test))

print("\n" + "=" * 70)
print("TABLE 5: Safety loading spread (k=0.5 loading factor)")
print(f"  Constant-phi: loading spread (std of loading ratio) = {spread_const:.6f}")
print(f"  GammaGBM:     loading spread (std of loading ratio) = {spread_gbm:.6f}")
if spread_const > 0:
    print(f"  GammaGBM produces {spread_gbm/spread_const:.1f}x more spread in safety loadings")

# COMMAND ----------

# ---------------------------------------------------------------------------
# 9. SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BENCHMARK SUMMARY — insurance-distributional v0.1.2")
print("=" * 70)
print(f"  Phi scale check: GBM range [{pred.phi.min():.3f}, {pred.phi.max():.3f}]")
print(f"                   True range [{phi_test_true.min():.3f}, {phi_test_true.max():.3f}]")
print(f"  Phi correlation (GBM vs true): {r_gbm:+.4f}")
print(f"  Log-likelihood improvement: {improvement:+.1f}%")
print(f"  Coverage at 80%:  baseline={cov_baseline[0.80]:.3f}, GBM={cov_gbm[0.80]:.3f}")
print(f"  Coverage at 90%:  baseline={cov_baseline[0.90]:.3f}, GBM={cov_gbm[0.90]:.3f}")
print(f"  Coverage at 95%:  baseline={cov_baseline[0.95]:.3f}, GBM={cov_gbm[0.95]:.3f}")
print(f"  Safety loading spread: {spread_gbm/max(spread_const,1e-10):.1f}x vs constant-phi")
print("=" * 70)

# COMMAND ----------

# Export results for API capture
import json as _json

_results = {
    "phi_gbm_min": float(pred.phi.min()),
    "phi_gbm_max": float(pred.phi.max()),
    "phi_true_min": float(phi_test_true.min()),
    "phi_true_max": float(phi_test_true.max()),
    "phi_correlation": float(r_gbm),
    "gamma_deviance_baseline": float(dev_ols),
    "gamma_deviance_gbm": float(dev_gbm),
    "loglik_baseline": float(ll_baseline),
    "loglik_gbm": float(ll_gbm),
    "loglik_improvement_pct": float(improvement),
    "coverage_80_baseline": float(cov_baseline[0.80]),
    "coverage_80_gbm": float(cov_gbm[0.80]),
    "coverage_90_baseline": float(cov_baseline[0.90]),
    "coverage_90_gbm": float(cov_gbm[0.90]),
    "coverage_95_baseline": float(cov_baseline[0.95]),
    "coverage_95_gbm": float(cov_gbm[0.95]),
    "loading_spread_baseline": float(spread_const),
    "loading_spread_gbm": float(spread_gbm),
    "loading_spread_ratio": float(spread_gbm / max(spread_const, 1e-10)),
}

dbutils.notebook.exit(_json.dumps(_results))
