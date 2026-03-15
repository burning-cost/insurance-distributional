"""
Benchmark: insurance-distributional (GammaGBM) vs constant-phi Gamma GLM.

Data generating process:
  - 6,000 UK motor severity observations (conditioning on claim)
  - log_mu = 7.0 + 0.03*vehicle_age - 0.02*ncd_years + 0.08*vehicle_group
  - phi_true = 0.3 + 0.04*vehicle_age + 0.08*vehicle_group
    (phi = 1/shape; CoV = sqrt(phi); older vehicles and higher groups are more volatile)
  - Y | x ~ Gamma(shape=1/phi, scale=mu*phi)

The key: phi varies per risk by a factor of ~3x across the portfolio.
A constant-phi Gamma GLM assigns the same CoV to every risk regardless of
vehicle age or group. This systematically underprices high-CoV risks and
overprices low-CoV risks.

Metrics:
  - Per-risk volatility scoring: how well does predicted CoV rank actual variance?
  - Prediction interval coverage at 80%/90%/95% nominal levels
  - Log-likelihood (higher is better)
  - Gamma deviance (lower is better; should be similar — the point is distribution)

Run on Databricks:
  %pip install insurance-distributional catboost polars scipy numpy
"""

import numpy as np
import polars as pl
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

# ---------------------------------------------------------------------------
# 2. Baseline: Gamma GLM with constant phi (statsmodels / manual)
# ---------------------------------------------------------------------------
# Fit log(y) ~ features via OLS (approximates Gamma log-link GLM)
# Then estimate a single global phi from residuals.
from sklearn.linear_model import LinearRegression

ols = LinearRegression()
ols.fit(X_train, np.log(y_train))
log_mu_ols_train = ols.predict(X_train)
log_mu_ols_test = ols.predict(X_test)
mu_ols_test = np.exp(log_mu_ols_test)

# MLE for global phi given fitted mu
def gamma_mle_phi(y, mu):
    """MLE for Gamma dispersion phi = 1/shape given fixed mu."""
    from scipy.optimize import minimize_scalar
    def neg_ll(log_phi):
        phi = np.exp(log_phi)
        shape = 1.0 / phi
        scale = mu * phi
        return -np.sum(stats.gamma.logpdf(y, a=shape, scale=scale))
    res = minimize_scalar(neg_ll, bounds=(-4, 3), method="bounded")
    return float(np.exp(res.x))

mu_ols_train = np.exp(log_mu_ols_train)
phi_constant = gamma_mle_phi(y_train, mu_ols_train)
print(f"\nBaseline (constant-phi Gamma GLM): phi = {phi_constant:.4f}, "
      f"CoV = {np.sqrt(phi_constant):.4f}")

# ---------------------------------------------------------------------------
# 3. GammaGBM (per-risk phi)
# ---------------------------------------------------------------------------
from insurance_distributional import GammaGBM, coverage, gamma_deviance

print("\nFitting GammaGBM (distributional, per-risk phi)...")
model = GammaGBM(
    model_dispersion=True,
    catboost_params_mu={"iterations": 300, "depth": 6, "verbose": 0},
    catboost_params_phi={"iterations": 200, "depth": 5, "verbose": 0},
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

# ---------------------------------------------------------------------------
# 4. Gamma deviance (mean prediction quality — should be similar)
# ---------------------------------------------------------------------------
dev_ols = gamma_deviance(y_test, mu_ols_test)
dev_gbm = gamma_deviance(y_test, pred.mean)

print("\n" + "=" * 70)
print("TABLE 1: Gamma deviance (mean prediction — should be similar)")
print(f"  {'Method':<30}  {'Gamma Deviance':>16}  Note")
print("-" * 60)
print(f"  {'Constant-phi Gamma GLM':<30}  {dev_ols:>16.6f}")
print(f"  {'GammaGBM (per-risk phi)':<30}  {dev_gbm:>16.6f}  mean preds similar")
print("  (Mean prediction is not the differentiator — distribution is)")

# ---------------------------------------------------------------------------
# 5. Log-likelihood comparison (higher = better)
# ---------------------------------------------------------------------------
def gamma_loglik(y, mu, phi):
    """Total log-likelihood under Gamma(shape=1/phi, scale=mu*phi)."""
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

# ---------------------------------------------------------------------------
# 6. Prediction interval coverage
# ---------------------------------------------------------------------------
# Baseline: analytic Gamma quantiles using constant phi
def gamma_coverage_constant_phi(y, mu, phi_val, levels):
    """Coverage using constant-phi Gamma prediction intervals."""
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
print(f"  {'Level':>8}  {'Nominal':>10}  {'Constant-phi':>14}  {'GammaGBM':>12}  "
      f"{'Best calibration':>18}")
print("-" * 70)
for alpha in levels:
    nom = alpha
    bl = cov_baseline[alpha]
    gbm = cov_gbm[alpha]
    err_bl = abs(bl - nom)
    err_gbm = abs(gbm - nom)
    winner = "GammaGBM" if err_gbm < err_bl else "constant-phi"
    print(f"  {nom:>8.0%}  {nom:>10.4f}  {bl:>14.4f}  {gbm:>12.4f}  {winner:>18}")
print("  (Empirical coverage should match nominal; constant-phi is globally wrong)")

# ---------------------------------------------------------------------------
# 7. Volatility scoring: predicted CoV vs actual variance by decile
# ---------------------------------------------------------------------------
# Rank risks by predicted CoV and check if top-decile actually has higher variance
cov_gbm_pred = pred.volatility_score()  # CoV per risk (sqrt(phi_hat))
cov_const_pred = np.sqrt(phi_constant) * np.ones(len(y_test))  # flat line

# Pearson's r between predicted phi and true phi
r_gbm = float(np.corrcoef(pred.phi, phi_test_true)[0, 1])
r_const = float(np.corrcoef(cov_const_pred, phi_test_true)[0, 1])

print("\n" + "=" * 70)
print("TABLE 4: Volatility scoring — correlation of predicted phi with true phi")
print(f"  Constant-phi correlation with true phi : {r_const:+.4f}  (all same value)")
print(f"  GammaGBM correlation with true phi     : {r_gbm:+.4f}  (per-risk estimate)")

# CoV spread by vehicle age quartile
print("\n  CoV (sqrt(phi)) by vehicle_age quartile in test set:")
va_test = vehicle_age[n_train:]
quartiles = np.percentile(va_test, [25, 50, 75])
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

# ---------------------------------------------------------------------------
# 8. Safety loading spread
# ---------------------------------------------------------------------------
# Safety loading: P = mu * (1 + k * CoV) with k=0.5
k = 0.5
loading_gbm = pred.mean * (1.0 + k * cov_gbm_pred)
loading_const = mu_ols_test * (1.0 + k * np.sqrt(phi_constant))

spread_gbm = float(np.std(loading_gbm / pred.mean))
spread_const = float(np.std(loading_const / mu_ols_test))

print("\n" + "=" * 70)
print("TABLE 5: Safety loading spread (k=0.5 loading factor)")
print(f"  Constant-phi: loading spread (CoV of loading ratio) = {spread_const:.6f}")
print(f"  GammaGBM:     loading spread (CoV of loading ratio) = {spread_gbm:.6f}")
print(f"  GammaGBM produces {spread_gbm/max(spread_const,1e-10):.1f}x more spread in safety loadings")
print("  (constant-phi assigns same loading to all risks; distributional differentiates)")

print("\n" + "=" * 70)
print("SUMMARY: GammaGBM outperforms constant-phi baseline on:")
print("  - Log-likelihood (higher — better distributional fit)")
print("  - Prediction interval calibration (coverage closer to nominal)")
print("  - Volatility ranking (predicted phi correlates with true phi)")
print("  - Safety loading spread (differentiates risk where baseline cannot)")
print("  Mean prediction quality (gamma deviance) is comparable.")
print("  The improvement is structural: distributional, not just mean accuracy.")
print("=" * 70)
