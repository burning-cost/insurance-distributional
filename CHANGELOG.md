# Changelog

## [0.3.0] - 2026-03-28

### Added
- ZeroInflatedTweedieGBM: first open-source implementation of So & Valdez (2024) ASTIN Best Paper Scenario 2
- Two-stage ZI-Tweedie: CatBoost classifier for P(Y=0) + CatBoost Tweedie regressor for E[Y|Y>0]
- predict_components() returns zero_prob, severity_mean, combined_mean separately
- 63-test suite covering fit/predict, components, exposure, sample_weight, pickle, edge cases
- Databricks demo notebook: 02_zero_inflated_tweedie_demo.py

## [0.2.1] - 2026-03-23

### Fixed
- Bumped numpy minimum version from >=1.24 to >=1.25 to ensure compatibility with scipy's use of numpy.exceptions (added in numpy 1.25)

## v0.2.0 (2026-03-21)
- Add cross-links to related libraries in README
- docs: replace pip install with uv add in README
- Add community CTA to README
- Fix three performance and correctness bugs in prediction.py / flexcode.py
- docs: add FlexCodeDensity README section; restore full remote source
- Add FlexCodeDensity: nonparametric conditional density estimation (v0.2.0)
- Fix: use np.argmax for CDF quantile inversion to support numpy<2.0
- Fix: add numpy<2.0 trapezoid compatibility shim to flexcode.py and scoring.py
- Fix: vectorise FlexCodePrediction methods and add log_epsilon sub-unit guard
- feat: add FlexCodeDensity for nonparametric conditional density estimation
- Add pdoc API documentation with GitHub Pages
- Add Google Colab quickstart notebook and Open-in-Colab badge
- Add quickstart notebook
- fix: README technical errors from quality review
- Add MIT license
- Quality fixes: DGP labelling, Tweedie p=1.5 note, NegBinom predict, version
- Add PyPI classifiers for financial/insurance audience
- Benchmark v0.1.3: cross-fitting fix restores coverage calibration
- fix: correct phi absolute scale via cross-fitting (v0.1.3)
- benchmarks: run Databricks benchmark for v0.1.2, update performance section
- fix: phi predictions on correct scale — apply CatBoost baseline at inference (P0-4)
- Update Performance section with post-review benchmark results
- Fix P0/P1 bugs: exposure prediction, ZIPGBM lambda init, NegBinom gradient
- Add benchmark: GammaGBM (per-risk phi) vs constant-phi Gamma GLM
- docs: add Databricks notebook link
