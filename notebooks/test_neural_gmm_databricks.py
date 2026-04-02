# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # NeuralGaussianMixture — Test Runner
# MAGIC
# MAGIC Runs `tests/test_neural_gmm.py` on Databricks serverless compute where torch is available.

# COMMAND ----------

# Install package in editable mode from the uploaded workspace files
import subprocess, sys

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet",
     "catboost", "polars", "scipy",
     "torch", "--index-url", "https://download.pytorch.org/whl/cpu"],
    capture_output=True, text=True
)
print(result.stdout[-2000:] if result.stdout else "")
print(result.stderr[-2000:] if result.stderr else "")

# COMMAND ----------

import subprocess, sys

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet",
     "-e", "/Workspace/insurance-distributional/"],
    capture_output=True, text=True
)
print(result.stdout[-2000:] if result.stdout else "")
print(result.stderr[-2000:] if result.stderr else "")

# COMMAND ----------

import subprocess, sys

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-distributional/tests/test_neural_gmm.py",
     "-v", "--tb=short", "--no-header"],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-distributional"
)
print(result.stdout)
print(result.stderr[-3000:] if result.stderr else "")
print("Return code:", result.returncode)
