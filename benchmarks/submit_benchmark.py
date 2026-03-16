"""
Submit insurance-distributional v0.1.3 benchmark to Databricks.

Embeds benchmark code as base64 in the notebook, decodes at runtime,
runs as subprocess, and returns output via dbutils.notebook.exit().

Usage:
    python submit_benchmark.py
"""

import os
import sys
import time
import base64
import requests

env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

DATABRICKS_HOST = os.environ["DATABRICKS_HOST"].rstrip("/")
DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]
HEADERS = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}

bench_path = os.path.join(os.path.dirname(__file__), "benchmark.py")
with open(bench_path, "rb") as f:
    bench_bytes = f.read()

bench_b64 = base64.b64encode(bench_bytes).decode()

# Build notebook without f-string to avoid escaping issues
# Use string concatenation for the parts that need the variable
launcher_lines = [
    "# Databricks notebook source",
    "# MAGIC %pip install insurance-distributional==0.1.3 catboost polars scipy numpy scikit-learn",
    "",
    "# COMMAND ----------",
    "",
    "import subprocess, sys, base64, os",
    "",
    "# Benchmark code embedded as base64",
    "_bench_b64 = '" + bench_b64 + "'",
    "_bench_code = base64.b64decode(_bench_b64)",
    "",
    "bench_path = '/tmp/benchmark_v013.py'",
    "with open(bench_path, 'wb') as f:",
    "    f.write(_bench_code)",
    "",
    "result = subprocess.run(",
    "    [sys.executable, bench_path],",
    "    capture_output=True,",
    "    text=True,",
    "    timeout=300,",
    ")",
    "",
    "output = result.stdout",
    "if result.returncode != 0:",
    "    output = output + chr(10) + '--- STDERR (rc=' + str(result.returncode) + ') ---' + chr(10) + result.stderr",
    "",
    "print(output)",
    "dbutils.notebook.exit(output[:65000])",
]
launcher = "\n".join(launcher_lines)

workspace_path = "/Workspace/Users/pricing.frontier@gmail.com/benchmark_distributional_v013_b64"
print(f"Uploading launcher notebook to {workspace_path} ...")

resp = requests.post(
    f"{DATABRICKS_HOST}/api/2.0/workspace/import",
    headers=HEADERS,
    json={
        "path": workspace_path,
        "format": "SOURCE",
        "language": "PYTHON",
        "content": base64.b64encode(launcher.encode()).decode(),
        "overwrite": True,
    },
)
if not resp.ok:
    print(f"Import failed: {resp.status_code} {resp.text}")
    sys.exit(1)
print("Upload done.")

print("Submitting run...")
run_resp = requests.post(
    f"{DATABRICKS_HOST}/api/2.1/jobs/runs/submit",
    headers=HEADERS,
    json={
        "run_name": "benchmark-distributional-v013-b64",
        "tasks": [{
            "task_key": "benchmark",
            "notebook_task": {
                "notebook_path": workspace_path,
                "base_parameters": {},
            },
        }],
    },
)
if not run_resp.ok:
    print(f"Submit failed: {run_resp.status_code} {run_resp.text}")
    sys.exit(1)

run_id = run_resp.json()["run_id"]
print(f"Parent run ID: {run_id}")

print("Polling (benchmark takes ~5-8 minutes)...")
while True:
    d = requests.get(
        f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get",
        headers=HEADERS,
        params={"run_id": run_id},
        timeout=30,
    ).json()
    lc = d.get("state", {}).get("life_cycle_state", "UNKNOWN")
    rs = d.get("state", {}).get("result_state")
    msg = d.get("state", {}).get("state_message", "")
    print(f"  {lc} / {rs}" + (f" — {msg}" if msg else ""))
    if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break
    time.sleep(30)

tasks = d.get("tasks", [])
task_run_id = tasks[0]["run_id"] if tasks else run_id

out = requests.get(
    f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get-output",
    headers=HEADERS,
    params={"run_id": task_run_id},
    timeout=30,
).json()

result_state = d.get("state", {}).get("result_state", "UNKNOWN")
result_text = out.get("notebook_output", {}).get("result", "")
error = out.get("error", "")
trace = out.get("error_trace", "")

if result_state == "SUCCESS":
    print("\n" + "=" * 70)
    print("BENCHMARK OUTPUT:")
    print("=" * 70)
    print(result_text)
    out_path = os.path.join(os.path.dirname(__file__), "benchmark_output_v013.txt")
    with open(out_path, "w") as f:
        f.write(result_text)
    print(f"\nSaved to {out_path}")
else:
    print(f"\nFAILED: {result_state}")
    if error:
        print(f"Error: {error}")
    if result_text:
        print(f"Partial output:\n{result_text[:3000]}")
    if trace:
        print(f"Trace:\n{trace[:5000]}")
    sys.exit(1)
