"""
Run the insurance-distributional test suite on Databricks serverless compute.

Embeds the project zip as base64 inside the notebook source, so no
external storage (DBFS/Volumes) is required.
"""
import os
import sys
import time
import base64
import tempfile
import zipfile
from pathlib import Path

# Load Databricks credentials
env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k] = v

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace, jobs

w = WorkspaceClient()

WORKSPACE_PATH = "/Workspace/insurance-distributional-tests"
NOTEBOOK_PATH = f"{WORKSPACE_PATH}/run_pytest"

PROJECT_ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Step 1: Build project zip and encode it
# ---------------------------------------------------------------------------
print("Building project zip...")

zip_path = Path(tempfile.mktemp(suffix=".zip"))
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for pattern in ["src/**/*.py", "tests/**/*.py", "pyproject.toml", "README.md"]:
        for f in PROJECT_ROOT.glob(pattern):
            arcname = str(f.relative_to(PROJECT_ROOT))
            zf.write(f, arcname)

zip_bytes = zip_path.read_bytes()
zip_b64 = base64.b64encode(zip_bytes).decode()
print(f"  Zip size: {len(zip_bytes) / 1024:.1f} KB  b64 length: {len(zip_b64)}")

# ---------------------------------------------------------------------------
# Step 2: Create notebook with embedded zip
# ---------------------------------------------------------------------------
print(f"Creating notebook at {NOTEBOOK_PATH}...")

try:
    w.workspace.mkdirs(path=WORKSPACE_PATH)
except Exception:
    pass

# NOTE: All curly braces in Python code inside this f-string must be doubled.
# The zip_b64 variable is interpolated directly.
NOTEBOOK_CONTENT = (
    "# Databricks notebook source\n"
    "# COMMAND ----------\n"
    "# MAGIC %pip install catboost polars scipy numpy pytest --quiet\n"
    "\n"
    "# COMMAND ----------\n"
    "import base64, zipfile, subprocess, sys, tempfile, os\n"
    "from pathlib import Path\n"
    "\n"
    f'PROJECT_ZIP_B64 = """{zip_b64}"""\n'
    "\n"
    "extract_dir = Path(tempfile.mkdtemp(prefix='ins-dist-'))\n"
    "\n"
    "zip_bytes = base64.b64decode(PROJECT_ZIP_B64)\n"
    "fd, zip_tmp = tempfile.mkstemp(suffix='.zip')\n"
    "os.close(fd)\n"
    "Path(zip_tmp).write_bytes(zip_bytes)\n"
    "with zipfile.ZipFile(zip_tmp, 'r') as zf:\n"
    "    zf.extractall(str(extract_dir))\n"
    "os.unlink(zip_tmp)\n"
    "print('Extracted to:', extract_dir)\n"
    "\n"
    "# COMMAND ----------\n"
    "result = subprocess.run(\n"
    "    [sys.executable, '-m', 'pip', 'install', '-e', str(extract_dir), '--quiet'],\n"
    "    capture_output=True, text=True\n"
    ")\n"
    "if result.returncode != 0:\n"
    "    msg = 'pip install failed:\\n' + result.stderr[-2000:]\n"
    "    dbutils.notebook.exit(msg)\n"
    "    raise RuntimeError(msg)\n"
    "print('Package installed OK')\n"
    "\n"
    "# COMMAND ----------\n"
    "result = subprocess.run(\n"
    "    [sys.executable, '-m', 'pytest', str(extract_dir / 'tests'), '-v', '--tb=short', '-x'],\n"
    "    capture_output=True, text=True, cwd=str(extract_dir)\n"
    ")\n"
    "combined = result.stdout\n"
    "if result.stderr:\n"
    "    combined += '\\nSTDERR:\\n' + result.stderr[-500:]\n"
    "status = 'PASSED' if result.returncode == 0 else f'FAILED (rc={result.returncode})'\n"
    "combined += f'\\n\\n=== pytest {status} ==='\n"
    "dbutils.notebook.exit(combined[-8000:])\n"
    "if result.returncode != 0:\n"
    "    raise RuntimeError(f'pytest {status}')\n"
)

nb_b64 = base64.b64encode(NOTEBOOK_CONTENT.encode()).decode()
w.workspace.import_(
    path=NOTEBOOK_PATH,
    format=workspace.ImportFormat.SOURCE,
    language=workspace.Language.PYTHON,
    content=nb_b64,
    overwrite=True,
)
print("  Notebook created.")

# ---------------------------------------------------------------------------
# Step 3: Submit as a serverless one-shot job run (no new_cluster)
# ---------------------------------------------------------------------------
print("Submitting serverless job run...")

run = w.jobs.submit(
    run_name="insurance-distributional-pytest",
    tasks=[
        jobs.SubmitTask(
            task_key="run_pytest",
            notebook_task=jobs.NotebookTask(
                notebook_path=NOTEBOOK_PATH,
                base_parameters={},
            ),
            timeout_seconds=1800,
        )
    ],
)

run_id = run.run_id
print(f"Submitted: run_id={run_id}")
print(f"URL: {os.environ['DATABRICKS_HOST']}#job/runs/{run_id}")

# ---------------------------------------------------------------------------
# Step 4: Poll for completion
# ---------------------------------------------------------------------------
print("Polling (takes ~3-8 min for serverless startup + tests)...")
start = time.time()
while True:
    state = w.jobs.get_run(run_id=run_id)
    life = state.state.life_cycle_state.value if state.state.life_cycle_state else "UNKNOWN"
    result_state = state.state.result_state.value if state.state.result_state else ""
    elapsed = int(time.time() - start)
    print(f"  [{elapsed:4d}s] {life} {result_state}", flush=True)

    if life in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        # Fetch task-level output
        run_detail = w.jobs.get_run(run_id=run_id)
        for t in (run_detail.tasks or []):
            try:
                out = w.jobs.get_run_output(run_id=t.run_id)
                if out.notebook_output and out.notebook_output.result:
                    print("\n--- Pytest output ---")
                    print(out.notebook_output.result)
                if out.error and out.error != out.notebook_output.result if out.notebook_output else True:
                    print("Error:", out.error[:500])
            except Exception as te:
                print(f"Task output error: {te}")

        if result_state == "SUCCESS":
            print("\nRun SUCCEEDED.")
            sys.exit(0)
        else:
            print(f"\nRun FAILED: {result_state}")
            sys.exit(1)

    time.sleep(20)
