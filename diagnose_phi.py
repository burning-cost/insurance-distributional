"""
Diagnostic: run GammaGBM on gamma_data and print phi statistics.
"""
import os, sys, time, base64, tempfile, zipfile
from pathlib import Path

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
PROJECT_ROOT = Path(__file__).parent
WORKSPACE_PATH = "/Workspace/insurance-distributional-tests"
NOTEBOOK_PATH = f"{WORKSPACE_PATH}/diagnose_phi6"

zip_path = Path(tempfile.mktemp(suffix=".zip"))
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for pattern in ["src/**/*.py", "pyproject.toml", "README.md"]:
        for f in PROJECT_ROOT.glob(pattern):
            zf.write(f, str(f.relative_to(PROJECT_ROOT)))
zip_b64 = base64.b64encode(zip_path.read_bytes()).decode()

# Diagnostic code to embed
DIAG_CODE = r"""
import numpy as np

# Try to import after sys.path manipulation
import sys, os
sys.path.insert(0, str(extract_dir / 'src'))
from insurance_distributional import GammaGBM

rng = np.random.default_rng(1)
n = 300
X = rng.standard_normal((n, 4))
mu_true = np.exp(6.5 + 0.5 * X[:, 0])
shape_true = 2.0
y = rng.gamma(shape=shape_true, scale=mu_true / shape_true)

model = GammaGBM()
model.fit(X, y)
pred = model.predict(X)

print('phi: mean={:.4f}, median={:.4f}'.format(np.mean(pred.phi), np.median(pred.phi)))
print('phi pcts:', np.percentile(pred.phi, [5,25,50,75,95]).tolist())
print('_log_phi_init:', model._log_phi_init)

phi_raw = model._model_phi.predict(X)
print('phi_raw (no base): mean={:.4f}, median={:.4f}'.format(np.mean(phi_raw), np.median(phi_raw)))

phi_wb = model._predict_catboost(
    model._model_phi, X,
    baseline=np.full(n, model._log_phi_init)
)
print('phi_with_base: mean={:.4f}, median={:.4f}'.format(np.mean(phi_wb), np.median(phi_wb)))

d_hat = np.clip((y - pred.mu)**2 / pred.mu**2, 1e-8, None)
print('d: mean={:.4f}, median={:.4f}'.format(np.mean(d_hat), np.median(d_hat)))
"""

NOTEBOOK_CONTENT = (
    "# Databricks notebook source\n"
    "# COMMAND ----------\n"
    "# MAGIC %pip install catboost polars scipy numpy --quiet\n"
    "\n"
    "# COMMAND ----------\n"
    "import base64, zipfile, sys, tempfile, os\n"
    "from pathlib import Path\n"
    "\n"
    f'PROJECT_ZIP_B64 = """{zip_b64}"""\n'
    "\n"
    "extract_dir = Path(tempfile.mkdtemp(prefix='ins-dist-'))\n"
    "zip_bytes = base64.b64decode(PROJECT_ZIP_B64)\n"
    "fd, zip_tmp = tempfile.mkstemp(suffix='.zip')\n"
    "os.close(fd)\n"
    "Path(zip_tmp).write_bytes(zip_bytes)\n"
    "with zipfile.ZipFile(zip_tmp, 'r') as zf:\n"
    "    zf.extractall(str(extract_dir))\n"
    "os.unlink(zip_tmp)\n"
    "\n"
    "# Add src to sys.path\n"
    "sys.path.insert(0, str(extract_dir / 'src'))\n"
    "\n"
    "import numpy as np\n"
    "from insurance_distributional import GammaGBM\n"
    "\n"
    "rng = np.random.default_rng(1)\n"
    "n = 300\n"
    "X = rng.standard_normal((n, 4))\n"
    "mu_true = np.exp(6.5 + 0.5 * X[:, 0])\n"
    "y = rng.gamma(shape=2.0, scale=mu_true / 2.0)\n"
    "\n"
    "model = GammaGBM()\n"
    "model.fit(X, y)\n"
    "pred = model.predict(X)\n"
    "\n"
    "print('phi: mean={:.4f}, median={:.4f}'.format(np.mean(pred.phi), np.median(pred.phi)))\n"
    "print('phi pcts:', np.percentile(pred.phi, [5,25,50,75,95]).tolist())\n"
    "print('_log_phi_init:', model._log_phi_init)\n"
    "\n"
    "phi_raw = model._model_phi.predict(X)\n"
    "print('phi_raw: mean={:.4f}, median={:.4f}'.format(np.mean(phi_raw), np.median(phi_raw)))\n"
    "\n"
    "phi_wb = model._predict_catboost(\n"
    "    model._model_phi, X,\n"
    "    baseline=np.full(n, model._log_phi_init)\n"
    ")\n"
    "print('phi_wb: mean={:.4f}, median={:.4f}'.format(np.mean(phi_wb), np.median(phi_wb)))\n"
    "\n"
    "d_hat = np.clip((y - pred.mu)**2 / pred.mu**2, 1e-8, None)\n"
    "print('d: mean={:.4f}, median={:.4f}'.format(np.mean(d_hat), np.median(d_hat)))\n"
    "\n"
    "out = 'phi={:.4f}/{:.4f},phi_raw={:.4f}/{:.4f},phi_wb={:.4f}/{:.4f},d={:.4f}/{:.4f},log_phi_init={:.4f}'.format(\n"
    "    np.mean(pred.phi), np.median(pred.phi),\n"
    "    np.mean(phi_raw), np.median(phi_raw),\n"
    "    np.mean(phi_wb), np.median(phi_wb),\n"
    "    np.mean(d_hat), np.median(d_hat),\n"
    "    model._log_phi_init\n"
    ")\n"
    "print(out)\n"
    "dbutils.notebook.exit(out)\n"
)

nb_b64 = base64.b64encode(NOTEBOOK_CONTENT.encode()).decode()
w.workspace.import_(
    path=NOTEBOOK_PATH,
    format=workspace.ImportFormat.SOURCE,
    language=workspace.Language.PYTHON,
    content=nb_b64,
    overwrite=True,
)
print("Notebook created")

run = w.jobs.submit(
    run_name="diagnose-phi6",
    tasks=[jobs.SubmitTask(
        task_key="diag",
        notebook_task=jobs.NotebookTask(notebook_path=NOTEBOOK_PATH),
        timeout_seconds=1200,
    )]
)
run_id = run.run_id
print(f"Submitted: {run_id}")

start = time.time()
while True:
    state = w.jobs.get_run(run_id=run_id)
    life = state.state.life_cycle_state.value if state.state.life_cycle_state else "UNKNOWN"
    result_state = state.state.result_state.value if state.state.result_state else ""
    elapsed = int(time.time() - start)
    print(f"  [{elapsed:3d}s] {life} {result_state}", flush=True)
    if life in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        for t in (state.tasks or []):
            try:
                out = w.jobs.get_run_output(run_id=t.run_id)
                if out.notebook_output and out.notebook_output.result:
                    print("\n--- Output ---")
                    print(out.notebook_output.result)
                if out.error:
                    print("Error:", out.error[:2000])
            except Exception as te:
                print(f"Task output error: {te}")
        sys.exit(0 if result_state == "SUCCESS" else 1)
    time.sleep(15)
