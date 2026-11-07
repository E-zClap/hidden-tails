#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCEPTED_PATH="${1:-${LENDINGCLUB_ACCEPTED:-}}"
REJECTED_PATH="${2:-${LENDINGCLUB_REJECTED:-}}"
RUN_STAGE="${RUN_STAGE:-all}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -z "${ACCEPTED_PATH}" || -z "${REJECTED_PATH}" ]]; then
  echo "Usage: bash run_gpu.sh /path/to/accepted.csv.gz /path/to/rejected.csv.gz" >&2
  echo "Alternatively set LENDINGCLUB_ACCEPTED and LENDINGCLUB_REJECTED." >&2
  exit 2
fi
if [[ ! -f "${ACCEPTED_PATH}" || ! -f "${REJECTED_PATH}" ]]; then
  echo "One or both LendingClub input files do not exist." >&2
  exit 2
fi

cd "${ROOT_DIR}"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found; an NVIDIA GPU/driver is required." >&2
  exit 3
fi
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "No accessible NVIDIA GPU was reported by nvidia-smi." >&2
  exit 3
fi
if [[ ! -d .venv ]]; then
  "${PYTHON_BIN}" -m venv .venv
fi
source .venv/bin/activate
python - <<'PY'
import sys

if sys.version_info < (3, 9):
    raise SystemExit(
        f"Python 3.9 or newer is required; found {sys.version.split()[0]} at {sys.executable}"
    )
print("Using virtual-environment Python", sys.version.split()[0], "from", sys.executable)
PY
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
fi

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED="20260727"

# SKIP_INSTALL is safe only when the environment exactly matches the lock file.
python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

problems = []
for raw in Path("requirements.txt").read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    name, expected = line.split("==", 1)
    try:
        actual = version(name)
    except PackageNotFoundError:
        problems.append(f"{name}: missing (expected {expected})")
    else:
        if actual != expected:
            problems.append(f"{name}: installed {actual}, expected {expected}")
if problems:
    raise SystemExit(
        "Dependency preflight failed:\n  - "
        + "\n  - ".join(problems)
        + "\nRerun without SKIP_INSTALL=1."
    )
print("Dependency versions match requirements.txt")
PY

# Fail on lightweight preprocessing/numerical regressions before touching the
# full CSV files or fitting any model.
python -m unittest discover -s tests -v

python - <<'PY'
import numpy as np
import pandas as pd
import xgboost as xgb
from real_cvar.config import load_config
from real_cvar.models import build_classifier

if not xgb.build_info().get("USE_CUDA", False):
    raise SystemExit("Installed XGBoost build does not include CUDA support")

cfg, _ = load_config("config.yaml")
cfg["xgboost"]["n_estimators"] = 2
cfg["xgboost"]["max_depth"] = 1
cfg["xgboost"]["n_jobs"] = 1
frame = pd.DataFrame(
    {
        "requested_amount": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
        "dti": [5, 10, 15, 20, 25, 30, 35, 40],
        "risk_score": [620, 640, 660, 680, 700, 720, 740, 760],
        "employment_years": [0.5, 1, 2, 3, 4, 5, 8, 10],
        "state": ["CA", "NY"] * 4,
        "zip3": ["941", "100"] * 4,
        "purpose": ["debt_consolidation", "credit_card"] * 4,
    }
)
target = np.asarray([0, 0, 0, 1, 0, 1, 1, 1], dtype=np.int32)
model = build_classifier("gradient_boosting", cfg, seed=7)
model.fit(frame, target)
probability = model.predict_proba(frame)[:, 1]
if probability.shape != (len(frame),) or not np.isfinite(probability).all():
    raise SystemExit("GPU sklearn-pipeline preflight returned invalid probabilities")
print("CUDA sklearn-pipeline preflight passed with XGBoost", xgb.__version__)
PY

python -m real_cvar.cli "${RUN_STAGE}" \
  --config "${ROOT_DIR}/config.yaml" \
  --accepted "${ACCEPTED_PATH}" \
  --rejected "${REJECTED_PATH}" \
  --device cuda
