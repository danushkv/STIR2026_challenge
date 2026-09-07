#!/bin/bash
# Evaluate one deterministic clip and validate the resulting shard.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/nct-zfs/TCO-Test/venkateda/miccai_challenges/stir/stir2026-nct}"
THIRDPARTY_ROOT="${THIRDPARTY_ROOT:-/mnt/nct-zfs/TCO-Test/venkateda/miccai_challenges/stir}"
VENV_ROOT="${VENV_ROOT:-/mnt/cluster/environments/venkateda}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_ROOT}/trackon_env/bin/python}"
export THIRDPARTY_ROOT
export STIRLOADER_ROOT="${STIRLOADER_ROOT:-${THIRDPARTY_ROOT}/STIRLoader}"
export LITETRACKER_ROOT="${LITETRACKER_ROOT:-${THIRDPARTY_ROOT}/lite-tracker-master}"
export STIR_INFERENCE_ROOT="${STIR_INFERENCE_ROOT:-${THIRDPARTY_ROOT}/stir-challenge-2026-inference}"
CHECKPOINT="${1:?usage: bash scripts/smoke_eval.sh <checkpoint> [config]}"
EVAL_CONFIG="${2:-${EVAL_CONFIG:-${REPO_ROOT}/configs/eval_reproduction.yaml}}"
SMOKE_CLIP="${SMOKE_CLIP:-01__left__seq03}"
OUT_DIR="${OUT_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/stir-eval.XXXXXX")}"
TAG="${TAG:-smoke_eval}"

[ -x "${PYTHON_BIN}" ] || { echo "Python is not executable: ${PYTHON_BIN}"; exit 1; }
[ -f "${CHECKPOINT}" ] || { echo "checkpoint not found: ${CHECKPOINT}"; exit 1; }
[ -f "${EVAL_CONFIG}" ] || { echo "config not found: ${EVAL_CONFIG}"; exit 1; }

"${PYTHON_BIN}" "${REPO_ROOT}/src/eval_sweep.py" --config "${EVAL_CONFIG}" \
  tag="${TAG}" student_checkpoint="${CHECKPOINT}" \
  eval_clip_ids="[${SMOKE_CLIP}]" max_eval_clips=1 \
  iters_list='[1]' do_drift=false out_dir="${OUT_DIR}" \
  < /dev/null

"${PYTHON_BIN}" - "${OUT_DIR}/${TAG}.json" "${OUT_DIR}/${TAG}.npz" <<'PY'
import json
import math
import os
import sys

json_path, npz_path = sys.argv[1:]
if not os.path.getsize(npz_path):
    raise SystemExit(f"empty shard: {npz_path}")
summary = json.load(open(json_path))
metrics = summary.get("iters", {}).get("1", {})
required = ("delta_avg_2d", "epe_mean_px")
for key in required:
    value = metrics.get(key)
    if value is None or not math.isfinite(float(value)):
        raise SystemExit(f"missing/non-finite {key}: {value}")
    print(f"  ok   {key}={value}")
print("EVALUATION SMOKE PASSED")
PY

echo "outputs: ${OUT_DIR}"
