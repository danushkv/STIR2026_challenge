#!/bin/bash
# Validate the current cluster inputs before any GPU time is spent.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/nct-zfs/TCO-Test/venkateda/miccai_challenges/stir/stir2026-nct}"
THIRDPARTY_ROOT="${THIRDPARTY_ROOT:-/mnt/nct-zfs/TCO-Test/venkateda/miccai_challenges/stir}"
DATA_ROOT="${DATA_ROOT:-/mnt/cluster/datasets}"
VENV_ROOT="${VENV_ROOT:-/mnt/cluster/environments/venkateda}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_ROOT}/trackon_env/bin/python}"
STIRLOADER_ROOT="${STIRLOADER_ROOT:-${THIRDPARTY_ROOT}/STIRLoader}"
LITETRACKER_ROOT="${LITETRACKER_ROOT:-${THIRDPARTY_ROOT}/lite-tracker-master}"
STIR_INFERENCE_ROOT="${STIR_INFERENCE_ROOT:-${THIRDPARTY_ROOT}/stir-challenge-2026-inference}"

PSEUDO_LABELS="${PSEUDO_LABELS:-${DATA_ROOT}/STIRprocessed/pseudo_labels_with_2024_agg}"
STIR_ROOT="${STIR_ROOT:-${DATA_ROOT}/STIRcombined}"
TEST_ROOT="${TEST_ROOT:-${DATA_ROOT}/STIRTest_2025}"
INIT_CKPT="${INIT_CKPT:-${THIRDPARTY_ROOT}/track_on/checkpoint/litetracker_finetuned.pth}"
RELEASED_CKPT="${RELEASED_CKPT:-${DATA_ROOT}/STIRprocessed/model_runs/fast_run_with2024_agg/student_e44.pth}"
RAW_TRACKS="${RAW_TRACKS:-${DATA_ROOT}/STIRprocessed/STIROrig_tracks}"
SMOKE_CLIP="${SMOKE_CLIP:-0__left__seq00}"
EVAL_SMOKE_CLIP="${EVAL_SMOKE_CLIP:-01__left__seq03}"
EXPECTED_INIT_SHA256="${EXPECTED_INIT_SHA256:-8c2082b8c15756913f8a1ad6b6bb1fd9b8c4790e8022450677f6ac19be816e0a}"
EXPECTED_RELEASED_SHA256="${EXPECTED_RELEASED_SHA256:-3b964f18793bab959d1537aa58b0411bb8454e2df5f30bf6d66039a215c8455c}"

failures=0
ok() { echo "  ok   $*"; }
bad() { echo "  FAIL $*"; failures=$((failures + 1)); }
need_file() { [ -f "$1" ] && ok "$2" || bad "$2: $1"; }
need_dir() { [ -d "$1" ] && ok "$2" || bad "$2: $1"; }

echo "=== executables ==="
if [ -x "${PYTHON_BIN}" ]; then
  ok "Python: ${PYTHON_BIN}"
else
  bad "Python is not executable: ${PYTHON_BIN} -> $(readlink "${PYTHON_BIN}" 2>/dev/null || true)"
fi
command -v ffmpeg >/dev/null 2>&1 && ok "ffmpeg" || bad "ffmpeg not found"
command -v nvidia-smi >/dev/null 2>&1 && ok "nvidia-smi" || bad "nvidia-smi not found"

echo "=== code and checkpoints ==="
need_file "${THIRDPARTY_ROOT}/co-tracker/cotracker/models/build_cotracker.py" "CoTracker training clone"
need_file "${STIRLOADER_ROOT}/STIRLoader/STIRLoader.py" "patched STIRLoader clone"
need_file "${LITETRACKER_ROOT}/src/lite_tracker.py" "LiteTracker runtime"
need_file "${STIR_INFERENCE_ROOT}/models/stereo/student_stereo.py" "external stereo wrapper"
need_file "${INIT_CKPT}" "2025 initialization checkpoint"
need_file "${RELEASED_CKPT}" "released epoch-44 checkpoint"

if [ -f "${INIT_CKPT}" ]; then
  got=$(sha256sum "${INIT_CKPT}" | awk '{print $1}')
  [ "${got}" = "${EXPECTED_INIT_SHA256}" ] && ok "initialization SHA-256" || bad "initialization SHA-256: ${got}"
fi
if [ -f "${RELEASED_CKPT}" ]; then
  got=$(sha256sum "${RELEASED_CKPT}" | awk '{print $1}')
  [ "${got}" = "${EXPECTED_RELEASED_SHA256}" ] && ok "released checkpoint SHA-256" || bad "released checkpoint SHA-256: ${got}"
fi

echo "=== data ==="
need_dir "${PSEUDO_LABELS}" "released pseudo-label directory"
need_dir "${STIR_ROOT}" "combined training dataset"
need_dir "${TEST_ROOT}" "evaluation dataset"
eval_patient="${EVAL_SMOKE_CLIP%%__*}"
eval_rest="${EVAL_SMOKE_CLIP#*__}"
eval_side="${eval_rest%%__*}"
eval_seq="${eval_rest#*__}"
need_dir "${TEST_ROOT}/${eval_patient}/${eval_side}/${eval_seq}" "evaluation smoke clip ${EVAL_SMOKE_CLIP}"
if [ -d "${PSEUDO_LABELS}" ]; then
  count=$(find "${PSEUDO_LABELS}" -type f -name '*.npz' | wc -l)
  [ "${count}" -eq 626 ] && ok "626 pseudo-label files" || bad "expected 626 pseudo-label files, found ${count}"
fi
if [ -d "${STIR_ROOT}" ]; then
  count=$(find "${STIR_ROOT}" -mindepth 1 -maxdepth 1 -type l | wc -l)
  [ "${count}" -eq 38 ] && ok "38 combined patient links" || bad "expected 38 combined patient links, found ${count}"
fi

patient="${SMOKE_CLIP%%__*}"
seq_part="${SMOKE_CLIP#*__}"
need_file "${PSEUDO_LABELS}/${patient}/${seq_part}.npz" "smoke pseudo-label ${SMOKE_CLIP}"
for teacher in alltracker cotracker3 trackon2 trackon_r locotrack mft; do
  need_file "${RAW_TRACKS}/${teacher}/${patient}/${seq_part}__${teacher}.npz" "smoke raw track: ${teacher}"
done

echo "=== Python environment ==="
if [ -x "${PYTHON_BIN}" ]; then
  export PYTHONPATH="${THIRDPARTY_ROOT}/co-tracker:${PYTHONPATH:-}"
  if "${PYTHON_BIN}" -c 'import cv2, numpy, omegaconf, scipy, torch; assert torch.cuda.is_available(); print("  ok   imports; torch", torch.__version__, "CUDA", torch.version.cuda, torch.cuda.get_device_name(0))'; then
    :
  else
    bad "Python imports or CUDA availability"
  fi
  if REPO_ROOT="${REPO_ROOT}" "${PYTHON_BIN}" -c 'import os, sys; sys.path.insert(0, os.path.join(os.environ["REPO_ROOT"], "src")); from model import merge_config_strict; merge_config_strict({"known": 1}, overrides=["unknown=2"])' >/dev/null 2>&1; then
    bad "strict configuration accepted an unknown key"
  else
    ok "strict configuration rejects unknown keys"
  fi
fi

echo
if [ "${failures}" -ne 0 ]; then
  echo "PREFLIGHT FAILED: ${failures} problem(s)"
  exit 1
fi
echo "PREFLIGHT PASSED"
