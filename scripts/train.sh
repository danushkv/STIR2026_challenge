#!/bin/bash
#SBATCH -o logs/train_%j.out
#SBATCH --job-name=stir_train
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --mem=80GB
#
# TRAIN THE RELEASED MODEL -- run `agg`, whose released checkpoint is epoch 44.
#
#   bash train.sh
#   RUN_NAME=my_run bash train.sh wandb.enabled=false
#
# This is the exact command the original run used, recovered from that run's
# wandb metadata and cross-checked against the config.yaml the trainer itself
# wrote into the run directory. Everything not listed here comes from
# src/train.yaml unchanged: lr 2e-5, window_len 16, train_iters 4,
# model_resolution 384x512, max_train_frames 64, max_points 384, seed 0,
# supervision `weighted`, skip 5.
#
# NOTE ON THE INITIALISATION. `checkpoint=` is NOT the stock CoTracker3
# `scaled_online.pth` that train.yaml defaults to -- this run started
# from `litetracker_finetuned.pth`, the STIR-2025 winning checkpoint. Same
# architecture and same state-dict keys, so it loads identically; it is a
# different starting point, not a different model.
#
# Output: ${out_dir}/${RUN_NAME}/{config.yaml,student_e2.pth,...,student_e44.pth,
# student_last.pth}. save_freq=2 is why only even epochs exist.
#
# RUN_NAME DEFAULTS TO A *NEW* NAME, NOT THE ORIGINAL. train.py builds
# its run directory as <out_dir>/<wandb.run_name> with exist_ok=True and writes
# student_e<N>.pth unconditionally -- so reusing `fast_run_with2024_agg` could
# overwrite the released run's source checkpoint. The guard below refuses to
# start if the directory already
# has checkpoints in it. Override deliberately with:
#   RUN_NAME=my_run bash train.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env}"
if [ -f "${ENV_FILE}" ]; then set -a; source "${ENV_FILE}"; set +a; fi
export THIRDPARTY_ROOT="${THIRDPARTY_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)}"
export STIRLOADER_ROOT="${STIRLOADER_ROOT:-${THIRDPARTY_ROOT}/STIRLoader}"
export LITETRACKER_ROOT="${LITETRACKER_ROOT:-${THIRDPARTY_ROOT}/lite-tracker-master}"
export STIR_METRICS_ROOT="${STIR_METRICS_ROOT:-${THIRDPARTY_ROOT}/stir-challenge-2026-metrics}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/reproduce.yaml}"
export PSEUDO_LABELS_DIR="${PSEUDO_LABELS_DIR:-${DATA_ROOT}/STIRprocessed/pseudo_labels_with_2024_agg}"
export STIR_TRAIN_ROOT="${STIR_TRAIN_ROOT:-${DATA_ROOT}/STIRcombined}"
export COTRACKER_ROOT="${COTRACKER_ROOT:-${THIRDPARTY_ROOT}/co-tracker}"
export INIT_CKPT="${INIT_CKPT:-${THIRDPARTY_ROOT}/track_on/checkpoint/litetracker_finetuned.pth}"

export PYTHONUNBUFFERED=1
[ -x "${PYTHON_BIN}" ] || {
  echo "Python is not executable: ${PYTHON_BIN}"
  echo "Run bash scripts/preflight.sh after recreating the environment, or set PYTHON_BIN."
  exit 1
}
[ -f "${CONFIG}" ] || { echo "config not found: ${CONFIG}"; exit 1; }

RUN_NAME="${RUN_NAME:-fast_run_with2024_agg_repro}"
OUT_DIR="${OUT_DIR:-${RUNS_ROOT:-${DATA_ROOT}/STIRprocessed/model_runs}}"
RUN_DIR="${OUT_DIR}/${RUN_NAME}"

if compgen -G "${RUN_DIR}/student_*.pth" > /dev/null; then
  echo "REFUSING TO START: ${RUN_DIR} already contains checkpoints."
  echo "Training would overwrite them. Pick another name:  RUN_NAME=... bash $0"
  exit 1
fi

echo "config:  ${CONFIG}"
echo "python:  ${PYTHON_BIN}"
echo "run dir: ${RUN_DIR}"

"${PYTHON_BIN}" "${REPO_ROOT}/src/train.py" --config "${CONFIG}" \
  "$@" \
  out_dir="${OUT_DIR}" \
  wandb.run_name="${RUN_NAME}"
