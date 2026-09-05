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
# TRAIN THE SHIPPED MODEL -- run `agg`, whose epoch-44 checkpoint is
# submission/weights/student.pth -- epoch 44 of the `agg` run.
#
#   bash train.sh
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
# student_e<N>.pth unconditionally -- so reusing `fast_run_with2024_agg` would
# overwrite student_e44.pth, which is the file submission/weights/student.pth
# was copied from. The guard below refuses to start if the directory already
# has checkpoints in it. Override deliberately with:
#   RUN_NAME=my_run bash train.sh
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/nct-zfs/TCO-Test/venkateda/miccai_challenges/stir/stir2026-nct}"
export THIRDPARTY_ROOT="${THIRDPARTY_ROOT:-/mnt/nct-zfs/TCO-Test/venkateda/miccai_challenges/stir}"
export STIRLOADER_ROOT="${STIRLOADER_ROOT:-${THIRDPARTY_ROOT}/STIRLoader}"
export LITETRACKER_ROOT="${LITETRACKER_ROOT:-${THIRDPARTY_ROOT}/lite-tracker-master}"
export STIR_METRICS_ROOT="${STIR_METRICS_ROOT:-${THIRDPARTY_ROOT}/stir-challenge-2026-metrics}"
DATA_ROOT="${DATA_ROOT:-/mnt/cluster/datasets}"
VENV_ROOT="${VENV_ROOT:-/mnt/cluster/environments/venkateda}"

export PYTHONUNBUFFERED=1
# shellcheck disable=SC1091
source "${VENV_ROOT}/trackon_env/bin/activate"
cd "${REPO_ROOT}/src"

RUN_NAME="${RUN_NAME:-fast_run_with2024_agg_repro}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/STIRprocessed/model_runs}"
RUN_DIR="${OUT_DIR}/${RUN_NAME}"

if compgen -G "${RUN_DIR}/student_*.pth" > /dev/null; then
  echo "REFUSING TO START: ${RUN_DIR} already contains checkpoints."
  echo "Training would overwrite them. Pick another name:  RUN_NAME=... bash $0"
  exit 1
fi

echo "run dir: ${RUN_DIR}"

python train.py --config train.yaml \
  --pseudo-labels-dir "${DATA_ROOT}/STIRprocessed/pseudo_labels_with_2024_agg/" \
  --stir-root         "${DATA_ROOT}/STIRcombined/" \
  checkpoint="${THIRDPARTY_ROOT}/track_on/checkpoint/litetracker_finetuned.pth" \
  out_dir="${OUT_DIR}" \
  epochs=45 \
  save_freq=5 \
  wandb.run_name="${RUN_NAME}"
