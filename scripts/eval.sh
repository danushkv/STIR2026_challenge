#!/bin/bash
#SBATCH -o logs/eval_repro_%j.out
#SBATCH --job-name=stir_eval_repro
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --mem=80GB
#
# Evaluate a REPRODUCED checkpoint and compare it to the original, paired.
#
#   bash eval.sh                      # defaults to the agg_e44 repro
#   RUN_NAME=my_run EPOCH=44 bash eval.sh
#
# Writes its shard into the SAME sweep directory as the original run. That is
# deliberate and it is the whole point: compare_ckpts.py pairs runs by scoring
# the same clips and the same points in the same order, so the reproduction and
# the original are compared on identical data with one shared resample matrix.
# An unpaired comparison of these two numbers would need ~21pp to mean anything;
# the paired one needs far less.
#
# The analysis goes to analysis_repro/ so the original analysis/ranking.md --
# the record every number in the report came from -- is left untouched.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/nct-zfs/TCO-Test/venkateda/miccai_challenges/stir/stir2026-nct}"
export THIRDPARTY_ROOT="${THIRDPARTY_ROOT:-/mnt/nct-zfs/TCO-Test/venkateda/miccai_challenges/stir}"
export STIRLOADER_ROOT="${STIRLOADER_ROOT:-${THIRDPARTY_ROOT}/STIRLoader}"
export LITETRACKER_ROOT="${LITETRACKER_ROOT:-${THIRDPARTY_ROOT}/lite-tracker-master}"
export STIR_METRICS_ROOT="${STIR_METRICS_ROOT:-${THIRDPARTY_ROOT}/stir-challenge-2026-metrics}"
export STIR_INFERENCE_ROOT="${STIR_INFERENCE_ROOT:-${THIRDPARTY_ROOT}/stir-challenge-2026-inference}"
DATA_ROOT="${DATA_ROOT:-/mnt/cluster/datasets}"
VENV_ROOT="${VENV_ROOT:-/mnt/cluster/environments/venkateda}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_ROOT}/trackon_env/bin/python}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/eval_reproduction.yaml}"

RUN_NAME="${RUN_NAME:-fast_run_with2024_agg_repro}"
EPOCH="${EPOCH:-44}"
TAG="${TAG:-agg_e${EPOCH}_repro}"
ITERS="${ITERS:-[1,2,4]}"

CKPT="${DATA_ROOT}/STIRprocessed/model_runs/${RUN_NAME}/student_e${EPOCH}.pth"
TEST_ROOT="${TEST_ROOT:-${DATA_ROOT}/STIRTest_2025}"
OUT="${OUT:-${DATA_ROOT}/STIRprocessed/eval_sweep}"

[ -f "${CKPT}" ] || { echo "no checkpoint at ${CKPT}"; exit 1; }
[ -f "${OUT}/agg_e44.npz" ] || echo "warning: ${OUT}/agg_e44.npz missing -- nothing to pair against"

export PYTHONUNBUFFERED=1
[ -x "${PYTHON_BIN}" ] || { echo "Python is not executable: ${PYTHON_BIN}"; exit 1; }
[ -f "${CONFIG}" ] || { echo "config not found: ${CONFIG}"; exit 1; }
mkdir -p "${OUT}/logs"

echo "tag   ${TAG}"
echo "ckpt  ${CKPT}"
echo "iters ${ITERS}"
echo "config ${CONFIG}"
echo

# </dev/null is NOT optional: STIRLoader decodes with ffmpeg, ffmpeg reads stdin,
# and anything that runs this inside a `while read` loop would have its input eaten.
"${PYTHON_BIN}" "${REPO_ROOT}/src/eval_sweep.py" --config "${CONFIG}" \
    tag="${TAG}" student_checkpoint="${CKPT}" \
    val_stir_root="${TEST_ROOT}" iters_list="${ITERS}" out_dir="${OUT}" \
    < /dev/null 2>&1 | tee "${OUT}/logs/${TAG}.log"

echo
echo "=== paired comparison against every existing shard ==="
"${PYTHON_BIN}" "${REPO_ROOT}/src/compare_ckpts.py" "${OUT}" \
    --n-boot 10000 --out-dir "${OUT}/analysis_repro"

echo
echo "read:  ${OUT}/analysis_repro/ranking.md"
echo "the row that matters is the paired delta between ${TAG} and agg_e44"
