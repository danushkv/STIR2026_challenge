#!/bin/bash
#SBATCH -o logs/collect_%j.out
#SBATCH --job-name=stir_collect
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --mem=80GB
#
# PHASE 1 -- run ONE teacher over one STIR collection and write raw trajectories.
#
#   bash collect_teacher.sh <teacher> [orig|2024]        # default collection: orig
#   sbatch collect_teacher.sh cotracker3 orig
#
# Teachers (all six that produced the shipped pseudo-labels):
#   cotracker3  alltracker  locotrack        -> track_on's ensemble/ wrappers, trackon_env
#   trackon2    trackon_r                    -> track_on's own model,          trackon_env
#   mft                                      -> the MFT repo,                  mft_env
#
# WHY ONE TEACHER PER INVOCATION: the teachers pin incompatible torch/timm
# versions and cannot be imported into one process. Label generation is offline,
# so they never need to run together -- only their OUTPUT has to land in the
# same place. Run them one at a time, in any order, in each one's own venv.
#
# OUTPUT LAYOUT (what run_phase2.py reads):
#   ${RAW_TRACKS_ROOT}/<teacher>/<patient>/<side>__<seq>__<teacher>.npz
set -euo pipefail

TEACHER="${1:?usage: bash collect_teacher.sh <teacher> [orig|2024]}"
COLLECTION="${2:-orig}"

# --- paths -----------------------------------------------------------------
# REPO_ROOT       : this repository (contains src/ and scripts/)
# THIRDPARTY_ROOT : where the cloned upstream repos live -- track_on/, MFT/,
#                   co-tracker/, lite-tracker-master/. See docs/ENVIRONMENTS.md.
REPO_ROOT="${REPO_ROOT:-/mnt/nct-zfs/TCO-Test/venkateda/miccai_challenges/stir/stir2026-nct}"
export THIRDPARTY_ROOT="${THIRDPARTY_ROOT:-/mnt/nct-zfs/TCO-Test/venkateda/miccai_challenges/stir}"
export STIRLOADER_ROOT="${STIRLOADER_ROOT:-${THIRDPARTY_ROOT}/STIRLoader}"
export LITETRACKER_ROOT="${LITETRACKER_ROOT:-${THIRDPARTY_ROOT}/lite-tracker-master}"
export STIR_METRICS_ROOT="${STIR_METRICS_ROOT:-${THIRDPARTY_ROOT}/stir-challenge-2026-metrics}"
DATA_ROOT="${DATA_ROOT:-/mnt/cluster/datasets}"

# The 2024 clips are read through the STIRcombined symlink farm rather than
# STIRChallenge_2024 directly, because the patient folder there is named `02`
# and STIR's own patient `2` already exists -- the clip ids would collide. The
# farm renames them `02_2024`, which is the patient name that ends up in every
# raw-track and pseudo-label filename. Build it with make_stir_combined.sh.
PATIENTS_2024=(02_2024 03_2024 04_2024 05_2024 06_2024 07_2024 08_2024 09_2024 11_2024)
SEL=()
case "${COLLECTION}" in
  orig) STIR_ROOT="${DATA_ROOT}/STIRDataset"
        RAW_TRACKS_ROOT="${DATA_ROOT}/STIRprocessed/STIROrig_tracks" ;;
  2024) STIR_ROOT="${DATA_ROOT}/STIRcombined"
        RAW_TRACKS_ROOT="${DATA_ROOT}/STIRprocessed/STIRChallenge2024_tracks"
        SEL=(--patients "${PATIENTS_2024[@]}") ;;
  *) echo "collection must be 'orig' or '2024'"; exit 1 ;;
esac

# --- environment + teacher-specific checkpoint ------------------------------
# The track_on teachers all share one venv; MFT has its own.
VENV_ROOT="${VENV_ROOT:-/mnt/cluster/environments/venkateda}"
CKPT_DIR="${THIRDPARTY_ROOT}/track_on/checkpoint"
EXTRA=()
case "${TEACHER}" in
  cotracker3)  VENV="${VENV_ROOT}/trackon_env"; REPO="${THIRDPARTY_ROOT}/track_on"
               ;;                                        # checkpoint auto-downloaded via torch.hub
  alltracker)  VENV="${VENV_ROOT}/trackon_env"; REPO="${THIRDPARTY_ROOT}/track_on"
               EXTRA=(--checkpoint "${CKPT_DIR}/alltracker.pth") ;;
  locotrack)   VENV="${VENV_ROOT}/trackon_env"; REPO="${THIRDPARTY_ROOT}/track_on"
               EXTRA=(--checkpoint "${CKPT_DIR}/locotrack_base.ckpt") ;;
  trackon2)    VENV="${VENV_ROOT}/trackon_env"; REPO="${THIRDPARTY_ROOT}/track_on"
               EXTRA=(--checkpoint "${CKPT_DIR}/trackon2_dinov3_checkpoint.pt") ;;
  trackon_r)   VENV="${VENV_ROOT}/trackon_env"; REPO="${THIRDPARTY_ROOT}/track_on"
               EXTRA=(--checkpoint "${CKPT_DIR}/track_on_r.pt") ;;
  mft)         VENV="${VENV_ROOT}/mft_env";     REPO="${THIRDPARTY_ROOT}/MFT" ;;
  *) echo "unknown teacher: ${TEACHER}"; exit 1 ;;
esac

OUT_DIR="${RAW_TRACKS_ROOT}/${TEACHER}"
mkdir -p "${OUT_DIR}"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
cd "${REPO_ROOT}/src"

echo "teacher=${TEACHER}  collection=${COLLECTION}"
echo "stir_root=${STIR_ROOT}"
echo "out_dir=${OUT_DIR}"

# --skip 5 is NOT optional: it must match the skip used for training
# (train.yaml: skip: 5). Changing it changes the inter-frame motion the
# pseudo-labels describe, and the trainer would then index the wrong frames.
python collect_tracks.py \
  --teacher "${TEACHER}" \
  --repo-root "${REPO}" \
  --stir-root "${STIR_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --skip 5 \
  "${SEL[@]+"${SEL[@]}"}" \
  "${EXTRA[@]+"${EXTRA[@]}"}"
  # --patients 0 1      # limit to specific patient folders (quick sanity run)
  # --overwrite         # redo clips that already have output
