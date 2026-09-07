#!/bin/bash
# PHASE 2 -- verifier-guided fusion of the six teachers' raw tracks into the
# pseudo-label set the SHIPPED model was trained on
# (`pseudo_labels_with_2024_agg`) that the released checkpoint was trained on.
#
#   bash run_phase2.sh
#
# No GPU, no model imports -- it only reads back the .npz files phase 1 wrote,
# plus the IR-tattoo endpoints, which are pulled live from the dataset through
# STIRLoader.getendcenters().
#
# TWO INVOCATIONS, ONE OUTPUT DIRECTORY. The two collections live under
# different raw-track roots (collect_teacher.sh writes them separately), and
# run_phase2.py takes a single --raw-tracks-root, so it is run once per
# collection into the SAME --out-dir. Patient names do not collide (2024 ones
# carry the `_2024` suffix), so the merged directory is exactly the union.
#
# --mode aggregate is the one knob that distinguishes this lineage: the verifier
# blends all covering teachers weighted by their per-frame score, instead of
# committing to the single best teacher per frame (`select`, the default, which
# is the alternative, and what our other ablations used).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env}"
if [ -f "${ENV_FILE}" ]; then set -a; source "${ENV_FILE}"; set +a; fi
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
# run_phase2.py pulls the IR-tattoo endpoints live through STIRLoader.
export THIRDPARTY_ROOT="${THIRDPARTY_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)}"
export STIRLOADER_ROOT="${STIRLOADER_ROOT:-${THIRDPARTY_ROOT}/STIRLoader}"
export LITETRACKER_ROOT="${LITETRACKER_ROOT:-${THIRDPARTY_ROOT}/lite-tracker-master}"
export STIR_METRICS_ROOT="${STIR_METRICS_ROOT:-${THIRDPARTY_ROOT}/stir-challenge-2026-metrics}"

PROC="${DATA_ROOT}/STIRprocessed"
OUT="${PSEUDO_LABELS_OUT:-${PROC}/pseudo_labels_with_2024_agg}"
TEACHERS=(alltracker cotracker3 trackon2 trackon_r locotrack mft)

cd "${REPO_ROOT}/src"

echo "=== STIR original =============================================="
"${PYTHON_BIN}" run_phase2.py \
  --raw-tracks-root "${PROC}/STIROrig_tracks" \
  --stir-root       "${DATA_ROOT}/STIRDataset" \
  --teachers        "${TEACHERS[@]}" \
  --mode            aggregate \
  --out-dir         "${OUT}"

echo "=== STIR Challenge 2024 ========================================"
"${PYTHON_BIN}" run_phase2.py \
  --raw-tracks-root "${PROC}/STIRChallenge2024_tracks" \
  --stir-root       "${DATA_ROOT}/STIRcombined" \
  --teachers        "${TEACHERS[@]}" \
  --mode            aggregate \
  --out-dir         "${OUT}"

echo
echo "pseudo-labels in ${OUT}"
find "${OUT}" -name '*.npz' | wc -l
