#!/bin/bash
# Re-fuse one cached six-teacher clip and compare it with the released label.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/nct-zfs/TCO-Test/venkateda/miccai_challenges/stir/stir2026-nct}"
THIRDPARTY_ROOT="${THIRDPARTY_ROOT:-/mnt/nct-zfs/TCO-Test/venkateda/miccai_challenges/stir}"
DATA_ROOT="${DATA_ROOT:-/mnt/cluster/datasets}"
VENV_ROOT="${VENV_ROOT:-/mnt/cluster/environments/venkateda}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_ROOT}/trackon_env/bin/python}"
SMOKE_CLIP="${SMOKE_CLIP:-0__left__seq00}"
RAW_TRACKS="${RAW_TRACKS:-${DATA_ROOT}/STIRprocessed/STIROrig_tracks}"
STIR_ROOT="${STIR_ROOT:-${DATA_ROOT}/STIRDataset}"
REFERENCE_ROOT="${REFERENCE_ROOT:-${DATA_ROOT}/STIRprocessed/pseudo_labels_with_2024_agg}"
export THIRDPARTY_ROOT
export STIRLOADER_ROOT="${STIRLOADER_ROOT:-${THIRDPARTY_ROOT}/STIRLoader}"

[ -x "${PYTHON_BIN}" ] || { echo "Python is not executable: ${PYTHON_BIN}"; exit 1; }
WORK="${SMOKE_OUT_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/stir-phase2.XXXXXX")}"
if [ "${KEEP_SMOKE_OUTPUT:-0}" != 1 ]; then
  trap 'rm -rf "${WORK}"' EXIT
fi

teachers=(alltracker cotracker3 trackon2 trackon_r locotrack mft)
"${PYTHON_BIN}" "${REPO_ROOT}/src/run_phase2.py" \
  --raw-tracks-root "${RAW_TRACKS}" \
  --stir-root "${STIR_ROOT}" \
  --teachers "${teachers[@]}" \
  --mode aggregate \
  --clip-ids "${SMOKE_CLIP}" \
  --out-dir "${WORK}"

patient="${SMOKE_CLIP%%__*}"
seq_part="${SMOKE_CLIP#*__}"
actual="${WORK}/${patient}/${seq_part}.npz"
reference="${REFERENCE_ROOT}/${patient}/${seq_part}.npz"

"${PYTHON_BIN}" - "${actual}" "${reference}" <<'PY'
import sys
import numpy as np

actual_path, reference_path = sys.argv[1:]
with np.load(actual_path) as actual, np.load(reference_path) as reference:
    if set(actual.files) != set(reference.files):
        raise SystemExit(f"key mismatch: {actual.files} != {reference.files}")
    for key in actual.files:
        a, b = actual[key], reference[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise SystemExit(
                f"{key}: shape/dtype mismatch {a.shape}/{a.dtype} != {b.shape}/{b.dtype}")
        if a.dtype.kind in "fc":
            same = np.array_equal(a, b, equal_nan=True)
        else:
            same = np.array_equal(a, b)
        if not same:
            raise SystemExit(f"{key}: regenerated values differ from released label")
        print(f"  ok   {key}: shape={a.shape} dtype={a.dtype}")
print("PHASE-2 SMOKE PASSED: regenerated label exactly matches the release")
PY

[ "${KEEP_SMOKE_OUTPUT:-0}" = 1 ] && echo "kept output: ${WORK}"
