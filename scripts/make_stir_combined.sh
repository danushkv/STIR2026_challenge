#!/bin/bash
# Build the STIRcombined symlink farm: STIR original + STIR-Challenge-2024, in
# ONE root with non-colliding patient names.
#
# WHY: training used both collections at once (train.yaml stir_root =
# STIRcombined). They cannot simply be merged -- STIRChallenge_2024's patients
# are 02..11 and STIR's are 0..28, so `02` and `2` would fight. The farm keeps
# the STIR names as-is and suffixes the 2024 ones with `_2024`. That suffixed
# name is what appears in every raw-track / pseudo-label filename, so it must
# match exactly or the trainer will not find frames for its labels.
#
#   bash make_stir_combined.sh
set -euo pipefail
DATA_ROOT="${DATA_ROOT:-/mnt/cluster/datasets}"
SRC_STIR="${DATA_ROOT}/STIRDataset"
SRC_2024="${DATA_ROOT}/STIRChallenge_2024"
DST="${DATA_ROOT}/STIRcombined"

mkdir -p "${DST}"
for p in "${SRC_STIR}"/*/; do
  ln -sfn "${p}" "${DST}/$(basename "${p}")"
done
for p in "${SRC_2024}"/*/; do
  ln -sfn "${p}" "${DST}/$(basename "${p}")_2024"
done
echo "${DST}:"; ls "${DST}" | tr '\n' ' '; echo
