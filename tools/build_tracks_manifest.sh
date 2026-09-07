#!/bin/bash
# Build the integrity/index manifest shipped with the Hugging Face track dataset.
set -euo pipefail

TRACK_ROOT="${1:?usage: bash tools/build_tracks_manifest.sh <STIROrig_tracks> [manifest.csv]}"
OUT="${2:-manifest.csv}"
[ -d "${TRACK_ROOT}" ] || { echo "track root not found: ${TRACK_ROOT}"; exit 1; }

TMP="$(mktemp "${TMPDIR:-/tmp}/stir-track-manifest.XXXXXX")"
trap 'rm -f "${TMP}"' EXIT
printf 'path,teacher,clip_id,bytes,sha256\n' > "${TMP}"

while IFS= read -r -d '' path; do
  rel="${path#${TRACK_ROOT}/}"
  teacher="${rel%%/*}"
  filename="${rel##*/}"
  clip_id="${filename%__${teacher}.npz}"
  patient="${rel#*/}"
  patient="${patient%%/*}"
  bytes="$(stat -c '%s' "${path}")"
  sha256="$(sha256sum "${path}")"
  sha256="${sha256%% *}"
  printf '%s,%s,%s__%s,%s,%s\n' \
    "${rel}" "${teacher}" "${patient}" "${clip_id}" "${bytes}" "${sha256}" >> "${TMP}"
done < <(find "${TRACK_ROOT}" -type f -name '*.npz' -print0 | sort -z)

mv "${TMP}" "${OUT}"
trap - EXIT
echo "wrote ${OUT} ($(($(wc -l < "${OUT}") - 1)) files)"
