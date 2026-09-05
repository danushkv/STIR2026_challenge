#!/bin/bash
# Save the submission images as .tar.
#
#   bash export_all.sh [out_dir]
#
# ONE image serves all three tracks, so there is one tar. The base is
# nvcr.io/nvidia/pytorch:24.10-py3 (~20GB uncompressed), so expect a large file
# and check free space -- printed below before writing.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TEAM="${TEAM:-nct}"
OUT="${HERE}/export"
for a in "$@"; do OUT="$a"; done
mkdir -p "$OUT"

IMG="stir-challenge-2026-inference-${TEAM}:latest"
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "missing image $IMG -- run build_all.sh"; exit 1; }

echo "free space on $(df -h "$OUT" | awk 'NR==2{print $6}'): $(df -h "$OUT" | awk 'NR==2{print $4}')"
docker images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}' | grep "${TEAM}" || true
echo

F="${OUT}/stir-challenge-2026-inference-${TEAM}.tar"
echo "saving ${IMG} -> ${F}"
docker save -o "$F" "$IMG"

ls -lh "$OUT"
echo
echo "verify before sending:  docker load -i <file>.tar"
