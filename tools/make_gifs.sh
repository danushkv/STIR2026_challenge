#!/bin/bash
# Build the README GIFs in assets/.
#
#   bash tools/make_gifs.sh                       # defaults below
#   CLIP=1__left__seq05 bash tools/make_gifs.sh
#
# Needs ffmpeg plus the project venv (STIRLoader, cv2, torch). Produces two
# GIFs, each capped at 480px / 8fps / ~6s so they stay a couple of MB -- GitHub
# will not animate a GIF over 5 MB, and a repository should not carry tens of
# MB of surgical video. The two-pass palette is not optional: ffmpeg's default
# 256-colour table turns red tissue to mud.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export THIRDPARTY_ROOT="${THIRDPARTY_ROOT:-$(dirname "${REPO_ROOT}")}"
export STIRLOADER_ROOT="${STIRLOADER_ROOT:-${THIRDPARTY_ROOT}/STIRLoader}"
export LITETRACKER_ROOT="${LITETRACKER_ROOT:-${THIRDPARTY_ROOT}/lite-tracker-master}"
DATA_ROOT="${DATA_ROOT:-/mnt/cluster/datasets}"

CLIP="${CLIP:-0__left__seq00}"
STIR_ROOT="${STIR_ROOT:-${DATA_ROOT}/STIRDataset}"
PL_DIR="${PL_DIR:-${DATA_ROOT}/STIRprocessed/pseudo_labels_with_2024_agg}"
CKPT=/mnt/nct-zfs/TCO-Test/venkateda/miccai_challenges/stir/submission/weights/student.pth
SECONDS_LEN="${SECONDS_LEN:-6}"
WIDTH="${WIDTH:-420}"
# Surgical video is high-entropy and GIF compresses it badly -- ~126 KB/frame at
# 480px/256 colours. Cutting the palette is the cheapest lever by far and costs
# almost nothing visually on tissue; width is the next one.
MAX_COLORS="${MAX_COLORS:-128}"
FPS="${FPS:-8}"

OUT="${REPO_ROOT}/assets"; mkdir -p "${OUT}"
WORK="$(mktemp -d)"; trap 'rm -rf "${WORK}"' EXIT

command -v ffmpeg >/dev/null || { echo "ffmpeg not found"; exit 1; }
[ -f "${CKPT}" ] || { echo "checkpoint not found: ${CKPT} (see submission/weights/README.md)"; exit 1; }

gif() {   # $1 in.mp4  $2 out.gif
  local VF="fps=${FPS},scale=${WIDTH}:-1:flags=lanczos"
  ffmpeg -v error -y -t "${SECONDS_LEN}" -i "$1" \
    -vf "${VF},palettegen=max_colors=${MAX_COLORS}:stats_mode=diff" "${WORK}/pal.png"
  ffmpeg -v error -y -t "${SECONDS_LEN}" -i "$1" -i "${WORK}/pal.png" -loop 0 \
    -lavfi "${VF}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3" "$2"
  local BYTES; BYTES=$(stat -c%s "$2")
  echo "   $(du -h "$2" | cut -f1)  $2"
  if [ "${BYTES}" -gt 5000000 ]; then
    echo "   !! over 5 MB -- GitHub will show a still frame, not an animation."
    echo "      Re-run with MAX_COLORS=64 or WIDTH=320 or SECONDS_LEN=4."
  elif [ "${BYTES}" -gt 2500000 ]; then
    echo "      (heavy for a repo; MAX_COLORS=64 usually halves it again)"
  fi
}

echo "1/2  pseudo-labels -- the verifier's fused output, i.e. what the student is trained on"
python3 "${REPO_ROOT}/src/visualize_pseudo_labels.py" \
    --pseudo-labels-dir "${PL_DIR}" --stir-root "${STIR_ROOT}" \
    --clip-id "${CLIP}" --skip 5 --out-dir "${WORK}/pl"
gif "${WORK}/pl/${CLIP}.mp4" "${OUT}/pseudo_labels.gif"

echo "2/2  student -- streaming inference on a ${GRID}px grid, the forward pass the container runs"
python3 "${REPO_ROOT}/tools/render_student.py" \
    --checkpoint "${CKPT}" --stir-root "${STIR_ROOT}" \
    --clip-id "${CLIP}" --out-dir "${WORK}/st" --iters 4 \
    --grid "${GRID}" --trail "${TRAIL}" --radius 3
gif "${WORK}/st/${CLIP}__student.mp4" "${OUT}/student_tracking.gif"

echo
echo "done:"; ls -la "${OUT}"
echo
echo "Tunables: WIDTH=${WIDTH} MAX_COLORS=${MAX_COLORS} FPS=${FPS} SECONDS_LEN=${SECONDS_LEN} GRID=${GRID}"
