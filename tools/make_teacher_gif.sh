#!/bin/bash
# Render a synchronized six-teacher comparison and compress it for GitHub.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env}"
if [ -f "${ENV_FILE}" ]; then set -a; source "${ENV_FILE}"; set +a; fi
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
RAW_TRACKS="${RAW_TRACKS:-${DATA_ROOT}/STIRprocessed/STIROrig_tracks}"
STIR_ROOT="${STIR_ROOT:-${DATA_ROOT}/STIRDataset}"

CLIP="${CLIP:-4__left__seq00}"
OUT="${OUT:-${REPO_ROOT}/assets/teacher_comparison.gif}"
WIDTH="${WIDTH:-900}"
FPS="${FPS:-7}"
SECONDS_LEN="${SECONDS_LEN:-6}"
MAX_COLORS="${MAX_COLORS:-96}"
PANEL_WIDTH="${PANEL_WIDTH:-360}"
TRAIL="${TRAIL:-12}"
MAX_POINTS="${MAX_POINTS:-16}"

[ -x "${PYTHON_BIN}" ] || { echo "Python is not executable: ${PYTHON_BIN}"; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg not found"; exit 1; }
[ -d "${RAW_TRACKS}" ] || { echo "raw tracks not found: ${RAW_TRACKS}"; exit 1; }
[ -d "${STIR_ROOT}" ] || { echo "STIR root not found: ${STIR_ROOT}"; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/stir-teachers.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT
mkdir -p "$(dirname "${OUT}")"

"${PYTHON_BIN}" "${REPO_ROOT}/tools/render_teachers.py" \
  --raw-tracks-root "${RAW_TRACKS}" \
  --stir-root "${STIR_ROOT}" \
  --clip-id "${CLIP}" \
  --out "${WORK}/teachers.mp4" \
  --fps "${FPS}" --panel-width "${PANEL_WIDTH}" \
  --trail "${TRAIL}" --max-points "${MAX_POINTS}"

VF="fps=${FPS},scale=${WIDTH}:-1:flags=lanczos"
ffmpeg -v error -y -t "${SECONDS_LEN}" -i "${WORK}/teachers.mp4" \
  -vf "${VF},palettegen=max_colors=${MAX_COLORS}:stats_mode=diff" "${WORK}/palette.png"
ffmpeg -v error -y -t "${SECONDS_LEN}" -i "${WORK}/teachers.mp4" \
  -i "${WORK}/palette.png" -loop 0 \
  -lavfi "${VF}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3" "${OUT}"

BYTES="$(stat -c '%s' "${OUT}")"
echo "wrote $(du -h "${OUT}" | cut -f1)  ${OUT}"
if [ "${BYTES}" -gt 5000000 ]; then
  echo "warning: over 5 MB; retry with WIDTH=760 MAX_COLORS=64 FPS=6"
elif [ "${BYTES}" -gt 3500000 ]; then
  echo "large for a README; MAX_COLORS=64 usually helps most"
fi
