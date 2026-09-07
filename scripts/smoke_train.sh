#!/bin/bash
# Run one real GPU optimizer step, save it, and strictly reload the checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env}"
if [ -f "${ENV_FILE}" ]; then set -a; source "${ENV_FILE}"; set +a; fi
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
OUT_DIR="${OUT_DIR:-${TMPDIR:-/tmp}/stir2026-smoke-runs}"
RUN_NAME="${RUN_NAME:-smoke_$(date +%Y%m%d_%H%M%S)}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/smoke.yaml}"
RUN_DIR="${OUT_DIR}/${RUN_NAME}"
LOG="${OUT_DIR}/${RUN_NAME}.log"

[ -x "${PYTHON_BIN}" ] || { echo "Python is not executable: ${PYTHON_BIN}"; exit 1; }
mkdir -p "${OUT_DIR}"

REPO_ROOT="${REPO_ROOT}" PYTHON_BIN="${PYTHON_BIN}" CONFIG="${CONFIG}" \
  OUT_DIR="${OUT_DIR}" RUN_NAME="${RUN_NAME}" \
  bash "${REPO_ROOT}/scripts/train.sh" "$@" 2>&1 | tee "${LOG}"

grep -Eq '^\[[^]]+\] loss [0-9]' "${LOG}" || {
  echo "no finite training loss found in ${LOG}"
  exit 1
}
if grep -Eqi 'loss (nan|inf)' "${LOG}"; then
  echo "non-finite loss found in ${LOG}"
  exit 1
fi
for file in config.yaml student_last.pth student_e1.pth; do
  [ -s "${RUN_DIR}/${file}" ] || { echo "missing output: ${RUN_DIR}/${file}"; exit 1; }
done

"${PYTHON_BIN}" "${REPO_ROOT}/src/student_lt_wrapper.py" \
  --weights "${RUN_DIR}/student_last.pth" --window-len 16 --iters 1

echo
echo "TRAINING SMOKE PASSED"
echo "checkpoint: ${RUN_DIR}/student_last.pth"
echo "log:        ${LOG}"
