#!/bin/bash
# Test all three submission images against the 2026 val set BEFORE sending them.
# This is the last chance to catch a wrong checkpoint path, a missing dep, a
# broken stereo match, or non-metric depths -- none of which the organisers will
# debug for you.
#
#   bash test_submission.sh [val_dir]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# The 2026 val set, unpacked outside this repository (it used to be a sibling
# of submission/). Pass a path as $1, or set THIRDPARTY_ROOT / VAL_DIR.
export THIRDPARTY_ROOT="${THIRDPARTY_ROOT:-/mnt/nct-zfs/TCO-Test/venkateda/miccai_challenges/stir}"
export STIRLOADER_ROOT="${STIRLOADER_ROOT:-${THIRDPARTY_ROOT}/STIRLoader}"
export LITETRACKER_ROOT="${LITETRACKER_ROOT:-${THIRDPARTY_ROOT}/lite-tracker-master}"
export STIR_METRICS_ROOT="${STIR_METRICS_ROOT:-${THIRDPARTY_ROOT}/stir-challenge-2026-metrics}"
VAL="${1:-${VAL_DIR:-${THIRDPARTY_ROOT}/val}}"
OUT="${HERE}/test_output"
TEAM="${TEAM:-nct}"
rm -rf "$OUT"; mkdir -p "$OUT"/{2d,2d_latency,3d}
FAIL=0

echo "=== 0. stereo matcher self-test (numpy only, no docker) ==="
python3 "${HERE}/code/models/stereo/student_stereo.py" "${VAL}/00" || FAIL=1

echo; echo "=== 1. weights are present and are the intended checkpoint ==="
ls -la "${HERE}/weights/student.pth"
cat "${HERE}/CHECKPOINT.txt" 2>/dev/null

IMG="stir-challenge-2026-inference-${TEAM}:latest"
if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  echo "image ${IMG} missing -- run build_all.sh first"; exit 1
fi

# Fail fast and loudly if the container cannot see a GPU. Without this the runs
# fall back to CPU, take hours instead of minutes, and -- worse -- the latency
# numbers would be meaningless while still looking like a successful test.
echo; echo "=== 1b. can the container see a GPU? ==="
GPU_ARGS=(--gpus "device=0")
if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
  GPU_ARGS+=(--runtime=nvidia)
fi
if docker run --rm "${GPU_ARGS[@]}" "$IMG" nvidia-smi -L 2>&1 | grep -q "GPU 0"; then
  docker run --rm "${GPU_ARGS[@]}" "$IMG" nvidia-smi -L 2>&1 | sed 's/^/   /'
else
  echo "   CONTAINER CANNOT SEE A GPU."
  echo "   Runs would silently fall back to CPU and the latency numbers would be"
  echo "   meaningless. Install nvidia-container-toolkit and register it:"
  echo "     sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
  echo "   Or test on a host that already has GPU docker working."
  exit 1
fi

# one image, three tracks -- exercise all three entry points
for SPEC in "2d:2d" "2d-latency:2d_latency" "3d:3d"; do
  TRACK="${SPEC%%:*}"; DIR="${SPEC#*:}"
  echo; echo "=== 2.${TRACK}: ${IMG} on ${VAL} ==="
  bash "${HERE}/run_docker_inference.sh" "$VAL" "${OUT}/${DIR}" "$IMG" "$TRACK" \
      > "${OUT}/${DIR}.log" 2>&1
  if [ $? -ne 0 ]; then
    echo "   RUN FAILED -- tail of ${OUT}/${DIR}.log:"; tail -20 "${OUT}/${DIR}.log"; FAIL=1; continue
  fi
  PREDS=$(find "${OUT}/${DIR}" -name preds.json | head -1)
  if [ -z "$PREDS" ]; then echo "   NO preds.json WRITTEN"; FAIL=1; continue; fi
  echo "   wrote $PREDS"
  # the iters actually used is echoed into modeltype by the wrapper, so a
  # mis-wired --model shows up here rather than silently scoring the wrong config
  grep -oE "StudentLiteTracker_iters[0-9]+|StudentStereo" "${OUT}/${DIR}.log" | sort -u | sed 's/^/   modeltype: /'
  python3 - "$PREDS" "$TRACK" <<'PY2' || FAIL=1
import json, sys
import numpy as np
preds, track = json.load(open(sys.argv[1])), sys.argv[2]
ok = True
for s in preds:
    frames = preds[s]
    f0 = frames[sorted(frames, key=int)[0]]
    print(f"   {s}: {len(frames)} frames, keys={sorted(f0)}")
    if "latency_ms" not in f0:
        print("   MISSING latency_ms -- the efficiency metric needs it"); ok = False
    if track == "3d":
        arr = np.array(f0.get("coords", f0.get("positions", [])), dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3:
            Z = arr[:, 2]; Zf = Z[np.isfinite(Z) & (Z > 0)]
            if Zf.size:
                print(f"   Z median {np.median(Zf)*1000:.0f} mm "
                      f"(range {Zf.min()*1000:.0f}-{Zf.max()*1000:.0f} mm)")
                if not (0.005 < float(np.median(Zf)) < 0.5):
                    print("   Z NOT a plausible working distance -- units bug?"); ok = False
            else:
                print("   no finite positive Z"); ok = False
        else:
            print("   3D coords not (N,3)"); ok = False
sys.exit(0 if ok else 1)
PY2
done

echo; echo "=== 3. p95 latency, measured the way run.py measures it ==="
for D in 2d 2d_latency 3d; do
  P=$(find "${OUT}/${D}" -name preds.json | head -1)
  [ -n "$P" ] && python3 "${HERE}/../src/latency_from_preds.py" "$P" 2>/dev/null | tail -3
done

echo; echo "==============================================="
[ "$FAIL" -eq 0 ] && echo "ALL CHECKS PASSED -- safe to run export_all.sh" \
                  || echo "FAILURES ABOVE -- do not submit until resolved"
exit $FAIL
