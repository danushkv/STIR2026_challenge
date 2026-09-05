#!/bin/bash
# Build the base image once, then the three submission images on top of it.
#   bash build_all.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# The organisers' inference repo, cloned OUTSIDE this repository. It used to
# be a sibling of submission/; THIRDPARTY_ROOT now says where it lives.
export THIRDPARTY_ROOT="${THIRDPARTY_ROOT:-/mnt/nct-zfs/TCO-Test/venkateda/miccai_challenges/stir}"
export STIRLOADER_ROOT="${STIRLOADER_ROOT:-${THIRDPARTY_ROOT}/STIRLoader}"
export LITETRACKER_ROOT="${LITETRACKER_ROOT:-${THIRDPARTY_ROOT}/lite-tracker-master}"
export STIR_METRICS_ROOT="${STIR_METRICS_ROOT:-${THIRDPARTY_ROOT}/stir-challenge-2026-metrics}"
UPSTREAM="${UPSTREAM:-${THIRDPARTY_ROOT}/stir-challenge-2026-inference}"
TEAM="${TEAM:-nct}"

# Proxy: the BASE image needs the network (apt-get, the uv installer, a git
# clone, then uv sync). HTTP_PROXY/HTTPS_PROXY/NO_PROXY are Docker PREDEFINED
# build args, so they work without any ARG line in the Dockerfile and are NOT
# baked into the resulting image -- which is what you want, since the eval
# environment has no proxy and no network. Both cases are passed because many
# tools read only the lowercase form.
PROXY_ARGS=()
for V in HTTP_PROXY HTTPS_PROXY NO_PROXY; do
    L=$(echo "$V" | tr "A-Z" "a-z")
    # accept either case from the environment
    VAL="${!V:-}"; [ -z "$VAL" ] && VAL="${!L:-}"
    if [ -n "$VAL" ]; then
        PROXY_ARGS+=(--build-arg "${V}=${VAL}" --build-arg "${L}=${VAL}")
    fi
done
if [ ${#PROXY_ARGS[@]} -gt 0 ]; then
    echo "proxy build args: ${PROXY_ARGS[*]}"
else
    echo "no HTTP_PROXY/HTTPS_PROXY/NO_PROXY in the environment -- building direct"
fi

echo "=== base image (upstream, unmodified) ==="
docker build "${PROXY_ARGS[@]+"${PROXY_ARGS[@]}"}" \
    -t stir-challenge-2026-inference:base "${UPSTREAM}"

echo; echo "=== submission image (one, serves all three tracks) ==="
docker build "${PROXY_ARGS[@]+"${PROXY_ARGS[@]}"}" -f "${HERE}/docker/Dockerfile" \
    -t "stir-challenge-2026-inference-${TEAM}:latest" "${HERE}"

echo; docker images | grep "stir-challenge-2026-inference" || true
echo; echo "next: bash test_submission.sh"
