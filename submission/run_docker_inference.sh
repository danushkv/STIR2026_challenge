#!/bin/bash
# Our copy of the upstream run_docker_inference.sh. One image serves all three
# tracks; the track is selected by the entry script + --model class.
#
#   bash run_docker_inference.sh <dataset_dir> <output_dir> <image> <track>
#     track = 2d | 3d | 2d-latency
set -euo pipefail
DATASETLOCATION=$1
OUTPUTLOCATION=$2
IMAGE=$3
TRACK=${4:-2d}

case "$TRACK" in
  # iters=4 on the two ACCURACY tracks, iters=1 on the latency track.
  #
  # i1 vs i4 is a statistical TIE on accuracy (2D -0.0017 CI [-0.0074,+0.0065];
  # 3D -0.0044 CI [-0.0170,+0.0043]) while latency differs 94.4 vs 52.1ms -- so
  # if a single entry had to serve both, iters=1 would win. It does not: these
  # are separate entries, and AJ/ATA/OA carry no latency penalty, so the
  # accuracy entries take the nominally better setting and the latency entry
  # takes the fast one.
  #
  # The deciding factor for 3D: every match_right() measurement (0.6863, and the
  # consistency-check runs) was made at iters=4. iters=1 on the SHIPPED stereo
  # pipeline has never been measured, and shipping an unmeasured config is what
  # this whole exercise exists to avoid.
  2d)         SCRIPT=mono.py;   MODEL=models.mono.student_lt.StudentLTWrapperAccurate ;;
  2d-latency) SCRIPT=mono.py;   MODEL=models.mono.student_lt.StudentLTWrapperFast ;;
  3d)         SCRIPT=stereo.py; MODEL=models.stereo.student_stereo.StudentStereoWrapper ;;
  *) echo "track must be one of: 2d | 3d | 2d-latency"; exit 1 ;;
esac
echo "track=${TRACK}  script=${SCRIPT}  model=${MODEL}"

# --runtime=nvidia only works if the nvidia runtime is REGISTERED with the
# daemon (nvidia-container-toolkit + `nvidia-ctk runtime configure`). The
# upstream script hardcodes it; on a host that only has the --gpus hook it
# fails with "unknown or invalid runtime name: nvidia" before the container
# even starts. Detect it instead: the organisers' host has it (their own script
# uses it) and will take this branch, while a dev box without it still runs.
RUNTIME_ARGS=()
if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
    RUNTIME_ARGS+=(--runtime=nvidia)
else
    echo "note: no 'nvidia' runtime registered -- using --gpus alone"
fi

docker run --rm --gpus "device=0" --net host \
  "${RUNTIME_ARGS[@]+"${RUNTIME_ARGS[@]}"}" --ipc=host \
  --cap-add=CAP_SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --mount src="$DATASETLOCATION",target=/workspace/data,type=bind \
  --mount src="$OUTPUTLOCATION",target=/workspace/output,type=bind \
  "$IMAGE" /bin/bash -c \
  "cd /workspace/stir-challenge-2026-inference && uv run ${SCRIPT} \
     --data_dir /workspace/data --output_dir /workspace/output --model ${MODEL}"
