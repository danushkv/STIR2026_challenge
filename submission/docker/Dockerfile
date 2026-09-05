# ONE image for all three submissions (2D, 3D, 2D+latency).
#
# The image content is identical for every track: same checkpoint, same
# dependency group (`lt` -- student_stereo.py's matcher is pure numpy, so the
# 3D track needs nothing extra), same code. What differs between tracks is only
# the entry script and the --model class, and both are RUN-COMMAND arguments,
# not image content. Refinement iters is pinned per wrapper CLASS rather than
# read from the environment, because run_docker_inference.sh passes no -e flags.
#
#   2D           mono.py    models.mono.student_lt.StudentLTWrapperAccurate  (iters 4)
#   2D+latency   mono.py    models.mono.student_lt.StudentLTWrapperFast      (iters 1)
#   3D           stereo.py  models.stereo.student_stereo.StudentStereoWrapper (iters 4)
ARG BASE=stir-challenge-2026-inference:base
FROM ${BASE}

# The weights MUST live inside the image: no -e flags are passed, only the
# dataset and output dirs are mounted, and there is no network at eval time.
# student_lt.py fails loudly if this file is absent rather than scoring garbage.
COPY weights/student.pth /workspace/weights/student.pth
COPY code/models/mono/student_lt.py \
     /workspace/stir-challenge-2026-inference/models/mono/student_lt.py
COPY code/models/stereo/student_stereo.py \
     /workspace/stir-challenge-2026-inference/models/stereo/student_stereo.py

ENV STUDENT_LT_WEIGHTS=/workspace/weights/student.pth
ENV STUDENT_LT_WINDOW_LEN=16
# Fallback only -- the shipped wrapper classes pin their own iters. This just
# means a bare StudentLTWrapper also gets the accuracy setting.
ENV STUDENT_LT_ITERS=4
