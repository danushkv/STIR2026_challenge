"""STIR 2026 mono submission: our pseudo-label-distilled CoTracker3 student,
run under the LiteTracker streaming runtime.

Same architecture as the reference LTWrapper (models/mono/lt.py) -- LiteTracker
IS CoTrackerThreeOnline's architecture with a causal/streaming forward pass, so
a fine-tuned CoTracker3-online checkpoint loads into it directly, no conversion.
The only difference from lt.py is that we load OUR fine-tuned weights instead of
downloading Meta's stock scaled_online.pth.

DOCKER SUBMISSION NOTES (why the path is hardcoded):
  run_docker_inference.sh passes NO -e environment variables and mounts only the
  dataset + output dirs, so the checkpoint must already be inside the image at a
  fixed path. WEIGHTS_PATH below must point at where the Dockerfile COPYs it.
  The env var is only an optional override for local testing outside Docker.

  There is also no network guarantee at eval time -- unlike lt.py, this file
  never downloads anything. If the weights are missing it fails loudly at
  construction rather than silently scoring garbage.

Interface required by mono.py: no-arg constructor, and
    track(frame: (H,W,3) uint8 RGB, queries: (N,2) float32 | None)
        -> (coords (N,2) float32, visibs (N,) bool)
A fresh instance is constructed per sequence by mono.py, which is what resets
LiteTracker's per-video streaming buffers.
"""

import os
from pathlib import Path

import numpy as np
import torch
from lite_tracker import LiteTracker

# Baked into the image by the Dockerfile. Change here AND in the Dockerfile COPY
# if you relocate it.
WEIGHTS_PATH = os.environ.get("STUDENT_LT_WEIGHTS", "/workspace/weights/student.pth")

# Must match the window_len the checkpoint was trained with (train_student.yaml
# window_len: 16). It sets the time_emb buffer shape -- a mismatch fails the load.
WINDOW_LEN = int(os.environ.get("STUDENT_LT_WINDOW_LEN", "16"))

# Refinement iterations per frame. LiteTracker's own default is 1; we trained at
# train_iters=4. This is the accuracy-vs-latency knob the efficiency benchmark
# scores -- set it to whatever your own benchmarking picked.
ITERS = int(os.environ.get("STUDENT_LT_ITERS", "4"))


class StudentLTWrapper:
    # Refinement iterations per frame. `None` -> take the module-level ITERS
    # (i.e. the STUDENT_LT_ITERS env var, default 4). Subclasses below pin it
    # instead, so ONE image can serve every track: which operating point runs is
    # chosen by the --model argument in the run command, not by an env var the
    # organisers would have to remember to set. run_docker_inference.sh passes
    # no -e flags, so anything env-dependent is a silent failure waiting to
    # happen.
    ITERS = None

    def __init__(self):
        self.modeltype = "StudentLiteTracker"

        weights_path = Path(WEIGHTS_PATH)
        assert weights_path.exists(), (
            f"checkpoint not found at {weights_path} -- it must be baked into the "
            f"Docker image (see the Dockerfile COPY step); no download is attempted."
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.dtype = (
            torch.bfloat16
            if self.device == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )

        iters = ITERS if self.ITERS is None else self.ITERS
        self.modeltype = f"{self.modeltype}_iters{iters}"
        self.model = LiteTracker(window_len=WINDOW_LEN, iters=iters)
        with open(weights_path, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        if "model" in state_dict:          # train_student*.py saves {"model": ...}
            state_dict = state_dict["model"]
        self.model.load_state_dict(state_dict)

        self.model = self.model.to(device=self.device)
        self.model.eval()
        self.model.init_video_online_processing()

        self.queries = None
        self.is_first_frame = True

    @torch.no_grad()
    def __preprocess_frame(self, im: np.ndarray) -> torch.Tensor:
        """(H,W,3) uint8 RGB -> [1,3,H,W] on device, still 0-255 and still NATIVE
        resolution: LiteTracker.forward resizes to model_resolution internally and
        rescales its output coords back to native, so do NOT pre-resize here."""
        return (
            torch.from_numpy(im)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(dtype=self.dtype, device=self.device)
        )

    @torch.no_grad()
    def track(self, frame: np.ndarray, queries: np.ndarray = None):
        with (
            torch.no_grad(),
            torch.autocast(device_type=self.device, dtype=self.dtype, enabled=True),
        ):
            frame_t = self.__preprocess_frame(frame)

            if self.is_first_frame:
                if queries is None:
                    raise ValueError("queries must be provided on the first track() call")
                q = torch.from_numpy(np.asarray(queries, dtype=np.float32))
                q = q.unsqueeze(0).to(device=self.device)               # [1,N,2]
                # LiteTracker wants [B,N,3] = (query_frame_idx, x, y)
                self.queries = torch.cat([torch.zeros_like(q[:, :, :1]), q], dim=-1)
                self.is_first_frame = False
                self.model(frame_t, queries=self.queries)
                # Frame 0 is the reference frame: echo the queries back, matching
                # lt.py. The metrics runner skips frame 0 anyway.
                n = np.asarray(queries).shape[0]
                return (np.asarray(queries, dtype=np.float32),
                        np.ones(n, dtype=bool))

            coords, vis, *_ = self.model(frame_t, queries=self.queries)
            # coords [B,T,N,2] with T==1, already in native pixels
            return (coords[0, 0].float().cpu().numpy().astype(np.float32),
                    vis[0, 0].cpu().numpy().astype(bool))


class StudentLTWrapperAccurate(StudentLTWrapper):
    """2D accuracy track: iters=4, the setting the checkpoint was trained at.
    2D delta_avg 0.8103, ~94ms p95."""
    ITERS = 4


class StudentLTWrapperFast(StudentLTWrapper):
    """2D + latency track: iters=1. delta_avg 0.8085 (-0.0018, inside noise)
    for 52ms p95 instead of 94ms -- the accuracy cost is noise, the latency
    saving is not."""
    ITERS = 1
