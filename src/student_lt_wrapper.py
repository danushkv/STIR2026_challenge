"""Run YOUR fine-tuned CoTracker3 student under the LiteTracker runtime, for the
STIR 2026 efficiency benchmark.

NO CHECKPOINT CONVERSION IS NEEDED -- and that is the whole point of this file.
LiteTracker is not a different model: it is the SAME architecture as
CoTrackerThreeOnline (verified module-by-module -- identical
fnet=BasicEncoder(3,128,stride), EfficientUpdateFormer(space_depth=3,
time_depth=3, input_dim=1110, hidden_size=384, output_dim=4, ...),
corr_mlp=Mlp(49*49,384,256), and one `time_emb` buffer; no other parameters in
either). What differs is the FORWARD PASS: LiteTracker is causal/streaming --
one frame per call, with persistent coords/vis/conf/corr_embs buffers, a
track_feat_cache, and an EMA-flow warm start -- instead of CoTracker3's sliding
window over a whole clip. Same weights, cheaper inference.

Proof it's weight-compatible, not just similar: the OFFICIAL challenge repo
(stir-challenge-2026-inference/models/mono/lt.py) loads Meta's stock
`scaled_online.pth` CoTracker3 checkpoint into `LiteTracker()` with
`load_state_dict(...)` at its default strict=True. Your fine-tuned checkpoint
has exactly the same keys, because model*.py fine-tunes that same
CoTrackerThreeOnline module.

So: your checkpoint IS a LiteTracker checkpoint already. Just load it.

THE EFFICIENCY KNOB: LiteTracker defaults to `iters=1` (one refinement
iteration per frame) whereas you TRAINED with train_iters=4. That default is
where much of the speedup comes from, but it is a real accuracy/latency
tradeoff and is NOT what the weights were optimized under -- benchmark
iters=1 vs 2 vs 4 rather than assuming 1 is free. Set via --iters / env
STUDENT_LT_ITERS.

RESOLUTION: unlike the training path (which resized to model_resolution and
rescaled coords by hand), LiteTracker.forward does this INTERNALLY -- feed it
native-resolution frames and native-resolution queries, and it returns coords
already rescaled back to native. Note model_resolution is (H, W) in the code
despite the docstring in lite_tracker.py saying "(width, height)"; the default
(384, 512) matches your training config either way.

--- 1. Verify your checkpoint loads (run this first) ---
    python student_lt_wrapper.py --weights /path/to/student_last.pth

--- 2. Use it in the official STIR 2026 mono benchmark ---
    export STUDENT_LT_WEIGHTS=/path/to/student_last.pth
    export STUDENT_LT_ITERS=1
    cd ../stir-challenge-2026-inference
    uv run mono.py --data_dir <dataset> --output_dir results \\
        --model student_lt_wrapper.StudentLTWrapper
    (ensure this file is importable from there, e.g. copy it in or set PYTHONPATH)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

# lite_tracker.py imports its own deps ABSOLUTELY ("from src.model_utils import ...",
# "from src.model_blocks import ..."), so the package root that must be on sys.path is
# lite-tracker-master/ (the PARENT of src/), not src/ itself. Adding only src/ makes
# `import lite_tracker` succeed and then fail inside it with
# "ModuleNotFoundError: No module named 'src'". Add both: the repo root so `src.*`
# resolves (src/ has no __init__.py, but PEP-420 namespace packages handle that), and
# src/ so environments that install/expose lite_tracker as a top-level module still work.
from _thirdparty import add_to_syspath, find_repo

_LITE_TRACKER_ROOT = find_repo(["lite-tracker-master", "litetracker-master"],
                               "LITETRACKER_ROOT", "src/lite_tracker.py", "LiteTracker")
add_to_syspath(_LITE_TRACKER_ROOT, _LITE_TRACKER_ROOT / "src")

try:  # noqa: E402  (after sys.path insert)
    from src.lite_tracker import LiteTracker
except ModuleNotFoundError:  # installed/exposed as a top-level module instead
    from lite_tracker import LiteTracker


def load_student_as_litetracker(weights, window_len: int = 16, iters: int = 1,
                                device=None, strict: bool = True):
    """Instantiate LiteTracker and load a CoTracker3-online checkpoint into it.

    weights: path to model*.py output (saved as {"model": state_dict})
             or any raw CoTracker3-online state dict (e.g. stock scaled_online.pth).
    window_len: MUST match what the checkpoint was trained with (16 for
             scaled_online and for train.yaml's default) -- it sets the
             `time_emb` buffer's shape, so a mismatch fails the strict load.
    iters: refinement iterations per frame at inference. LiteTracker's own
             default is 1; you trained at 4. Benchmark, don't assume.
    Returns (model, info) with info describing the load for verification.
    """
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"checkpoint not found: {weights}")

    model = LiteTracker(window_len=window_len, iters=iters)

    state_dict = torch.load(weights, map_location="cpu")
    wrapped = isinstance(state_dict, dict) and "model" in state_dict
    if wrapped:
        state_dict = state_dict["model"]

    missing, unexpected = [], []
    if strict:
        model.load_state_dict(state_dict)          # raises on any mismatch
    else:
        res = model.load_state_dict(state_dict, strict=False)
        missing, unexpected = list(res.missing_keys), list(res.unexpected_keys)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device=device)
    model.eval()

    info = {
        "weights": str(weights),
        "wrapped_in_model_key": wrapped,
        "n_ckpt_keys": len(state_dict),
        "n_model_keys": len(model.state_dict()),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "window_len": window_len,
        "iters": iters,
        "device": str(device),
    }
    return model, info


class StudentLTWrapper:
    """Drop-in replacement for the official LTWrapper, using YOUR checkpoint.

    Implements the exact interface stir-challenge-2026-inference/mono.py needs:
    a no-arg constructor plus track(frame, queries=None) -> (coords [N,2] float32,
    visibs [N,] bool). Because the constructor takes no arguments (mono.py does
    `tracker_cls()` fresh per sequence), configuration comes from env vars:

        STUDENT_LT_WEIGHTS     path to the checkpoint (REQUIRED)
        STUDENT_LT_ITERS       refinement iters per frame (default 1)
        STUDENT_LT_WINDOW_LEN  must match training (default 16)

    A fresh instance per sequence is what resets the streaming state -- LiteTracker
    keeps per-video buffers, so reusing one instance across sequences would leak
    the previous video's tracks. mono.py already constructs one per sequence.
    """

    def __init__(self):
        weights = os.environ.get("STUDENT_LT_WEIGHTS", "")
        if not weights:
            raise SystemExit(
                "STUDENT_LT_WEIGHTS is not set -- point it at your fine-tuned "
                "checkpoint, e.g. export STUDENT_LT_WEIGHTS=/path/to/student_last.pth")
        iters = int(os.environ.get("STUDENT_LT_ITERS", "1"))
        window_len = int(os.environ.get("STUDENT_LT_WINDOW_LEN", "16"))

        self.modeltype = "StudentLiteTracker"
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.dtype = (
            torch.bfloat16
            if self.device == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )

        self.model, _ = load_student_as_litetracker(
            weights, window_len=window_len, iters=iters, device=self.device)
        self.model.init_video_online_processing()   # fresh streaming state

        self.queries = None
        self.is_first_frame = True

    @torch.no_grad()
    def _preprocess(self, im: np.ndarray) -> torch.Tensor:
        """(H,W,3) uint8 RGB -> [1,3,H,W] at model dtype/device, still 0-255 and
        still NATIVE resolution (LiteTracker.forward resizes internally)."""
        return (
            torch.from_numpy(im)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(dtype=self.dtype, device=self.device)
        )

    @torch.no_grad()
    def track(self, frame: np.ndarray, queries: np.ndarray | None = None):
        """One streaming step. queries [N,2] float32 (native px) is REQUIRED on
        the first call and ignored after. Returns (coords [N,2] float32,
        visibs [N,] bool) in native pixels."""
        with torch.autocast(device_type=self.device, dtype=self.dtype, enabled=True):
            frame_t = self._preprocess(frame)

            if self.is_first_frame:
                if queries is None:
                    raise ValueError("queries must be provided on the first track() call")
                q = torch.from_numpy(np.asarray(queries, dtype=np.float32))
                q = q.unsqueeze(0).to(device=self.device)          # [1,N,2]
                # LiteTracker expects [B,N,3] = (query_frame_idx, x, y)
                self.queries = torch.cat([torch.zeros_like(q[:, :, :1]), q], dim=-1)
                self.is_first_frame = False
                self.model(frame_t, queries=self.queries)
                # frame 0 is the reference frame: report the queries themselves,
                # matching the official LTWrapper (and mono.py skips frame 0 in
                # scoring anyway -- see stir-challenge-2026-metrics/run.py).
                n = self.queries.shape[1]
                return np.asarray(queries, dtype=np.float32), np.ones(n, dtype=bool)

            coords, vis, *_ = self.model(frame_t, queries=self.queries)
            # coords [B,T,N,2] with T==1, already rescaled to native px by forward()
            return (coords[0, 0].float().cpu().numpy().astype(np.float32),
                    vis[0, 0].cpu().numpy().astype(bool))


def _verify(args):
    """Load the checkpoint into LiteTracker and report whether it's ready."""
    print(f"Loading {args.weights}\n  window_len={args.window_len} iters={args.iters}")
    try:
        model, info = load_student_as_litetracker(
            args.weights, window_len=args.window_len, iters=args.iters,
            device="cpu", strict=True)
    except RuntimeError as e:
        print("\nSTRICT LOAD FAILED -- retrying non-strict to show the mismatch:\n")
        print(f"  {e}\n")
        _, info = load_student_as_litetracker(
            args.weights, window_len=args.window_len, iters=args.iters,
            device="cpu", strict=False)
        print(f"  missing keys   ({len(info['missing_keys'])}): {info['missing_keys'][:10]}")
        print(f"  unexpected keys({len(info['unexpected_keys'])}): {info['unexpected_keys'][:10]}")
        print("\nIf window_len differs from training, fix that first -- it changes "
              "the time_emb buffer shape.")
        return 1

    n_params = sum(p.numel() for p in model.parameters())
    print("\n=== checkpoint IS LiteTracker-compatible (strict load succeeded) ===")
    print(f"  checkpoint keys : {info['n_ckpt_keys']} (wrapped in 'model': {info['wrapped_in_model_key']})")
    print(f"  model keys      : {info['n_model_keys']}")
    print(f"  parameters      : {n_params:,}")
    print(f"  time_emb shape  : {tuple(model.time_emb.shape)}")
    print("\nNo conversion needed. Use it in the 2026 benchmark with:")
    print(f"  export STUDENT_LT_WEIGHTS={Path(args.weights).resolve()}")
    print(f"  export STUDENT_LT_ITERS={args.iters}")
    print("  uv run mono.py --data_dir <dataset> --output_dir results \\")
    print("      --model student_lt_wrapper.StudentLTWrapper")
    print("\nThen benchmark iters=1/2/4 -- you trained at 4, LiteTracker defaults to 1.")
    return 0


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True,
                   help="fine-tuned checkpoint from model*.py (or scaled_online.pth)")
    p.add_argument("--window-len", type=int, default=16,
                   help="must match training (train.yaml window_len; default 16)")
    p.add_argument("--iters", type=int, default=1,
                   help="refinement iters per frame at inference (LiteTracker default 1; "
                        "you trained at train_iters=4)")
    raise SystemExit(_verify(p.parse_args()))


if __name__ == "__main__":
    main()
