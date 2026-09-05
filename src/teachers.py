"""Teacher tracker wrappers.

Every teacher exposes the SAME interface so the ensemble runner and verifier are
agnostic to the underlying model:

    result = teacher.track(frames, queries)          # forward pass
    fwd, bwd = teacher.track_cycle(frames, queries)  # forward + aligned backward

`frames`  : [T, H, W, 3] uint8/float
`queries` : [N, 2] (x, y) at frame 0
returns   : TrackResult

The concrete classes are thin adapters. The Track-On-R repo
(github.com/gorkaydemir/track_on) already ships clean inference wrappers for
CoTracker3, LocoTrack, TAPIR/BootsTAPIR, AllTracker, etc. — the fastest path is
to call those inside `track()`. MFT (github.com/serycjon/MFT) needs a RAFT-style
flow backend and is the one heavier dependency; it is optional (see run.py).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from trajectories import TrackResult
from config import TeacherConfig


class Teacher:
    """Base adapter. Subclasses implement `_forward` only."""

    def __init__(self, cfg: TeacherConfig):
        self.cfg = cfg
        self.name = cfg.name

    # --- implement this per tracker ---------------------------------------- #
    def _forward(self, frames: np.ndarray, queries: np.ndarray) -> TrackResult:
        raise NotImplementedError

    # --- shared logic ------------------------------------------------------ #
    def track(self, frames: np.ndarray, queries: np.ndarray) -> TrackResult:
        return self._forward(frames, queries)

    def track_cycle(self, frames: np.ndarray,
                    queries: np.ndarray) -> Tuple[TrackResult, TrackResult]:
        """Forward pass, plus a backward pass aligned to forward time.

        Backward = reverse the clip, seed with the forward endpoint, track, then
        flip the result back so index t means the same frame in both passes.
        Two inferences total (not T), which is the cheap cycle-consistency check.
        """
        fwd = self._forward(frames, queries)                 # [N, T, 2]
        end_query = fwd.coords[:, -1, :]                      # [N, 2]
        rev = self._forward(frames[::-1].copy(), end_query)  # tracked on reversed clip
        bwd = TrackResult(
            coords=rev.coords[:, ::-1, :].copy(),            # flip back to forward time
            visibility=rev.visibility[:, ::-1].copy(),
            name=self.name,
            confidence=rev.confidence[:, ::-1].copy(),
        )
        return fwd, bwd


# --------------------------------------------------------------------------- #
# Concrete adapters — fill in the marked lines with the real repo calls.
# --------------------------------------------------------------------------- #
    # (removed: CoTracker3Teacher and LocoTrackTeacher stubs. Both are now
    # sourced from track_on's own ensemble wrappers via TrackOnRTeacher below --
    # track_on already wraps CoTracker3 (auto-downloads its checkpoint through
    # torch.hub, no separate co-tracker install needed for the TEACHER role)
    # and LocoTrack with the exact same uniform interface as BootsTAPIR and
    # AllTracker. This means 4 of 6 teachers now come from ONE environment.
    #
    # NOTE: the standalone facebookresearch/co-tracker repo is still needed
    # separately -- not for this teacher role, but because train.py
    # fine-tunes CoTracker3's actual trainable weights, which requires its real
    # model code, not just this frozen inference wrapper.


class MFTIQTeacher(Teacher):
    """MFTIQ (Serych, Neoral, Matas, WACV 2025) -- extends MFT with an
    independent matching-quality module; supports swappable flow backbones
    (RAFT, RoMa, FlowFormer++, GMFlow, ...). Ships a pretrained checkpoint via
    `download_model.sh` in github.com/serycjon/MFTIQ -- no training needed to
    use as a teacher.

    IMPORTANT: in the published STIR 2024 results, off-the-shelf MFTIQ did NOT
    surpass the plain MFT baseline on this domain. Include it for ensemble
    diversity, not because it's presumed stronger than MFT -- the verifier's
    endpoint-anchoring + cycle-consistency should down-weight it automatically
    on the clips/frames where it's actually wrong.

    Same API shape as MFTTeacher (same author, same init/track pattern).
    """
    def __init__(self, cfg: TeacherConfig):
        super().__init__(cfg)
        import sys
        if cfg.repo_root and cfg.repo_root not in sys.path:
            sys.path.insert(0, cfg.repo_root)
        from MFTIQ.config import load_config  # noqa: import after sys.path insert
        config_path = cfg.config_path or "configs/MFTIQ4_RAFT_200k_cfg.py"
        import os
        orig_cwd = os.getcwd()
        try:
            if cfg.repo_root:
                os.chdir(cfg.repo_root)
            self._config = load_config(config_path)
        finally:
            os.chdir(orig_cwd)
        self._tracker_class = self._config.tracker_class

    def _forward(self, frames, queries):
        import os
        import torch
        from MFTIQ.point_tracking import convert_to_point_tracking
        from MFTIQ.utils.misc import ensure_numpy

        orig_cwd = os.getcwd()
        try:
            if self.cfg.repo_root:
                os.chdir(self.cfg.repo_root)
            tracker = self._tracker_class(self._config)
            q = torch.from_numpy(queries).float().cuda()

            coords_per_frame, occl_per_frame = [], []
            for i, frame in enumerate(frames):
                meta = tracker.init(frame) if i == 0 else tracker.track(frame)
                coords, occlusions = convert_to_point_tracking(meta.result, q)
                coords_per_frame.append(ensure_numpy(coords))
                occl_per_frame.append(ensure_numpy(occlusions))
        finally:
            os.chdir(orig_cwd)

        coords = np.stack(coords_per_frame, axis=1).astype(np.float32)  # [N, T, 2]
        occl = np.stack(occl_per_frame, axis=1).astype(np.float32)      # [N, T]
        return TrackResult(coords, visibility=1.0 - occl, name=self.name)


class TrackOnRTeacher(Teacher):
    """Generic adapter over Track-On-R's uniformly-wrapped baseline trackers
    (github.com/gorkaydemir/track_on/ensemble/). Confirmed directly from
    ensemble/ensemble_predictor.py and the individual predictor classes: every
    model there shares ONE call signature --
        (B,T,C,H,W) 0-255 video + (B,N,3) [t,x,y] queries
        -> (B,T,N,2) tracks, (B,T,N) visibility
    -- so one adapter class covers all of them; only construction differs.
    None need training; all use the authors' official checkpoints.

    cfg.repo_root  -> path to the cloned track_on repo (added to sys.path)
    cfg.checkpoint -> path to the .pt/.pth checkpoint for THIS specific model

    Supported `kind`s: "cotracker3" (auto-downloads its checkpoint via
    torch.hub; pass cfg.checkpoint to override with a fine-tuned checkpoint
    instead, e.g. litetracker_finetuned.pth -- leave unset to use the stock
    weights), "locotrack" (needs cfg.checkpoint,
    e.g. locotrack_base.ckpt), "bootstapir"/"tapir" (same predictor class --
    it switches architecture variant based on whether "bootstapir" appears in
    the checkpoint FILENAME, so use the real bootstapir_checkpoint_v2.pt
    name), and "alltracker". TAPNext/BootsTAPNext/Anthro-LocoTrack follow the
    same pattern if you want to add them later.
    """
    def __init__(self, cfg: TeacherConfig, kind: str):
        super().__init__(cfg)
        import sys
        if cfg.repo_root and cfg.repo_root not in sys.path:
            sys.path.insert(0, cfg.repo_root)

        self.kind = kind
        if kind in ("bootstapir", "tapir"):
            from ensemble.bootstapir.bootstapir_predictor import TAPIRPredictor
            self._model = TAPIRPredictor(cfg.checkpoint).cuda().eval()
        elif kind == "alltracker":
            from ensemble.alltracker.alltracker_predictor import AllTrackerPredictor
            self._model = AllTrackerPredictor(checkpoint_path=cfg.checkpoint).cuda().eval()
        elif kind == "cotracker3":
            from ensemble.cotracker import CoTracker_Predictor
            self._model = CoTracker_Predictor(windowed=True, checkpoint_path=cfg.checkpoint or None).cuda().eval()
        elif kind == "locotrack":
            from ensemble.locotrack.locotrack_predictor import LocoTrackPredictor
            self._model = LocoTrackPredictor(cfg.checkpoint).cuda().eval()
        else:
            raise ValueError(f"unknown track_on kind {kind!r}")

    def _forward(self, frames, queries):
        import torch
        video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().cuda()  # (1,T,3,H,W)
        N = len(queries)
        t0 = torch.zeros((N, 1), dtype=torch.float32)
        q = torch.cat([t0, torch.from_numpy(queries).float()], dim=1)[None].cuda()  # (1,N,3) [t,x,y]

        with torch.no_grad():
            tracks, vis = self._model(video, q)  # (1,T,N,2), (1,T,N)

        coords = tracks[0].permute(1, 0, 2).cpu().numpy().astype(np.float32)        # [N,T,2]
        visibility = vis[0].permute(1, 0).float().cpu().numpy().astype(np.float32)  # [N,T]
        return TrackResult(coords, visibility, name=self.name)


def _make_trackon_teacher(kind: str):
    def factory(cfg: TeacherConfig) -> Teacher:
        return TrackOnRTeacher(cfg, kind=kind)
    return factory


class TrackOnTeacher(Teacher):
    """Track-On2 / Track-On-R's OWN tracker (github.com/gorkaydemir/track_on,
    model/trackon_predictor.py::Predictor) -- NOT to be confused with
    TrackOnRTeacher above, which wraps track_on's *baseline* adapters
    (CoTracker3, LocoTrack, ...) from its ensemble/ dir. This class wraps
    track_on's own model.

    Registered under both "trackon2" and "trackon_r": identical architecture,
    only the checkpoint differs -- Track-On-R is Track-On2 fine-tuned on
    real-world video via verifier-guided pseudo-labeling (i.e. what this whole
    pipeline produces), so it's a meaningful teacher/consistency check once
    you have a checkpoint for it. Pick the registry name matching whichever
    checkpoint you're pointing --checkpoint at.

    `Predictor.forward` internally resizes frames to its configured
    input_size and rescales predictions back to the input resolution, and
    resets its internal memory at the start of each call -- so no manual
    resize/rescale/reset is needed here (confirmed against the repo's demo.py).

    Needs the DINOv3 backbone weights (facebook/dinov3-vits16plus-pretrain-
    lvd1689m by default) reachable via HF `AutoModel.from_pretrained` --
    gated on HF Hub, so either `huggingface-cli login` with access granted,
    or set DINOV3_LOCAL_DIR to a local copy (see dinov3_vit_adapter.py).

    cfg.repo_root  -> path to the cloned track_on repo
    cfg.checkpoint -> track_on_r.pt or trackon2_dinov3_checkpoint.pt (required)
    """
    def __init__(self, cfg: TeacherConfig):
        super().__init__(cfg)
        if not cfg.checkpoint:
            raise ValueError(
                f"teacher {cfg.name!r} needs --checkpoint pointing at "
                "track_on_r.pt or trackon2_dinov3_checkpoint.pt"
            )
        import sys
        if cfg.repo_root and cfg.repo_root not in sys.path:
            sys.path.insert(0, cfg.repo_root)
        from model.trackon_predictor import Predictor  # noqa: import after sys.path insert
        self._model = Predictor(checkpoint_path=cfg.checkpoint).cuda().eval()

    def _forward(self, frames, queries):
        import torch
        video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().cuda()  # (1,T,3,H,W), 0-255
        N = len(queries)
        t0 = torch.zeros((N, 1), dtype=torch.float32)
        q = torch.cat([t0, torch.from_numpy(queries).float()], dim=1)[None].cuda()  # (1,N,3) [t,x,y]

        with torch.no_grad():
            tracks, vis = self._model(video, q)  # (1,T,N,2), (1,T,N) bool

        coords = tracks[0].permute(1, 0, 2).cpu().numpy().astype(np.float32)        # [N,T,2]
        visibility = vis[0].permute(1, 0).float().cpu().numpy().astype(np.float32)  # [N,T]
        return TrackResult(coords, visibility, name=self.name)


class MFTTeacher(Teacher):
    """Optical-flow-chaining tracker (Neoral & Serych, WACV 2024). Ships a ready
    checkpoint -- checkpoints/raft-things-sintel-kubric-splitted-occlusion-
    uncertainty-non-occluded-base-sintel.pth in github.com/serycjon/MFT -- so no
    training is needed to use it. Accurate, slower -> teacher only, never the student.

    Setup: `git clone https://github.com/serycjon/MFT` and set cfg.repo_root to
    that path; cfg.config_path defaults to the repo's own default config.
    """
    def __init__(self, cfg: TeacherConfig):
        super().__init__(cfg)
        import sys
        if cfg.repo_root and cfg.repo_root not in sys.path:
            sys.path.insert(0, cfg.repo_root)
        from MFT.config import load_config  # noqa: import after sys.path insert
        config_path = cfg.config_path or "configs/MFT_cfg.py"
        import os
        orig_cwd = os.getcwd()
        try:
            if cfg.repo_root:
                os.chdir(cfg.repo_root)
            self._config = load_config(config_path)
        finally:
            os.chdir(orig_cwd)
        self._tracker_class = self._config.tracker_class

    def _forward(self, frames, queries):
        import os
        import torch
        from MFT.point_tracking import convert_to_point_tracking
        from MFT.utils.misc import ensure_numpy

        orig_cwd = os.getcwd()
        try:
            if self.cfg.repo_root:
                os.chdir(self.cfg.repo_root)
            tracker = self._tracker_class(self._config)
            q = torch.from_numpy(queries).float().cuda()

            coords_per_frame, occl_per_frame = [], []
            for i, frame in enumerate(frames):
                meta = tracker.init(frame) if i == 0 else tracker.track(frame)
                coords, occlusions = convert_to_point_tracking(meta.result, q)
                coords_per_frame.append(ensure_numpy(coords))
                occl_per_frame.append(ensure_numpy(occlusions))
        finally:
            os.chdir(orig_cwd)

        coords = np.stack(coords_per_frame, axis=1).astype(np.float32)  # [N, T, 2]
        occl = np.stack(occl_per_frame, axis=1).astype(np.float32)      # [N, T]
        return TrackResult(coords, visibility=1.0 - occl, name=self.name)


class DummyTeacher(Teacher):
    """Runs with no dependencies so the pipeline is executable end-to-end before
    any real tracker is wired. Produces a linear guess plus per-teacher noise."""
    def __init__(self, cfg: TeacherConfig, drift: float = 0.0, noise: float = 1.0, seed: int = 0):
        super().__init__(cfg)
        self.drift, self.noise = drift, noise
        self.rng = np.random.default_rng(seed)

    def _forward(self, frames, queries):
        T = len(frames)
        N = len(queries)
        base = np.repeat(queries[:, None, :], T, axis=1).astype(np.float32)  # [N,T,2]
        base += np.linspace(0, self.drift, T)[None, :, None]
        base += self.rng.normal(0, self.noise, base.shape)
        return TrackResult(base, np.ones((N, T), np.float32), self.name)


_REGISTRY = {
    "mft": MFTTeacher,
    "mftiq": MFTIQTeacher,
    "cotracker3": _make_trackon_teacher("cotracker3"),
    "locotrack": _make_trackon_teacher("locotrack"),
    "bootstapir": _make_trackon_teacher("bootstapir"),
    "tapir": _make_trackon_teacher("tapir"),
    "alltracker": _make_trackon_teacher("alltracker"),
    "trackon2": TrackOnTeacher,
    "trackon_r": TrackOnTeacher,
    "dummy": DummyTeacher,
    # "litetracker" is deliberately absent: it's not a separate weight-bearing
    # model, it's a runtime wrapper around a CoTracker3 checkpoint (stock or
    # fine-tuned). See student_lt_wrapper.py -- applied after training, at
    # evaluation and submission time only.
}


def build_teacher(cfg: TeacherConfig) -> Teacher:
    if cfg.name not in _REGISTRY:
        raise KeyError(f"unknown teacher {cfg.name!r}; known: {list(_REGISTRY)}")
    return _REGISTRY[cfg.name](cfg)
