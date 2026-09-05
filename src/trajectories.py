"""Shared data structures for the STIR verifier-guided pseudo-labeling pipeline.

All trajectories use image-pixel coordinates (x, y) to match STIR's delta-at-
pixel-threshold metric. Shapes are documented as [N, T, 2] where N = number of
query points in a clip and T = number of frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TrackResult:
    """One teacher's output for a clip.

    coords:      [N, T, 2] predicted (x, y) per query per frame, frame 0 == query frame.
    visibility:  [N, T]    in [0, 1]; 1 == visible. Teachers that don't predict
                           visibility should return ones.
    confidence:  [N, T]    optional native confidence from the teacher (in [0, 1]).
                           Used as a soft prior only; the verifier does not trust it blindly.
    name:        teacher identifier, e.g. "mft", "cotracker3".
    """
    coords: np.ndarray
    visibility: np.ndarray
    name: str
    confidence: Optional[np.ndarray] = None

    def __post_init__(self):
        assert self.coords.ndim == 3 and self.coords.shape[-1] == 2, self.coords.shape
        N, T, _ = self.coords.shape
        assert self.visibility.shape == (N, T), self.visibility.shape
        if self.confidence is None:
            self.confidence = np.ones((N, T), dtype=np.float32)


@dataclass
class PseudoLabel:
    """The verifier's cleaned output for a clip. This is what the student trains on.

    coords:      [N, T, 2] refined trajectory (per-frame selected/aggregated).
    visibility:  [N, T]    aggregated visibility.
    weight:      [N, T]    per-frame supervision weight in [0, 1]. Frames the
                           verifier distrusts get ~0 and are effectively masked
                           out of the loss.
    source:      [N, T]    int index of the teacher chosen at each frame (for
                           debugging / ablation analysis; -1 if aggregated).
    clip_id:     provenance.
    """
    coords: np.ndarray
    visibility: np.ndarray
    weight: np.ndarray
    source: np.ndarray
    clip_id: str = ""
    meta: dict = field(default_factory=dict)


def save_track_result_pair(path, fwd: TrackResult, bwd: Optional[TrackResult] = None) -> None:
    """Write one teacher's forward (+ optional backward/cycle) pass to disk.

    This is the hand-off point between environments: a teacher's own env runs
    `collect_tracks.py`, which calls this; a separate, dependency-free env
    later reads it back with `load_track_result_pair` to run the verifier.
    Nothing here needs any teacher's actual model code.
    """
    payload = {"fwd_coords": fwd.coords, "fwd_vis": fwd.visibility,
               "fwd_conf": fwd.confidence, "name": fwd.name}
    if bwd is not None:
        payload.update({"bwd_coords": bwd.coords, "bwd_vis": bwd.visibility,
                         "bwd_conf": bwd.confidence})
    np.savez_compressed(path, **payload)


def load_track_result_pair(path) -> "tuple[TrackResult, Optional[TrackResult]]":
    """Inverse of save_track_result_pair. Pure numpy -- safe to call from any
    environment, including one with none of the teachers' dependencies installed."""
    d = np.load(path, allow_pickle=True)
    name = str(d["name"])
    fwd = TrackResult(d["fwd_coords"], d["fwd_vis"], name=name, confidence=d["fwd_conf"])
    bwd = None
    if "bwd_coords" in d:
        bwd = TrackResult(d["bwd_coords"], d["bwd_vis"], name=name, confidence=d["bwd_conf"])
    return fwd, bwd
