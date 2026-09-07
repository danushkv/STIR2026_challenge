"""Metric helpers shared by the streaming evaluators.

These definitions are shared by the streaming evaluation pipeline,
which ran CoTracker3's whole-clip sliding-window forward. That is not the
forward pass the challenge scores, so those two scripts were dropped and only
the pieces that define the METRIC were kept here -- unchanged, so numbers stay
comparable with everything already published from this pipeline.

Thresholds match STIRMetrics: `calculate_error_from_json2d.py` for pixels and
`calculate_error_from_json3d.py` for millimetres.
"""
from __future__ import annotations

import os

import numpy as np

STIR_THRESHOLDS_PX = [4, 8, 16, 32, 64]
STIR_THRESHOLDS_MM = [2, 4, 8, 16, 32]


def nn_dist(pred, gt):
    """Per-point distance from each prediction to its NEAREST ground-truth point.

    KDTree, unordered: STIR gives no persistent point identities, so a
    prediction is scored against whichever GT point it is closest to. Same
    convention as STIRMetrics' `pointlossunidirectional`, in 2D and in 3D.
    """
    from scipy.spatial import KDTree
    d, _ = KDTree(gt, leafsize=10).query(pred)
    return d


def load_val_clip(seq_path, STIRStereoClip, skip, val_max_frames):
    """One clip's left frames plus its start/end IR-tattoo centres.

    Returns (frames [T,H,W,3] uint8, start [N,2], end [M,2]) in native pixels.
    """
    if skip > 1:
        os.environ["SKIP"] = str(skip)
    clip = STIRStereoClip(seq_path)
    left_frames, _ = clip.extractallframes()
    frames = np.stack(left_frames, axis=0)
    start = np.array(clip.getstartcenters(left=True), dtype=np.float32)
    end = np.array(clip.getendcenters(left=True), dtype=np.float32)

    T = len(frames)
    if val_max_frames and T > val_max_frames:
        # temporal stride keeping first & last frame, so the tracked endpoint is
        # still the real endpoint (memory safety for very long clips)
        idx = np.unique(np.linspace(0, T - 1, val_max_frames).round().astype(int))
        frames = frames[idx]
    return frames, start, end


def triangulate_mm(u, v, disparity, clip):
    """(u, v) left pixels + disparity -> [N,3] XYZ in MILLIMETRES, via STIR's own Q.

    Mirrors STIRLoader's `get3DSegmentationPositions` exactly: unscaled K at
    scale 1.0, Q built from baseline_mm, homogeneous divide. Millimetres is the
    STIRMetrics convention -- NOT the 2026 stereo convention, which is metres.
    """
    # STIRLoader is already on sys.path by the time this is called (the caller
    # has run collect_tracks._import_stirloader).
    from STIRLoader.STIRLoader import getKfromcameramat, getQ
    unscaledK = getKfromcameramat(clip.leftcameramat, 1.0)
    Q = getQ(clip.baseline_mm, unscaledK)
    pts = np.stack([u, v, disparity], axis=-1).astype(np.float64)
    h = np.pad(pts, ((0, 0), (0, 1)), "constant", constant_values=1) @ Q.T
    return (h[:, :3] / h[:, 3:4]).astype(np.float32)
