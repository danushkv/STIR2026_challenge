"""Verifier-guided pseudo-label construction for STIR.

This is the intellectual core of the pipeline. It turns candidate trajectories
from several teacher trackers into one cleaned pseudo-label trajectory plus a
per-frame supervision weight, using three cheap trust signals:

  1. ENDPOINT ANCHORING (STIR-specific, requires known start/end tattoo points)
       The true point location at the final frame is known from the IR tattoo.
       A teacher whose trajectory misses that endpoint is untrustworthy for the
       WHOLE clip. This gives a per-teacher global trust multiplier.

  2. CYCLE CONSISTENCY (self-supervised, works on any clip)
       Track forward (query -> frame T), then backward (from the forward endpoint
       back toward frame 0). Where the forward and backward passes disagree, the
       track is unreliable. This gives a per-teacher PER-FRAME reliability.

  3. TEACHER AGREEMENT (self-supervised, needs >=2 teachers)
       At each frame, teachers that cluster near the robust median are more
       trustworthy than outliers. This gives a second per-frame reliability.

The verifier combines these into a per-frame score for each teacher, then either
selects the best teacher per frame ("select") or takes a confidence-weighted
robust aggregate ("aggregate"), and emits a confidence weight for the loss.

"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from trajectories import PseudoLabel, TrackResult
from config import VerifierConfig


# --------------------------------------------------------------------------- #
# Trust signals
# --------------------------------------------------------------------------- #
def endpoint_error(track: TrackResult, end_centers: np.ndarray) -> np.ndarray:
    """L2 distance (pixels) between each track's final position and its NEAREST
    ground-truth end-frame IR-tattoo center.

    STIR's start/end segmentation centers (STIRLoader's getstartcenters /
    getendcenters) are independently detected contours per frame -- there is no
    persistent point ID tying query i's start center to end_centers[i]. STIR's
    own evaluation code (STIRMetrics testutil.pointlossunidirectional) never
    assumes index correspondence either: it nearest-neighbor-matches a tracker's
    OWN prediction against the raw end-frame centers. Do the same here.

    track.coords: [N, T, 2]; end_centers: [M, 2] (M need not equal N -- it's
    just every detected end-frame tattoo center, unordered). Returns [N].
    """
    final = track.coords[:, -1, :]                                  # [N, 2]
    diff = final[:, None, :] - end_centers[None, :, :]              # [N, M, 2]
    return np.linalg.norm(diff, axis=-1).min(axis=1)                # [N]


def endpoint_trust(track: TrackResult, end_centers: np.ndarray, tau: float) -> np.ndarray:
    """Per-query global trust multiplier in (0, 1] from endpoint agreement.

    Uses a soft gate exp(-err / tau) so that hitting (the nearest) endpoint ->
    ~1 and drifting off it -> ~0. tau is in pixels (tie it to your delta
    thresholds, e.g. tau ~= 8-16 px). Returns [N].
    """
    err = endpoint_error(track, end_centers)           # [N]
    return np.exp(-err / max(tau, 1e-6))               # [N]


def cycle_distance(forward: TrackResult, backward: TrackResult) -> np.ndarray:
    """Raw forward/backward disagreement in PIXELS, [N, T].

    Two-pass approximation (cheap: 2 inferences per teacher instead of T):
      forward.coords[:, t]  = position at frame t tracked from the query at frame 0.
      backward.coords[:, t] = position at frame t tracked from the forward endpoint
                              back through the reversed clip, re-indexed to forward time.

    NOTE on indexing: the caller is responsible for running the backward pass on
    the time-reversed clip and flipping it back so backward.coords is aligned to
    forward time (see teachers.track_cycle). At t=0 this reduces to the classic
    return-to-origin check ||backward[:,0] - query||.

    Split out from cycle_reliability so a learned scorer could consume the
    distance directly: exp(-d/tau) saturates to ~0 past a few tau, which throws
    away exactly the "how badly did this teacher fail" resolution a trained model
    can use.
    """
    return np.linalg.norm(forward.coords - backward.coords, axis=-1)       # [N, T]


def median_distance(tracks: Sequence[TrackResult]) -> np.ndarray:
    """Raw per-teacher distance to the per-frame across-teacher median, PIXELS,
    [K, N, T]. All zeros for K == 1 (no consensus signal available).

    Split out from agreement_reliability for the same reason as cycle_distance.
    """
    coords = np.stack([t.coords for t in tracks], axis=0)   # [K, N, T, 2]
    if len(tracks) == 1:
        return np.zeros(coords.shape[:3], dtype=np.float32)
    median = np.median(coords, axis=0)                      # [N, T, 2]
    return np.linalg.norm(coords - median[None], axis=-1)   # [K, N, T]


def cycle_reliability(forward: TrackResult, backward: TrackResult,
                      tau: float) -> np.ndarray:
    """Per-frame reliability in (0, 1] from forward/backward agreement.

    Soft gate exp(-d/tau) over cycle_distance; tau in pixels. Returns [N, T].
    """
    return np.exp(-cycle_distance(forward, backward) / max(tau, 1e-6))     # [N, T]


def agreement_reliability(tracks: Sequence[TrackResult], tau: float) -> np.ndarray:
    """Per-teacher per-frame reliability from clustering around the robust median.

    Returns [K, N, T] where K = number of teachers. A teacher whose prediction sits
    near the per-frame median across teachers is trusted; outliers are down-weighted.
    With a single teacher this returns all-ones (no consensus signal available).
    """
    if len(tracks) == 1:
        N, T, _ = tracks[0].coords.shape
        return np.ones((1, N, T), dtype=np.float32)
    return np.exp(-median_distance(tracks) / max(tau, 1e-6))               # [K, N, T]


# --------------------------------------------------------------------------- #
# Cheap verifier
# --------------------------------------------------------------------------- #
class CheapVerifier:
    """Zero-training verifier built from endpoint anchoring + cycle + agreement."""

    def __init__(self, cfg: VerifierConfig):
        self.cfg = cfg

    def _per_teacher_frame_score(
        self,
        tracks: List[TrackResult],
        cycles: Optional[List[TrackResult]],
        endpoints: Optional[np.ndarray],
    ) -> np.ndarray:
        """Combine the three signals into a score [K, N, T] per teacher per frame."""
        cfg = self.cfg
        K = len(tracks)
        N, T, _ = tracks[0].coords.shape
        score = np.ones((K, N, T), dtype=np.float32)

        # (2) cycle consistency — per-frame
        if cfg.use_cycle_consistency and cycles is not None:
            for k, (fwd, bwd) in enumerate(zip(tracks, cycles)):
                score[k] *= cycle_reliability(fwd, bwd, cfg.cycle_tau) ** cfg.w_cycle

        # (3) teacher agreement — per-frame
        if cfg.use_agreement and K > 1:
            agree = agreement_reliability(tracks, cfg.agree_tau)   # [K, N, T]
            score *= agree ** cfg.w_agreement

        # (1) endpoint anchoring — per-teacher global multiplier broadcast over T
        if cfg.use_endpoint_anchoring and endpoints is not None:
            for k, trk in enumerate(tracks):
                trust = endpoint_trust(trk, endpoints, cfg.endpoint_tau)  # [N]
                score[k] *= (trust[:, None] ** cfg.w_endpoint)

        # native teacher confidence, as a weak prior
        if cfg.use_native_confidence:
            for k, trk in enumerate(tracks):
                score[k] *= np.clip(trk.confidence, 1e-3, 1.0) ** cfg.w_native

        return score  # [K, N, T]

    def build(
        self,
        tracks: List[TrackResult],
        clip_id: str = "",
        cycles: Optional[List[TrackResult]] = None,
        endpoints: Optional[np.ndarray] = None,
    ) -> PseudoLabel:
        """Produce one cleaned PseudoLabel for a clip.

        endpoints: [M, 2] raw end-frame IR-tattoo centers for this clip (e.g.
        STIRLoader's getendcenters), M need not equal N -- see endpoint_error's
        docstring for why this is nearest-neighbor-matched, not index-aligned.
        """
        cfg = self.cfg
        K = len(tracks)
        N, T, _ = tracks[0].coords.shape
        coords = np.stack([t.coords for t in tracks], axis=0)          # [K, N, T, 2]
        vis = np.stack([t.visibility for t in tracks], axis=0)         # [K, N, T]
        score = self._per_teacher_frame_score(tracks, cycles, endpoints)  # [K, N, T]

        # A teacher can emit NaN/inf for a lost point (and a NaN coord poisons the
        # median in agreement_reliability, hence score, hence every teacher's
        # weight). "select" survives that by accident -- argmax just returns some
        # index -- but "aggregate" blends every teacher at every frame, so one bad
        # sample contaminates the whole clip and lands in the visibility BCE
        # target. Zero out the offending samples AND their weights: masking the
        # weight alone is not enough, since 0 * NaN is still NaN.
        usable = np.isfinite(coords).all(axis=-1) & np.isfinite(vis)   # [K, N, T]
        coords = np.where(usable[..., None], coords, 0.0)
        vis = np.where(usable, vis, 0.0)
        score = np.where(usable, np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        if not usable.all():
            print(f"[warn] {clip_id or '<clip>'}: dropped {int((~usable).sum())}/"
                  f"{usable.size} non-finite teacher sample(s) before combining")

        if cfg.mode == "select":
            best = np.argmax(score, axis=0)                            # [N, T]
            ii, jj = np.meshgrid(np.arange(N), np.arange(T), indexing="ij")
            out_coords = coords[best, ii, jj]                          # [N, T, 2]
            out_vis = vis[best, ii, jj]                                # [N, T]
            out_source = best.astype(np.int32)
            frame_conf = np.max(score, axis=0)                         # [N, T]
        elif cfg.mode == "aggregate":
            w = score / (score.sum(axis=0, keepdims=True) + 1e-8)      # [K, N, T]
            out_coords = (w[..., None] * coords).sum(axis=0)           # [N, T, 2]
            out_vis = (w * vis).sum(axis=0)                            # [N, T]
            out_source = np.full((N, T), -1, dtype=np.int32)
            # confidence = agreement-weighted score mass (spread-out mass => low conf)
            frame_conf = (w * score).sum(axis=0)
        else:
            raise ValueError(f"unknown mode {cfg.mode!r}")

        # visibility is a BCE *target* downstream (sequence_BCE_loss applies no
        # valid mask), so it must be in [0,1] even if a teacher reports an
        # unnormalized occlusion score.
        out_vis = np.clip(out_vis, 0.0, 1.0)

        # normalize confidence to [0,1] and threshold into a supervision weight
        frame_conf = frame_conf / (frame_conf.max() + 1e-8)
        weight = np.where(frame_conf >= cfg.weight_floor, frame_conf, 0.0)
        # don't supervise position where the aggregate says the point is occluded
        weight = weight * (out_vis >= cfg.visibility_floor)

        return PseudoLabel(
            coords=out_coords.astype(np.float32),
            visibility=out_vis.astype(np.float32),
            weight=weight.astype(np.float32),
            source=out_source,
            clip_id=clip_id,
            meta={"kept_frac": float((weight > 0).mean())},
        )


def make_verifier(cfg: VerifierConfig):
    return CheapVerifier(cfg)


# --------------------------------------------------------------------------- #
# Self-test: runs the cheap verifier on synthetic trajectories, no trackers needed
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, T = 4, 20
    # ground-truth trajectory: a smooth drift
    t = np.linspace(0, 1, T)
    gt = np.stack([np.stack([q0 + 30 * t, q1 + 10 * np.sin(3 * t)], axis=-1)
                   for q0, q1 in [(50, 50), (100, 80), (200, 120), (150, 200)]], axis=0)
    endpoints = gt[:, -1, :]

    # teacher A: accurate (MFT-like) with small noise
    a = TrackResult(gt + rng.normal(0, 1.0, gt.shape), np.ones((N, T)), "accurate")
    # teacher B: fast but drifts badly on 2 of the 4 points late in the clip
    b_coords = gt + rng.normal(0, 1.5, gt.shape)
    b_coords[1, T // 2:] += np.linspace(0, 40, T - T // 2)[:, None]  # drift
    b_coords[3, T // 2:] += np.linspace(0, 25, T - T // 2)[:, None]
    b = TrackResult(b_coords, np.ones((N, T)), "fast")

    # cheap forward/backward: fake a backward pass that agrees except where B drifts
    a_cyc = TrackResult(a.coords + rng.normal(0, 1.0, gt.shape), np.ones((N, T)), "accurate")
    b_cyc_coords = b.coords.copy()
    b_cyc_coords[1, T // 2:] -= np.linspace(0, 40, T - T // 2)[:, None]  # inconsistent
    b_cyc_coords[3, T // 2:] -= np.linspace(0, 25, T - T // 2)[:, None]
    b_cyc = TrackResult(b_cyc_coords, np.ones((N, T)), "fast")

    from config import VerifierConfig
    cfg = VerifierConfig(mode="select")
    v = CheapVerifier(cfg)
    pl = v.build([a, b], clip_id="synthetic",
                 cycles=[a_cyc, b_cyc], endpoints=endpoints)

    # error of the pseudo-label vs ground truth, vs each teacher alone
    def rmse(x): return float(np.sqrt(((x - gt) ** 2).sum(-1).mean()))
    print(f"teacher 'accurate' rmse : {rmse(a.coords):6.2f}")
    print(f"teacher 'fast'     rmse : {rmse(b.coords):6.2f}")
    print(f"pseudo-label       rmse : {rmse(pl.coords):6.2f}  "
          f"(should beat the drifting teacher)")
    print(f"frames kept            : {pl.meta['kept_frac']:.0%}")
    print(f"teacher chosen per frame (point 1, the one 'fast' drifts on):")
    print(f"  {pl.source[1]}   (0=accurate, 1=fast)")
