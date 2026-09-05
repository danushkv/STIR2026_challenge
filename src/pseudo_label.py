"""Orchestration: for each clip, run the teacher ensemble (+ cycle passes),
hand the candidates to the verifier, and write out the cleaned pseudo-label.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from trajectories import PseudoLabel, TrackResult, load_track_result_pair
from config import PipelineConfig
from teachers import Teacher, build_teacher
from verifier import make_verifier


def save_pseudo_label(pl: PseudoLabel, out_dir: str, teacher_names: List[str]) -> Path:
    """Write one clip's PseudoLabel to <out_dir>/<patient>/<side__seq>.npz --
    same patient/seq folder structure collect_tracks.py uses for raw_tracks.
    """
    patient, seq_part = pl.clip_id.split("__", 1)
    patient_dir = Path(out_dir) / patient
    patient_dir.mkdir(parents=True, exist_ok=True)
    out_path = patient_dir / f"{seq_part}.npz"
    np.savez_compressed(
        out_path,
        coords=pl.coords, visibility=pl.visibility,
        weight=pl.weight, source=pl.source,
        teacher_names=np.array(teacher_names),  # source[i] indexes into this
    )
    return out_path


def generate_for_clip(
    frames: np.ndarray,
    queries: np.ndarray,
    teachers: List[Teacher],
    verifier,
    run_cycle: bool,
    endpoints: Optional[np.ndarray] = None,
    clip_id: str = "",
) -> PseudoLabel:
    """frames [T,H,W,3], queries [N,2], optional endpoints [N,2] (IR tattoo at frame T)."""
    tracks: List[TrackResult] = []
    cycles: Optional[List[TrackResult]] = [] if run_cycle else None
    for tt in teachers:
        if run_cycle:
            fwd, bwd = tt.track_cycle(frames, queries)
            tracks.append(fwd)
            cycles.append(bwd)
        else:
            tracks.append(tt.track(frames, queries))
    return verifier.build(tracks, clip_id=clip_id, cycles=cycles, endpoints=endpoints)


def run(cfg: PipelineConfig, dataset, out_dir: Optional[str] = None) -> None:
    """dataset yields dicts: {frames, queries, endpoints (or None), clip_id}."""
    out_dir = out_dir or cfg.pseudo_label_out
    os.makedirs(out_dir, exist_ok=True)

    teachers = [build_teacher(tc) for tc in cfg.teachers if tc.enabled]
    teacher_names = [t.name for t in teachers]
    verifier = make_verifier(cfg.verifier)
    run_cycle = cfg.verifier.use_cycle_consistency

    kept = []
    for clip in dataset:
        pl = generate_for_clip(
            frames=clip["frames"],
            queries=clip["queries"],
            teachers=teachers,
            verifier=verifier,
            run_cycle=run_cycle,
            endpoints=clip.get("endpoints"),
            clip_id=clip["clip_id"],
        )
        save_pseudo_label(pl, out_dir, teacher_names)
        kept.append(pl.meta.get("kept_frac", 1.0))
        print(f"[{clip['clip_id']}] kept {pl.meta.get('kept_frac', 1.0):.0%} of frames")

    if kept:
        print(f"\nmean frames kept across {len(kept)} clips: {np.mean(kept):.0%}")


# --------------------------------------------------------------------------- #
# Phase 2 (multi-environment path): verify from pre-computed, on-disk teacher
# trajectories instead of live in-process Teacher objects. Use this whenever
# your teachers don't all share one importable environment -- run
# collect_tracks.py once per teacher (phase 1, in that teacher's own env),
# then this (phase 2) in any plain environment with just numpy.
# --------------------------------------------------------------------------- #
def generate_for_clip_from_disk(
    clip_id: str,
    teacher_names: List[str],
    raw_tracks_dir: str,
    verifier,
    endpoints: Optional[np.ndarray] = None,
) -> PseudoLabel:
    """Load each teacher's pre-computed (fwd, bwd) pair for one clip and verify.
    Has NO model dependencies -- pure numpy -- so it runs in any environment,
    independent of every teacher's own.
    """
    tracks, cycles = [], []
    # clip_id format: "patient__side__seq" ->
    # raw_tracks/<teacher>/<patient>/<side__seq>__<teacher>.npz -- one top-level
    # folder per teacher, since each teacher runs in its own collect_tracks.py
    # invocation (own --out-dir) in its own environment.
    patient, seq_part = clip_id.split("__", 1)
    for name in teacher_names:
        path = Path(raw_tracks_dir) / name / patient / f"{seq_part}__{name}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path} -- did you run collect_tracks.py for teacher "
                f"'{name}' on clip '{clip_id}' yet?"
            )
        fwd, bwd = load_track_result_pair(path)
        tracks.append(fwd)
        cycles.append(bwd)  # may be None if that teacher's collection used --no-cycle

    have_all_cycles = all(c is not None for c in cycles)
    return verifier.build(
        tracks, clip_id=clip_id,
        cycles=cycles if have_all_cycles else None,
        endpoints=endpoints,
    )


def run_from_disk(
    cfg: PipelineConfig,
    clip_ids: List[str],
    raw_tracks_dir: str,
    endpoints_by_clip: Optional[dict] = None,
    out_dir: Optional[str] = None,
) -> None:
    """Phase 2 entry point. Requires collect_tracks.py to have already been run,
    once per teacher in cfg.teachers, in each teacher's own environment, each
    writing into its own raw_tracks_dir/<teacher>/ subfolder (its --out-dir).
    endpoints_by_clip: {clip_id: np.ndarray [M,2]} of raw end-frame IR-tattoo
    centers (e.g. STIRLoader's getendcenters) for clips with known ground
    truth -- M need not equal the clip's N queries, see verifier.endpoint_error.
    Omit or map to None for unlabeled clips (endpoint anchoring is simply
    skipped for those, per verifier.py).
    """
    out_dir = out_dir or cfg.pseudo_label_out
    os.makedirs(out_dir, exist_ok=True)
    verifier = make_verifier(cfg.verifier)
    teacher_names = [tc.name for tc in cfg.teachers if tc.enabled]

    kept = []
    for clip_id in clip_ids:
        endpoints = (endpoints_by_clip or {}).get(clip_id)
        pl = generate_for_clip_from_disk(clip_id, teacher_names, raw_tracks_dir, verifier, endpoints)
        save_pseudo_label(pl, out_dir, teacher_names)
        kept.append(pl.meta.get("kept_frac", 1.0))
        print(f"[{clip_id}] kept {pl.meta.get('kept_frac', 1.0):.0%} of frames")

    if kept:
        print(f"\nmean frames kept across {len(kept)} clips: {np.mean(kept):.0%}")
