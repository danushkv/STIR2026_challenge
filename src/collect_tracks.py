"""Phase 1 of a two-phase, multi-environment pipeline: run ONE teacher, inside
its own environment, over all clips, and write raw trajectories to disk.

This exists because your teachers don't all live in one importable Python
environment (MFT, MFTIQ, and track_on likely pin incompatible torch/torchvision/
timm/transformers versions) -- but they never actually need to run together.
Label generation is offline: each teacher can run completely separately, on its
own schedule, and only its OUTPUT (a numpy array) needs to reach a shared place.

TWO clip sources are supported -- use whichever fits your workflow:

  A) --stir-root  (recommended, no intermediate files)
     Reads directly from the STIR dataset via STIRLoader. No .npz prep step needed.

        (mft-env)  python collect_tracks.py --teacher mft \\
                       --repo-root ~/MFT --stir-root /data/STIR --out-dir raw_tracks/

        --skip 3   takes every 3rd frame (~10fps from 30fps video)
        --patients 0 1   process only these patient folders

  B) --clips-dir  (legacy: pre-converted .npz clips)
     Reads pre-converted .npz files (frames [T,H,W,3] uint8, queries [N,2] float32).

        (mft-env)  python collect_tracks.py --teacher mft \\
                       --repo-root ~/MFT --clips-dir data/clips --out-dir raw_tracks/

All teachers write into the SAME --out-dir. Run one teacher at a time, each in
its own environment, in any order -- the verifier in phase 2 reads them all.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from config import TeacherConfig
from teachers import build_teacher
from trajectories import TrackResult, save_track_result_pair


# --------------------------------------------------------------------------- #
# Clip iterators
# --------------------------------------------------------------------------- #
def _iter_npz(clips_dir: str) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
    """Yield (clip_id, frames [T,H,W,3], queries [N,2]) from pre-converted .npz files."""
    clip_paths = sorted(Path(clips_dir).glob("*.npz"))
    if not clip_paths:
        raise SystemExit(f"No .npz clips found in {clips_dir}")
    for clip_path in clip_paths:
        d = np.load(clip_path)
        yield clip_path.stem, d["frames"], d["queries"]


def _import_stirloader():
    """Locate STIRLoader: installed in the environment first, else a cloned repo.

    The clone must be the PATCHED one -- see patches/stirloader-streaming-skip.patch.
    Stock upstream applies SKIP after decoding the whole clip, which OOMs on
    STIR's long sequences; the patch applies it while streaming from ffmpeg,
    keeping the same frames. Every result in this repo was produced with it.

    Search order:
      1. an installed `STIRLoader` package
      2. $STIRLOADER_ROOT                      (the clone's top level)
      3. $THIRDPARTY_ROOT/STIRLoader
      4. ../STIRLoader and ./STIRLoader relative to this file
    """
    try:
        from STIRLoader.STIRLoader import getviddirs2d_STIR, STIRStereoClip
        return getviddirs2d_STIR, STIRStereoClip
    except ImportError:
        pass
    from _thirdparty import find_repo
    root = find_repo("STIRLoader", "STIRLOADER_ROOT",
                     "STIRLoader/STIRLoader.py", "STIRLoader")
    sys.path.insert(0, str(root))
    from STIRLoader.STIRLoader import getviddirs2d_STIR, STIRStereoClip
    return getviddirs2d_STIR, STIRStereoClip


def _clip_id(seq_path: Path) -> str:
    return "__".join(seq_path.parts[-3:])


def load_clip_frames(stir_root: str, clip_id: str, skip: int = 1) -> np.ndarray:
    """[T,H,W,3] RGB uint8 frames for one exact clip_id, at the given --skip.

    `skip` must match whatever --skip was used when that clip's tracks or
    pseudo-labels were originally generated -- the returned T has to line up
    frame-for-frame with data collected earlier at a different sampling rate.
    Shared by visualize_pseudo_labels.py and train.py so there's one
    place that knows how to map a clip_id back to real STIR video frames.
    """
    getviddirs2d_STIR, STIRStereoClip = _import_stirloader()
    if skip > 1:
        os.environ["SKIP"] = str(skip)
    for seq_path in getviddirs2d_STIR(stir_root):
        if _clip_id(seq_path) == clip_id:
            clip = STIRStereoClip(seq_path)
            left_frames, _ = clip.extractallframes()
            return np.stack(left_frames, axis=0)
    raise FileNotFoundError(f"clip_id {clip_id!r} not found under {stir_root}")


def _iter_stir(stir_root: str, skip: int = 1, patients: list = None,
               random_queries_n: int = 0, seed: int = 0
               ) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
    """Yield (clip_id, frames [T,H,W,3], queries [N,2]) directly from STIRLoader.

    Clips with no IR-tattoo segmentation (no start-center query points) are
    normally skipped, since there's nothing to seed tracking with. If
    random_queries_n > 0, such clips are kept: random_queries_n random points
    on frame 0 are used as queries instead. Meant for bootstrapping a cheap,
    general-purpose teacher (e.g. cotracker3) on clips that can't be endpoint-
    anchored -- these clips still have no ground truth, so treat them as
    unlabeled/self-labeling data, not as a verified teacher signal.
    """
    getviddirs2d_STIR, STIRStereoClip = _import_stirloader()
    rng = np.random.default_rng(seed)

    if skip > 1:
        os.environ["SKIP"] = str(skip)

    seq_paths = getviddirs2d_STIR(stir_root)
    if not seq_paths:
        raise SystemExit(f"No sequences found in {stir_root}")

    if patients is not None:
        allowed = set(patients)
        seq_paths = [p for p in seq_paths if p.parts[-3] in allowed]
        print(f"Filtered to patients {sorted(allowed)}: {len(seq_paths)} sequences")

    for seq_path in seq_paths:
        clip_id = _clip_id(seq_path)
        try:
            clip = STIRStereoClip(seq_path)
        except (AssertionError, IndexError) as e:
            print(f"[skip] {clip_id}: {e}")
            continue

        left_frames, _ = clip.extractallframes()
        if not left_frames:
            print(f"[skip] {clip_id}: no frames")
            continue
        frames = np.stack(left_frames, axis=0)  # [T, H, W, 3] uint8

        queries = np.array(clip.getstartcenters(left=True), dtype=np.float32)
        if len(queries) == 0:
            if random_queries_n <= 0:
                print(f"[skip] {clip_id}: no segmentation points")
                continue
            H, W = frames.shape[1:3]
            queries = rng.uniform([0, 0], [W, H], size=(random_queries_n, 2)).astype(np.float32)
            print(f"[{clip_id}] no segmentation points -> seeding {random_queries_n} random queries")

        yield clip_id, frames, queries


# --------------------------------------------------------------------------- #
# Resolution reduction (optional) -- run the teacher at a smaller resolution to
# cut memory/compute, while keeping saved coordinates in the ORIGINAL frame's
# pixel space (trajectories.py's contract that every TrackResult uses native
# pixel coordinates, since STIR's delta-at-pixel-threshold metric and the
# verifier's ensemble merge both assume that).
# --------------------------------------------------------------------------- #
def _downscale(frames: np.ndarray, queries: np.ndarray, track_res: Tuple[int, int]):
    _, H, W, _ = frames.shape
    target_h, target_w = track_res
    small_frames = np.stack(
        [cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA) for f in frames],
        axis=0,
    )
    scale_x, scale_y = W / target_w, H / target_h
    small_queries = queries.copy()
    small_queries[:, 0] /= scale_x
    small_queries[:, 1] /= scale_y
    return small_frames, small_queries, scale_x, scale_y


def _upscale_result(result: TrackResult, scale_x: float, scale_y: float) -> TrackResult:
    coords = result.coords.copy()
    coords[..., 0] *= scale_x
    coords[..., 1] *= scale_y
    return TrackResult(coords, result.visibility, name=result.name, confidence=result.confidence)


# --------------------------------------------------------------------------- #
# Temporal chunking (optional) -- some teachers (AllTracker, LocoTrack) are NOT
# causal/windowed: they process the whole clip's frames in one shot, so memory
# scales with clip LENGTH regardless of resolution or query count -- unlike
# MFT/MFTIQ/Track-On2/R (frame-by-frame) or cotracker3 (windowed). Very long
# clips OOM these teachers no matter the GPU. Fix: chunk along time and reseed
# each chunk's queries at the previous chunk's last known point position --
# the same trick Teacher.track_cycle already uses to reseed its backward pass.
# --------------------------------------------------------------------------- #
def _run_chunked(teacher, frames: np.ndarray, queries: np.ndarray, chunk_size: int,
                 use_cycle: bool) -> Tuple[TrackResult, Optional[TrackResult]]:
    T = len(frames)
    fwd_coords, fwd_vis, fwd_conf = [], [], []
    bwd_coords, bwd_vis, bwd_conf = [], [], []
    cur_queries = queries

    for start in range(0, T, chunk_size):
        chunk = frames[start:start + chunk_size]
        if use_cycle:
            fwd, bwd = teacher.track_cycle(chunk, cur_queries)
        else:
            fwd, bwd = teacher.track(chunk, cur_queries), None

        fwd_coords.append(fwd.coords)
        fwd_vis.append(fwd.visibility)
        fwd_conf.append(fwd.confidence)
        if bwd is not None:
            bwd_coords.append(bwd.coords)
            bwd_vis.append(bwd.visibility)
            bwd_conf.append(bwd.confidence)

        cur_queries = fwd.coords[:, -1, :]  # reseed next chunk at last known position

    fwd_result = TrackResult(np.concatenate(fwd_coords, axis=1),
                             np.concatenate(fwd_vis, axis=1),
                             name=teacher.name,
                             confidence=np.concatenate(fwd_conf, axis=1))
    bwd_result = None
    if use_cycle:
        bwd_result = TrackResult(np.concatenate(bwd_coords, axis=1),
                                 np.concatenate(bwd_vis, axis=1),
                                 name=teacher.name,
                                 confidence=np.concatenate(bwd_conf, axis=1))
    return fwd_result, bwd_result


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", required=True,
                   help="name registered in teachers._REGISTRY, e.g. mft, mftiq, "
                        "cotracker3, locotrack, bootstapir, tapir, alltracker")
    p.add_argument("--repo-root", default="", help="path to that teacher's cloned repo")
    p.add_argument("--checkpoint", default="", help="checkpoint path (ignored by some teachers)")
    p.add_argument("--config-path", default="", help="e.g. configs/MFTIQ4_RAFT_200k_cfg.py, relative to --repo-root")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--no-cycle", action="store_true",
                   help="skip the backward pass (faster, disables cycle-consistency)")
    p.add_argument("--overwrite", action="store_true")

    # Clip source — one of these two is required
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--stir-root",
                     help="Root of the STIR dataset (reads directly, no .npz needed)")
    src.add_argument("--clips-dir",
                     help="Directory of pre-converted .npz clip files")

    # STIRLoader options (only used with --stir-root)
    p.add_argument("--skip", type=int, default=1,
                   help="Take every Nth frame, e.g. --skip 3 for ~10fps (only with --stir-root)")
    p.add_argument("--patients", nargs="+", default=None,
                   help="Only process these patient folders, e.g. --patients 0 1 (only with --stir-root)")
    p.add_argument("--track-res", type=int, nargs=2, default=None, metavar=("H", "W"),
                   help="Run the teacher at this resolution instead of native, e.g. "
                        "--track-res 384 512 (matches student training resolution). "
                        "Saved coordinates are always rescaled back to the native "
                        "frame resolution, so this is purely a memory/speed knob.")
    p.add_argument("--random-queries-n", type=int, default=0,
                   help="If a clip has no segmentation-derived query points, seed this "
                        "many random points on frame 0 instead of skipping it (only "
                        "with --stir-root). 0 = keep skipping such clips (default). "
                        "Meant for a cheap, general-purpose teacher like cotracker3 -- "
                        "these clips have no ground truth, so treat the output as "
                        "self-labeling data, not a verified teacher signal.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for --random-queries-n")
    p.add_argument("--chunk-frames", type=int, default=None,
                   help="Split the clip into chunks of this many frames along time "
                        "before tracking, reseeding each chunk's queries at the "
                        "previous chunk's last known point position. For non-causal "
                        "teachers (AllTracker, LocoTrack) whose memory scales with clip "
                        "length -- use this for very long clips that OOM even at "
                        "reduced resolution.")

    args = p.parse_args()

    cfg = TeacherConfig(args.teacher, checkpoint=args.checkpoint,
                        repo_root=args.repo_root, config_path=args.config_path)
    print(f"Building teacher '{args.teacher}'...")
    teacher = build_teacher(cfg)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.stir_root:
        clips = _iter_stir(args.stir_root, skip=args.skip, patients=args.patients,
                           random_queries_n=args.random_queries_n, seed=args.seed)
    else:
        clips = _iter_npz(args.clips_dir)

    for clip_id, frames, queries in clips:
        # clip_id format: "patient__side__seq" (e.g. "0__left__seq00")
        # Output: <out_dir>/<patient>/<side__seq>__<teacher>.npz
        patient, seq_part = clip_id.split("__", 1)
        patient_dir = Path(args.out_dir) / patient
        patient_dir.mkdir(parents=True, exist_ok=True)
        out_path = patient_dir / f"{seq_part}__{args.teacher}.npz"

        if out_path.exists() and not args.overwrite:
            print(f"[{args.teacher}] {clip_id}: already exists, skipping (--overwrite to redo)")
            continue

        if args.track_res:
            run_frames, run_queries, scale_x, scale_y = _downscale(frames, queries, tuple(args.track_res))
        else:
            run_frames, run_queries, scale_x, scale_y = frames, queries, 1.0, 1.0

        if args.chunk_frames:
            fwd, bwd = _run_chunked(teacher, run_frames, run_queries, args.chunk_frames,
                                    use_cycle=not args.no_cycle)
        elif args.no_cycle:
            fwd, bwd = teacher.track(run_frames, run_queries), None
        else:
            fwd, bwd = teacher.track_cycle(run_frames, run_queries)

        if args.track_res:
            fwd = _upscale_result(fwd, scale_x, scale_y)
            if bwd is not None:
                bwd = _upscale_result(bwd, scale_x, scale_y)

        save_track_result_pair(out_path, fwd, bwd)
        print(f"[{args.teacher}] {clip_id}: wrote {out_path}")

    print(f"\nDone. {args.teacher}'s trajectories are in {args.out_dir}/<patient>/*__{args.teacher}.npz")


if __name__ == "__main__":
    main()
