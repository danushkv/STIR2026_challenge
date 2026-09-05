"""Add EXTRA random query points to clips that already have pseudo-labels, and
run one teacher (peng_tracker / cotracker3) on just those new points.

Motivation: run_phase2.py already produced a pseudo-label per clip with some set
of tracked points (from segmentation and/or earlier random queries). This script
seeds MORE random points on frame 0 of the same clips -- guaranteed distinct
from the existing pseudo-label points AND from each other (rejection sampling
with a minimum pixel distance) -- then tracks them with the teacher, exactly like
collect_tracks.py's --random-queries-n path. Output is written in the SAME
raw-tracks layout collect_tracks.py uses, so run_phase2.py can pick it up:

    <out_dir>/<patient>/<side__seq>__<teacher_name>.npz

Because these new points have no ground truth (no IR endpoint), treat them as
self-labeling data -- run_phase2.py will copy a single-teacher clip straight
through (no cross-teacher verification available for the extra points alone).

Usage:
    python collect_extra_points.py \\
        --pseudo-labels-dir /mnt/cluster/datasets/STIRprocessed/pseudo_labels \\
        --stir-root /mnt/cluster/datasets/STIRDataset \\
        --repo-root ../track_on \\
        --out-dir /mnt/cluster/datasets/STIRprocessed/extra_points \\
        --teacher cotracker3 --teacher-name peng_tracker \\
        --num-points 128 --min-dist 10 --skip 5
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np

from collect_tracks import _clip_id, _import_stirloader, _run_chunked
from config import TeacherConfig
from teachers import build_teacher
from trajectories import save_track_result_pair


def _frames_from_seq(seq_path, STIRStereoClip, skip: int) -> np.ndarray:
    """[T,H,W,3] RGB uint8 for one seq_path at the given skip."""
    if skip > 1:
        os.environ["SKIP"] = str(skip)
    clip = STIRStereoClip(seq_path)
    left_frames, _ = clip.extractallframes()
    return np.stack(left_frames, axis=0)


def sample_distinct_points(existing: np.ndarray, n: int, H: int, W: int,
                           min_dist: float, rng: np.random.Generator,
                           max_tries_factor: int = 50) -> np.ndarray:
    """n random [x,y] points in [0,W]x[0,H], each at least min_dist pixels from
    every `existing` point AND from every already-accepted new point (rejection
    sampling). Returns [<=n, 2] float32 -- fewer than n if the space is too
    crowded to place them all within the try budget."""
    existing = np.asarray(existing, dtype=np.float32).reshape(-1, 2)
    min_d2 = float(min_dist) ** 2
    accepted = []
    tries = 0
    max_tries = n * max_tries_factor + 100

    while len(accepted) < n and tries < max_tries:
        tries += 1
        cand = rng.uniform([0.0, 0.0], [W, H]).astype(np.float32)  # [x, y]
        if existing.shape[0] and np.min(np.sum((existing - cand) ** 2, axis=1)) < min_d2:
            continue
        if accepted:
            arr = np.asarray(accepted, dtype=np.float32)
            if np.min(np.sum((arr - cand) ** 2, axis=1)) < min_d2:
                continue
        accepted.append(cand)

    if not accepted:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack(accepted, axis=0).astype(np.float32)


def sample_around_points(existing: np.ndarray, n: int, radius: float, H: int, W: int,
                         min_dist: float, rng: np.random.Generator,
                         max_tries_factor: int = 50) -> np.ndarray:
    """n new [x,y] points sampled NEAR the real points instead of uniformly over
    the frame: each candidate picks a random `existing` point and lands at a
    uniform-random offset INSIDE a disk of `radius` pixels around it (uniform in
    area, not bunched at the center), clipped to the frame. Same rejection as
    sample_distinct_points -- each accepted point stays >= min_dist from every
    real point AND every already-accepted new point, so nothing overlaps.

    Densifies the supervision around the actual tracked tissue rather than on
    random background. Needs radius > min_dist to have room to place anything;
    returns [<=n, 2] (fewer if the disks are too crowded for the try budget)."""
    existing = np.asarray(existing, dtype=np.float32).reshape(-1, 2)
    if existing.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    min_d2 = float(min_dist) ** 2
    accepted = []
    tries = 0
    max_tries = n * max_tries_factor + 100

    while len(accepted) < n and tries < max_tries:
        tries += 1
        center = existing[rng.integers(existing.shape[0])]
        # uniform in a disk: r = radius*sqrt(u), theta = 2*pi*v
        rr = radius * np.sqrt(rng.random())
        theta = 2.0 * np.pi * rng.random()
        cand = center + np.array([rr * np.cos(theta), rr * np.sin(theta)], dtype=np.float32)
        cand[0] = np.clip(cand[0], 0.0, W)
        cand[1] = np.clip(cand[1], 0.0, H)
        if np.min(np.sum((existing - cand) ** 2, axis=1)) < min_d2:
            continue
        if accepted:
            arr = np.asarray(accepted, dtype=np.float32)
            if np.min(np.sum((arr - cand) ** 2, axis=1)) < min_d2:
                continue
        accepted.append(cand.astype(np.float32))

    if not accepted:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack(accepted, axis=0).astype(np.float32)


def _render_extra_viz(frames, real_coords, real_vis, new_coords, new_vis, radius,
                      out_path, gt_end=None, fps=10, vis_threshold=0.5, point_radius=3):
    """Per-clip sanity video, BOTH point groups tracked over the full clip (not
    just frame 0): real/existing pseudo-label points (blue start, magenta
    current/end, cyan trail -- their own already-verified track from
    run_phase2.py) and the new points just collected here (green start, red
    current/end, yellow trail). real_coords/new_coords are [M or N, T, 2]
    native pixels, real_vis/new_vis [M or N, T]. The sampling-radius circle (if
    radius > 0) follows each real point's CURRENT position each frame, not its
    frame-0 spot.

    gt_end: optional [K, 2] TRUE end-frame IR-tattoo centers (STIRLoader's
    getendcenters -- unordered, no per-point correspondence to any track, see
    verifier.endpoint_error's docstring for why). Drawn as a white X-marker
    (not a filled dot, so it can never be mistaken for a tracked point),
    persistent every frame, so you can visually judge how close the real/new
    points' tracked endpoints (magenta/red) land relative to it -- exactly the
    gap evaluate_labels.py measures numerically.
    """
    import cv2
    T, H, W, _ = frames.shape
    M, N = real_coords.shape[0], new_coords.shape[0]
    blue, magenta, cyan = (255, 0, 0), (255, 0, 255), (255, 255, 0)      # BGR -- real points
    green, red, yellow = (0, 255, 0), (0, 0, 255), (0, 255, 255)          # BGR -- new points
    white = (255, 255, 255)                                                # BGR -- GT end target
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    def _draw_track(f, coords, vis, i, t, start_color, cur_color, trail_color):
        pts, vi = coords[i, : t + 1], vis[i, : t + 1]
        for a in range(len(pts) - 1):
            if vi[a] < vis_threshold or vi[a + 1] < vis_threshold:
                continue
            cv2.line(f, tuple(pts[a].astype(int)), tuple(pts[a + 1].astype(int)),
                     trail_color, 1, cv2.LINE_AA)
        cv2.circle(f, tuple(coords[i, 0].astype(int)), point_radius, start_color, -1, cv2.LINE_AA)
        if vi[-1] >= vis_threshold:
            cv2.circle(f, tuple(coords[i, t].astype(int)), point_radius, cur_color, -1, cv2.LINE_AA)

    try:
        for t in range(T):
            f = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
            for i in range(M):
                _draw_track(f, real_coords, real_vis, i, t, blue, magenta, cyan)
                if radius > 0 and real_vis[i, t] >= vis_threshold:
                    c = tuple(real_coords[i, t].astype(int))
                    cv2.circle(f, c, int(radius), blue, 1, cv2.LINE_AA)
            for i in range(N):
                _draw_track(f, new_coords, new_vis, i, t, green, red, yellow)
            if gt_end is not None:
                for gx, gy in gt_end:
                    cv2.drawMarker(f, (int(gx), int(gy)), white,
                                   markerType=cv2.MARKER_TILTED_CROSS,
                                   markerSize=2 * point_radius + 6, thickness=2)
            writer.write(f)
    finally:
        writer.release()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pseudo-labels-dir", required=True,
                   help="run_phase2.py output: existing points + which clips to process")
    p.add_argument("--stir-root", help="STIR dataset root, for frames",
                   default="/mnt/cluster/datasets/STIRDataset")
    p.add_argument("--out-dir", required=True, help="where to write the new raw tracks")
    # Absolute: this default used to be "../track_on", which only resolved from
    # the old working directory. Same clone, spelled out.
    p.add_argument("--repo-root",
                   default="/mnt/nct-zfs/TCO-Test/venkateda/miccai_challenges/stir/track_on",
                   help="teacher's cloned repo (track_on for cotracker3)")
    p.add_argument("--checkpoint", default="/mnt/nct-zfs/TCO-Test/venkateda/miccai_challenges/stir/lite-tracker-master/model/scaled_online.pth",
                    help="teacher checkpoint (ignored by cotracker3)")
    p.add_argument("--teacher", default="cotracker3",
                   help="registry name to BUILD (default cotracker3 == peng_tracker's model)")
    p.add_argument("--teacher-name", default="",
                   help="name to SAVE as in filenames (default: same as --teacher, "
                        "e.g. set cotracker3)")
    p.add_argument("--num-points", type=int, nargs="+", default=[128],
                   help="points to add per clip: ONE value for a fixed count, or "
                        "TWO for an inclusive random range drawn per clip, e.g. "
                        "--num-points 7 36")
    p.add_argument("--min-dist", type=float, default=10.0,
                   help="minimum pixel distance between any two points (existing or new)")
    p.add_argument("--around-radius", type=float, default=0.0,
                   help="if > 0, sample new points INSIDE a disk of this many pixels "
                        "around each real point (densify near tracked tissue) instead "
                        "of uniformly over the whole frame. Needs > --min-dist to have "
                        "room. 0 (default) = old uniform-over-frame behavior.")
    p.add_argument("--limit", type=int, default=0,
                   help="process only the first N clips (0 = all). Use --limit 1 to "
                        "try a single sequence first.")
    p.add_argument("--viz-dir", default="",
                   help="if set, write a per-clip sanity mp4 here: real points in blue "
                        "(+ sampling circle), new points' tracks in green/red/yellow.")
    p.add_argument("--skip", type=int, default=5,
                   help="must match the --skip used for the existing pseudo-labels")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-cycle", action="store_true",
                   help="skip the backward pass (faster, disables cycle-consistency)")
    p.add_argument("--chunk-frames", type=int, default=None,
                   help="Split each clip into chunks of this many frames along time "
                        "before tracking, reseeding each chunk's queries at the "
                        "previous chunk's last known position -- avoids OOM on very "
                        "long clips (e.g. STIR has some 700s+ outliers) that blow up "
                        "memory regardless of teacher, same mechanism as "
                        "collect_tracks.py's --chunk-frames.")
    p.add_argument("--patients", nargs="+", default=None,
                   help="only process these patient folders, e.g. --patients 0 1")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    teacher_name = args.teacher_name or args.teacher
    rng = np.random.default_rng(args.seed)

    if len(args.num_points) == 1:
        npts_low = npts_high = args.num_points[0]
    elif len(args.num_points) == 2:
        npts_low, npts_high = sorted(args.num_points)
    else:
        raise SystemExit("--num-points takes 1 value (fixed) or 2 (a range)")
    print(f"Points per clip: {npts_low}"
          + (f"-{npts_high} (random per clip)" if npts_high != npts_low else " (fixed)"))

    if args.around_radius > 0:
        print(f"Sampling new points within {args.around_radius}px disks around real points")
        if args.around_radius <= args.min_dist:
            print(f"[warn] --around-radius {args.around_radius} <= --min-dist "
                  f"{args.min_dist}: little/no room to place points inside the disks")
    if args.viz_dir:
        os.makedirs(args.viz_dir, exist_ok=True)

    cfg = TeacherConfig(args.teacher, checkpoint=args.checkpoint, repo_root=args.repo_root)
    print(f"Building teacher '{args.teacher}' (saving as '{teacher_name}')...")
    teacher = build_teacher(cfg)

    getviddirs2d_STIR, STIRStereoClip = _import_stirloader()
    seq_map = {_clip_id(sp): sp for sp in getviddirs2d_STIR(args.stir_root)}

    pl_paths = sorted(glob.glob(os.path.join(args.pseudo_labels_dir, "**", "*.npz"), recursive=True))
    if not pl_paths:
        raise SystemExit(f"No pseudo-labels found under {args.pseudo_labels_dir}")

    allowed = set(args.patients) if args.patients else None
    os.makedirs(args.out_dir, exist_ok=True)

    total_added = 0
    n_clips = 0
    for pl_path in pl_paths:
        patient = os.path.basename(os.path.dirname(pl_path))
        seq_part = os.path.splitext(os.path.basename(pl_path))[0]
        clip_id = f"{patient}__{seq_part}"
        if allowed is not None and patient not in allowed:
            continue

        out_path = Path(args.out_dir) / patient / f"{seq_part}__{teacher_name}.npz"
        if out_path.exists() and not args.overwrite:
            print(f"[{clip_id}] already exists, skipping (--overwrite to redo)")
            continue

        seq_path = seq_map.get(clip_id)
        if seq_path is None:
            print(f"[skip] {clip_id}: not found under {args.stir_root}")
            continue

        with np.load(pl_path) as d:
            existing_coords = d["coords"].astype(np.float32)      # [M,T,2] full track, for viz
            existing_vis = d["visibility"].astype(np.float32)     # [M,T]
            existing_pts = existing_coords[:, 0, :]                # [M,2] frame 0, for sampling only

        try:
            frames = _frames_from_seq(seq_path, STIRStereoClip, args.skip)
        except (AssertionError, IndexError) as e:
            print(f"[skip] {clip_id}: {e}")
            continue

        H, W = frames.shape[1:3]
        target_n = int(rng.integers(npts_low, npts_high + 1))  # inclusive range per clip
        if args.around_radius > 0:
            new_pts = sample_around_points(existing_pts, target_n, args.around_radius,
                                           H, W, args.min_dist, rng)
        else:
            new_pts = sample_distinct_points(existing_pts, target_n, H, W,
                                             args.min_dist, rng)
        if len(new_pts) == 0:
            print(f"[skip] {clip_id}: could not place any distinct points")
            continue
        if len(new_pts) < target_n:
            print(f"[{clip_id}] only placed {len(new_pts)}/{target_n} distinct "
                  f"points (space crowded at min_dist={args.min_dist})")

        if args.chunk_frames:
            fwd, bwd = _run_chunked(teacher, frames, new_pts, args.chunk_frames,
                                    use_cycle=not args.no_cycle)
        elif args.no_cycle:
            fwd, bwd = teacher.track(frames, new_pts), None
        else:
            fwd, bwd = teacher.track_cycle(frames, new_pts)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_track_result_pair(out_path, fwd, bwd)
        total_added += len(new_pts)
        n_clips += 1
        print(f"[{teacher_name}] {clip_id}: {len(new_pts)} new points -> {out_path}")

        if args.viz_dir:
            if frames.shape[0] != existing_coords.shape[1]:
                print(f"[{clip_id}] viz skipped: {frames.shape[0]} frames now vs. "
                      f"{existing_coords.shape[1]} in the pseudo-label -- is --skip "
                      f"{args.skip} the same value used to build the pseudo-labels?")
            else:
                try:
                    gt_end = np.array(STIRStereoClip(seq_path).getendcenters(left=True),
                                      dtype=np.float32)
                    if len(gt_end) == 0:
                        gt_end = None
                except (AssertionError, IndexError):
                    gt_end = None  # no end segmentation for this clip -- viz without it

                viz_path = Path(args.viz_dir) / f"{clip_id}__{teacher_name}.mp4"
                _render_extra_viz(frames, existing_coords, existing_vis,
                                  fwd.coords, fwd.visibility, args.around_radius, viz_path,
                                  gt_end=gt_end)
                print(f"    viz -> {viz_path}"
                     + (f" ({len(gt_end)} GT end point(s) marked)" if gt_end is not None else ""))

        if args.limit and n_clips >= args.limit:
            print(f"[limit] stopping after {n_clips} clip(s)")
            break

    print(f"\nDone. Added {total_added} points across {n_clips} clips into "
          f"{args.out_dir}/<patient>/*__{teacher_name}.npz")


if __name__ == "__main__":
    main()
