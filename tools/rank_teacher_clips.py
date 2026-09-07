"""Rank STIROrig clips for a six-teacher comparison GIF.

This is a visualization selector, not an evaluation metric. It considers only
clips present for every requested teacher, forms a robust median trajectory
across teachers, and favours clips with visible, sustained point motion. Large
cross-teacher disagreement is reported and mildly penalized so a single broken
tracker does not make a clip look artificially exciting.

Example:

    python tools/rank_teacher_clips.py \
        --raw-tracks-root data/STIROrig_tracks --top 20
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_TEACHERS = (
    "cotracker3", "alltracker", "locotrack",
    "mft", "trackon2", "trackon_r",
)


@dataclass(frozen=True)
class Candidate:
    clip_id: str
    points: int
    frames: int
    excursion_px: float
    endpoint_px: float
    path_px: float
    disagreement_px: float
    visible_fraction: float
    score: float


def files_for_teacher(root: Path, teacher: str) -> dict[str, Path]:
    teacher_root = root / teacher
    if not teacher_root.is_dir():
        raise FileNotFoundError(f"teacher directory not found: {teacher_root}")

    suffix = f"__{teacher}"
    files = {}
    for path in teacher_root.rglob(f"*{suffix}.npz"):
        rel = path.relative_to(teacher_root)
        if len(rel.parts) < 2:
            continue
        stem = path.stem
        if not stem.endswith(suffix):
            continue
        sequence = stem[:-len(suffix)]
        clip_id = f"{rel.parts[0]}__{sequence}"
        files[clip_id] = path
    return files


def score_clip(clip_id: str, paths: list[Path], vis_threshold: float) -> Candidate:
    coordinates = []
    visibilities = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            coordinates.append(np.asarray(data["fwd_coords"], dtype=np.float32))
            visibilities.append(np.asarray(data["fwd_vis"], dtype=np.float32))

    shapes = {array.shape for array in coordinates}
    if len(shapes) != 1:
        raise ValueError(f"coordinate shapes differ: {sorted(shapes)}")
    points, frames, xy = coordinates[0].shape
    if xy != 2:
        raise ValueError(f"expected [N,T,2], got {coordinates[0].shape}")
    if any(vis.shape != (points, frames) for vis in visibilities):
        raise ValueError("visibility shape differs from coordinates")

    tracks = np.stack(coordinates, axis=0)       # [teacher,N,T,2]
    visibility = np.stack(visibilities, axis=0)  # [teacher,N,T]
    consensus = np.median(tracks, axis=0)        # [N,T,2]

    displacement = np.linalg.norm(consensus - consensus[:, :1], axis=-1)
    excursion = float(np.median(np.max(displacement, axis=1)))
    endpoint = float(np.median(displacement[:, -1]))
    steps = np.linalg.norm(np.diff(consensus, axis=1), axis=-1)
    path_length = float(np.median(np.sum(steps, axis=1)))
    disagreement = float(np.median(
        np.linalg.norm(tracks - consensus[None], axis=-1)))
    visible_fraction = float(np.mean(visibility >= vis_threshold))

    # Excursion drives the ranking. Endpoint displacement rewards a clear
    # start-to-finish change, while path length adds only a small bonus because
    # it can also grow through jitter. Disagreement is a mild visual-quality
    # penalty, not a scientific claim about accuracy.
    score = excursion + 0.25 * endpoint + 0.05 * path_length - 0.15 * disagreement
    return Candidate(
        clip_id, points, frames, excursion, endpoint, path_length,
        disagreement, visible_fraction, score,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-tracks-root", required=True)
    parser.add_argument("--teachers", nargs=6, default=DEFAULT_TEACHERS)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--min-points", type=int, default=4)
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=160)
    parser.add_argument("--min-visible", type=float, default=0.60)
    parser.add_argument("--vis-threshold", type=float, default=0.50)
    args = parser.parse_args()

    root = Path(args.raw_tracks_root)
    teachers = tuple(args.teachers)
    by_teacher = {teacher: files_for_teacher(root, teacher) for teacher in teachers}
    common = set.intersection(*(set(files) for files in by_teacher.values()))
    if not common:
        raise SystemExit(f"no clips are shared by all six teachers under {root}")

    candidates = []
    rejected = 0
    errors = []
    for clip_id in sorted(common):
        paths = [by_teacher[teacher][clip_id] for teacher in teachers]
        try:
            candidate = score_clip(clip_id, paths, args.vis_threshold)
        except (KeyError, ValueError) as error:
            errors.append((clip_id, str(error)))
            continue
        if not (candidate.points >= args.min_points and
                args.min_frames <= candidate.frames <= args.max_frames and
                candidate.visible_fraction >= args.min_visible):
            rejected += 1
            continue
        candidates.append(candidate)

    candidates.sort(key=lambda item: (-item.score, item.clip_id))
    shown = candidates[:max(0, args.top)]

    print(f"{len(common)} clips covered by all {len(teachers)} teachers; "
          f"{len(candidates)} pass visualization filters; {rejected} filtered; "
          f"{len(errors)} malformed")
    print()
    print(f"{'rank':>4}  {'clip_id':<27} {'pts':>3} {'frm':>4} "
          f"{'exc_px':>7} {'end_px':>7} {'path_px':>8} {'dis_px':>7} "
          f"{'vis':>6} {'score':>8}")
    for rank, item in enumerate(shown, 1):
        print(f"{rank:>4}  {item.clip_id:<27} {item.points:>3} {item.frames:>4} "
              f"{item.excursion_px:>7.1f} {item.endpoint_px:>7.1f} "
              f"{item.path_px:>8.1f} {item.disagreement_px:>7.1f} "
              f"{item.visible_fraction:>6.1%} {item.score:>8.1f}")

    if shown:
        print()
        print(f"Suggested render: CLIP={shown[0].clip_id} bash tools/make_teacher_gif.sh")
    if errors:
        print(f"\nFirst malformed clip: {errors[0][0]}: {errors[0][1]}")


if __name__ == "__main__":
    main()
