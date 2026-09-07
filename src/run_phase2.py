"""Phase 2 driver: build endpoints_by_clip from STIRLoader, then verify.

Layout this reads (written by collect_tracks.py, one folder per teacher):
    raw_tracks_root/<teacher>/<patient>/<side>__<seq>__<teacher>.npz

Different teachers can cover different clip sets -- some skip clips with no
IR-tattoo segmentation entirely, some (cotracker3 --random-queries-n,
peng_tracker_all, or any other future random-queries-style run) cover those
clips too, using random points on frame 0 instead. There's no fixed pattern
to how many "extra" clips any given teacher has, so clips are grouped by
whichever EXACT subset of --teachers actually has output for them:
  - clips covered by >=2 teachers go through the verifier, cross-checked
    against each other.
  - clips covered by exactly ONE teacher (e.g. a peng_tracker_all-exclusive
    sequence no other teacher processed) have nothing to verify against, so
    that teacher's raw track is copied straight through as the pseudo-label
    instead -- see _copy_through().
Nothing here is hardcoded to a specific teacher name.

Endpoint ground truth (STIRLoader.getendcenters) is looked up per clip
regardless of grouping: clips with real segmentation get it, clips without
naturally don't (endpoint anchoring is then skipped for those, per
verifier.py) -- independent of which teachers happen to cover the clip.

Usage:
    python run_phase2.py \\
        --raw-tracks-root /mnt/cluster/datasets/STIRprocessed \\
        --stir-root /mnt/cluster/datasets/STIRDataset \\
        --teachers mft mftiq cotracker3 locotrack bootstapir alltracker peng_tracker_all \\
        --out-dir data/pseudo_labels
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Set

import numpy as np

from collect_tracks import _clip_id, _import_stirloader
from config import PipelineConfig, TeacherConfig, VerifierConfig
from pseudo_label import run_from_disk, save_pseudo_label
from trajectories import PseudoLabel, load_track_result_pair


def _discover_clip_ids(raw_tracks_root: str, teacher: str) -> Set[str]:
    """clip_ids ('patient__side__seq') for which `teacher` wrote output."""
    teacher_dir = Path(raw_tracks_root) / teacher
    if not teacher_dir.is_dir():
        return set()
    suffix = f"__{teacher}.npz"
    clip_ids = set()
    for patient_dir in teacher_dir.iterdir():
        if not patient_dir.is_dir():
            continue
        for f in patient_dir.glob(f"*{suffix}"):
            clip_ids.add(f"{patient_dir.name}__{f.name[: -len(suffix)]}")
    return clip_ids


def _diagnose_empty_teacher(raw_tracks_root: str, teacher: str, patients) -> None:
    """Called when a teacher found 0 clips (optionally after --patients filtering)
    but the user expects files to be there. Prints exactly what _discover_clip_ids
    looked for vs. what's actually on disk, so a naming mismatch is visible
    instead of just silently showing up as "missing" everywhere downstream.
    """
    teacher_dir = Path(raw_tracks_root) / teacher
    print(f"[diagnose] {teacher!r}: expected folder {teacher_dir}")
    if not teacher_dir.is_dir():
        print(f"[diagnose] {teacher!r}: that folder does NOT exist (or isn't a directory)")
        return

    patient_dirs = sorted(p.name for p in teacher_dir.iterdir() if p.is_dir())
    print(f"[diagnose] {teacher!r}: patient subfolders found: {patient_dirs}")

    check_patients = [p for p in (patients or [])] or patient_dirs[:1]
    suffix = f"__{teacher}.npz"
    for patient in check_patients:
        patient_dir = teacher_dir / patient
        if not patient_dir.is_dir():
            print(f"[diagnose] {teacher!r}: no subfolder named {patient!r} "
                  f"under {teacher_dir} (looked for exact match)")
            continue
        files = sorted(f.name for f in patient_dir.iterdir())
        matching = [f for f in files if f.endswith(suffix)]
        print(f"[diagnose] {teacher!r}/{patient!r}: {len(files)} file(s) present: {files}")
        print(f"[diagnose] {teacher!r}/{patient!r}: expected suffix {suffix!r} -> "
              f"{len(matching)} match(es)")


def _build_endpoints_by_clip(stir_root: str, clip_ids: Set[str]) -> Dict[str, np.ndarray]:
    """clip_id -> [M, 2] raw end-frame IR-tattoo centers, via STIRLoader.

    NOT index-aligned with any teacher's queries -- see verifier.endpoint_error.
    Clips with no segmentation simply get no entry (endpoint anchoring is then
    skipped for them by the verifier).
    """
    getviddirs2d_STIR, STIRStereoClip = _import_stirloader()
    seq_paths = {_clip_id(p): p for p in getviddirs2d_STIR(stir_root)}

    endpoints_by_clip = {}
    for clip_id in clip_ids:
        seq_path = seq_paths.get(clip_id)
        if seq_path is None:
            print(f"[warn] {clip_id}: not found under {stir_root}, skipping endpoints")
            continue
        try:
            clip = STIRStereoClip(seq_path)
            end_centers = np.array(clip.getendcenters(left=True), dtype=np.float32)
        except (AssertionError, IndexError) as e:
            print(f"[warn] {clip_id}: could not load end centers ({e})")
            continue
        if len(end_centers) > 0:
            endpoints_by_clip[clip_id] = end_centers
    return endpoints_by_clip


def _copy_through(clip_id: str, teacher_name: str, raw_tracks_dir: str) -> PseudoLabel:
    """No other teacher covers this clip, so there's nothing to verify against --
    copy that teacher's raw forward track straight through as the pseudo-label,
    weighted by its own visibility (no cycle/agreement scoring applied).
    """
    patient, seq_part = clip_id.split("__", 1)
    path = Path(raw_tracks_dir) / teacher_name / patient / f"{seq_part}__{teacher_name}.npz"
    fwd, _ = load_track_result_pair(path)
    N, T, _ = fwd.coords.shape
    return PseudoLabel(
        coords=fwd.coords,
        visibility=fwd.visibility,
        weight=fwd.visibility.astype(np.float32),
        source=np.zeros((N, T), dtype=np.int32),
        clip_id=clip_id,
        meta={"kept_frac": float(fwd.visibility.mean()), "copied_from": teacher_name},
    )


def _group_by_coverage(teachers: List[str],
                       clip_ids_by_teacher: Dict[str, Set[str]]
                       ) -> Dict[FrozenSet[str], List[str]]:
    """clip_id -> which exact subset of `teachers` has output for it, grouped."""
    all_clip_ids = set.union(*clip_ids_by_teacher.values()) if clip_ids_by_teacher else set()
    groups: Dict[FrozenSet[str], List[str]] = defaultdict(list)
    for clip_id in all_clip_ids:
        covering = frozenset(t for t in teachers if clip_id in clip_ids_by_teacher[t])
        groups[covering].append(clip_id)
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-tracks-root", required=True,
                   help="parent of the per-teacher folders written by collect_tracks.py, "
                        "e.g. /mnt/cluster/datasets/STIRprocessed")
    p.add_argument("--stir-root", required=True, help="STIR dataset root, for endpoints")
    p.add_argument("--teachers", nargs="+", required=True,
                   help="e.g. mft mftiq cotracker3 locotrack bootstapir alltracker "
                        "peng_tracker_all")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mode", choices=["select", "aggregate"], default="select",
                   help="per-frame pick the single best teacher ('select') or "
                        "score-weighted blend of all teachers ('aggregate')")
    p.add_argument("--min-teachers", type=int, default=2,
                   help="clips covered by fewer teachers than this are copy-through'd "
                        "instead of verified when ==1 (default 2: only >=2 covering "
                        "teachers actually get cross-checked)")
    p.add_argument("--patients", nargs="+", default=None,
                   help="Only process these patient folders, e.g. --patients 0 1. "
                        "For a quick sanity check before running the full dataset.")
    p.add_argument("--clip-ids", nargs="+", default=None,
                   help="Only process these exact clip ids, e.g. "
                        "--clip-ids 0__left__seq00. Intended for smoke tests.")
    args = p.parse_args()

    clip_ids_by_teacher = {t: _discover_clip_ids(args.raw_tracks_root, t) for t in args.teachers}

    if args.patients is not None:
        allowed = set(args.patients)
        clip_ids_by_teacher = {
            t: {c for c in ids if c.split("__", 1)[0] in allowed}
            for t, ids in clip_ids_by_teacher.items()
        }
        print(f"Filtered to patients {sorted(allowed)}")
    if args.clip_ids is not None:
        allowed = set(args.clip_ids)
        clip_ids_by_teacher = {
            t: ids & allowed for t, ids in clip_ids_by_teacher.items()
        }
        found = set.union(*clip_ids_by_teacher.values()) if clip_ids_by_teacher else set()
        missing = sorted(allowed - found)
        if missing:
            raise SystemExit(f"requested clip id(s) not found: {missing}")
        print(f"Filtered to clip ids {sorted(allowed)}")

    for t, ids in clip_ids_by_teacher.items():
        if not ids:
            _diagnose_empty_teacher(args.raw_tracks_root, t, args.patients)

    groups = _group_by_coverage(args.teachers, clip_ids_by_teacher)

    all_clip_ids = set.union(*clip_ids_by_teacher.values()) if clip_ids_by_teacher else set()
    endpoints_by_clip = _build_endpoints_by_clip(args.stir_root, all_clip_ids)
    print(f"{len(all_clip_ids)} total clips across {len(args.teachers)} teachers "
          f"in {len(groups)} coverage group(s); {len(endpoints_by_clip)} have endpoint ground truth")

    for covering, clip_ids in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        missing = sorted(set(args.teachers) - covering)
        note = f" (missing: {missing})" if missing else ""

        if len(covering) == 1:
            (teacher_name,) = covering
            print(f"{len(clip_ids)} clip(s) covered ONLY by {teacher_name!r}{note} "
                  f"-> copying straight through, no verification")
            for clip_id in sorted(clip_ids):
                pl = _copy_through(clip_id, teacher_name, args.raw_tracks_root)
                save_pseudo_label(pl, args.out_dir, teacher_names=[teacher_name])
                print(f"[{clip_id}] copied from {teacher_name!r} "
                      f"({pl.meta['kept_frac']:.0%} visible)")
            continue

        if len(covering) < args.min_teachers:
            print(f"[skip] {len(clip_ids)} clip(s) covered by only {sorted(covering)}{note} "
                  f"({len(covering)} < --min-teachers {args.min_teachers})")
            continue

        print(f"{len(clip_ids)} clip(s) covered by {sorted(covering)}{note}")
        cfg = PipelineConfig(teachers=[TeacherConfig(t) for t in args.teachers if t in covering],
                             verifier=VerifierConfig(mode=args.mode))
        run_from_disk(cfg, sorted(clip_ids), args.raw_tracks_root,
                      endpoints_by_clip=endpoints_by_clip, out_dir=args.out_dir)


if __name__ == "__main__":
    main()


# python run_phase2.py \
#   --raw-tracks-root /mnt/cluster/datasets/STIRprocessed \
#   --stir-root /mnt/cluster/datasets/STIRDataset \
#   --teachers mft mftiq cotracker3 locotrack bootstapir alltracker peng_tracker_all \
#   --out-dir /tmp/pseudo_labels_sanity \
#   --patients 16 \
#   --min-teachers 1
