"""Draw pseudo-label tracks over a clip's frames and save as a video.

Blue dot = each point's start position (frame 0). Red dot = its final
position (last frame). A yellow trail traces the point's predicted path
between them, growing frame by frame, skipping segments where the
pseudo-label marks the point occluded (visibility < --vis-threshold).

Usage:
    python visualize_pseudo_labels.py \\
        --pseudo-labels-dir data/pseudo_labels \\
        --stir-root /mnt/cluster/datasets/STIRDataset \\
        --clip-id 1__left__seq05 \\
        --skip 5 \\
        --out-dir viz/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from collect_tracks import load_clip_frames

_BLUE = (255, 0, 0)     # BGR
_RED = (0, 0, 255)
_TRAIL = (0, 255, 255)  # yellow


def _load_pseudo_label(pseudo_labels_dir: str, clip_id: str):
    patient, seq_part = clip_id.split("__", 1)
    path = Path(pseudo_labels_dir) / patient / f"{seq_part}.npz"
    if not path.exists():
        raise FileNotFoundError(f"No pseudo-label at {path}")
    d = np.load(path)
    return d["coords"], d["visibility"]


def render(frames: np.ndarray, coords: np.ndarray, visibility: np.ndarray,
          out_path: Path, fps: int, max_points, point_radius: int,
          trail_thickness: int, vis_threshold: float) -> None:
    T, H, W, _ = frames.shape
    N = coords.shape[0]

    idx = np.arange(N)
    if max_points is not None and N > max_points:
        idx = np.random.default_rng(0).choice(N, size=max_points, replace=False)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    try:
        for t in range(T):
            frame_bgr = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)

            for i in idx:
                pts = coords[i, : t + 1]
                vis = visibility[i, : t + 1]

                for a in range(len(pts) - 1):
                    if vis[a] < vis_threshold or vis[a + 1] < vis_threshold:
                        continue
                    p0 = tuple(pts[a].astype(int))
                    p1 = tuple(pts[a + 1].astype(int))
                    cv2.line(frame_bgr, p0, p1, _TRAIL, trail_thickness, cv2.LINE_AA)

                start_pt = tuple(coords[i, 0].astype(int))
                cv2.circle(frame_bgr, start_pt, point_radius, _BLUE, -1, cv2.LINE_AA)

                end_pt = tuple(coords[i, -1].astype(int))
                cv2.circle(frame_bgr, end_pt, point_radius, _RED, -1, cv2.LINE_AA)

            writer.write(frame_bgr)
    finally:
        writer.release()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pseudo-labels-dir", required=True)
    p.add_argument("--stir-root", required=True)
    p.add_argument("--clip-id", required=True, help="e.g. 1__left__seq05")
    p.add_argument("--skip", type=int, default=1,
                   help="must match the --skip used by collect_tracks.py for this clip")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--max-points", type=int, default=None,
                   help="randomly subsample to this many points for readability "
                        "(default: draw all)")
    p.add_argument("--point-radius", type=int, default=3)
    p.add_argument("--trail-thickness", type=int, default=1)
    p.add_argument("--vis-threshold", type=float, default=0.5)
    args = p.parse_args()

    coords, visibility = _load_pseudo_label(args.pseudo_labels_dir, args.clip_id)
    frames = load_clip_frames(args.stir_root, args.clip_id, args.skip)

    if frames.shape[0] != coords.shape[1]:
        raise ValueError(
            f"Frame count mismatch: {frames.shape[0]} frames loaded vs. "
            f"{coords.shape[1]} in the pseudo-label -- is --skip {args.skip} the same "
            f"value used when this clip was collected?"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.clip_id}.mp4"
    render(frames, coords, visibility, out_path, fps=args.fps,
          max_points=args.max_points, point_radius=args.point_radius,
          trail_thickness=args.trail_thickness, vis_threshold=args.vis_threshold)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
