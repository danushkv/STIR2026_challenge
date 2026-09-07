"""Render six cached teacher trajectories as a synchronized 2x3 comparison.

The input is the phase-1 Hugging Face/raw-track layout:

    <raw_tracks_root>/<teacher>/<patient>/<side>__<seq>__<teacher>.npz

Example:

    python tools/render_teachers.py \
        --raw-tracks-root data/STIROrig_tracks \
        --stir-root data/STIRDataset \
        --clip-id 4__left__seq00 \
        --out assets/teacher_comparison.mp4

Every panel shows the same video frame and query points. Point colours are held
constant across panels, making disagreement visible as the coloured trails
separate. A hollow ring marks each starting location; dim crosses mark a
teacher's currently invisible points.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from collect_tracks import load_clip_frames  # noqa: E402


DEFAULT_TEACHERS = (
    "cotracker3", "alltracker", "locotrack",
    "mft", "trackon2", "trackon_r",
)
DISPLAY_NAMES = {
    "cotracker3": "CoTracker3",
    "alltracker": "AllTracker",
    "locotrack": "LocoTrack",
    "mft": "MFT",
    "trackon2": "Track-On2",
    "trackon_r": "Track-On-R",
}
# BGR colours chosen for contrast on red/pink tissue. The same query receives
# the same colour in every teacher panel.
POINT_COLOURS = (
    (64, 224, 255), (84, 255, 135), (255, 172, 70), (255, 100, 220),
    (245, 240, 65), (95, 145, 255), (190, 105, 255), (80, 255, 235),
    (255, 210, 120), (145, 255, 185), (255, 145, 145), (210, 210, 255),
)
ACCENTS = {
    "cotracker3": (244, 151, 76),
    "alltracker": (107, 173, 255),
    "locotrack": (114, 215, 160),
    "mft": (206, 154, 238),
    "trackon2": (83, 193, 241),
    "trackon_r": (126, 126, 255),
}


def track_path(root: Path, teacher: str, clip_id: str) -> Path:
    patient, sequence = clip_id.split("__", 1)
    return root / teacher / patient / f"{sequence}__{teacher}.npz"


def load_tracks(root: Path, teachers: tuple[str, ...], clip_id: str):
    tracks = {}
    for teacher in teachers:
        path = track_path(root, teacher, clip_id)
        if not path.is_file():
            raise FileNotFoundError(f"missing {teacher} track for {clip_id}: {path}")
        with np.load(path, allow_pickle=False) as data:
            tracks[teacher] = (
                np.asarray(data["fwd_coords"], dtype=np.float32),
                np.asarray(data["fwd_vis"], dtype=np.float32),
            )
    return tracks


def point_indices(n: int, max_points: int) -> np.ndarray:
    if max_points <= 0 or n <= max_points:
        return np.arange(n)
    # Evenly spaced and deterministic, unlike a random visual subsample.
    return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))


def draw_panel(frame: np.ndarray, coords: np.ndarray, visibility: np.ndarray,
               t: int, panel_width: int, trail: int, vis_threshold: float,
               max_points: int) -> tuple[np.ndarray, int, int]:
    source_h, source_w = frame.shape[:2]
    image_h = int(round(panel_width * source_h / source_w))
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    panel = cv2.resize(frame_bgr, (panel_width, image_h), interpolation=cv2.INTER_AREA)
    scale = np.array([panel_width / source_w, image_h / source_h], np.float32)
    idx = point_indices(coords.shape[0], max_points)
    visible_now = 0

    for colour_index, point_index in enumerate(idx):
        colour = POINT_COLOURS[colour_index % len(POINT_COLOURS)]
        start = tuple(np.round(coords[point_index, 0] * scale).astype(int))
        cv2.circle(panel, start, 6, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.circle(panel, start, 8, (25, 32, 45), 1, cv2.LINE_AA)

        first = max(0, t - trail)
        for step in range(first, t):
            if (visibility[point_index, step] < vis_threshold or
                    visibility[point_index, step + 1] < vis_threshold):
                continue
            p0 = tuple(np.round(coords[point_index, step] * scale).astype(int))
            p1 = tuple(np.round(coords[point_index, step + 1] * scale).astype(int))
            age = (step - first + 1) / max(1, t - first)
            faded = tuple(int(channel * (0.42 + 0.58 * age)) for channel in colour)
            cv2.line(panel, p0, p1, faded, 2, cv2.LINE_AA)

        current = tuple(np.round(coords[point_index, t] * scale).astype(int))
        if visibility[point_index, t] >= vis_threshold:
            visible_now += 1
            cv2.circle(panel, current, 6, (18, 24, 35), -1, cv2.LINE_AA)
            cv2.circle(panel, current, 4, colour, -1, cv2.LINE_AA)
        else:
            cv2.line(panel, (current[0] - 4, current[1] - 4),
                     (current[0] + 4, current[1] + 4), (135, 145, 160), 2, cv2.LINE_AA)
            cv2.line(panel, (current[0] + 4, current[1] - 4),
                     (current[0] - 4, current[1] + 4), (135, 145, 160), 2, cv2.LINE_AA)
    return panel, visible_now, len(idx)


def render(frames: np.ndarray, tracks, teachers: tuple[str, ...], out_path: Path,
           fps: int, panel_width: int, trail: int, vis_threshold: float,
           max_points: int, clip_id: str) -> None:
    if len(teachers) != 6:
        raise ValueError("the comparison layout requires exactly six teachers")
    frame_count = frames.shape[0]
    shapes = {teacher: tracks[teacher][0].shape for teacher in teachers}
    for teacher, (coords, visibility) in tracks.items():
        if coords.ndim != 3 or coords.shape[-1] != 2:
            raise ValueError(f"{teacher}: expected coords [N,T,2], got {coords.shape}")
        if visibility.shape != coords.shape[:2]:
            raise ValueError(
                f"{teacher}: visibility {visibility.shape} != coords {coords.shape[:2]}")
        if coords.shape[1] != frame_count:
            raise ValueError(
                f"{teacher}: {coords.shape[1]} track frames != {frame_count} video frames; "
                "check --skip")

    source_h, source_w = frames.shape[1:3]
    image_h = int(round(panel_width * source_h / source_w))
    header_h, gap, outer, footer_h = 42, 12, 18, 40
    panel_h = header_h + image_h
    canvas_w = outer * 2 + panel_width * 3 + gap * 2
    canvas_h = outer * 2 + panel_h * 2 + gap + footer_h
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (canvas_w, canvas_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cv2 could not open a writer for {out_path}")

    try:
        for t in range(frame_count):
            canvas = np.full((canvas_h, canvas_w, 3), (20, 27, 40), np.uint8)
            for panel_index, teacher in enumerate(teachers):
                row, col = divmod(panel_index, 3)
                x = outer + col * (panel_width + gap)
                y = outer + row * (panel_h + gap)
                coords, visibility = tracks[teacher]
                panel, visible, shown = draw_panel(
                    frames[t], coords, visibility, t, panel_width, trail,
                    vis_threshold, max_points,
                )
                accent = ACCENTS[teacher]
                cv2.rectangle(canvas, (x, y), (x + panel_width, y + header_h),
                              (31, 41, 57), -1)
                cv2.rectangle(canvas, (x, y), (x + 6, y + header_h), accent, -1)
                cv2.putText(canvas, DISPLAY_NAMES.get(teacher, teacher),
                            (x + 19, y + 28), cv2.FONT_HERSHEY_DUPLEX,
                            0.68, (248, 250, 252), 1, cv2.LINE_AA)
                status = f"{visible}/{shown} visible"
                (text_w, _), _ = cv2.getTextSize(
                    status, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)
                cv2.putText(canvas, status,
                            (x + panel_width - text_w - 13, y + 27),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                            (190, 199, 213), 1, cv2.LINE_AA)
                canvas[y + header_h:y + panel_h, x:x + panel_width] = panel
                cv2.rectangle(canvas, (x, y), (x + panel_width, y + panel_h),
                              (70, 82, 101), 1)

            footer_y = canvas_h - footer_h
            caption = (f"{clip_id}   |   same query points   |   frame {t + 1}/{frame_count}"
                       f"   |   temporal stride 5")
            cv2.putText(canvas, caption, (outer, footer_y + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (203, 213, 225), 1, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()

    print(f"wrote {out_path}")
    print("track shapes:", ", ".join(f"{key}={value}" for key, value in shapes.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-tracks-root", required=True)
    parser.add_argument("--stir-root", required=True)
    parser.add_argument("--clip-id", default="4__left__seq00")
    parser.add_argument("--out", required=True)
    parser.add_argument("--teachers", nargs=6, default=DEFAULT_TEACHERS)
    parser.add_argument("--skip", type=int, default=5)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--panel-width", type=int, default=360)
    parser.add_argument("--trail", type=int, default=12)
    parser.add_argument("--vis-threshold", type=float, default=0.5)
    parser.add_argument("--max-points", type=int, default=16,
                        help="0 draws every query; default 16 keeps panels legible")
    args = parser.parse_args()

    teachers = tuple(args.teachers)
    tracks = load_tracks(Path(args.raw_tracks_root), teachers, args.clip_id)
    frames = load_clip_frames(args.stir_root, args.clip_id, args.skip)
    render(frames, tracks, teachers, Path(args.out), args.fps,
           args.panel_width, args.trail, args.vis_threshold, args.max_points,
           args.clip_id)


if __name__ == "__main__":
    main()
