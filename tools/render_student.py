"""Render the student's streaming predictions over a STIR clip, as an mp4.

This is the only script here whose output is a picture of the MODEL, not of the
training data: it streams a clip through LiteTracker exactly the way the
submitted container does -- one frame at a time, causal, with the persistent
memory buffers -- and draws each tracked point plus a fading trail.

    python tools/render_student.py \
        --checkpoint <run>/student_e44.pth \
        --stir-root  /path/to/STIRDataset \
        --clip-id    0__left__seq00 \
        --out-dir    assets/raw

QUERY POINTS. By default it tracks the IR tattoos -- a median of 6 per clip,
which is what the model is scored on but looks sparse. `--grid N` instead seeds
a point every N pixels, which is what the 2026 harness actually hands the
container (`queries.json` is an arbitrary 64 px grid), so it is both denser to
look at and closer to the real workload. `--grid 64 --with-tattoos` draws both.

Ground truth, when the clip has it, is drawn as hollow circles: green at the
frame-0 IR tattoo (where tracking starts) and blue at the frame-T tattoo (where
it should end up). The gap between a track and its nearest blue circle at the
end IS the endpoint error the tables report. Grid points have no ground truth,
so those circles simply are not drawn for them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from collect_tracks import _clip_id, _import_stirloader          # noqa: E402
from student_lt_wrapper import load_student_as_litetracker       # noqa: E402
from eval_common import load_val_clip                            # noqa: E402

# BGR, cycled per point. Chosen to stay distinguishable on red/pink tissue.
_PALETTE = [(60, 220, 255), (80, 255, 120), (255, 160, 60),
            (255, 90, 200), (240, 240, 60), (90, 130, 255)]


def grid_queries(H, W, spacing, margin=24):
    """A point every `spacing` px, inset by `margin` from the frame edge.

    Mirrors the 2026 harness, which hands the container a fixed grid rather than
    annotation-derived points. The margin keeps queries off the vignetted border
    where there is no tissue to track.
    """
    ys = np.arange(margin, H - margin, spacing, dtype=np.float32)
    xs = np.arange(margin, W - margin, spacing, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)


def stream_and_record(model, frames, start_pts, device, dtype):
    """Same forward pass as eval_2d.track_clip_streaming, but keeps
    every frame's coords instead of only the last. Returns [N,T,2] native px.
    """
    model.init_video_online_processing()
    q = torch.from_numpy(np.asarray(start_pts, np.float32)).unsqueeze(0).to(device)
    queries = torch.cat([torch.zeros_like(q[:, :, :1]), q], dim=-1)  # [B,N,3] = (t,x,y)

    per_frame = []
    with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
        for t in range(frames.shape[0]):
            frame_t = (torch.from_numpy(frames[t]).permute(2, 0, 1)
                       .unsqueeze(0).to(dtype=dtype, device=device))
            coords, _vis, *_ = model(frame_t, queries=queries)
            per_frame.append(coords[0, 0].float().cpu().numpy())
    return np.stack(per_frame, axis=1).astype(np.float32)          # [N,T,2]


def render(frames, tracks, out_path, fps, trail, start_gt=None, end_gt=None,
           radius=4):
    T, H, W = frames.shape[:3]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"cv2 could not open a writer for {out_path}")
    try:
        for t in range(T):
            canvas = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR).copy()
            if start_gt is not None:
                for x, y in start_gt:
                    cv2.circle(canvas, (int(x), int(y)), radius + 4, (120, 255, 120), 1)
            if end_gt is not None:
                for x, y in end_gt:
                    cv2.circle(canvas, (int(x), int(y)), radius + 4, (255, 170, 90), 1)
            for n in range(tracks.shape[0]):
                col = _PALETTE[n % len(_PALETTE)]
                for s in range(max(0, t - trail), t):
                    p0 = tuple(np.round(tracks[n, s]).astype(int))
                    p1 = tuple(np.round(tracks[n, s + 1]).astype(int))
                    cv2.line(canvas, p0, p1, col, 1, cv2.LINE_AA)
                c = tuple(np.round(tracks[n, t]).astype(int))
                cv2.circle(canvas, c, radius, col, -1, cv2.LINE_AA)
                cv2.circle(canvas, c, radius, (20, 20, 20), 1, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="student_e<N>.pth")
    p.add_argument("--stir-root", required=True)
    p.add_argument("--clip-id", required=True, help="e.g. 0__left__seq00")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--skip", type=int, default=1,
                   help="1 (default) streams every frame, as the container does")
    p.add_argument("--max-frames", type=int, default=0,
                   help="0 = whole clip; else temporally stride down to this many")
    p.add_argument("--iters", type=int, default=4,
                   help="LiteTracker refinement iterations (4 = accuracy tracks)")
    p.add_argument("--window-len", type=int, default=16)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--trail", type=int, default=25, help="trail length in frames")
    p.add_argument("--grid", type=int, default=0, metavar="PX",
                   help="track a point every PX pixels instead of the IR tattoos. "
                        "64 matches the 2026 harness. 0 (default) = tattoos only")
    p.add_argument("--with-tattoos", action="store_true",
                   help="with --grid, also track the tattoos and draw their GT")
    p.add_argument("--radius", type=int, default=4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported()
             else torch.float32)
    model, info = load_student_as_litetracker(args.checkpoint,
                                              window_len=args.window_len,
                                              iters=args.iters, device=device)
    print(f"LiteTracker: iters={info['iters']} window_len={info['window_len']} "
          f"device={device} dtype={dtype}")

    getviddirs2d_STIR, STIRStereoClip = _import_stirloader()
    seq_path = next((sp for sp in getviddirs2d_STIR(args.stir_root)
                     if _clip_id(sp) == args.clip_id), None)
    if seq_path is None:
        raise SystemExit(f"clip {args.clip_id!r} not found under {args.stir_root}")

    frames, start, end = load_val_clip(seq_path, STIRStereoClip,
                                       args.skip, args.max_frames)

    # n_gt leading points are the tattoos, and only those get GT circles drawn
    if args.grid > 0:
        H, W = frames.shape[1:3]
        grid = grid_queries(H, W, args.grid)
        queries = np.concatenate([start, grid], 0) if args.with_tattoos else grid
        n_gt = len(start) if args.with_tattoos else 0
        print(f"{args.clip_id}: {frames.shape[0]} frames, {len(queries)} query points "
              f"({args.grid}px grid" + (f" + {len(start)} tattoos)" if n_gt else ")"))
    else:
        queries, n_gt = start, len(start)
        print(f"{args.clip_id}: {frames.shape[0]} frames, {len(queries)} IR tattoos")

    tracks = stream_and_record(model, frames, queries, device, dtype)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.clip_id}__student.mp4"
    render(frames, tracks, out_path, args.fps, args.trail,
           start if n_gt else None, end if n_gt else None, radius=args.radius)

    msg = f"wrote {out_path}"
    if n_gt:
        # nearest-GT distance, the same unordered convention the metrics use
        from eval_common import nn_dist
        err = nn_dist(tracks[:n_gt, -1], end)
        msg += f"  (median endpoint error {np.median(err):.2f} px over {n_gt} tattoos)"
    print(msg)


if __name__ == "__main__":
    main()
