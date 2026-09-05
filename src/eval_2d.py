"""Same STIR endpoint metric as eval_2d.py, but run through the
LITETRACKER runtime instead of CoTracker3's whole-clip forward.

WHY A SEPARATE FILE: eval_2d.py calls CoTrackerThreeOnline with the
entire clip at once (sliding 16-frame windows, non-causal within each window).
Your submission runs LiteTracker: one frame at a time, causal, with persistent
coords/vis/conf/corr_embs buffers, an EMA-flow warm start, and visibility
thresholded at 0.6. Identical weights, DIFFERENT forward pass -- so the two can
disagree, and only this one reflects what the challenge will actually score.
Run both and compare; a large gap means the streaming runtime is costing you
accuracy, which is a real finding rather than a bug.

Metric is unchanged from eval_2d.py so the numbers are comparable:
track each clip's start-frame IR-tattoo centers to the last frame, KDTree
nearest-neighbour match against the end-frame centers (unordered -- no
persistent point IDs), report delta-accuracy at [4,8,16,32,64] px and mean
endpoint error.

TWO DELIBERATE DEFAULTS THAT DIFFER FROM eval_2d.py:
  skip=1  -- LiteTracker is causal, so memory is CONSTANT in clip length; there
      is no reason to subsample time. The real eval streams every frame, so
      that is what we do. (eval_2d.py's whole-clip forward had to cap
      frames to bound GPU memory.) Override with skip=N to match an old run.
  val_max_frames=0 -- no temporal striding, same reason. Set it non-zero only
      if you are deliberately reproducing eval_2d.py's protocol.
  Both of these change the motion between consecutive frames, so a run with
  skip=5 is NOT comparable to a run with skip=1. Keep them fixed when comparing
  checkpoints.

lt_iters: LiteTracker refinement iterations per frame (its own default is 1;
you TRAINED at train_iters=4). This is the accuracy half of the same knob
latency_from_preds.py measures the cost of -- sweep it here, read latency there.

    python eval_2d.py --config train.yaml \\
        student_checkpoint=/path/to/student_last.pth \\
        val_patients=[15,16] lt_iters=1
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from eval_common import STIR_THRESHOLDS_PX, load_val_clip, nn_dist
from collect_tracks import _clip_id, _import_stirloader
from student_lt_wrapper import load_student_as_litetracker
from model import DEFAULT_CONFIG, normalize_overrides


def load_config():
    argv = sys.argv[1:]
    config_path = None
    raw = []
    i = 0
    while i < len(argv):
        if argv[i] == "--config":
            config_path = argv[i + 1]
            i += 2
        else:
            raw.append(argv[i])
            i += 1
    overrides = normalize_overrides(raw)
    cli_keys = {o.split('=', 1)[0] for o in overrides}

    base = dict(DEFAULT_CONFIG)
    base["student_checkpoint"] = "???"   # the trained student to evaluate
    base["val_stir_root"] = ""           # optional separate test root
    base["lt_iters"] = 1                 # LiteTracker refinement iters per frame
    base["skip"] = 1                     # stream every frame (see docstring)
    base["val_max_frames"] = 0           # no striding (see docstring)
    cfg = OmegaConf.create(base)
    if config_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(config_path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))


    # train.yaml is a TRAINING config (skip: 5, val_max_frames: 250) and
    # merges AFTER these defaults, so without this it silently overrides the
    # streaming protocol this file's docstring promises. Restore unless the CLI
    # asked otherwise.
    for _k, _v in (("skip", 1), ("val_max_frames", 0)):
        if _k not in cli_keys and _k in cfg and cfg[_k] != _v:
            print(f"[protocol] forcing {_k}={_v} (config said {cfg[_k]})")
            cfg[_k] = _v
    missing = [k for k in ("stir_root", "student_checkpoint")
               if OmegaConf.is_missing(cfg, k)]
    if missing:
        raise SystemExit(f"Missing required config keys: {missing}.")
    return cfg


@torch.no_grad()
def track_clip_streaming(model, frames, start_pts, device, dtype):
    """Stream `frames` [T,H,W,3] uint8 through LiteTracker one frame at a time,
    tracking `start_pts` [N,2] (native px). Returns final positions [N,2], native px.

    Frames go in at NATIVE resolution: LiteTracker.forward resizes to
    model_resolution internally and rescales its output coords back to native,
    so nothing here should pre-resize or post-scale.
    """
    model.init_video_online_processing()   # reset per-video streaming state

    q = torch.from_numpy(np.asarray(start_pts, dtype=np.float32)).unsqueeze(0).to(device)
    # LiteTracker wants [B,N,3] = (query_frame_idx, x, y)
    queries = torch.cat([torch.zeros_like(q[:, :, :1]), q], dim=-1)

    coords = None
    with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
        for t in range(frames.shape[0]):
            frame_t = (
                torch.from_numpy(frames[t])
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(dtype=dtype, device=device)
            )
            coords, _vis, *_ = model(frame_t, queries=queries)

    return coords[0, 0].float().cpu().numpy().astype(np.float32)


def main():
    cfg = load_config()
    print(OmegaConf.to_yaml(cfg))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    model, info = load_student_as_litetracker(
        cfg.student_checkpoint, window_len=cfg.window_len, iters=cfg.lt_iters,
        device=device)
    print(f"LiteTracker runtime: window_len={info['window_len']} iters={info['iters']} "
          f"device={info['device']} dtype={dtype}")

    stir_root = cfg.val_stir_root or cfg.stir_root
    val_patients = set(str(p) for p in cfg.val_patients)
    getviddirs2d_STIR, STIRStereoClip = _import_stirloader()
    seq_map = {_clip_id(p): p for p in getviddirs2d_STIR(stir_root)
               if (not val_patients) or _clip_id(p).split("__", 1)[0] in val_patients}
    print(f"Evaluating {len(seq_map)} clip(s) from {stir_root}"
          + (f", patients {sorted(val_patients)}" if val_patients else " (all patients)")
          + f" | skip={cfg.skip} val_max_frames={cfg.val_max_frames}")

    dists, n_clips = [], 0
    for cid, seq_path in sorted(seq_map.items()):
        try:
            frames, start, end = load_val_clip(
                seq_path, STIRStereoClip, cfg.skip, cfg.val_max_frames)
        except (AssertionError, IndexError) as e:
            print(f"[skip] {cid}: {e}")
            continue
        if len(start) == 0 or len(end) == 0:
            print(f"[skip] {cid}: no start/end centers")
            continue
        # No window_len minimum here: LiteTracker is causal and produces output
        # from the very first frame, unlike CoTracker3's sliding window.
        if frames.shape[0] < 2:
            print(f"[skip] {cid}: {frames.shape[0]} frame(s)")
            continue

        final = track_clip_streaming(model, frames, start, device, dtype)
        d = nn_dist(final, end)
        dists.append(d)
        n_clips += 1
        print(f"[val-lt] {cid}: endpoint_error {float(np.mean(d)):.2f}px "
              f"({len(start)} pts, {frames.shape[0]} frames)")

    if not dists:
        print("[val-lt] no usable validation clips")
        return

    dist = np.concatenate(dists)
    metrics = {f"delta_{t}px": float(np.mean(dist <= t)) for t in STIR_THRESHOLDS_PX}
    metrics["delta_avg"] = float(np.mean([metrics[f"delta_{t}px"] for t in STIR_THRESHOLDS_PX]))
    metrics["endpoint_error_px"] = float(np.mean(dist))
    metrics["n_clips"] = n_clips

    print(f"\n=== STIR validation, LiteTracker runtime (iters={cfg.lt_iters}) ===")
    for t in STIR_THRESHOLDS_PX:
        print(f"  delta_{t}px:        {metrics[f'delta_{t}px']:.4f}")
    print(f"  delta_avg:          {metrics['delta_avg']:.4f}")
    print(f"  endpoint_error_px:  {metrics['endpoint_error_px']:.2f}")
    print(f"  clips evaluated:    {metrics['n_clips']}")
    print("\nCompare against eval_2d.py (CoTracker3 whole-clip forward) on "
          "the SAME patients/skip -- a large gap means the streaming runtime, not "
          "the weights, is what's costing accuracy.")


if __name__ == "__main__":
    main()
