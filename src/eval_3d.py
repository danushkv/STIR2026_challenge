"""Same 3D metric as eval_3d.py, but run through the LITETRACKER runtime.

Relationship to the other three scripts:
                    CoTracker3 whole-clip      LiteTracker streaming
    2D (px)         eval_2d.py        eval_2d.py
    3D (mm)         eval_3d.py         THIS FILE

WHY: eval_3d.py tracks with CoTracker3's sliding-window forward over the
whole clip. Your submission streams frame-by-frame through LiteTracker (causal,
memory buffers, EMA-flow init, visibility thresholded at 0.6). Same weights,
different forward pass, so the 3D numbers can differ -- and only this one matches
what gets scored.

Method is otherwise identical to eval_3d.py, so numbers are comparable:
STIR is calibrated stereo, so depth is triangulation, not estimation.
getsegsstereo gives start points matched in BOTH views, so the student tracks
twice -- left centers through the left video, right centers through the right --
then disparity = (u_L + disparitypad*scale) - u_R, triangulated with STIR's own Q.
No RAFT-Stereo, no MFT, no monocular depth network.

UNITS: millimetres (Q from baseline_mm, as get3DSegmentationPositions does) --
the STIRMetrics convention, NOT the 2026 stereo convention (metres, 0.002-0.032).

CONTROL is the "never moved" baseline (start GT vs end GT). GT depth comes from
NCC patch matching so it carries its own error; read Model relative to CONTROL.

skip=1 / val_max_frames=0 by default: LiteTracker is causal, so memory is
constant in clip length and there is no reason to subsample time. Keep these
fixed when comparing checkpoints -- changing skip changes inter-frame motion.

    python eval_3d.py --config train.yaml \\
        student_checkpoint=/path/to/student_last.pth \\
        test_patients=[15,16] lt_iters=1
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from eval_common import STIR_THRESHOLDS_MM, nn_dist, triangulate_mm
from collect_tracks import _clip_id, _import_stirloader
from student_lt_wrapper import load_student_as_litetracker
from model import DEFAULT_CONFIG, merge_config_strict, normalize_overrides
from eval_2d import track_clip_streaming


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
    base["student_checkpoint"] = "???"
    base["val_stir_root"] = ""
    base["test_patients"] = []
    base["lt_iters"] = 1     # LiteTracker refinement iters per frame
    base["skip"] = 1         # stream every frame (see docstring)
    cfg = merge_config_strict(base, config_path, overrides)


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
    patients = set(str(p) for p in cfg.test_patients)
    getviddirs2d_STIR, STIRStereoClip = _import_stirloader()
    seq_paths = {_clip_id(p): p for p in getviddirs2d_STIR(stir_root)
                 if (not patients) or _clip_id(p).split("__", 1)[0] in patients}
    print(f"Evaluating {len(seq_paths)} clip(s) from {stir_root}"
          + (f", patients {sorted(patients)}" if patients else " (all patients)")
          + f" | skip={cfg.skip}")

    if cfg.skip > 1:
        os.environ["SKIP"] = str(cfg.skip)

    model_d, control_d, n_clips = [], [], 0
    for cid, seq_path in sorted(seq_paths.items()):
        try:
            clip = STIRStereoClip(seq_path)
            start_l, start_r = (np.array(x, dtype=np.float32)
                                for x in clip.getsegsstereo(start=True))
            _, _, gt_start_3d = clip.get3DSegmentationPositions(start=True)
            _, _, gt_end_3d = clip.get3DSegmentationPositions(start=False)
        except (AssertionError, IndexError, ValueError) as e:
            print(f"[skip] {cid}: {e}")
            continue
        if len(start_l) == 0 or len(gt_end_3d) == 0:
            print(f"[skip] {cid}: no stereo-matched segmentation points")
            continue

        try:
            left_frames, right_frames = clip.extractallframes()
        except (AssertionError, IndexError) as e:
            print(f"[skip] {cid}: {e}")
            continue
        left_frames = np.stack(left_frames, axis=0)
        right_frames = np.stack(right_frames, axis=0)
        if left_frames.shape[0] < 2:
            print(f"[skip] {cid}: {left_frames.shape[0]} frame(s)")
            continue

        # stream each view separately; track_clip_streaming resets the model's
        # per-video state at the start of each call
        final_l = track_clip_streaming(model, left_frames, start_l, device, dtype)
        final_r = track_clip_streaming(model, right_frames, start_r, device, dtype)

        disparity = (final_l[:, 0] + clip.disparitypad * clip.scale) - final_r[:, 0]
        pred_3d = triangulate_mm(final_l[:, 0], final_l[:, 1], disparity, clip)

        d = nn_dist(pred_3d, gt_end_3d)
        c = nn_dist(np.asarray(gt_start_3d, dtype=np.float32), gt_end_3d)
        model_d.append(d)
        control_d.append(c)
        n_clips += 1
        print(f"[{cid}] mean 3D error {float(np.mean(d)):7.2f} mm  "
              f"(control {float(np.mean(c)):7.2f} mm, {len(start_l)} pts, "
              f"{left_frames.shape[0]} frames)")

    if not model_d:
        raise SystemExit("no usable clips (need stereo-matched segmentation + both views)")

    def report(name, dists):
        a = np.concatenate(dists)
        accs = []
        print(f"\n{name}\n-----\nAccuracy\n-----")
        for t in STIR_THRESHOLDS_MM:
            acc = float(np.mean(a <= t))
            accs.append(acc)
            print(f"{t:2d} mm:\t{acc:0.5f}")
        print(f"Avg:\t{np.mean(accs):0.5f}")
        print(f"mean/median error: {a.mean():.2f} / {np.median(a):.2f} mm")

    print(f"\n=== STIR 3D (endpoint), LiteTracker runtime "
          f"(iters={cfg.lt_iters}), {n_clips} clips ===")
    report("CONTROL", control_d)
    report("Model", model_d)
    print("\nCompare against eval_3d.py (CoTracker3 whole-clip forward) on "
          "the SAME patients/skip to isolate the streaming runtime's effect.")


if __name__ == "__main__":
    main()
