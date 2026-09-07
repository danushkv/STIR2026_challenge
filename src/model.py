"""The student model: how it is built, configured, and fed.

Shared core for training and evaluation. It does NOT train anything -- the
trainer is `train.py`. Everything here is used by both, which is why it lives
apart from either.

  DEFAULT_CONFIG        every knob, with the value it takes if `train.yaml`
                        and the command line both stay silent
  normalize_overrides   accept `--key value`, `--key=value` and `key=value`
                        interchangeably, mapping hyphens in KEYS to underscores
  build_student         instantiate the real CoTrackerThreeOnline module from
                        the cloned co-tracker repo -- upstream's own
                        `build_cotracker`, not a reimplementation
  subsample_for_training  cap frames and points per clip, which is what bounds
                        GPU memory (batch size is 1 clip, always)
"""
from __future__ import annotations

import sys

import numpy as np
from omegaconf import OmegaConf

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

DEFAULT_CONFIG = {
    "pseudo_labels_dir": "???", "stir_root": "???", "repo_root": "???",
    "checkpoint": "", "skip": 5, "out_dir": "runs/student_ft",
    "model_resolution": [384, 512],
    "lr": 2.0e-5, "epochs": 10, "window_len": 16, "train_iters": 4,
    "supervision": "weighted", "max_train_frames": 64, "max_points": 384, "seed": 0,
    "train_clip_ids": [], "max_train_clips": 0,
    "val_patients": [], "val_every": 1, "val_max_frames": 250,
    "save_freq": 1, "num_workers": 4,
    "wandb": {"enabled": False, "project": "stir-student", "entity": None,
              "run_name": None, "mode": "online"},
}


def normalize_overrides(tokens):
    """Accept BOTH argparse-style (--key value, --key=value) and OmegaConf
    dotlist (key=value), returning clean 'key=value' dotlist strings. Hyphens in
    KEYS are mapped to underscores so --pseudo-labels-dir and pseudo_labels_dir
    both reach the same config key. Values are left untouched."""
    out = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            body = tok[2:]
            if "=" in body:
                key, val = body.split("=", 1)
                out.append(f"{key.replace('-', '_')}={val}")
                i += 1
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                out.append(f"{body.replace('-', '_')}={tokens[i + 1]}")
                i += 2
            else:  # bare flag -> boolean true
                out.append(f"{body.replace('-', '_')}=true")
                i += 1
        else:  # already dotlist; normalize hyphens in the key part only
            if "=" in tok:
                key, val = tok.split("=", 1)
                out.append(f"{key.replace('-', '_')}={val}")
            else:
                out.append(tok)
            i += 1
    return out


def merge_config_strict(defaults, config_path=None, overrides=()):
    """Merge a YAML file and CLI dotlist into known defaults only.

    OmegaConf normally accepts unknown keys. That made a historical typo such
    as ``pseudo-data-sir`` silently leave ``pseudo_labels_dir`` unchanged. A
    structured config turns the same typo into an immediate, readable error.
    Entry points must put all of their supported keys in ``defaults`` before
    calling this helper.
    """
    cfg = OmegaConf.create(defaults)
    OmegaConf.set_struct(cfg, True)
    if config_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(config_path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg

def build_student(checkpoint: str, repo_root: str, window_len: int = 16):
    """Builds the actual trainable CoTrackerThreeOnline module via the co-tracker
    repo's own build_cotracker() -- NOT CoTrackerOnlinePredictor, which is an
    inference-only (@torch.no_grad) wrapper. window_len=16 matches the released
    scaled_online.pth checkpoint; changing it misaligns the time embedding.
    """
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from cotracker.models.build_cotracker import build_cotracker  # noqa: after sys.path insert

    model = build_cotracker(
        checkpoint=checkpoint or None, offline=False, window_len=window_len
    ).cuda()
    model.train()
    return model


def subsample_for_training(frames, coords, visibility, weight, max_frames, max_points, rng):
    """Bounds clip length and point count to the scale Meta's own real-data run
    uses (sequence_len=64, traj_per_sample=384) -- training on a full clip's
    frames+points in one is_train=True call retains every window x refinement
    iteration for backprop, which is what exhausts GPU memory.

    Random contiguous sub-window of <= max_frames frames, then only points
    visible at that sub-window's first frame, then random <= max_points of those.
    """
    T = len(frames)
    if max_frames and T > max_frames:
        start = int(rng.integers(0, T - max_frames + 1))
        frames = frames[start:start + max_frames]
        coords = coords[:, start:start + max_frames]
        visibility = visibility[:, start:start + max_frames]
        weight = weight[:, start:start + max_frames]

    visible_at_start = visibility[:, 0] > 0.5
    coords, visibility, weight = coords[visible_at_start], visibility[visible_at_start], weight[visible_at_start]

    N = coords.shape[0]
    if max_points and N > max_points:
        idx = rng.choice(N, size=max_points, replace=False)
        coords, visibility, weight = coords[idx], visibility[idx], weight[idx]

    return frames, coords, visibility, weight
