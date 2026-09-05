"""Same student fine-tuning as train.py, but with a prefetching input
pipeline so the GPU stops starving.

Why this exists: train.py loads each clip synchronously in the training
loop -- a full STIR directory rescan + a full ffmpeg video decode per clip,
serially, with the GPU idle the whole time (~20% utilization). Here the decode +
subsample + resize happen in DataLoader worker processes that run AHEAD of the
GPU, and the seq->path map is built ONCE instead of per clip. The GPU loop does
only the forward/backward.

Everything else -- model, losses, config system, wandb logging, run dir,
checkpointing, val_patients holdout -- matches train.py, whose helpers
are imported directly. Validation still lives in eval_2d.py.

    python train.py --config train.yaml \\
        num_workers=8 wandb.run_name=fast_run
"""
from __future__ import annotations

import glob
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf

from collect_tracks import _clip_id, _import_stirloader
from model import (
    DEFAULT_CONFIG, build_student, normalize_overrides, subsample_for_training,
)

# num_workers is the one config key train.py's DEFAULT_CONFIG lacks.
FAST_DEFAULT_CONFIG = dict(DEFAULT_CONFIG, num_workers=4)


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

    cfg = OmegaConf.create(FAST_DEFAULT_CONFIG)
    if config_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(config_path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    missing = [k for k in ("pseudo_labels_dir", "stir_root", "repo_root")
               if OmegaConf.is_missing(cfg, k)]
    if missing:
        raise SystemExit(f"Missing required config keys: {missing}.")
    return cfg


def _load_left_frames_only(stir_cls, clip, skip: int) -> np.ndarray:
    """[T,H,W,3] RGB uint8 for just the LEFT video of one clip.

    STIRStereoClip.extractallframes() always decodes BOTH stereo videos in
    full before returning (leftvidframes, rightvidframes) -- callers here only
    ever use the left half. That's ~2x wasted ffmpeg decode time and peak
    memory on every clip, in every DataLoader worker, which is what caused the
    OOM (concurrent workers each briefly holding both channels of a long clip
    at native 1280x1024 resolution). extractfullvideopipe() is a staticmethod
    that only needs a filename, so call it directly on clip.leftvidname and
    skip the right video entirely -- same decode path, same frame-selection
    logic (every `skip`-th frame + final frame), just half the work.

    Deliberately NOT a STIRLoader.py change: train.py is running
    against the shared loader code right now, so this stays local to
    train.py instead.
    """
    return np.stack(
        stir_cls.extractfullvideopipe(clip.leftvidname, "visible", skip=skip), axis=0
    )


def _prepare_clip_cpu(frames, coords, visibility, weight, model_resolution, clip_id=""):
    """CPU-side prep (runs in DataLoader workers): resize frames + rescale the
    pseudo-label trajectory to model resolution. Returns the tensors run_loss
    needs, all on CPU. video is uint8 to keep worker->main IPC ~4x smaller than
    float32; run_loss casts it to float on the GPU (the model normalizes anyway).
    """
    T, H, W, _ = frames.shape
    h, w = model_resolution

    video = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()
    video = F.interpolate(video, size=(h, w), mode="bilinear", align_corners=False)
    video = video.round().clamp_(0, 255).to(torch.uint8)[None]  # [1,T,3,h,w] uint8

    coords_scaled = coords.astype(np.float32).copy()
    coords_scaled[..., 0] *= w / W
    coords_scaled[..., 1] *= h / H

    N = coords_scaled.shape[0]
    queries = torch.zeros((1, N, 3))
    queries[0, :, 1:] = torch.from_numpy(coords_scaled[:, 0, :])

    traj_gt = torch.from_numpy(coords_scaled).float().permute(1, 0, 2)[None]  # [1,T,N,2]
    vis_gt = torch.from_numpy(visibility).float().permute(1, 0)[None]          # [1,T,N]
    weight_gt = torch.from_numpy(weight).float().permute(1, 0)[None]           # [1,T,N]
    # vis_gt is a binary_cross_entropy TARGET and sequence_BCE_loss applies no
    # valid mask, so anything outside [0,1] (or NaN) becomes a device-side assert
    # deep in a CUDA kernel with no indication of which clip caused it. Fail here
    # instead, where the clip is still identifiable.
    if not torch.isfinite(vis_gt).all() or vis_gt.min() < 0 or vis_gt.max() > 1:
        raise ValueError(
            f"[{clip_id}] visibility must be finite and in [0,1] to use as a BCE "
            f"target; got range [{vis_gt.min()}, {vis_gt.max()}], "
            f"{int((~torch.isfinite(vis_gt)).sum())} non-finite value(s)")
    return video, queries, traj_gt, vis_gt, weight_gt


class PseudoLabelClipDataset(Dataset):
    """One item == one clip: load its pseudo-label, decode its frames (via the
    precomputed seq_path -- no dataset rescan), subsample, and CPU-prep. Returns
    None for clips too short / with no visible points at the sampled window start
    (a custom collate turns that into a skipped step)."""

    def __init__(self, items, skip, max_frames, max_points, min_frames,
                 model_resolution, base_seed):
        self.items = items  # list of (clip_id, pl_path, seq_path)
        self.skip = skip
        self.max_frames = max_frames
        self.max_points = max_points
        self.min_frames = min_frames
        self.model_resolution = model_resolution
        self.base_seed = base_seed
        self.epoch = 0
        self._STIRStereoClip = None

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return len(self.items)

    def _stir_cls(self):
        if self._STIRStereoClip is None:
            _, self._STIRStereoClip = _import_stirloader()
        return self._STIRStereoClip

    def __getitem__(self, idx):
        clip_id, pl_path, seq_path = self.items[idx]
        with np.load(pl_path) as d:
            coords, visibility, weight = d["coords"], d["visibility"], d["weight"]

        stir_cls = self._stir_cls()
        clip = stir_cls(seq_path)
        frames = _load_left_frames_only(stir_cls, clip, self.skip)

        if frames.shape[0] != coords.shape[1]:
            raise ValueError(
                f"[{clip_id}] frame count mismatch: {frames.shape[0]} frames vs "
                f"{coords.shape[1]} in the pseudo-label -- skip mismatch?")

        # per-(epoch, idx) seeding: reproducible, but a different crop each epoch
        rng = np.random.default_rng([self.base_seed, self.epoch, idx])
        frames, coords, visibility, weight = subsample_for_training(
            frames, coords, visibility, weight, self.max_frames, self.max_points, rng)

        if frames.shape[0] < self.min_frames or coords.shape[0] == 0:
            return None

        video, queries, traj_gt, vis_gt, weight_gt = _prepare_clip_cpu(
            frames, coords, visibility, weight, self.model_resolution, clip_id)
        return {"clip_id": clip_id, "video": video, "queries": queries,
                "traj_gt": traj_gt, "vis_gt": vis_gt, "weight_gt": weight_gt}


def _collate(batch):
    """batch_size is 1; pass the single sample through (or None to skip it)."""
    return batch[0]


def run_loss(model, sample, use_verifier_weight, window_len, train_iters):
    """GPU forward + loss for one prepared clip. Same math as train.py's
    compute_loss, but consuming the CPU tensors the dataset produced."""
    from cotracker.models.core.cotracker.losses import (
        sequence_loss, sequence_BCE_loss, sequence_prob_loss,
    )
    device = next(model.parameters()).device
    video = sample["video"].to(device, non_blocking=True).float()  # uint8 -> float 0-255
    queries = sample["queries"].to(device, non_blocking=True)
    traj_gt = sample["traj_gt"].to(device, non_blocking=True)
    vis_gt = sample["vis_gt"].to(device, non_blocking=True)
    weight_gt = sample["weight_gt"].to(device, non_blocking=True)

    _, _, _, train_data = model(video=video, queries=queries, iters=train_iters, is_train=True)
    coord_predictions, vis_predictions, confidence_predictions, valid_mask = train_data

    T = video.shape[1]
    S = window_len
    base_valid = torch.ones_like(vis_gt)
    if use_verifier_weight:
        base_valid = base_valid * weight_gt

    traj_gts, vis_gts, valids_gts = [], [], []
    for ind in range(0, T - S // 2, S // 2):
        traj_gts.append(traj_gt[:, ind:ind + S])
        vis_gts.append(vis_gt[:, ind:ind + S])
        valids_gts.append(base_valid[:, ind:ind + S] * valid_mask[:, ind:ind + S])

    coord_loss = sequence_loss(
        coord_predictions, traj_gts, valids_gts, vis=vis_gts,
        gamma=0.8, add_huber_loss=False, loss_only_for_visible=True,
    ).mean()
    vis_loss = sequence_BCE_loss(vis_predictions, vis_gts).mean()
    conf_loss = sequence_prob_loss(
        coord_predictions, confidence_predictions, traj_gts, vis_gts
    ).mean()

    total = coord_loss * 0.05 + vis_loss + conf_loss
    components = {"coord_loss": coord_loss.item(), "vis_loss": vis_loss.item(),
                  "conf_loss": conf_loss.item()}
    return total, components


def _build_items(cfg, heldout):
    """(clip_id, pl_path, seq_path) for every trainable clip -- the seq->path map
    is built ONCE here, not rescanned per clip."""
    getviddirs2d_STIR, _ = _import_stirloader()
    seq_map = {_clip_id(sp): sp for sp in getviddirs2d_STIR(cfg.stir_root)}

    items = []
    missing = 0
    for path in sorted(glob.glob(os.path.join(cfg.pseudo_labels_dir, "**", "*.npz"), recursive=True)):
        patient = os.path.basename(os.path.dirname(path))
        if patient in heldout:
            continue
        seq_part = os.path.splitext(os.path.basename(path))[0]
        clip_id = f"{patient}__{seq_part}"
        seq_path = seq_map.get(clip_id)
        if seq_path is None:
            missing += 1
            continue
        items.append((clip_id, path, seq_path))
    if missing:
        print(f"[warn] {missing} pseudo-label clip(s) had no matching STIR video, skipped")
    return items


def train(cfg, wandb_run=None):
    run_name = cfg.wandb.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))
    print(f"Run directory: {run_dir}")

    model = build_student(cfg.checkpoint, cfg.repo_root, window_len=cfg.window_len)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5, eps=1e-8)

    model_resolution = tuple(cfg.model_resolution)
    heldout = set(str(p) for p in cfg.val_patients)
    if heldout:
        print(f"Holding patients {sorted(heldout)} OUT of training (for eval_2d.py)")
    use_verifier_weight = (cfg.supervision == "weighted")
    min_frames = cfg.window_len // 2 + 1

    items = _build_items(cfg, heldout)
    print(f"{len(items)} trainable clips")
    dataset = PseudoLabelClipDataset(
        items, cfg.skip, cfg.max_train_frames, cfg.max_points, min_frames,
        model_resolution, cfg.seed)

    loader_kwargs = dict(batch_size=1, num_workers=cfg.num_workers,
                         collate_fn=_collate, pin_memory=True)
    if cfg.num_workers > 0:
        loader_kwargs.update(prefetch_factor=2, persistent_workers=False)
    loader = DataLoader(dataset, **loader_kwargs)

    global_step = 0
    for epoch in range(cfg.epochs):
        model.train()
        dataset.set_epoch(epoch)  # workers re-fork per epoch (persistent_workers=False) and see it
        sums = defaultdict(float)
        n = 0
        for sample in loader:
            if sample is None:  # too-short / no-visible-points clip
                continue

            loss, components = run_loss(
                model, sample, use_verifier_weight=use_verifier_weight,
                window_len=cfg.window_len, train_iters=cfg.train_iters)

            opt.zero_grad()
            loss.backward()
            opt.step()

            step_loss = loss.item()
            sums["loss"] += step_loss
            for k, v in components.items():
                sums[k] += v
            n += 1
            global_step += 1
            print(f"[{sample['clip_id']}] loss {step_loss:.4f}")

            if wandb_run is not None:
                step_log = {"train/loss": step_loss, "epoch": epoch}
                for k, v in components.items():
                    step_log[f"train/{k}"] = v
                wandb_run.log(step_log, step=global_step)

        mean_loss = sums["loss"] / max(n, 1)
        print(f"epoch {epoch} done, mean train loss {mean_loss:.4f}")
        if wandb_run is not None:
            epoch_log = {f"train_epoch/{k}": v / max(n, 1) for k, v in sums.items()}
            epoch_log["epoch"] = epoch
            wandb_run.log(epoch_log, step=global_step)

        torch.save({"model": model.state_dict()},
                  os.path.join(run_dir, "student_last.pth"))
        if cfg.save_freq and (epoch + 1) % cfg.save_freq == 0:
            torch.save({"model": model.state_dict()},
                      os.path.join(run_dir, f"student_e{epoch + 1}.pth"))


def main():
    cfg = load_config()
    print(OmegaConf.to_yaml(cfg))

    wandb_run = None
    if cfg.wandb.enabled:
        import wandb
        wandb_run = wandb.init(
            project=cfg.wandb.project, entity=cfg.wandb.entity,
            name=cfg.wandb.run_name, mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    try:
        train(cfg, wandb_run=wandb_run)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
