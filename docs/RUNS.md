# Run ledger

Every checkpoint behind the ablation board, with the command that produced it.

Sources, in order of authority:
1. `model_runs/<run>/config.yaml` — written by the trainer itself at start of
   run. This is what the run *actually* used, after all overrides.
2. that run's wandb metadata (`wandb-metadata.json` `program` + `args`) — the
   only record of **which training script** was invoked.

Where the two disagree, (1) wins and the disagreement is called out.

## Constant across every run

`lr 2e-5` · `window_len 16` · `train_iters 4` · `supervision weighted` ·
`model_resolution 384×512` · `max_train_frames 64` · `max_points 384` ·
`skip 5` · `seed 0` · `val_patients []` · batch = 1 clip

Two initialisations appear:

- **`scaled_online.pth`** — Meta's stock CoTracker3-Online, at
  `lite-tracker-master/model/scaled_online.pth`.
- **`litetracker_finetuned.pth`** — the **STIR-2025 winning** checkpoint, at
  `track_on/checkpoint/litetracker_finetuned.pth`. Same architecture and state
  dict, different starting point. This is also the `peng_baseline` row.

## The board

| tag | run directory | script | pseudo-labels | frames root | init | epochs |
|---|---|---|---|---|---|---|
| **`agg_e40`, `agg_e44`** | `fast_run_with2024_agg` | `train.py` | `pseudo_labels_with_2024_agg/` | `STIRcombined/` | `litetracker_finetuned` | 45 |
| `agg_around_e30`, `agg_around_e44` | `fast_run_with2024_agg_extra_aroundpoints` | `train.py` | `pseudo_labels_with2024_extra_aroundpoints_agg/` | `STIRcombined/` | `litetracker_finetuned` | 45 |
| `agg_around_thr_e30` | `fast_run_with2024_agg_extra_aroundpoints_thresholdloss` | `train.py` | `pseudo_labels_with2024_extra_aroundpoints_agg/` | `STIRcombined/` | `litetracker_finetuned` | 46 |
| `around_peng_e42` | `fast_run_with2024_extra_aroundpoints_pengchkpt` | `train.py` | `pseudo_with2024_cotracker3_extra_aroundpoints/` | `STIRcombined/` | `litetracker_finetuned` | 45 |
| `around_peng_thr_e44` | `fast_run_with2024_extra_aroundpoints_pengchkpt_thresholdloss` | `train.py` | `pseudo_with2024_cotracker3_extra_aroundpoints/` | `STIRcombined/` | `litetracker_finetuned` | 46 |
| `extra_e40` | `fast_run_with2024_extrapoints` | `train.py` | `pseudo_labels/` ⚠ | `STIRcombined/` | `scaled_online` | 45 |
| `extra_thr_e30`, `extra_thr_e42` | `fast_run_with2024_extrapoints_thresholdloss` | `train.py` | `pseudo_with_2024_cotracker3_extra/` | `STIRcombined/` | `scaled_online` | 45 |
| `extra_thr_peng_e40` | `fast_run_with2024_extrapoints_thresholdloss_pengchkpt` | `train.py` | `pseudo_with_2024_cotracker3_extra/` | `STIRcombined/` | `litetracker_finetuned` | 45 |
| `trial_e45` | `fast_run_trial_continueepoch45` | `train.py` | `pseudo_labels/` | `STIRDataset/` | chained ↓ | 25 |
| `peng_baseline` | — not trained here — | — | — | — | `track_on/checkpoint/litetracker_finetuned.pth` | — |

`trial_e45` is the third leg of a chain, each stage resuming from the previous
`student_last.pth`, all on `pseudo_labels/` + `STIRDataset/`:
`fast_run_trial` (10 ep, from `scaled_online`) →
`fast_run_trial_continue` (10 ep) →
`fast_run_trial_continueepoch45` (25 ep). It is the only lineage with **no
patient overlap with `STIRTest_2025`**, which is why it is kept despite ranking
mid-table.

⚠ **`extra_e40` did not train on what its name says.** The launch line carried
`pseudo-data-sir=…/pseudo_with_2024_cotracker3_extra/` — a typo. OmegaConf
accepted it as a new key `pseudo_data_sir`, so `pseudo_labels_dir` kept its YAML
default and the run trained on `pseudo_labels/` (STIR-original-only, select
mode) while reading frames from `STIRcombined/`. The stray key is still visible
in that run's `config.yaml`. Whatever `extra_e40` measures, it is not
densification.

## The shipped run, verbatim

```bash
python train.py --config train.yaml \
  --pseudo-labels-dir "$PSEUDO_LABELS_DIR" \
  --stir-root         "$STIR_TRAIN_ROOT" \
  checkpoint="$INIT_CKPT" \
  epochs=45 save_freq=2 wandb.run_name=fast_run_with2024_agg
```

Wrapped as `scripts/train.sh`. The original used the relative form
`checkpoint=../track_on/checkpoint/litetracker_finetuned.pth`, which resolved
from the old working directory. The public command names that same file through
the machine-local `INIT_CKPT` variable.

`--key value` and `key=value` are interchangeable —
`model.normalize_overrides` maps hyphens in *keys* to underscores and
feeds everything to OmegaConf as a dotlist. That is also why a typo becomes a
silent new key rather than an error; see `extra_e40` above.

## Excluded models

Two trained models are deliberately absent from the board: 3D acc_avg
0.510 and 0.474, against a 0.61 CONTROL ("the point never moved") and 0.71–0.74
for everything listed. Broken, not marginal.

- `fast_run_pseudo_cotracker3_continue/student_e44.pth` — 0.51013
- `fast_run_with2024_continue/student_e44.pth` — 0.47401

`model_runs/20260724_152332/` is not a model either: a truncated command line
(`wandb.run_name` cut to a bare `w`) made the trainer name the run after its
timestamp. Ignore it.
