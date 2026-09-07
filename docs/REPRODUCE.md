# Reproducing the shipped model

Target: `agg_e44` — epoch 44 of `fast_run_with2024_agg`, which is
the released checkpoint (md5 `4709eaefce5d4d45c7222c8617d0890f`, SHA-256
`3b964f18793bab959d1537aa58b0411bb8454e2df5f30bf6d66039a215c8455c`).

Five steps. Steps 1–2 are the expensive part (GPU-days); if the artifacts they
produce already exist on the cluster you can start at step 3.

Copy `.env.example` to `.env` and set the local dataset, clone, interpreter,
and checkpoint paths. Shell drivers load it automatically. No private cluster
paths are encoded in the public configurations.

---

## 0. Prerequisites

Environments and clones: [ENVIRONMENTS.md](ENVIRONMENTS.md). Dataset layout:
[DATA.md](DATA.md).

Build the combined dataset root (idempotent, symlinks only):

```bash
bash scripts/make_stir_combined.sh
```

Start with the static/synthetic checks:

```bash
python src/verifier.py      # self-test on synthetic tracks: the fused label
                            # beats the teacher that drifts. No models, no data.
python tools/check_repo.py  # imports, links, scripts -- all still wired up
```

Then run the local preflight and the real-data smoke path before submitting a
17-hour job:

```bash
bash scripts/preflight.sh
bash scripts/smoke_phase2.sh
bash scripts/smoke_train.sh
bash scripts/smoke_eval.sh /tmp/stir2026-smoke-runs/<run>/student_last.pth
```

Phase 2 and training use `0__left__seq00` from STIR original; evaluation uses
`01__left__seq03` from `STIRTest_2025`. Together they check cached-track fusion,
one GPU forward/backward/update/save cycle, strict checkpoint reload, and one
streaming evaluation shard. Configuration details are in
[`../configs/README.md`](../configs/README.md).

---

## 1. Phase 1 — collect six teachers' raw tracks

One invocation per (teacher, collection): 6 × 2 = 12 runs. They are independent
and can go in any order; five use `trackon_env`, MFT uses `mft_env`. The script
picks the right venv and checkpoint per teacher.

```bash
for t in cotracker3 alltracker locotrack trackon2 trackon_r mft; do
  sbatch scripts/collect_teacher.sh "$t" orig
  sbatch scripts/collect_teacher.sh "$t" 2024
done
```

Writes `STIRprocessed/{STIROrig_tracks,STIRChallenge2024_tracks}/<teacher>/<patient>/<side>__<seq>__<teacher>.npz`.
`--skip 5` is pinned inside the script and must not be changed — see
[DATA.md](DATA.md).

Check the collections are complete and consistent across teachers before going on:

```bash
cd src && python sanity_check.py
```

## 2. Phase 2 — verify and fuse into pseudo-labels

CPU only, numpy only — this step imports no tracker. It reads back the phase-1
`.npz` files and pulls the IR-tattoo endpoints live from the dataset through
`STIRLoader.getendcenters()`.

```bash
bash scripts/run_phase2.sh
```

Two invocations (one per collection) into **one** output directory,
`STIRprocessed/pseudo_labels_with_2024_agg/`. Expect **626 files, 6 548 points**,
547 clips fused from all six teachers and 79 copied through from a single
covering teacher.

`--mode aggregate` is the one knob that defines this lineage: the verifier
blends all covering teachers weighted by their per-frame score, rather than
committing to the single best teacher per frame (`select` — `run_phase2.py`'s
default, and what produced every non-`agg` run).

Optional, and worth doing — score the labels against the endpoint GT to confirm
the fusion beats its own best teacher:

```bash
cd src
python evaluate_labels.py \
  --raw-tracks-root   "$DATA_ROOT/STIRprocessed/STIROrig_tracks" \
  --pseudo-labels-dir "$PSEUDO_LABELS_DIR" \
  --stir-root         "$DATA_ROOT/STIRDataset"
```

## 3. Train the student

```bash
bash scripts/train.sh          # or: sbatch scripts/train.sh
```

The run directory defaults to **`fast_run_with2024_agg_repro`**, not the
original name. `model.py` builds it as `<out_dir>/<wandb.run_name>` with
`exist_ok=True` and writes `student_e<N>.pth` unconditionally, so reusing
`fast_run_with2024_agg` could overwrite the released run's source checkpoint.
The script refuses to start if the target directory already
holds checkpoints. Override with `RUN_NAME=… bash scripts/train.sh`.

Weights & Biases is on by default (`wandb.mode: online`) and needs
`WANDB_API_KEY` in the environment. To run without it, append
`wandb.enabled=false`.

This is the recovered original command (cross-checked against the `config.yaml`
the trainer wrote into the run directory). Everything not named in it comes from
`src/train.yaml` unchanged:

| | |
|---|---|
| init | `track_on/checkpoint/litetracker_finetuned.pth` — the STIR-2025 winning checkpoint, **not** stock `scaled_online.pth` |
| labels | `pseudo_labels_with_2024_agg/` |
| frames | `STIRcombined/` |
| loss | CoTracker3's `sequence_loss`, `0.05·coord + vis + conf`, `add_huber_loss=False` |
| supervision | `weighted` — the verifier's per-frame weight multiplies the coordinate loss's validity mask |
| lr / epochs | `2e-5` / 45, `save_freq=2` |
| window_len / train_iters | 16 / 4 |
| model_resolution | 384 × 512 |
| max_train_frames / max_points | 64 / 384 |
| skip / seed | 5 / 0 |

Output: `STIRprocessed/model_runs/${RUN_NAME}/student_e44.pth`. `save_freq=2`
means only even epochs are kept — 23 files, ~2.2 GB.

### The reference run, to compare against

From that run's own wandb record (`fast_run_with2024_agg`, 2026-08-03):

| | |
|---|---|
| hardware | 1 × NVIDIA A100 80 GB PCIe, Python 3.12.13 |
| wall clock | **17.2 h** for 45 epochs (25 910 steps) |
| final `train_epoch/loss` | 0.09222 |
| final `train_epoch/coord_loss` | 0.41038 |
| final `train_epoch/vis_loss` | 0.05243 |
| final `train_epoch/conf_loss` | 0.01927 |

Watch the epoch losses rather than waiting 17 h for the eval — if they are
tracking these within a few percent by epoch 10, the data path is right.
(Note there are two wandb runs under this name ten minutes apart; the first died
before logging anything. `eroymege` is the real one.)

> **Bit-exactness.** Do not expect an identical checkpoint. Batch size is 1 and
> the seed is fixed, but the run used a multi-worker `DataLoader` and cuDNN
> autotuning, neither of which is pinned; and it was not run under
> `torch.use_deterministic_algorithms`. Judge the reproduction on the
> evaluation numbers in step 4, not on file hashes.

## 4. Evaluate under the streaming runtime

The numbers in the report come from `eval_sweep.py`, which streams every
frame through LiteTracker — the same forward pass as the submitted container,
not a whole-clip offline pass.

Single checkpoint:

```bash
cd src
python eval_sweep.py --config train.yaml \
  tag=agg_e44 \
  student_checkpoint="$STUDENT_CHECKPOINT" \
  val_stir_root="$STIR_TEST_ROOT" \
  iters_list='[1,2,4]' \
  out_dir="$EVAL_ROOT" < /dev/null
```

The `< /dev/null` is **not** optional when this runs inside a `while read` loop:
STIRLoader decodes with ffmpeg, ffmpeg reads stdin, and it will silently eat the
loop's input.

Expected for `agg_e44` (see [results/ranking.md](results/ranking.md)):

| iters | 2D δ_avg | 2D median | 3D acc_avg | p95 latency (RTX A5000) |
|---|---|---|---|---|
| 1 | 0.8085 | 4.55 px | 0.7339 | 48.7 ms |
| 2 | 0.8077 | 4.60 px | 0.7366 | — |
| 4 | 0.8103 | 4.60 px | 0.7383 | 128.7 ms |

Those 3D figures use `getsegsstereo` for the right-view start points, which is
annotation the 2026 harness does not provide. What the container ships scores
**0.7031** — see `submission/CHECKPOINT.txt`.

### Comparing a reproduced checkpoint against the original

```bash
bash scripts/eval.sh
```

Evaluates `fast_run_with2024_agg_repro/student_e44.pth` at iters 1/2/4 and then
runs the paired bootstrap against every shard already in the sweep directory.
It writes its shard **into the same sweep directory** on purpose:
`compare_ckpts.py` can only pair runs that scored the same clips and the same
points in the same order. The analysis lands in `analysis_repro/` so the
original `analysis/ranking.md` — the record the report was written from — is
left untouched.

Read the **paired delta between `agg_e44_repro` and `agg_e44`**, not the two
individual numbers. Their individual 95% CIs are ~±0.055 wide and will overlap
almost completely no matter what; the paired interval is what says whether the
reproduction landed. A delta whose CI spans zero is a successful reproduction.

`compare_ckpts.py` merges every shard in a sweep directory and ranks them with a
paired clip-level bootstrap. **Read the paired deltas, not the individual CIs.**
The evaluation is 234 points over 32 clips with an intra-clip correlation of
0.40 and a design effect of 3.54, so an *unpaired* comparison needs ~21 pp to
mean anything; the paired one needs far less.

Our own 13-checkpoint ablation board was produced by running the same two
commands once per checkpoint. The batch drivers that did it were tied to our
cluster paths and are not part of this repository; the resulting board is
preserved in [results/ranking.md](results/ranking.md) and the per-checkpoint
commands in [RUNS.md](RUNS.md).

## Submission-protocol evaluation

Challenge submission packaging is maintained separately and intentionally is
not tracked in this repository. The measurable behavior is pinned in
[`../configs/eval_submission.yaml`](../configs/eval_submission.yaml): image-based
right-view matching, four refinement iterations, and the measured disparity
repair (`k=8`, absolute tolerance 8 px, MAD multiplier 3).

---

## Did it reproduce? (measured)

We ran this guide end to end from step 3 and compared the result to the released
checkpoint, paired on identical clips and points:

| | iters | released | reproduction | delta | 95% CI | |
|---|---|---|---|---|---|---|
| 2D δ_avg | 1 | 0.8085 | 0.7991 | −0.0094 | [−0.0210, +0.0010] | tie |
| | 2 | 0.8077 | 0.8009 | −0.0068 | [−0.0134, +0.0013] | tie |
| | 4 | 0.8103 | 0.8017 | −0.0085 | [−0.0149, −0.0012] | separable |
| 3D acc_avg | 1 | 0.7339 | 0.7304 | −0.0035 | [−0.0229, +0.0167] | tie |
| | 2 | 0.7366 | 0.7304 | −0.0062 | [−0.0236, +0.0079] | tie |
| | 4 | 0.7383 | 0.7313 | −0.0070 | [−0.0249, +0.0070] | tie |

The one separable cell is entirely the **4 px bin**: per-threshold deltas at
iters=4 are −0.0342 (4 px) then −0.0043 / 0.0000 / −0.0043 / 0.0000. That is 8 of
234 points crossing the tightest threshold, with median endpoint error unchanged
(4.60 → 4.59 px). Sub-pixel jitter, not a worse model.

The 3D **mean** error looks alarming (6.16 → 16.98 mm) and is two points: median
3.01 → 2.99 mm and p90 14.79 → 14.31 mm both hold, and one triangulation blew up
to 2246 mm on `05__left__seq58`. Excluding the two points above 100 mm gives
0.7422 vs 0.7378.

Cycle drift — which needs no ground truth and catches points that wander off and
return — was *lower* in the reproduction at every iters setting (4.11 / 4.67 /
4.68 px vs 6.74 / 6.24 / 6.48).

Full table: [results/ranking_reproduction.md](results/ranking_reproduction.md).
