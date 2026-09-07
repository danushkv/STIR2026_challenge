# Caveats

Things found while assembling this repository that a reproducer needs to know,
and that the report text does not currently say.

## 1. The densification runs replaced the verified labels rather than adding to them

The report describes seeding extra query points "in addition to" the IR tattoo
points, treated as self-labelling data at lower weight. That is not what the
label sets on disk contain.

`experiments/collect_extra_points.py` samples points that are distinct from the existing
pseudo-label points, and then tracks **only the new points** with one teacher
(`cotracker3`). Its output goes to a fresh raw-tracks root holding a single
teacher directory. `run_phase2.py` reading that root sees one covering teacher
per clip, so it takes the copy-through path — and writes a pseudo-label that
contains the new points *and nothing else*.

Checked on `02_2024__left__seq26`:

| label set | points | teachers | frame-0 coordinates |
|---|---|---|---|
| `pseudo_labels_with_2024_agg` | 3 | all six, fused | the three IR tattoos |
| `pseudo_labels_with2024_extra_aroundpoints_agg` | 16 | `cotracker3` only | 16 sampled points, none of them a tattoo |

Across the whole set, every file in all three densification label sets has
exactly one teacher (626/626), where the verified sets have six on 547 of 626.

So the `extra_*` / `*_around*` runs are not "verified labels plus extra points".
They are "one teacher, no endpoint anchoring, no cross-teacher agreement, no
tattoo points at all". Two variables moved at once — point placement **and** the
loss of the entire verifier — so the report's negative result for densification
is confounded. The honest statement is that these runs did not beat aggregate
fusion; they do not isolate the effect of adding points.

Fixing it means running `run_phase2.py` once with the six teacher roots *and*
the extra-points root passed together, so verified and unverified points land in
one label file. Nothing in the code prevents that; it was not what was run.

## 2. The report says STIROrig; training used STIROrig **and** STIR-Challenge-2024

The submitted PDF says "We ran six trackers over the STIROrig dataset". The
artifacts on disk say otherwise, unambiguously: the released pseudo-label set
`pseudo_labels_with_2024_agg/` contains patients `02_2024` … `11_2024` alongside
`0` … `28`, and the run's own `config.yaml` records
`stir_root: <data>/STIRcombined/`, the symlink farm over both
collections. 626 clips, where STIROrig alone gives 566.

This matters for anyone reproducing: training on STIROrig alone will not
reproduce the released checkpoint. It also matters for the held-out claim — the
2024 patients overlap `STIRTest_2025` at patient level, which is exactly why
`trial_e45` (the STIROrig-only lineage) is kept in the manifest despite ranking
mid-table. See [DATA.md](DATA.md) and [RUNS.md](RUNS.md).

## 3. The shipped model is initialised from the 2025 winning checkpoint

`fast_run_with2024_agg` started from `litetracker_finetuned.pth`, i.e. the
`peng_baseline` weights — not from stock CoTracker3. Table 1 of the report marks
`peng` only on some rows and not on the `agg` ones. Six of the nine trained
models in the manifest started from that checkpoint; only `trial_e45`,
`extra_e40` and `extra_thr_e{30,42}` started from `scaled_online.pth`.

This matters for the claim "the distillation beats the 2025 winning checkpoint
in 3D": the comparison is a fine-tune of that checkpoint against the checkpoint
itself, which is a fair claim about the fine-tuning, but not a from-scratch one.

## 4. `extra_e40` trained on the wrong label set

A typo in its launch line (`pseudo-data-sir=`) became a new OmegaConf key, so
the run silently used the YAML default `pseudo_labels/`. See
[RUNS.md](RUNS.md).

## 5. Only two of the five "aggregate" runs were actually fused by aggregation

Section "Aggregate fusion beats per-frame selection in 2D" pools five runs
marked `M = a` against seven marked `M = s`. But `--mode` only has an effect on
clips that **more than one teacher covered**, and three of those five runs
(`agg_around_e30`, `agg_around_e44`, `agg_around_thr_e30`) trained on
`pseudo_labels_with2024_extra_aroundpoints_agg/`, which is single-teacher
copy-through on all 626 clips — the mode never ran. Only `agg_e40` and `agg_e44`
were genuinely aggregate-fused.

The two groups also differ in initialisation, label set and loss (see
[RUNS.md](RUNS.md)). The report already flags this as "a strong association
rather than a controlled ablation"; the ledger here is what makes it checkable.
A controlled version is one `run_phase2.py` re-run with `--mode select` on the
*same* raw tracks, and one training run differing in nothing else.

## 6. `select` vs `aggregate` is set at label-generation time, not training time

`--mode` is an argument to `run_phase2.py`. There is no training-time switch.
Reproducing a `select` run means regenerating the labels.

## 7. Reported 3D numbers use annotation that the harness does not provide

Every 3D number from `eval_sweep.py` takes the right-view start points from
STIRLoader's `getsegsstereo`, which contour-matches the tattoo segmentation in
both views. The 2026 harness gives left-view queries only. The shipped container
recovers the right-view points from pixels by NCC along the epipolar row, which
costs 0.052 acc_avg; the geometric consistency check recovers 0.017 of that.

Expect **0.7031**, not 0.7383. The exact measurable submission settings are in
[`../configs/eval_submission.yaml`](../configs/eval_submission.yaml).

## 8. Training is not bit-reproducible

Batch size is 1 and `seed: 0` is set, but the runs used a multi-worker
`DataLoader` with no worker seeding pinned, cuDNN autotuning was left on, and
`torch.use_deterministic_algorithms` was never enabled. Compare evaluation
metrics, not checkpoint hashes.

## 9. There is no per-frame ground truth anywhere in this pipeline

STIR annotates the first and last frame only. Every number reported — 2D
δ_avg and 3D acc — is an **endpoint**
proxy. Visibility is entirely unvalidated. `eval_sweep.py`'s cycle-drift
column exists precisely because endpoint error cannot see a point that wanders
off and drifts back.
