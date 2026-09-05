# Data and artifact layout

Everything below lives under `${DATA_ROOT}` = `/mnt/cluster/datasets`.

## Source datasets

| path | what |
|---|---|
| `STIRDataset/` | STIR original. Patients `0`–`28`. |
| `STIRChallenge_2024/` | STIR Challenge 2024. Patients `02`–`11` (9 of them). |
| `STIRcombined/` | **symlink farm** over the two above — the root training used. |
| `STIRTest_2025/` | the 32 held-out clips every reported number is measured on. |

### Why `STIRcombined` exists

The two collections cannot simply be merged: 2024's patient `02` and STIR's
patient `2` would produce colliding clip ids. The farm keeps the STIR names
as-is and suffixes the 2024 ones, so `STIRChallenge_2024/02` becomes
`STIRcombined/02_2024`. That suffixed name is what ends up in **every**
raw-track and pseudo-label filename, so it has to match exactly or the trainer
will not find frames for its labels.

Rebuild it with `scripts/make_stir_combined.sh`.

## Clip ids

One clip = one patient + one side + one sequence. The id is
`<patient>__<side>__<seq>`, e.g. `02_2024__left__seq26`, and it is what pairs a
raw track with a pseudo-label with a set of frames. On disk the patient is the
directory and the rest is the filename.

```
raw track     <raw_tracks_root>/<teacher>/<patient>/<side>__<seq>__<teacher>.npz
pseudo-label  <pseudo_labels_dir>/<patient>/<side>__<seq>.npz
```

## Raw teacher tracks — phase 1 output

| path | collection | contents |
|---|---|---|
| `STIRprocessed/STIROrig_tracks/<teacher>/` | STIR original | 6 teachers × 29 patients |
| `STIRprocessed/STIRChallenge2024_tracks/<teacher>/` | STIR 2024 | 6 teachers × 9 patients (`02_2024` … `11_2024`) |

Teachers, in the order they appear in the fused labels' `teacher_names`:
`alltracker`, `cotracker3`, `trackon2`, `trackon_r`, `locotrack`, `mft`.

All were collected at `--skip 5` (every 5th frame). **This must match
`skip: 5` in `train.yaml`** — the trainer indexes frames by that stride,
so a mismatch silently supervises the wrong frames.

## Pseudo-labels — phase 2 output

Each `.npz` holds `coords [N,T,2]`, `visibility [N,T]`, `weight [N,T]`,
`source [N,T]` (an index into `teacher_names`) and `teacher_names`.

| directory | mode | clips | points | teachers/clip | used by |
|---|---|---|---|---|---|
| `pseudo_labels/` | select | 566 | 6 052 | 6 (79 clips: 1) | `trial_*` runs |
| `pseudo_labels_with_2024/` | select | 626 | 6 548 | 6 (79 clips: 1) | `fast_run_with2024*` |
| **`pseudo_labels_with_2024_agg/`** | **aggregate** | **626** | **6 548** | **6 (79 clips: 1)** | **the shipped model** |
| `pseudo_with_2024_cotracker3_extra/` | copy-through | 626 | 13 204 | 1 | `extra_*` runs |
| `pseudo_with2024_cotracker3_extra_aroundpoints/` | copy-through | 626 | 12 409 | 1 | `around_peng_*` runs |
| `pseudo_labels_with2024_extra_aroundpoints_agg/` | copy-through | 626 | 21 487 | 1 | `agg_around_*` runs |

The 79 single-teacher clips in the verified sets are clips only one teacher
managed to cover; `run_phase2.py` copies those straight through, since there is
nothing to cross-check them against.

**The three densification sets are single-teacher throughout — read
[CAVEATS.md](CAVEATS.md) before interpreting them.**

Intermediate raw-track roots for densification (`collect_extra_points.py`
output, one `cotracker3/` teacher directory each):
`pseudo_labels_with_extra_beforeprocessing/`,
`pseudo_labels_with2024_beforeprocess_aroundpoints/`,
`pseudo_labels_with2024_agg_beforeprocess_aroundpoints/`.

## Training runs

`STIRprocessed/model_runs/<run_name>/` holds `config.yaml` (written by the
trainer — the authoritative record of what that run actually used),
`student_e<N>.pth` every `save_freq` epochs, and `student_last.pth`.

`docs/RUNS.md` maps each board tag to its run directory and the
command that produced it.

## Evaluation artifacts

`STIRprocessed/eval_sweep/` — one `<tag>.json` + `<tag>.npz` per evaluated
checkpoint, `logs/<tag>.log`, and `analysis/{ranking.md,summary.json,per_point.npz}`
from `compare_ckpts.py`. Copies of the analysis outputs are checked in here as
`results/ranking.md` / `sweep_summary.json`.
