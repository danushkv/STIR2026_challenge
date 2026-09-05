# What changed when this repository was assembled

The reproduction check compares this repo against the results produced from the
old working tree (`…/stir/files/`). This is the complete list of differences, so
nothing has to be taken on trust.

## Python: three edits, none on a result-producing path

`diff` against the originals is clean apart from these.

| file | change | why |
|---|---|---|
| `src/train.yaml` | `repo_root: ../co-tracker/` → the absolute path to **the same clone** | the relative form only resolved when the working directory was `files/` |
| `src/collect_extra_points.py` | `--repo-root` default `../track_on` → absolute path to **the same clone** | same reason |
| `src/run.py` | demo clip id `synthetic_000` → `synthetic__clip000` | the dependency-free demo crashed on its own output: `save_pseudo_label` splits the clip id on `__` to pick a subdirectory, and the demo's id had no `__`. Pre-existing bug, demo only. |

Everything else under `src/` is byte-identical to `files/`. In particular
`verifier.py`, `pseudo_label.py`, `run_phase2.py`, `teachers.py`,
`collect_tracks.py`, all three `train.py`, `eval_sweep.py` and
`compare_ckpts.py` are untouched.

The modules stay in **one flat directory** for exactly this reason: they import
each other by bare name (`from verifier import ...`), so any repackaging into
subpackages would have meant editing every import.

## Shell: path plumbing only

- `cd "${REPO_ROOT}/files"` → `cd "${REPO_ROOT}/src"` everywhere.
- `REPO_ROOT=<hardcoded>` → `REPO_ROOT="${REPO_ROOT:-<this repo>}"`, plus a new
  `THIRDPARTY_ROOT` for the cloned upstream repos (`track_on/`, `MFT/`,
  `co-tracker/`, `lite-tracker-master/`), which are no longer siblings of the
  code. `DATA_ROOT` and `VENV_ROOT` likewise.
- `submission/build_all.sh`: `${HERE}/../stir-challenge-2026-inference` →
  `${THIRDPARTY_ROOT}/stir-challenge-2026-inference`.
- `submission/test_submission.sh`: `${HERE}/../val` → `${THIRDPARTY_ROOT}/val`
  (still overridable as `$1`), and `${HERE}/../files/latency_from_preds.py` →
  `${HERE}/../src/latency_from_preds.py`.
- `run_sweep.sh --help` printed lines 2–26 of itself, which after the path block
  moved would have included shell code; now 2–21.

## Shell: four collect scripts merged into one

`slurm_collect_tracks_mft.sh`, `slurm_collect_tracks_cotracker.sh`,
`trackon.sh` and `trackon2.sh` were near-identical copies covering four of the
six teachers, and their `--out-dir` values were stale — they wrote to flat
directories (`STIRprocessed/co_tracker3/`, `STIRprocessed/trackon_r/`) that do
not match the layout the tracks actually ended up in
(`STIRprocessed/STIROrig_tracks/<teacher>/`). `collect_trackon2.sh` also pointed
its output at `trackon_r/`, which would have overwritten Track-On-R's tracks.

They are replaced by `scripts/collect_teacher.sh <teacher> [orig|2024]`, which
covers all six teachers, selects the right venv and checkpoint per teacher, and
writes the layout `run_phase2.py` actually reads. The `collect_tracks.py`
invocation itself — teacher, repo root, stir root, `--skip 5` — is unchanged.

## A break the restructure introduced, and its fix

`collect_tracks._import_stirloader` located STIRLoader by walking **up one
directory** from itself: `files/collect_tracks.py` → `files/..` → `stir/` →
`stir/STIRLoader`. STIRLoader is not installed in either venv, so this fallback
was the only thing that ever made it importable — for training, evaluation and
the paper-figure script alike.

Moving the modules to `stir2026-nct/src/` put a directory between them and the
clone, so that hop resolved to `stir2026-nct/STIRLoader`, which does not exist,
and every entry point that touches STIR video would have died on
`ImportError: STIRLoader not found`.

**Five modules had the same assumption**, found by sweeping for `__file__`
arithmetic rather than one at a time:

| module | wanted | fired when |
|---|---|---|
| `collect_tracks.py` | `STIRLoader` | any clip decode — training included |
| `prepare_stir_clips.py` | `STIRLoader` | legacy clip prep |
| `student_lt_wrapper.py` | `lite-tracker-master` | every streaming evaluation |
| `eval_sweep.py` | `stir-challenge-2026-inference` | only `right_queries=match` |
| `test_student_2026.py` | `stir-challenge-2026-metrics` | AJ/ATA/OA |

All five now go through one resolver, `src/_thirdparty.py::find_repo`, which
tries in order: the repo-specific variable (`$STIRLOADER_ROOT`,
`$LITETRACKER_ROOT`, `$STIR_METRICS_ROOT`, `$STIR_INFERENCE_ROOT`), then
`$THIRDPARTY_ROOT/<name>`, then the original relative candidates — `../<name>`
**and** `../../<name>`, so the clones in their current location are still found
with no environment set at all. On failure it names every path it tried instead
of surfacing a bare `ModuleNotFoundError` from inside the upstream package.
Every script that launches Python exports all four variables.

`student_lt_wrapper.py` keeps its two-entry `sys.path` insert
(`lite-tracker-master/` **and** its `src/`) — `lite_tracker.py` imports its own
modules absolutely as `src.model_utils`, so the parent must be importable.
Checked that this does not shadow anything: no module name in `src/` collides
with `lite-tracker-master/` or `lite-tracker-master/src/`.

The STIRLoader clone is also **patched** (`patches/stirloader-streaming-skip.patch`,
base `7e7f87c`) — upstream would OOM at `--skip 5`. See
[ENVIRONMENTS.md](ENVIRONMENTS.md#stirloader).

## New files

Nothing here existed before; nothing here affects a result.

- `scripts/make_stir_combined.sh` — rebuilds the `STIRcombined` symlink farm,
  which previously existed only on disk with no script behind it.
- `scripts/run_phase2.sh` — the phase-2 invocation for the shipped label
  set. Reconstructed: no shell script or history recorded it, so the teacher
  list and `--mode aggregate` were recovered from the label files themselves
  (`teacher_names`, teacher counts per clip) and from the directory name.
- `scripts/train.sh` — the shipped training command, recovered from that
  run's wandb metadata and cross-checked against the `config.yaml` the trainer
  wrote into the run directory.
- `scripts/README.md`, `docs/*`, `README.md`, `.gitignore`,
  `submission/weights/README.md`.

## Files deliberately left behind

| file | why |
|---|---|
| `view_lerobot_dataset.py` | LeRobot dataset viewer, unrelated to STIR |
| `train_litetracker.py` | a third party's script, hardcoded to their home directory; nothing in this pipeline calls it |
| `student_runtime.py` | a from-scratch scaffold of LiteTracker's inference tricks, written before the real repo was available. Superseded by `student_lt_wrapper.py`, which loads the real thing. Imported by nothing. |
| `eval_student.py` | early evaluation script, superseded by `eval_2d.py` (streaming, matches what the challenge scores) |
| `wandb/`, `logs/`, `viz/`, `thumbs/`, `*.out`, `*.png`, `__pycache__/` | run artifacts |
| `files.zip`, `files(1).zip` | old snapshots of the same directory |
| `report/` (LaTeX source + `make_figures.py`) | superseded by the submitted PDF at the repository root, which it no longer matches (different team name, different densification wording). Kept in the original working tree. |
| `prepare_stir_clips.py` | legacy `.npz` clip prep; every run used `--stir-root` instead |
| `stats.txt` / `stats3d.txt` | pre-sweep logs, superseded by `docs/results/ranking.md` |



## Not yet done

Path genericisation. Absolute cluster paths are still the defaults, on purpose,
so the reproduction check runs against exactly what produced the results. Once
it passes, `REPO_ROOT` / `THIRDPARTY_ROOT` / `DATA_ROOT` / `VENV_ROOT` become
relative or required-argument, and the same treatment applies to the defaults in
`src/train.yaml`, `src/ckpt_manifest.txt` and the `argparse` defaults in
`collect_extra_points.py`.


---

# The open-source cleanup

Everything above describes lifting the code out of the working tree. This
section describes the second pass: cutting it down to what a reader actually
needs. `src/` went from 30 files to 22, `scripts/` from 10 to 5.

## Deleted — nothing imported them and no released result depends on them

| removed | why |
|---|---|
| `learned_verifier.py` | a trained replacement for the hand-tuned score. Never used by any released checkpoint. `verifier.py`'s `LearnedVerifier` subclass and the `verifier: "cheap"\|"learned"` config knobs went with it (79 lines). |
| `train_student_threshold.py` | the threshold-aware-loss ablation. Reported as a negative result; the released model does not use it. |
| `test_student_2026.py` | AJ/ATA/OA on endpoint-only ground truth. The report states these cannot be measured meaningfully without the 2026 `annotations.json`. |
| `dataset_stats.py` | point-count histogram, a one-off diagnostic. |
| `visualize_tracks.py` | sanity check for `collect_tracks.py`'s downscale plumbing, not a result visualisation. |
| `run.py` | synthetic `DummyTeacher` demo. `python src/verifier.py` is the real dependency-free smoke test and it stays. |
| `prepare_stir_clips.py` | legacy `.npz` clip prep; every run used `--stir-root`. |
| `ckpt_manifest.txt` | 13 absolute cluster paths, meaningless outside our filesystem. |
| `run_sweep.sh`, `run_stage1.sh`, `run_stage2.sh`, `slurm_eval_sweep.sh`, `slurm_analyse.sh`, `smoke_test_sweep.sh` | the manifest-driven batch drivers for our ablation board. The board itself is preserved in [results/ranking.md](results/ranking.md), and the per-checkpoint commands in [RUNS.md](RUNS.md); reproducing a row is two commands, in [REPRODUCE.md](REPRODUCE.md). |

## Merged

`validate_student.py` and `eval_student_3d.py` ran CoTracker3's whole-clip
sliding-window forward — **not** the pass the challenge scores. Nobody should
run them, but three helpers each were imported by the streaming evaluators.
Those helpers moved verbatim into **`eval_common.py`** and the two scripts were
deleted. The duplicated `_nn_dist` (identical KDTree query in both) collapsed
into one.

`train_student.py` was two things at once: a trainer nobody ran, and the shared
model/config core everything imports. The trainer half is gone; the core is now
**`model.py`** (114 lines, down from 407). The trainer is **`train.py`**, which
was already self-contained — its own config loader, dataset, loss and loop.

## Renamed, for names that say what the file does

| was | is |
|---|---|
| `train_student.py` | `model.py` (core only) |
| `train_student_fast.py` | `train.py` |
| `train_student.yaml` | `train.yaml` |
| `validate_student_lt.py` | `eval_2d.py` |
| `eval_student_3d_lt.py` | `eval_3d.py` |
| `eval_sweep_lt.py` | `eval_sweep.py` |
| `scripts/train_agg.sh` | `scripts/train.sh` |
| `scripts/run_phase2_agg.sh` | `scripts/run_phase2.sh` |
| `scripts/eval_repro.sh` | `scripts/eval.sh` |

The `_lt` suffixes marked "LiteTracker streaming runtime" back when a
non-streaming variant existed alongside. It doesn't any more, so the suffix
only said "the one you should use".

Weights & Biases metadata for the released run records the old name
`train_student_fast.py`. That is `train.py`.

## How this was verified

- symbol-level cross-module check: every `from X import y` resolves to a real
  top-level definition in `X`
- `python -m compileall` over `src/` and `tools/`
- `python src/verifier.py` — the self-test prints the same numbers as before the
  surgery (teacher RMSE 1.36 / 13.71, pseudo-label 1.81, 88% frames kept)
- no file mentions a module that no longer exists
