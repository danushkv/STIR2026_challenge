# Environments and upstream repositories

## Why there is more than one environment

MFT and the track_on family pin incompatible `torch` / `torchvision` / `timm`
versions and cannot be imported into one Python process. They never need to be:
label generation is offline, so each teacher runs separately, on its own
schedule, in its own venv — only its **output** (a `.npz`) has to land in the
same directory. `teachers.py` imports each teacher's real dependencies lazily,
inside that teacher's `__init__`, so a process never touches another teacher's
environment.

| venv | used by |
|---|---|
| `trackon_env` | everything except MFT: the five track_on-based teachers, **all training**, all evaluation, the sweep, the analysis |
| `mft_env` | `collect_tracks.py --teacher mft` only |

On the cluster these live at `${VENV_ROOT}` = `/mnt/cluster/environments/venkateda`:

```bash
source /mnt/cluster/environments/venkateda/trackon_env/bin/activate
source /mnt/cluster/environments/venkateda/mft_env/bin/activate
```

`trackon_env` needs, beyond track_on's own requirements: `omegaconf`, `wandb`,
`opencv-python`, `scipy`, `matplotlib`, plus `STIRLoader` importable (see below).

> **These venvs are not self-contained.** Both were created with `uv` and their
> `bin/python` is a symlink into `~/.local/share/uv/python/`. `trackon_env`
> points at `cpython-3.12.13-linux-x86_64-gnu`, which is present on the cluster
> nodes but not on every workstation — where it is missing, the venv is dead and
> `source .../activate` leaves you with no `python` on `PATH` at all, rather
> than an error. Check with
> `ls -l ${VENV_ROOT}/trackon_env/bin/python` before blaming the code.
> No `requirements.txt` is checked in because the exact resolved versions were
> never frozen; freeze them with `uv pip freeze` from a node where the venv
> works, and commit the result.

## Upstream repositories

All are **cloned, not vendored**. `THIRDPARTY_ROOT` (default
`/mnt/nct-zfs/TCO-Test/venkateda/miccai_challenges/stir`) is the directory that
holds them.

| directory | repository | why it is needed |
|---|---|---|
| `track_on/` | `github.com/gorkaydemir/track_on` | five of the six teachers. `ensemble/` ships inference wrappers for CoTracker3 / LocoTrack / AllTracker; `model/trackon_predictor.py` is Track-On2 / Track-On-R itself. |
| `MFT/` | `github.com/serycjon/MFT` | the sixth teacher. Ships its own RAFT checkpoint, so nothing to train. |
| `co-tracker/` | `github.com/facebookresearch/co-tracker` | **training only.** `train.py` builds the real `CoTrackerThreeOnline` module and calls upstream's own `sequence_loss` — not a reimplementation. |
| `lite-tracker-master/` | LiteTracker | the streaming runtime (its `src/lite_tracker.py`) that `eval_2d.py`, `eval_3d.py`, `eval_sweep.py` and the submission wrappers load the student into. Also ships `model/scaled_online.pth`, Meta's stock CoTracker3-Online weights. |
| `STIRLoader/` | `github.com/athaddius/STIRLoader` | decodes STIR clips and reads the IR-tattoo start/end segmentation. **Must be the patched clone — see below.** Located at import time by `collect_tracks._import_stirloader`. |
| `STIRMetrics/` | `github.com/athaddius/STIRMetrics` | reference for the 2D/3D metric conventions. |
| `stir-challenge-2026-inference/` | challenge organisers | the container the submission is built on top of. |
| `stir-challenge-2026-metrics/` | challenge organisers | the official AJ / ATA / OA formulas, and the latency definition `latency_from_preds.py` mirrors. |

## Checkpoints that must exist before phase 1

Under `${THIRDPARTY_ROOT}/track_on/checkpoint/`:

| file | teacher | source |
|---|---|---|
| `alltracker.pth` | AllTracker | upstream release |
| `locotrack_base.ckpt` | LocoTrack | upstream release |
| `track_on_r.pt` | Track-On-R | upstream release |
| `trackon2_dinov3_checkpoint.pt` | Track-On2 | upstream release |
| `litetracker_finetuned.pth` | — | the **STIR-2025 winning** checkpoint. Two roles: the `peng_baseline` row on the ablation board, and the initialisation of the shipped run. |

CoTracker3 as a teacher needs no file — `ensemble/cotracker` fetches it through
`torch.hub`. Track-On2 / Track-On-R additionally need the DINOv3 backbone
(`facebook/dinov3-vits16plus-pretrain-lvd1689m`), which is gated on the HF Hub:
either `huggingface-cli login` with access granted, or set `DINOV3_LOCAL_DIR`.

`${THIRDPARTY_ROOT}/lite-tracker-master/model/scaled_online.pth` is the stock
CoTracker3-Online checkpoint and the default `checkpoint:` in
`src/train.yaml`.


## STIRLoader

Every result in this repository was produced with a **locally patched**
STIRLoader that was **never installed** into either venv. Both facts matter.

### It is patched

Base commit `7e7f87c4f2c2aac525207f499ff18221f7936822`, plus
`patches/stirloader-streaming-skip.patch` (25 insertions, 19 deletions in
`STIRLoader/STIRLoader.py`).

Stock upstream decodes the **entire** clip into a Python list and only then
applies `os.environ["SKIP"]` by slicing. The patch moves the stride into the
ffmpeg read loop.

**It does not change which frames you get.** Upstream keeps `f[::SKIP]`, plus the
final frame re-appended when `len(f) % SKIP != 1`; the patched loop reproduces
exactly that set. Checked exhaustively for every clip length 1–600 against
SKIP ∈ {1,2,3,5,7}: identical output, no exceptions. So the patch is a
**memory** fix, not a correctness one, and results are comparable either way —
if you can afford the memory.

What it changes is peak RAM. The longest clip in the evaluation set is 2074
frames at 1280×1024×3:

| | per view | stereo pair |
|---|---|---|
| upstream — decode all, then slice | 8.2 GB | 16.3 GB |
| patched — stride while decoding, `skip=5` | 1.6 GB | 3.3 GB |

`pip install git+https://github.com/athaddius/STIRLoader` therefore does **not**
reproduce these results. Apply the patch:

```bash
git clone https://github.com/athaddius/STIRLoader "${THIRDPARTY_ROOT}/STIRLoader"
cd "${THIRDPARTY_ROOT}/STIRLoader"
git checkout 7e7f87c
git apply /path/to/stir2026-nct/patches/stirloader-streaming-skip.patch
```

### It was never installed

It is not in `trackon_env`'s site-packages and there is no `.pth` for it.
`collect_tracks._import_stirloader` found it on `sys.path` by relative path, and
that is how every run — training, evaluation, figure generation — reached
it. Nothing depended on the environment.

Two ways to make it importable; the first needs no install at all.

**1. Point at the clone (what the scripts do).** Every script that launches
Python now exports:

```bash
export STIRLOADER_ROOT="${THIRDPARTY_ROOT}/STIRLoader"
```

`_import_stirloader` tries, in order: an installed `STIRLoader` package,
`$STIRLOADER_ROOT`, `$THIRDPARTY_ROOT/STIRLoader`, then `../STIRLoader` and
`./STIRLoader` relative to `src/`.

**2. Install it into the venv — with `--no-deps`.**

```bash
source "${VENV_ROOT}/trackon_env/bin/activate"
pip install --no-deps -e "${THIRDPARTY_ROOT}/STIRLoader"
```

> `--no-deps` is not optional. STIRLoader's `pyproject.toml` pins
> `numpy==1.23.5` and `opencv-contrib-python==4.5.5.64`. `trackon_env` has
> numpy 2.4.4, opencv-python 5.0.0.93 and torch 2.4.1+cu121 — a plain
> `pip install .` would downgrade numpy by a major version and add a second,
> conflicting OpenCV, which breaks torch. `-e` also keeps the patch live rather
> than baking a copy.
