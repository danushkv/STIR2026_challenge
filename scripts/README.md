# scripts/

One script per pipeline stage. Each defaults its paths to our cluster and lets
you override them:

```bash
REPO_ROOT=… THIRDPARTY_ROOT=… DATA_ROOT=… VENV_ROOT=… bash <script>
```

Run in this order. [docs/REPRODUCE.md](../docs/REPRODUCE.md) is the narrative
version, with the expected numbers at each step.

| script | stage | cost |
|---|---|---|
| `make_stir_combined.sh` | build the `STIRcombined` symlink farm over STIROrig + STIR-2024 | seconds |
| `collect_teacher.sh <teacher> [orig\|2024]` | **phase 1** — run one teacher over one collection | GPU-hours, ×12 |
| `run_phase2.sh` | **phase 2** — verify and aggregate-fuse into pseudo-labels | CPU, minutes |
| `train.sh` | train the student | ~17 h on one A100 80GB |
| `eval.sh` | evaluate a checkpoint at iters 1/2/4, then the paired bootstrap | ~1.8 h |

Before a long run, use the real-data smoke path:

```bash
bash scripts/preflight.sh
bash scripts/smoke_phase2.sh
bash scripts/smoke_train.sh
bash scripts/smoke_eval.sh /tmp/stir2026-smoke-runs/<run>/student_last.pth
```

`preflight.sh` checks the exact local inputs and hashes. The remaining three
commands regenerate one pseudo-label, perform one GPU optimizer step, strictly
reload its checkpoint, and evaluate one clip. See [`configs/`](../configs/README.md)
for the exact released, smoke, and evaluation configurations.

`logs/` is where the SLURM headers write; it must exist before `sbatch`.

After editing anything here, run `python tools/check_repo.py`.

The `#SBATCH` headers carry our cluster's partitions and a `--mail-user`.
Change or delete them before running elsewhere.
