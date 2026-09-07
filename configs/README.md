# Configurations

These files separate the recorded experiment from small validation runs.

| file | purpose |
|---|---|
| `reproduce.yaml` | effective configuration of the released 45-epoch run |
| `smoke.yaml` | one clip, one epoch, no W&B; proves a forward/backward/update/save cycle |
| `eval_reproduction.yaml` | reported offline protocol using annotation-derived stereo starts |
| `eval_submission.yaml` | submitted 3D protocol using image matching and disparity repair |

Machine-local paths use OmegaConf's `oc.env` resolver. Copy `.env.example` to
`.env`; the shell drivers load and export it automatically. Each value also has
a repository-relative fallback so opening a config does not expose a private
filesystem path.

Command-line overrides are strict. Unknown keys fail instead of silently
creating a new OmegaConf entry.
