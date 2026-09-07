# Configurations

These files separate the recorded experiment from small validation runs.

| file | purpose |
|---|---|
| `reproduce.yaml` | effective configuration of the released 45-epoch run |
| `smoke.yaml` | one clip, one epoch, no W&B; proves a forward/backward/update/save cycle |
| `eval_reproduction.yaml` | reported offline protocol using annotation-derived stereo starts |
| `eval_submission.yaml` | submitted 3D protocol using image matching and disparity repair |

The paths are intentionally still the current cluster paths. They remain here
until the local preflight and smoke runs pass. Portability is a separate pass:
do not change paths and training behavior at the same time.

Command-line overrides are strict. Unknown keys fail instead of silently
creating a new OmegaConf entry.
