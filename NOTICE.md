# Third-party licenses

The MIT licence in [LICENSE](LICENSE) covers **the code in this repository**,
all of which is our own. No upstream source is vendored here — every external
project is cloned separately and imported at run time.

The released **checkpoint is not MIT**, and that distinction matters.

## The checkpoint

`student.pth` is a fine-tune of CoTracker3-Online weights, reached through the
STIR-2025 `litetracker_finetuned.pth` initialisation. CoTracker3 and LiteTracker
are released under **CC BY-NC 4.0**, so the derived weights inherit that
licence: free to use, share and adapt **with attribution, for non-commercial
purposes only**. Nothing in this repository can relicense them, and the MIT
grant above does not extend to them.

Practically: research and teaching are fine; a commercial product is not,
without permission from the upstream rights holders.

## What each dependency is licensed under

| project | licence | how this repo uses it |
|---|---|---|
| [co-tracker](https://github.com/facebookresearch/co-tracker) | CC BY-NC 4.0 | imported by `model.py` / `train.py` for the model class and loss; the student's lineage |
| LiteTracker ([arXiv:2504.09904](https://arxiv.org/abs/2504.09904)) | CC BY-NC 4.0 | imported by `student_lt_wrapper.py` for the streaming runtime |
| [MFT](https://github.com/serycjon/MFT) | CC BY-NC-SA 4.0 | teacher, label generation only |
| [track_on](https://github.com/gorkaydemir/track_on) | MIT | teachers: CoTracker3 / AllTracker / LocoTrack wrappers, Track-On2, Track-On-R |
| [STIRLoader](https://github.com/athaddius/STIRLoader) | MIT | dataset loading; used with our patch |
| [STIRMetrics](https://github.com/athaddius/STIRMetrics) | MIT | metric conventions |
| [stir-challenge-2026-inference](https://github.com/mertkaraoglu/stir-challenge-2026-inference) / [-metrics](https://github.com/mertkaraoglu/stir-challenge-2026-metrics) | no licence file at time of writing | the submission container and the official metric formulas |

Teachers were used **offline, to generate pseudo-labels**. Their weights are not
redistributed here, and none of their code is copied into this repository.

## Dataset

The STIR dataset carries its own terms; obtain it from the sources in
[docs/DATA.md](docs/DATA.md) and cite it — see [CITATION.cff](CITATION.cff) and
the README.

## If you need a permissively licensed checkpoint

Retrain from a permissive initialisation. `scripts/train.sh` takes
`checkpoint=` — point it at weights whose licence suits you. Everything else in
the pipeline (verifier, pseudo-labels, training loop) is MIT.
