# STIR 2026 submission

> `weights/student.pth` is not in git — see [weights/README.md](weights/README.md)
> for the one command that puts it there. Everything else needed to build the
> images is present.

**ONE Docker image serves all three tracks.** The image content is identical for
every track — same checkpoint, same dependency group (`lt`), same code. What
differs is only the entry script and the `--model` class, and both are
run-command arguments. Refinement iters is pinned per wrapper *class* rather
than read from the environment, because `run_docker_inference.sh` passes no
`-e` flags and anything env-dependent is a silent failure waiting to happen.

| track | `bash run_docker_inference.sh … <track>` | script + model | iters | p95 |
|---|---|---|---|---|
| 2D accuracy | `2d` | `mono.py` · `…student_lt.StudentLTWrapperAccurate` | 4 | ~94 ms |
| 2D + latency | `2d-latency` | `mono.py` · `…student_lt.StudentLTWrapperFast` | 1 | ~52 ms |
| 3D | `3d` | `stereo.py` · `…student_stereo.StudentStereoWrapper` | 4 | — |

The wrapper stamps the iters it actually used into `modeltype`
(`StudentLiteTracker_iters4`), so a mis-wired `--model` is visible in the log
rather than silently scoring the wrong operating point.

Checkpoint and the evidence for it: see `CHECKPOINT.txt`.

## Layout

```
weights/student.pth                     baked into every image at /workspace/weights/
code/models/mono/student_lt.py          2D wrapper (unchanged from the eval)
code/models/stereo/student_stereo.py    3D wrapper: dual tracking + frame-0 matcher
docker/Dockerfile                       one image, all three tracks
build_all.sh · test_submission.sh · export_all.sh
run_docker_inference.sh                 our copy; takes a track argument
```

## Use

```bash
bash build_all.sh          # base image, then the submission image on top
bash test_submission.sh    # runs all three tracks against ../val — DO THIS FIRST
bash export_all.sh         # one .tar for submission

# behind a proxy: build_all.sh picks up HTTP_PROXY / HTTPS_PROXY / NO_PROXY from
# your environment (either case) and passes them as Docker predefined build args,
# so they are NOT baked into the image — the eval environment has no network.
```

`test_submission.sh` checks the things that actually go wrong: the matcher
self-test on real frames, that each track produces `preds.json`, that
`latency_ms` is present (the efficiency metric needs it), and that stereo `Z` is
a plausible surgical working distance in **metres** — a units slip there would
be scored as a total failure rather than an error.

## How the 3D track works

Dual tracking, no MFT. The same checkpoint tracks the query points in the left
view and their stereo correspondents in the right view; `disparity = u_L - u_R`
back-projects to metres via `Z = fx·baseline/d`. This is the pipeline our 3D
numbers measured, and the design that won STIR 2025.

**The one part the offline numbers did not exercise.** The eval took right-view
start points from STIRLoader's `getsegsstereo`, which contour-matches the IR
tattoo segmentation in *both* views — annotation data that exists only in the
STIR release. The 2026 harness gives both images every frame but `queries.json`
carries `left_x`/`left_y` only. So `match_right()` recovers the right-view start
points from the images by NCC along the epipolar row (valid because the 2026
pair is rectified to a common principal point: `cx_left == cx_right` exactly).

**This cost has now been measured**, by re-running the sweep with
`right_queries=match` so the eval used this exact function instead of
`getsegsstereo`:

| right-view start points | 3D acc_avg | mean / median |
|---|---|---|
| `getsegsstereo` (annotation, not available at submission) | 0.7383 | 6.16 / 3.01 mm |
| `match_right()` (**what this image ships**) | **0.6863** | 31.09 / 3.18 mm |
| _CONTROL — point never moved_ | 0.6115 | 8.99 / 5.09 mm |

So **0.6863 is the number to expect**, not 0.7383. The matcher is good in the
median (frame-0 |dx| median 0.88 px, 98.2% pass the NCC gate) but has a tail:
12.8% of points miss by >3 px, and that tail is what moves the mean from 6 mm to
31 mm.

Two notes on why it was left as is. Raising `MIN_NCC` does **not** fix this:
`tracking_accuracy_per_threshold` in the metrics scores position regardless of
predicted visibility, so gating a bad point avoids an AJ false positive but
still loses it in ATA. And `student_stereo.py` carries two optional refinements
(L-R consistency `STUDENT_STEREO_LR_TOL`, disparity repair from confident
neighbours `STUDENT_STEREO_REPAIR_NCC`) which are **off by default** because
they are unmeasured — enable them and re-run the sweep before trusting either.

## Notes for whoever runs this

- Weights are baked in at a fixed path: `run_docker_inference.sh` passes no `-e`
  flags and there is no network at eval time. `student_lt.py` fails loudly if the
  file is missing rather than silently scoring garbage.
- `STUDENT_LT_ITERS` is set via `ENV` in each Dockerfile, so it survives a
  `docker run` with no environment flags.
- The stereo image still uses `--group lt`; the matcher is pure numpy, so it
  needs no dependency the mono image lacks.
