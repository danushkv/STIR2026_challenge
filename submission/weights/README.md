# submission/weights/

`student.pth` goes here. It is **not** in git: 97 MB, right at GitHub's 100 MB
hard limit, and it is a build artifact of `scripts/train_agg.sh`.

It is epoch 44 of the `agg` run — verified byte-identical (md5
`4709eaefce5d4d45c7222c8617d0890f`) to the file the submitted images were built
from:

```bash
cp /mnt/cluster/datasets/STIRprocessed/model_runs/fast_run_with2024_agg/student_e44.pth \
   submission/weights/student.pth
```

`docker/Dockerfile` copies it to `/workspace/weights/student.pth`, which is
where `student_lt.py` looks. The weights must be baked into the image:
`run_docker_inference.sh` passes no `-e` flags and there is no network at eval
time. `student_lt.py` fails loudly if the file is missing rather than silently
scoring garbage.

Provenance and the evidence for choosing this checkpoint: `../CHECKPOINT.txt`.
