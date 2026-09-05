# assets/

GIFs used by the top-level README. **They are not generated yet** — build them
with:

```bash
bash tools/make_gifs.sh
```

on a machine with `ffmpeg`, the project venv and the STIR dataset. It writes
`pseudo_labels.gif` and `student_tracking.gif` here, sized to stay a couple of
MB each.

| file | what it shows |
|---|---|
| `pseudo_labels.gif` | the verifier's fused trajectories over a real clip — the supervision the student is trained on |
| `student_tracking.gif` | the trained student under the streaming LiteTracker runtime — the same forward pass the submitted container runs — tracking a **64 px grid, 320 points**, which is what the 2026 harness actually hands the container. `GRID=96` for a less busy picture, `--with-tattoos` to also draw the ground-truth circles |

The pair is the point of the method: the left GIF is *all* the annotation STIR
provides, the right is a dense grid tracked by a model trained on nothing else.

## Keeping them small

Surgical video is high-entropy and GIF compresses it badly — a first pass at
480 px with a full 256-colour palette came out at 3.2 MB for 26 frames
(126 KB/frame). GitHub renders a GIF over **5 MB** as a still image, and two
heavy GIFs would outweigh the rest of the repository.

Defaults are now `WIDTH=420 MAX_COLORS=128 FPS=8 SECONDS_LEN=6`. Cutting the
palette is the cheapest lever and costs almost nothing visually on tissue; the
script warns if a file lands over 2.5 MB and tells you what to turn down:

```bash
MAX_COLORS=64 bash tools/make_gifs.sh     # usually halves it again
WIDTH=320 SECONDS_LEN=4 bash tools/make_gifs.sh
```
