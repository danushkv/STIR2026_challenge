# patches/

Small compatibility patches against upstream projects.

| file | what |
|---|---|
| `stirloader-streaming-skip.patch` | apply the frame stride while streaming from ffmpeg, instead of decoding the whole clip and then slicing |
| `stirloader-base-commit.txt` | the commit it applies to: `7e7f87c4f2c2aac525207f499ff18221f7936822` |
| `track-on-cotracker-compat.patch` | optional checkpoint loading and short-clip support against track_on commit `7e838e84ae6accf02294752da9e9fe25ec5835c4` |

```bash
git clone https://github.com/athaddius/STIRLoader && cd STIRLoader
git checkout 7e7f87c
git apply /path/to/this/repo/patches/stirloader-streaming-skip.patch
pip install --no-deps -e .
```

The track_on patch is required by `src/teachers.py`:

```bash
git clone https://github.com/gorkaydemir/track_on && cd track_on
git checkout 7e838e84ae6accf02294752da9e9fe25ec5835c4
git apply /path/to/this/repo/patches/track-on-cotracker-compat.patch
```

It deliberately omits the machine-local cache-directory edits present in the
original checkout; cache locations do not affect predictions and belong in
environment configuration.

## Is it needed?

**For correctness, no. For memory, yes.**

The frames you get are identical either way. Upstream keeps `f[::SKIP]` plus the
final frame when `len(f) % SKIP != 1`; the patched read loop produces exactly
that set — checked for every clip length 1–600 against `SKIP` ∈ {1,2,3,5,7},
with no disagreement.

What differs is peak memory. The longest clip in the evaluation set is 2074
frames of 1280×1024×3:

| | per view | stereo pair |
|---|---|---|
| upstream | 8.2 GB | 16.3 GB |
| patched, `skip=5` | 1.6 GB | 3.3 GB |

Every result in this repository was produced with the patch applied. If your
machine can hold 16 GB of decoded video per clip, stock STIRLoader will give you
the same numbers.

## Why a patch and not a fork

It is nine lines of upstream, and pinning a fork would make it harder to see
that we changed nothing about *which* frames are used — which is the thing that
would silently invalidate a comparison.
