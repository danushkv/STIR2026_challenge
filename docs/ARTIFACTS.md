# Released artifacts

Large artifacts live on Hugging Face rather than in Git. Replace the two
placeholder repository IDs below once the uploads finish; these are the only
release links that still need to be filled in.

| artifact | Hugging Face repository | expected contents |
|---|---|---|
| trained student checkpoint | [pre-trained chkpt](https://huggingface.co/nct-tso/VG-Track/tree/main) | `student.pth` |
| STIROrig raw teacher tracks | [`nct-tso/STIR_pseudo_tracks`](https://huggingface.co/datasets/nct-tso/STIR_pseudo_tracks) | one directory per teacher plus the dataset card |

The released `student.pth` SHA-256 is
`3b964f18793bab959d1537aa58b0411bb8454e2df5f30bf6d66039a215c8455c`.
Verify it after downloading:

```bash
sha256sum artifacts/student.pth
```

With the Hugging Face CLI installed, download the artifacts without cloning
their Git repositories:

```bash
hf download nct-tso/VG-Track/ student.pth \
  --local-dir artifacts

hf download nct-tso/STIR_pseudo_tracks \
  --repo-type dataset --local-dir data/STIROrig_tracks
```

The raw-track download should have this layout:

```text
data/STIROrig_tracks/
  alltracker/<patient>/<side>__<seq>__alltracker.npz
  cotracker3/<patient>/<side>__<seq>__cotracker3.npz
  locotrack/<patient>/<side>__<seq>__locotrack.npz
  mft/<patient>/<side>__<seq>__mft.npz
  trackon2/<patient>/<side>__<seq>__trackon2.npz
  trackon_r/<patient>/<side>__<seq>__trackon_r.npz
```

Each raw-track `.npz` contains `fwd_coords`, `fwd_vis`, `fwd_conf`, and `name`,
plus `bwd_coords`, `bwd_vis`, and `bwd_conf` when the cycle pass completed.
Coordinates use native-image `(x, y)` pixels. The temporal stride is 5.

