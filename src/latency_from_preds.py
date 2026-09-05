"""p95 frame latency straight from preds.json -- no ground truth needed.

WHY: stir-challenge-2026-metrics/run.py computes p95 latency alongside AJ/ATA/OA,
but it dies before reporting anything if the dataset has no annotations.json
(`ValueError: need at least one array to concatenate`), which is the case for a
validation set that ships inputs only. Latency, however, is recorded by mono.py
into preds.json itself and needs no GT -- so it's measurable today.

The computation mirrors run.py exactly: skip frames with index <= WARMUP_FRAMES
(10), pool the remaining per-frame latencies across all sequences, take the 95th
percentile. Same number run.py's `p95_latency_ms` would report.

Use it to pick the accuracy/latency operating point for the efficiency benchmark
(e.g. STUDENT_LT_ITERS=1 vs 2 vs 4) while accuracy selection waits on GT.

    # one run
    python latency_from_preds.py results_val/mono/StudentLTWrapper/preds.json

    # compare several (e.g. an iters sweep), sorted fastest-first
    python latency_from_preds.py results_iters*/mono/*/preds.json

    # per-sequence breakdown too
    python latency_from_preds.py --per-sequence <preds.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

WARMUP_FRAMES = 10  # must match stir-challenge-2026-metrics/run.py


def latencies_from_preds(preds_path: Path):
    """(pooled_latencies [ms], {seq_id: latencies}) excluding warmup frames."""
    with open(preds_path) as fh:
        all_tracks = json.load(fh)

    per_seq, pooled = {}, []
    for seq_id, seq_tracks in all_tracks.items():
        lat = [
            v["latency_ms"]
            for k, v in sorted((int(k), v) for k, v in seq_tracks.items())
            if int(k) > WARMUP_FRAMES
        ]
        per_seq[seq_id] = lat
        pooled.extend(lat)
    return pooled, per_seq


def summarize(lat) -> dict:
    a = np.asarray(lat, dtype=np.float64)
    return {
        "n_frames": int(a.size),
        "p95_latency_ms": float(np.percentile(a, 95)),
        "p50_latency_ms": float(np.percentile(a, 50)),
        "mean_latency_ms": float(a.mean()),
        "max_latency_ms": float(a.max()),
        "fps_at_p50": float(1000.0 / np.percentile(a, 50)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("preds", nargs="+", help="one or more preds.json paths")
    p.add_argument("--per-sequence", action="store_true",
                   help="also print a per-sequence breakdown")
    args = p.parse_args()

    rows = []
    for path in args.preds:
        path = Path(path)
        pooled, per_seq = latencies_from_preds(path)
        if not pooled:
            print(f"[skip] {path}: no frames beyond the {WARMUP_FRAMES}-frame warmup")
            continue
        s = summarize(pooled)
        # label by the model dir name (results/mono/<ModelName>/preds.json), and
        # include the results-root so an iters sweep across output dirs stays legible
        label = f"{path.parent.parent.parent.name}/{path.parent.name}"
        rows.append((label, s, per_seq))

    if not rows:
        raise SystemExit("no usable preds.json files")

    rows.sort(key=lambda r: r[1]["p95_latency_ms"])

    w = max(len(r[0]) for r in rows)
    print(f"\n{'model':<{w}}  {'p95_ms':>8} {'p50_ms':>8} {'mean_ms':>8} "
          f"{'max_ms':>8} {'fps@p50':>8} {'frames':>7}")
    print("-" * (w + 56))
    for label, s, _ in rows:
        print(f"{label:<{w}}  {s['p95_latency_ms']:>8.2f} {s['p50_latency_ms']:>8.2f} "
              f"{s['mean_latency_ms']:>8.2f} {s['max_latency_ms']:>8.2f} "
              f"{s['fps_at_p50']:>8.1f} {s['n_frames']:>7d}")

    if args.per_sequence:
        for label, _, per_seq in rows:
            print(f"\n--- {label} ---")
            for seq_id, lat in sorted(per_seq.items()):
                if not lat:
                    print(f"  [{seq_id}] no frames past warmup")
                    continue
                s = summarize(lat)
                print(f"  [{seq_id}] p95 {s['p95_latency_ms']:7.2f} ms  "
                      f"p50 {s['p50_latency_ms']:7.2f} ms  ({s['n_frames']} frames)")

    print(f"\nNote: excludes the first {WARMUP_FRAMES} frames per sequence, matching "
          f"run.py's WARMUP_FRAMES -- so these match its p95_latency_ms.")


if __name__ == "__main__":
    main()
