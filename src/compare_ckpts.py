"""Merge every shard from the eval_sweep.py array job, rank the checkpoints
with a PAIRED clip-level bootstrap, and write the results to disk.

WHY NOT JUST TAKE THE ARGMAX (the mistake stats.txt invites):

  The eval is ~230 points over ~32 clips. Points inside one clip share a scene, a
  tattoo pattern and a motion, so they are nowhere near independent and the naive
  binomial SE understates the real uncertainty badly. This script MEASURES that
  inflation (the design effect, by one-way ANOVA) instead of assuming it, and
  resamples whole CLIPS -- the unit that is actually independent.

  It then compares checkpoints PAIRWISE ON THE SAME CLIPS, reusing one fixed
  resample matrix so every run sees identical resamples. Checkpoints from the
  same lineage fail on the same hard clips, so their errors are strongly
  correlated and the variance of the DIFFERENCE is far smaller than that of
  either mean. Overlapping individual CIs therefore do NOT mean
  "indistinguishable" -- the paired interval is what decides it. Reading two
  independent CIs instead is the standard way to talk yourself out of a real
  effect, or into a fake one.

WRITES (to --out-dir):
  summary.json   every metric for every (checkpoint, iters), machine-readable
  ranking.md     the human table: scores, CIs, paired deltas vs the leader
  per_point.npz  merged per-point errors, for any further analysis

    python compare_ckpts.py <sweep_dir> [--n-boot 10000] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def delta_avg(d, thresholds):
    """STIR's headline number: accuracy averaged over the threshold ladder."""
    return float(np.mean([np.mean(d <= t) for t in thresholds]))


def boot_scores(dists, by_clip, idx_mat, thresholds):
    """delta_avg over each clip-resample. idx_mat is shared across runs, which
    is exactly what makes the pairwise deltas paired rather than independent."""
    out = np.empty(len(idx_mat))
    for b, row in enumerate(idx_mat):
        out[b] = delta_avg(dists[np.concatenate([by_clip[c] for c in row])], thresholds)
    return out


def design_effect(dists, owner, n_clips, t):
    """deff = 1 + (m_bar-1)*ICC by one-way ANOVA on the within-threshold
    indicator. deff=3 means real error bars are sqrt(3)~1.7x wider than a
    binomial calculator claims."""
    y = (dists <= t).astype(float)
    groups = [y[owner == c] for c in range(n_clips) if np.any(owner == c)]
    sizes = np.array([len(g) for g in groups], float)
    if len(groups) < 2 or y.std() == 0:
        return 1.0, 0.0, float(sizes.mean() if len(sizes) else 0)
    k, N, gm = len(groups), sizes.sum(), y.mean()
    ms_b = sum(len(g) * (g.mean() - gm) ** 2 for g in groups) / (k - 1)
    ms_w = sum(((g - g.mean()) ** 2).sum() for g in groups) / max(N - k, 1)
    m0 = (N - (sizes ** 2).sum() / N) / (k - 1)
    den = ms_b + (m0 - 1) * ms_w
    icc = float(np.clip((ms_b - ms_w) / den, 0, 1)) if den > 0 else 0.0
    return 1.0 + (sizes.mean() - 1.0) * icc, icc, float(sizes.mean())


def load_shards(sweep_dir):
    """Load every <tag>.npz, asserting the clip alignment that pairing needs."""
    paths = sorted(glob.glob(os.path.join(sweep_dir, "*.npz")))
    paths = [p for p in paths if not p.endswith("per_point.npz")]
    if not paths:
        raise SystemExit(f"no shard .npz found in {sweep_dir}")
    shards, ref = [], None
    for p in paths:
        z = np.load(p, allow_pickle=False)
        key = (list(z["clip_ids_2d"]), list(z["clip_ids_3d"]))
        if ref is None:
            ref = key
        elif key != ref:
            raise SystemExit(
                f"{p} scored a different clip set than {paths[0]} -- pairing would "
                f"be invalid. Re-run the shards against the same val_stir_root.")
        shards.append((str(z["tag"]), z))
        print(f"  loaded {str(z['tag']):<24} {os.path.basename(p)}")
    return shards


def analyse(shards, mode, unit, n_boot, rng, out):
    z0 = shards[0][1]
    owner, ids = z0[f"owner_{mode}"], z0[f"clip_ids_{mode}"]
    nframes, thresholds = z0[f"clip_nframes_{mode}"], z0[f"thresholds_{mode}"]
    n_clips = len(ids)
    runs = {}
    for tag, z in shards:
        for it in z["iters_list"]:
            k = f"d{mode}__i{it}"
            if k in z.files:
                runs[f"{tag}|i{it}"] = z[k]
    if not runs or n_clips == 0:
        return []

    lines = [f"\n## {mode.upper()} — {n_clips} clips, {len(owner)} points, "
             f"{len(runs)} runs, {n_boot} clip-bootstrap resamples\n"]

    t_mid = float(thresholds[1])
    ref = next(iter(runs.values()))
    deff, icc, m_bar = design_effect(ref, owner, n_clips, t_mid)
    p = float(np.mean(ref <= t_mid))
    se = np.sqrt(p * (1 - p) / len(ref))
    lines += [f"Measured noise floor at the {t_mid:g}{unit} threshold:\n",
              f"- points/clip {m_bar:.1f}, intra-clip correlation {icc:.3f}, "
              f"design effect {deff:.2f}",
              f"- naive binomial SE {100*se:.2f}pp → clustered SE {100*se*np.sqrt(deff):.2f}pp",
              f"- **unpaired** comparisons need ~{100*2*1.96*se*np.sqrt(deff):.1f}pp to mean "
              f"anything; the paired deltas below need far less.\n"]

    by_clip = [np.flatnonzero(owner == c) for c in range(n_clips)]
    idx_mat = rng.integers(0, n_clips, size=(n_boot, n_clips))
    scores = {r: delta_avg(d, thresholds) for r, d in runs.items()}
    boots = {r: boot_scores(d, by_clip, idx_mat, thresholds) for r, d in runs.items()}
    order = sorted(runs, key=lambda r: -scores[r])

    lines += [f"| run | {'delta_avg' if mode=='2d' else 'acc_avg'} | 95% CI | "
              f"mean {unit} | median {unit} |", "|---|---|---|---|---|"]
    for r in order:
        lo, hi = np.percentile(boots[r], [2.5, 97.5])
        d = runs[r]
        lines.append(f"| {r} | {scores[r]:.4f} | [{lo:.4f}, {hi:.4f}] | "
                     f"{d.mean():.2f} | {np.median(d):.2f} |")
        out["runs"].setdefault(r, {})[mode] = {
            "score": scores[r], "ci95": [float(lo), float(hi)],
            "mean": float(d.mean()), "median": float(np.median(d))}
    if mode == "3d" and "control_3d" in z0.files:
        c = z0["control_3d"]
        lines.append(f"| _CONTROL (never moved)_ | {delta_avg(c, thresholds):.4f} | — | "
                     f"{c.mean():.2f} | {np.median(c):.2f} |")

    lead = order[0]
    lines += [f"\n**Paired vs leader ({lead})** — same clips, same resamples:\n",
              "| run | delta | 95% CI | P(beats leader) | separable? |",
              "|---|---|---|---|---|"]
    for r in order[1:]:
        diff = boots[r] - boots[lead]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        sep = "no — tie" if lo <= 0 <= hi else "yes"
        lines.append(f"| {r} | {scores[r]-scores[lead]:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                     f"{np.mean(diff > 0):.3f} | {sep} |")
        out["runs"][r].setdefault(mode, {})["paired_vs_leader"] = {
            "leader": lead, "delta": float(scores[r] - scores[lead]),
            "ci95": [float(lo), float(hi)], "p_beats": float(np.mean(diff > 0)),
            "separable": bool(lo > 0 or hi < 0)}
    lines.append("\nA CI straddling 0 means that run is **not separable** from the "
                 "leader. Among ties, prefer the faster / cleaner-trained model.\n")

    d = runs[lead]
    per_clip = np.array([np.mean(d[owner == c]) for c in range(n_clips)])
    if n_clips > 2 and nframes.std() > 0:
        slope = np.polyfit(nframes, per_clip, 1)[0]
        r_ = np.corrcoef(nframes, per_clip)[0, 1]
        lines.append(f"Endpoint error vs clip length ({lead}): slope {slope:+.4f} "
                     f"{unit}/frame, r={r_:+.2f} over {nframes.min()}–{nframes.max()} "
                     f"frames. A clearly positive slope means drift accumulates and "
                     f"per-frame ATA will be worse than these endpoint numbers.\n")
        out["drift_vs_length"] = out.get("drift_vs_length", {})
        out["drift_vs_length"][mode] = {"slope": float(slope), "r": float(r_)}
    return lines



def _jsonable(o):
    """numpy scalars/arrays leak into the summary from several places; json
    refuses them silently until the very last line of a long job."""
    if isinstance(o, (np.bool_, np.integer)):
        return o.item()
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    out_dir = a.out_dir or os.path.join(a.sweep_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    print(f"merging shards from {a.sweep_dir}")
    shards = load_shards(a.sweep_dir)
    rng = np.random.default_rng(a.seed)
    out = {"n_boot": a.n_boot, "seed": a.seed, "runs": {}}

    md = ["# STIR checkpoint ranking — LiteTracker streaming runtime\n",
          f"Merged {len(shards)} checkpoint shard(s) from `{a.sweep_dir}`. "
          f"Paired clip-level bootstrap, {a.n_boot} resamples.\n"]
    md += analyse(shards, "2d", "px", a.n_boot, rng, out)
    md += analyse(shards, "3d", "mm", a.n_boot, rng, out)

    # drift + visibility from the per-shard json summaries
    md += ["\n## Drift and visibility (measurable without 2026 GT)\n",
           "| run | cycle mean px | cycle/endpoint | invisible % | flicker/100f | "
           "AJ pred-vis | AJ forced-vis | gain if forced |",
           "|---|---|---|---|---|---|---|---|"]
    for tag, _ in shards:
        jp = os.path.join(a.sweep_dir, f"{tag}.json")
        if not os.path.exists(jp):
            continue
        s = json.load(open(jp))
        for it, m in sorted(s["iters"].items(), key=lambda kv: int(kv[0])):
            if not m:
                continue
            r = f"{tag}|i{it}"
            md.append(
                f"| {r} | {m.get('cycle_mean_px', float('nan')):.2f} | "
                f"{m.get('cycle_over_endpoint', float('nan')):.2f}x | "
                f"{100*m.get('invisible_rate', float('nan')):.2f} | "
                f"{m.get('flicker_per_100f', float('nan')):.2f} | "
                f"{m.get('aj_endpoint_pred_vis', float('nan')):.4f} | "
                f"{m.get('aj_endpoint_forced_vis', float('nan')):.4f} | "
                f"{m.get('aj_gain_if_forced', float('nan')):+.4f} |")
            out["runs"].setdefault(r, {})["aux"] = m
    md.append("\n`gain if forced` > 0 means the visibility head is COSTING you AJ: "
              "setting `visibs=True` in the wrapper is legal and strictly better. "
              "An invisible rate under ~1% means the head is inert (AJ ≈ ATA), which "
              "is the safe failure mode.\n")
    md.append("Lowest `cycle mean px` at equal endpoint error = the model that "
              "actually tracks, rather than one the nearest-neighbour match "
              "rescues. Read cycle error comparatively, not as an error budget.\n")

    with open(os.path.join(out_dir, "ranking.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=_jsonable)
    merged = {}
    for tag, z in shards:
        for k in z.files:
            if k.startswith(("d2d__", "d3d__", "cyc__", "visend__")):
                merged[f"{tag}__{k}"] = z[k]
            elif tag == shards[0][0]:
                merged[k] = z[k]
    np.savez_compressed(os.path.join(out_dir, "per_point.npz"), **merged)

    print("\n".join(md))
    print(f"\nwrote {out_dir}/ranking.md, summary.json, per_point.npz")


if __name__ == "__main__":
    main()
