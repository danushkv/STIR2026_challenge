"""Diagnose the pseudo-label ceiling: score every teacher's raw tracks AND the
verifier's fused labels against the endpoint ground truth, per clip and pooled.

WHY THIS EXISTS: a distilled student cannot out-track its own supervision. If
the fused pseudo-labels are not better than the best single teacher, the
verifier -- not the student, not the loss -- is the ceiling, and student-side
work is polishing the wrong stage. This script settles that question with the
GT you already have (the IR-tattoo endpoints), no annotation needed.

Reads the same artifacts the pipeline already produces (same conventions/helpers
as run_phase2.py, so this stays in sync with it automatically):
  --raw-tracks-root    collect_tracks.py output, nested:
                        <root>/<teacher>/<patient>/<side__seq>__<teacher>.npz
  --pseudo-labels-dir  run_phase2.py output: <dir>/<patient>/<side__seq>.npz  [optional]
  --stir-root          STIR dataset root -- endpoints are pulled live via
                        STIRLoader.getendcenters(), same as run_phase2.py's
                        _build_endpoints_by_clip(). No separate endpoints file.

Reports, per teacher and for the fused labels:
  * endpoint L2 error (mean / median / p90), in native pixels
  * delta@{2,4,8,16,32} px on the endpoint frame, and their average
  * per-teacher win rate: fraction of points where that teacher's endpoint
    is the closest of all teachers (reveals correlated-majority effects)

Interpretation guide (the three outcomes that matter):
  1. fused >= best teacher on delta_avg  -> verifier is doing its job; the
     margin lives in student training / data volume / more rounds.
  2. fused < best teacher, and the best teacher's win rate is high while its
     selection rate in the labels (source stats) is low -> the agreement
     median is outvoting your best teacher with correlated weaker ones. Fix:
     teacher_priors for the minority-but-accurate teacher, lower w_agreement,
     raise w_endpoint, and/or drop redundant same-family teachers.
  3. all teachers cluster tightly -> the ensemble has no diversity worth
     verifying; gains must come from better/domain-adapted teachers instead.

NOTE: endpoint error is a proxy -- it checks trajectory ENDS, not middles. A
teacher can be endpoint-accurate but sloppy mid-clip. Cycle-consistency stats
mid-clip complement it; but for STIR's actual metric (accuracy at the queried
endpoint) this proxy is exactly the quantity being scored.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_phase2 import _build_endpoints_by_clip, _discover_clip_ids
from trajectories import load_track_result_pair

DELTAS = (2, 4, 8, 16, 32)


def _raw_track_path(raw_tracks_root, teacher: str, clip_id: str) -> Path:
    """<raw_tracks_root>/<teacher>/<patient>/<side__seq>__<teacher>.npz -- the
    SAME nested layout collect_tracks.py writes and run_phase2.py reads (see
    run_phase2._copy_through)."""
    patient, seq_part = clip_id.split("__", 1)
    return Path(raw_tracks_root) / teacher / patient / f"{seq_part}__{teacher}.npz"


def _pseudo_label_path(pseudo_labels_dir, clip_id: str) -> Path:
    """<pseudo_labels_dir>/<patient>/<side__seq>.npz -- pseudo_label.save_pseudo_label's
    layout (no teacher suffix; fused labels are one file per clip)."""
    patient, seq_part = clip_id.split("__", 1)
    return Path(pseudo_labels_dir) / patient / f"{seq_part}.npz"


def nn_endpoint_error(final_coords: np.ndarray, end_centers: np.ndarray) -> np.ndarray:
    """[N,2] final positions vs [M,2] UNORDERED end centers -> [N] distances to
    the nearest center. STIR's GT has no point IDs (start/end centers are
    independently detected), and the challenge eval NN-matches the same way, so
    this is the metric-aligned error -- not index-aligned subtraction.
    Caveat it inherits from the metric: a point that lands on the WRONG tattoo
    scores as accurate. Interpret near-tattoo-spacing errors with that in mind.
    """
    D = np.linalg.norm(final_coords[:, None, :] - end_centers[None, :, :], axis=-1)
    return D.min(axis=1)


# --------------------------------------------------------------------------- #
# Headroom test: does the LABEL-FREE cheap score predict true endpoint error?
# --------------------------------------------------------------------------- #
def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties averaged -- enough for AUROC/Spearman here."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ties
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def rank_auroc(scores: np.ndarray, is_good: np.ndarray) -> float:
    """AUROC via Mann-Whitney U: P(score of a good track > score of a bad one)."""
    n_pos, n_neg = int(is_good.sum()), int((~is_good).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _ranks(scores)
    u = r[is_good].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(_ranks(x), _ranks(y))[0, 1])


def headroom_test(raw_tracks_root, endpoints_by_clip, teachers, clip_ids,
                  good_px: float) -> None:
    """For every (teacher, point, labeled clip): the cheap verifier's LABEL-FREE
    score at the final frame (cycle + agreement + native confidence -- endpoint
    anchoring deliberately EXCLUDED, else the test is circular) vs the track's
    true NN endpoint error. High AUROC -> the cheap signals already predict
    failure well and a learned verifier has little headroom. Low AUROC -> real
    reliability information is invisible to the geometric signals, which is the
    learned verifier's opportunity. This is also exactly the deployment
    condition on unlabeled clips (no GT -> no anchoring), where a learned
    verifier would matter most.
    """
    from config import VerifierConfig
    from verifier import CheapVerifier

    cfg = VerifierConfig(use_endpoint_anchoring=False)
    scorer = CheapVerifier(cfg)

    per_teacher = defaultdict(lambda: ([], []))   # name -> (scores, errors)
    pooled_scores, pooled_errs = [], []
    skipped_cycles = set()

    for clip_id in clip_ids:
        gt = endpoints_by_clip.get(clip_id)
        if gt is None:
            continue

        tracks, cycles, names = [], [], []
        for t in teachers:
            path = _raw_track_path(raw_tracks_root, t, clip_id)
            if not path.exists():
                continue
            fwd, bwd = load_track_result_pair(path)
            tracks.append(fwd)
            cycles.append(bwd)
            names.append(t)
        if not tracks:
            continue
        if any(c is None for c in cycles):
            skipped_cycles.update(n for n, c in zip(names, cycles) if c is None)
            cycles = None  # consistent with pseudo_label: all-or-nothing

        score = scorer._per_teacher_frame_score(tracks, cycles, endpoints=None)  # [K,N,T]
        for k, (trk, name) in enumerate(zip(tracks, names)):
            s = score[k, :, -1]                        # label-free score, final frame
            e = nn_endpoint_error(trk.coords[:, -1, :], gt)
            per_teacher[name][0].append(s)
            per_teacher[name][1].append(e)
            pooled_scores.append(s)
            pooled_errs.append(e)

    if not pooled_scores:
        print("headroom test: no usable (tracks + endpoints) pairs found")
        return

    print(f"\nHeadroom test: label-free cheap score vs true endpoint error "
          f"(good = err < {good_px:g} px; anchoring excluded from score)\n" + "-" * 100)
    for name, (ss, es) in sorted(per_teacher.items()):
        s, e = np.concatenate(ss), np.concatenate(es)
        auc = rank_auroc(s, e < good_px)
        rho = spearman(s, -e)
        print(f"{name:>14s} | n={len(s):5d} | AUROC {auc:.3f} | Spearman(score, -err) {rho:+.3f} "
              f"| good frac {(e < good_px).mean():.0%}")
    s, e = np.concatenate(pooled_scores), np.concatenate(pooled_errs)
    auc = rank_auroc(s, e < good_px)
    rho = spearman(s, -e)
    print("-" * 100)
    print(f"{'POOLED':>14s} | n={len(s):5d} | AUROC {auc:.3f} | Spearman {rho:+.3f}")
    if skipped_cycles:
        print(f"note: cycle signal disabled (missing backward pass for: "
              f"{', '.join(sorted(skipped_cycles))}) -- score used agreement + confidence only")
    print("\nReading it: AUROC ~0.85+ -> cheap signals already predict failure; "
          "learned verifier has little headroom.\n            AUROC ~0.70 or below "
          "-> real headroom for a learned verifier (it can see failure evidence "
          "the geometric signals can't).")


def delta_stats(err: np.ndarray) -> dict:
    out = {f"d{t}": float((err < t).mean()) for t in DELTAS}
    out["d_avg"] = float(np.mean([out[f"d{t}"] for t in DELTAS]))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-tracks-root", required=True,
                   help="parent of the per-teacher folders written by collect_tracks.py "
                        "(same argument as run_phase2.py --raw-tracks-root)")
    p.add_argument("--stir-root", required=True,
                   help="STIR dataset root, for endpoints (same as run_phase2.py --stir-root)")
    p.add_argument("--pseudo-labels-dir", default=None,
                   help="optional: also score the fused labels from run_phase2.py")
    p.add_argument("--headroom-test", action="store_true",
                   help="also run the learned-verifier headroom test: AUROC of the "
                        "label-free cheap score (cycle+agreement+confidence, NO "
                        "endpoint anchoring) predicting true endpoint error")
    p.add_argument("--good-px", type=float, default=8.0,
                   help="threshold (px) defining a 'good' endpoint for the AUROC test")
    args = p.parse_args()

    raw_root = Path(args.raw_tracks_root)

    # teachers = per-teacher subfolders (nested layout, no filename parsing needed);
    # clip_ids = union of what each teacher actually wrote, via run_phase2's own helper
    teachers = sorted(p.name for p in raw_root.iterdir() if p.is_dir())
    if not teachers:
        raise SystemExit(f"no per-teacher folders found in {raw_root}")
    clip_ids = sorted(set().union(
        *(_discover_clip_ids(str(raw_root), t) for t in teachers)))
    if not clip_ids:
        raise SystemExit(f"no raw tracks found under {raw_root}/<teacher>/<patient>/...")

    endpoints_by_clip = _build_endpoints_by_clip(args.stir_root, set(clip_ids))
    print(f"{len(clip_ids)} clip(s) across {len(teachers)} teacher(s); "
          f"{len(endpoints_by_clip)} have endpoint ground truth")

    per_teacher_err = defaultdict(list)     # teacher -> [errors over all points/clips]
    fused_err, fused_err_unsnapped, n_snapped = [], [], 0
    win_counts = defaultdict(int)
    total_points = 0
    fused_source_counts = defaultdict(int)  # which teacher the verifier picked (select mode)

    for clip_id in clip_ids:
        gt = endpoints_by_clip.get(clip_id)
        if gt is None:
            print(f"[skip] {clip_id}: no endpoint ground truth")
            continue

        errs_this_clip = {}
        for t in teachers:
            path = _raw_track_path(raw_root, t, clip_id)
            if not path.exists():
                continue
            fwd, _ = load_track_result_pair(path)
            e = nn_endpoint_error(fwd.coords[:, -1, :], gt)  # [N], NN vs unordered centers
            errs_this_clip[t] = e
            per_teacher_err[t].append(e)

        if errs_this_clip:
            names = list(errs_this_clip)
            E = np.stack([errs_this_clip[t] for t in names], axis=0)  # [K,N]
            winners = np.argmin(E, axis=0)
            for w in winners:
                win_counts[names[w]] += 1
            total_points += E.shape[1]

        if args.pseudo_labels_dir:
            pl_path = _pseudo_label_path(args.pseudo_labels_dir, clip_id)
            if pl_path.exists():
                pl = np.load(pl_path)
                e_all = nn_endpoint_error(pl["coords"][:, -1, :], gt)
                fused_err.append(e_all)
                if "source" in pl:
                    snapped = pl["source"][:, -1] == -2      # GT written onto label
                    if (~snapped).any():
                        fused_err_unsnapped.append(e_all[~snapped])
                    n_snapped += int(snapped.sum())
                    src_last_ignored = pl["source"][:, :-1]  # exclude snapped GT frame
                    for s in np.unique(src_last_ignored):
                        fused_source_counts[int(s)] += int((src_last_ignored == s).sum())

    # ---- report --------------------------------------------------------- #
    def row(name, errs):
        e = np.concatenate(errs)
        ds = delta_stats(e)
        deltas = " ".join(f"d{t}={ds[f'd{t}']:.3f}" for t in DELTAS)
        return (f"{name:>14s} | mean {e.mean():7.2f}  med {np.median(e):7.2f}  "
                f"p90 {np.percentile(e, 90):7.2f} px | {deltas} | d_avg={ds['d_avg']:.3f}")

    print(f"\nEndpoint-frame accuracy vs IR-tattoo GT "
          f"({total_points} points, {len(clip_ids)} clips)\n" + "-" * 100)
    ranked = sorted(per_teacher_err, key=lambda t: -delta_stats(np.concatenate(per_teacher_err[t]))["d_avg"])
    for t in ranked:
        wr = win_counts[t] / max(total_points, 1)
        print(row(t, per_teacher_err[t]) + f" | win {wr:.0%}")
    if fused_err:
        print("-" * 100)
        print(row("FUSED LABELS", fused_err))
        if fused_err_unsnapped:
            print(row("  (unsnapped)", fused_err_unsnapped) + f" | {n_snapped} pts snapped to GT (score 0 by construction)")
        best = ranked[0]
        best_davg = delta_stats(np.concatenate(per_teacher_err[best]))["d_avg"]
        fused_davg = delta_stats(np.concatenate(fused_err))["d_avg"]
        print(f"\nfused d_avg {fused_davg:.3f} vs best teacher ({best}) {best_davg:.3f} -> "
              + ("verifier ADDS value; the ceiling is elsewhere (data volume, student training, rounds)."
                 if fused_davg >= best_davg else
                 "verifier is BELOW its best teacher: labels are the ceiling. See interpretation guide in the docstring."))
        if fused_source_counts:
            tot = sum(fused_source_counts.values())
            frac = ", ".join(f"idx{k}={v/tot:.0%}" for k, v in sorted(fused_source_counts.items()))
            print(f"selection shares (teacher index order as passed to the verifier; -1=aggregated): {frac}")

    print("\nNOTE: endpoint error is evaluated at the final frame only -- exactly "
          "STIR's scored quantity, but blind to mid-clip quality. If the fused "
          "labels look fine here yet the student underperforms, check mid-clip "
          "cycle-consistency next.")

    if args.headroom_test:
        headroom_test(raw_root, endpoints_by_clip, teachers, clip_ids, args.good_px)


if __name__ == "__main__":
    main()
