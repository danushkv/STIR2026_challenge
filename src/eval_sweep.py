"""Evaluate ONE student checkpoint across an iters sweep under the LITETRACKER
streaming runtime, measuring everything the 2026 harness scores that we can
measure without 2026 ground truth. Writes a .npz (per-point, for paired
statistics) + a .json summary. One invocation = one checkpoint = one SLURM array
task; compare_ckpts.py merges the shards.

MEASURED PER (checkpoint, iters), all in a single pass over each decoded clip:

  2D endpoint   left start centers -> last frame, KDTree NN vs end centers. [px]
  3D endpoint   stereo-matched points through BOTH views, disparity =
                (u_L + disparitypad*scale) - u_R, triangulated.            [mm]
  DRIFT         forward-backward cycle consistency: track 0->T-1, then the
                frames REVERSED from the tracked endpoint back to frame 0, and
                compare to the original query. No ground truth needed. Endpoint
                error cannot see a point that wanders off and drifts back, nor
                one the NN match rescues by snapping it to a neighbour's target;
                cycle error sees both. High cycle + low endpoint is exactly the
                model that scores well here and badly on per-frame ATA.
  VISIBILITY    invisible rate, flicker (state changes/100 frames), and endpoint
                AJ under predicted visibility vs forced all-visible. LiteTracker
                thresholds vis at 0.6 inside forward(), so this is byte-for-byte
                what StudentLTWrapper hands the harness -- and nothing in
                stats.txt or stats3d.txt has ever looked at it.

WHY SHARD BY CHECKPOINT: pairing (which is what gives compare_ckpts.py its
power) needs every run to score the SAME clips and the SAME points in the SAME
order. Clip order is `sorted(clip_id)` and the points come from the data, not
the model, so shards stay aligned by construction -- compare_ckpts.py asserts it.
Re-decoding per shard costs ~2 min and buys full cluster parallelism.

Protocol matches eval_2d.py / eval_3d.py exactly:
streaming/causal, skip=1, no temporal striding. `model.iters` is a plain
attribute read at forward time, so the iters sweep mutates it in place.

    python eval_sweep.py --config train.yaml \
        tag=A_trial_e45 student_checkpoint=/path/student_last.pth \
        val_stir_root=/mnt/cluster/datasets/STIRTest_2025 \
        iters_list=[1,2,4] out_dir=/path/sweep
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from eval_common import STIR_THRESHOLDS_MM, STIR_THRESHOLDS_PX, nn_dist, triangulate_mm
from collect_tracks import _clip_id, _import_stirloader
from student_lt_wrapper import load_student_as_litetracker
from model import DEFAULT_CONFIG, merge_config_strict, normalize_overrides


def _load_match_right():
    """Import match_right() from the 2026 stereo wrapper by path.

    That module keeps its torch/lite_tracker import inside StudentStereoWrapper
    .__init__, so loading it here costs nothing but numpy -- and it guarantees we
    are measuring the EXACT function the submission ships, not a copy of it.
    """
    import importlib.util
    from pathlib import Path
    here = Path(__file__).resolve().parent
    rel = Path("models") / "stereo" / "student_stereo.py"
    # Search order mirrors _import_stirloader: explicit env var, THIRDPARTY_ROOT,
    # then the original relative candidates. The organisers' inference repo is
    # cloned OUTSIDE this repository, so the bare `../` hop no longer finds it.
    cands = []
    if os.environ.get("STIR_INFERENCE_ROOT"):
        cands.append(Path(os.environ["STIR_INFERENCE_ROOT"]) / rel)
    if os.environ.get("THIRDPARTY_ROOT"):
        cands.append(Path(os.environ["THIRDPARTY_ROOT"])
                     / "stir-challenge-2026-inference" / rel)
    cands += [here.parent / "stir-challenge-2026-inference" / rel,
              here.parent.parent / "stir-challenge-2026-inference" / rel,
              # the copy this repo ships, as a last resort
              here.parent / "submission" / "code" / rel]
    p = next((c for c in cands if c.exists()), None)
    if p is None:
        raise SystemExit("cannot find the stereo wrapper (student_stereo.py). Set "
                         "STIR_INFERENCE_ROOT to the cloned stir-challenge-2026-"
                         "inference repo. Tried:\n  "
                         + "\n  ".join(str(c) for c in cands))
    spec = importlib.util.spec_from_file_location("student_stereo", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.match_right, m.MIN_NCC, m.repair_disparity


MONO_THRESHOLDS = (2, 4, 8, 16, 32)   # stir-challenge-2026-metrics/run.py


def load_config():
    argv = sys.argv[1:]
    config_path, raw, i = None, [], 0
    while i < len(argv):
        if argv[i] == "--config":
            config_path, i = argv[i + 1], i + 2
        else:
            raw.append(argv[i]); i += 1
    base = dict(DEFAULT_CONFIG)
    base["tag"] = "???"                 # short name for this checkpoint
    base["student_checkpoint"] = "???"
    base["val_stir_root"] = ""
    base["test_patients"] = []
    base["eval_clip_ids"] = []
    base["max_eval_clips"] = 0
    base["iters_list"] = [1, 2, 4]
    base["out_dir"] = "sweep"
    base["skip"] = 1                    # stream every frame; fixed for comparability
    base["do_drift"] = True
    # segs  = right-view start points from STIRLoader.getsegsstereo (contour
    #         matching on the IR segmentation in BOTH views -- annotation data,
    #         which the 2026 harness will NOT give you)
    # match = match_right() from the submission wrapper, i.e. NCC on the images,
    #         exactly what the shipped StudentStereoWrapper does
    base["right_queries"] = "segs"
    # plausible TRUE disparity range in px. STIRLoader itself gates start points
    # to [8,105]; this is that with margin.
    base["disp_min_true"] = 4.0
    base["disp_max_true"] = 120.0
    # --- the two fixes from Peng's 2025 write-up, measurable here before they ship
    # frame-0: reject ambiguous matches (L->R->L) and repair them from neighbours
    base["match_lr_tol"] = 0.0        # px; 0 disables
    base["match_repair_ncc"] = 0.0    # NCC below which frame-0 disparity is repaired
    # final frame: geometric consistency of the triangulated disparity
    base["disp_repair_k"] = 0         # neighbours; 0 disables
    # Repair ONLY gross outliers. A good point sits within ~1px of its
    # neighbours, so a tight tolerance overwrites correct measurements with a
    # coarser neighbourhood median -- which is what cost 2mm/4mm accuracy at
    # abs=3.0. Raise it until only the catastrophic points are touched.
    base["disp_repair_abs"] = 12.0    # px
    base["disp_repair_mad"] = 6.0     # x robust sigma
    ov = normalize_overrides(raw)
    cli_keys = {o.split("=", 1)[0] for o in ov}
    cfg = merge_config_strict(base, config_path, ov)

    # train.yaml is a TRAINING config: it sets skip: 5 (to match how the
    # pseudo-labels were collected) and merges AFTER these defaults, so it would
    # silently override the streaming protocol and evaluate every 5th frame. The
    # 2026 harness streams EVERY frame, so restore the protocol unless the CLI
    # explicitly asked otherwise. eval_2d.py and eval_3d.py
    # have the same latent bug -- their docstrings promise skip=1 and they do not
    # deliver it.
    for k, v in (("skip", 1), ("val_max_frames", 0)):
        if k not in cli_keys and k in cfg and cfg[k] != v:
            print(f"[protocol] forcing {k}={v} (train.yaml said {cfg[k]}) "
                  f"-- pass {k}=<n> on the CLI to override deliberately")
            cfg[k] = v
    for k in ("tag", "student_checkpoint"):
        if OmegaConf.is_missing(cfg, k):
            raise SystemExit(f"{k} is required")
    return cfg


@torch.no_grad()
def stream_track(model, frames, start_pts, device, dtype, collect_vis=False):
    """Stream frames [T,H,W,3] uint8 through LiteTracker one frame at a time.

    Returns (final_coords [N,2] native px, vis [T,N] float 0/1 or None). Same
    forward path as eval_2d.track_clip_streaming; this variant can
    also retain per-frame visibility, which that one discards.
    """
    model.init_video_online_processing()
    q = torch.from_numpy(np.asarray(start_pts, dtype=np.float32)).unsqueeze(0).to(device)
    queries = torch.cat([torch.zeros_like(q[:, :, :1]), q], dim=-1)  # [B,N,3]=(t,x,y)

    coords, vis_seq = None, []
    with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
        for t in range(frames.shape[0]):
            frame_t = (torch.from_numpy(frames[t]).permute(2, 0, 1).unsqueeze(0)
                       .to(dtype=dtype, device=device))
            coords, vis, *_ = model(frame_t, queries=queries)
            # vis is ALREADY thresholded to bool at 0.6 inside forward() -- this is
            # byte-for-byte what StudentLTWrapper hands the 2026 harness.
            if collect_vis:
                vis_seq.append(vis[0, 0].float().cpu().numpy())
    return (coords[0, 0].float().cpu().numpy().astype(np.float32),
            np.stack(vis_seq) if collect_vis else None)


def aj_at(dist, vis_pred, thresholds):
    """AJ as in stir-challenge-2026-metrics/utils.counts_per_threshold, with
    vis_gt=True everywhere (STIR's end-frame tattoos are annotated, so visible):
        tp = within & vis_pred   fp = vis_pred & ~within   fn = ~vis_pred | ~within
    """
    out = []
    for t in thresholds:
        w = dist < t
        tp = int((w & vis_pred).sum())
        fp = int((vis_pred & ~w).sum())
        fn = int((~vis_pred | ~w).sum())
        out.append(tp / (tp + fp + fn) if (tp + fp + fn) else 0.0)
    return float(np.mean(out))


def main():
    cfg = load_config()
    print(OmegaConf.to_yaml(cfg))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported()
             else torch.float32)
    model, info = load_student_as_litetracker(
        cfg.student_checkpoint, window_len=cfg.window_len, iters=1, device=device)
    print(f"[{cfg.tag}] {info['weights']}  device={device} dtype={dtype}")

    stir_root = cfg.val_stir_root or cfg.stir_root
    patients = set(str(p) for p in cfg.test_patients)
    getviddirs2d_STIR, STIRStereoClip = _import_stirloader()
    seqs = {_clip_id(p): p for p in getviddirs2d_STIR(stir_root)
            if (not patients) or _clip_id(p).split("__", 1)[0] in patients}
    requested = set(str(c) for c in cfg.eval_clip_ids)
    if requested:
        absent = sorted(requested - set(seqs))
        if absent:
            raise ValueError(f"requested evaluation clip(s) not found: {absent}")
        seqs = {cid: path for cid, path in seqs.items() if cid in requested}
    if cfg.max_eval_clips < 0:
        raise ValueError("max_eval_clips must be >= 0")
    if cfg.max_eval_clips:
        seqs = dict(sorted(seqs.items())[:int(cfg.max_eval_clips)])
    if not seqs:
        raise ValueError("no evaluation clips found; check paths and clip filters")
    if cfg.skip > 1:
        os.environ["SKIP"] = str(cfg.skip)
    iters_list = [int(i) for i in cfg.iters_list]
    print(f"[{cfg.tag}] {len(seqs)} clip(s) from {stir_root} | iters {iters_list}")

    acc = {it: {"d2d": [], "d3d": [], "cyc": [], "visend": [],
                "inv": [], "flk": []} for it in iters_list}
    ids2d, n2d, f2d = [], [], []
    ids3d, n3d, f3d, ctrl = [], [], [], []

    use_match = str(cfg.right_queries).lower() == "match"
    match_right = min_ncc = repair_disparity = None
    if use_match:
        match_right, min_ncc, repair_disparity = _load_match_right()
        print(f"[{cfg.tag}] right-view start points from match_right() "
              f"(MIN_NCC={min_ncc}) -- the SUBMISSION path, not getsegsstereo")
    else:
        print(f"[{cfg.tag}] right-view start points from getsegsstereo "
              f"(annotation-derived; not available at submission time)")
    match_dx, match_ncc = [], []      # frame-0 diagnostics vs the annotation

    for cid, seq_path in sorted(seqs.items()):
        t0 = time.time()
        try:
            clip = STIRStereoClip(seq_path)
            lf, rf = clip.extractallframes()
            left, right = np.stack(lf, axis=0), np.stack(rf, axis=0)
        except (AssertionError, IndexError, ValueError) as e:
            print(f"[skip] {cid}: {e}"); continue
        if left.shape[0] < 2:
            print(f"[skip] {cid}: {left.shape[0]} frame(s)"); continue

        try:                                   # 2D: left start/end IR centers
            s2 = np.array(clip.getstartcenters(left=True), dtype=np.float32)
            e2 = np.array(clip.getendcenters(left=True), dtype=np.float32)
            ok2 = len(s2) > 0 and len(e2) > 0
        except (AssertionError, IndexError, ValueError):
            ok2 = False
        try:                                   # 3D: stereo-matched + 3D GT (mm)
            sl, sr = (np.array(x, dtype=np.float32)
                      for x in clip.getsegsstereo(start=True))
            _, _, g0 = clip.get3DSegmentationPositions(start=True)
            _, _, g1 = clip.get3DSegmentationPositions(start=False)
            ok3 = len(sl) > 0 and len(g1) > 0
        except (AssertionError, IndexError, ValueError):
            ok3 = False
        if not (ok2 or ok3):
            print(f"[skip] {cid}: no usable annotation"); continue

        if ok2:
            ids2d.append(cid); n2d.append(len(s2)); f2d.append(left.shape[0])
        if ok3 and use_match:
            # STIR's pair is NOT rectified to a common principal point: the true
            # disparity is (u_L + pad) - u_R with pad = disparitypad*scale ~ +92px,
            # so the RAW offset u_L-u_R that an image matcher sees is shifted by
            # -pad and is mostly negative. Search in raw terms. (On 2026 data
            # cx_left == cx_right, so pad = 0 and no shift is needed -- this is a
            # STIR artefact, not something the submission has to handle.)
            pad = float(clip.disparitypad) * float(clip.scale)
            sr_m, ncc_m = match_right(left[0], right[0], sl,
                                      min_disp=float(cfg.disp_min_true) - pad,
                                      max_disp=float(cfg.disp_max_true) - pad,
                                      lr_tol=float(cfg.match_lr_tol),
                                      repair_ncc=float(cfg.match_repair_ncc))
            # how far the image match lands from the annotation-derived one
            match_dx.append(np.abs(sr_m[:, 0] - sr[:, 0]))
            match_ncc.append(ncc_m)
            sr = sr_m

        if ok3:
            ids3d.append(cid); n3d.append(len(sl)); f3d.append(left.shape[0])
            ctrl.append(nn_dist(np.asarray(g0, dtype=np.float32), g1))

        for it in iters_list:
            model.iters = it               # read by forward(); no rebuild needed
            a = acc[it]
            if ok2:
                fwd, vis = stream_track(model, left, s2, device, dtype, collect_vis=True)
                a["d2d"].append(nn_dist(fwd, e2))
                v = vis > 0.5
                a["visend"].append(v[-1])
                a["inv"].append(1.0 - float(v.mean()))
                a["flk"].append(float(np.mean(np.abs(np.diff(v.astype(int), axis=0)).sum(0)))
                                / max(v.shape[0] - 1, 1) * 100.0)
                if cfg.do_drift:
                    back, _ = stream_track(model, left[::-1].copy(), fwd, device, dtype)
                    a["cyc"].append(np.linalg.norm(back - s2, axis=1))
            if ok3:
                fl, _ = stream_track(model, left, sl, device, dtype)
                fr, _ = stream_track(model, right, sr, device, dtype)
                disp = (fl[:, 0] + clip.disparitypad * clip.scale) - fr[:, 0]
                if repair_disparity is not None and int(cfg.disp_repair_k) > 0:
                    # same function the wrapper applies every frame; here it acts
                    # on the endpoint, which is what this metric scores
                    disp, _rep = repair_disparity(
                        disp, fl, k=int(cfg.disp_repair_k),
                        abs_tol=float(cfg.disp_repair_abs),
                        mad_tol=float(cfg.disp_repair_mad))
                a["d3d"].append(nn_dist(triangulate_mm(fl[:, 0], fl[:, 1], disp, clip), g1))
        print(f"[{cfg.tag}] {cid} {left.shape[0]:>4}f 2d={'y' if ok2 else '-'} "
              f"3d={'y' if ok3 else '-'} {time.time()-t0:.1f}s")

    os.makedirs(cfg.out_dir, exist_ok=True)
    out = {
        "tag": np.array(cfg.tag), "ckpt": np.array(str(cfg.student_checkpoint)),
        "iters_list": np.array(iters_list),
        "clip_ids_2d": np.array(ids2d), "clip_npts_2d": np.array(n2d, dtype=int),
        "clip_nframes_2d": np.array(f2d, dtype=int),
        "clip_ids_3d": np.array(ids3d), "clip_npts_3d": np.array(n3d, dtype=int),
        "clip_nframes_3d": np.array(f3d, dtype=int),
        "owner_2d": np.repeat(np.arange(len(ids2d)), n2d) if ids2d else np.array([], int),
        "owner_3d": np.repeat(np.arange(len(ids3d)), n3d) if ids3d else np.array([], int),
        "thresholds_2d": np.array(STIR_THRESHOLDS_PX, float),
        "thresholds_3d": np.array(STIR_THRESHOLDS_MM, float),
    }
    if ctrl:
        out["control_3d"] = np.concatenate(ctrl)
    out["right_queries"] = np.array(str(cfg.right_queries))
    out["fix_config"] = np.array(f"lr{cfg.match_lr_tol}_rep{cfg.match_repair_ncc}"
                                 f"_k{cfg.disp_repair_k}_a{cfg.disp_repair_abs}"
                                 f"_m{cfg.disp_repair_mad}")
    if match_dx:
        dx = np.concatenate(match_dx); nc = np.concatenate(match_ncc)
        out["match_dx_px"] = dx
        out["match_ncc"] = nc
        summary_match = {
            "n_points": int(dx.size),
            "gate_pass_rate": float(np.mean(nc >= min_ncc)),
            "ncc_median": float(np.median(nc)),
            # |x_match - x_getsegsstereo| at frame 0: the matcher's own error,
            # isolated from any tracking. This is what the offline 3D numbers
            # got for free and the submission has to earn.
            "dx_median_px": float(np.median(dx)),
            "dx_p90_px": float(np.percentile(dx, 90)),
            "dx_within_1px": float(np.mean(dx <= 1.0)),
            "dx_within_3px": float(np.mean(dx <= 3.0)),
        }
    else:
        summary_match = None

    summary = {"tag": cfg.tag, "ckpt": str(cfg.student_checkpoint),
               "right_queries": str(cfg.right_queries),
               "frame0_match": summary_match,
               "n_clips_2d": len(ids2d), "n_clips_3d": len(ids3d), "iters": {}}
    for it in iters_list:
        a, s = acc[it], {}
        if a["d2d"]:
            d = np.concatenate(a["d2d"]); out[f"d2d__i{it}"] = d
            s["delta_2d"] = {f"delta_{t}px": float(np.mean(d <= t)) for t in STIR_THRESHOLDS_PX}
            s["delta_avg_2d"] = float(np.mean(list(s["delta_2d"].values())))
            s["epe_mean_px"], s["epe_median_px"] = float(d.mean()), float(np.median(d))
            ve = np.concatenate(a["visend"]); out[f"visend__i{it}"] = ve
            s["invisible_rate"] = float(np.mean(a["inv"]))
            s["flicker_per_100f"] = float(np.mean(a["flk"]))
            s["aj_endpoint_pred_vis"] = aj_at(d, ve, MONO_THRESHOLDS)
            s["aj_endpoint_forced_vis"] = aj_at(d, np.ones_like(ve, bool), MONO_THRESHOLDS)
            s["aj_gain_if_forced"] = s["aj_endpoint_forced_vis"] - s["aj_endpoint_pred_vis"]
        if a["cyc"]:
            c = np.concatenate(a["cyc"]); out[f"cyc__i{it}"] = c
            s["cycle_mean_px"], s["cycle_median_px"] = float(c.mean()), float(np.median(c))
            s["cycle_over_endpoint"] = float(c.mean() / max(np.concatenate(a["d2d"]).mean(), 1e-6))
        if a["d3d"]:
            d3 = np.concatenate(a["d3d"]); out[f"d3d__i{it}"] = d3
            s["acc_3d"] = {f"{t}mm": float(np.mean(d3 <= t)) for t in STIR_THRESHOLDS_MM}
            s["acc_avg_3d"] = float(np.mean(list(s["acc_3d"].values())))
            s["err_mean_mm"], s["err_median_mm"] = float(d3.mean()), float(np.median(d3))
        summary["iters"][str(it)] = s

    npz_path = os.path.join(cfg.out_dir, f"{cfg.tag}.npz")
    json_path = os.path.join(cfg.out_dir, f"{cfg.tag}.json")

    # MERGE with an earlier run of this same tag, so an iters sweep can be split
    # across jobs (screen at iters=1 first, add 2 and 4 for the finalists later)
    # without redoing the ones already measured. Only results for iters NOT in
    # this run are carried over, and only if the clip set is identical -- a
    # different val_stir_root would silently mix incomparable numbers.
    if os.path.exists(npz_path):
        try:
            prev = np.load(npz_path, allow_pickle=False)
            # right_queries must match too: segs and match results are NOT
            # interchangeable, and merging them under one tag would silently
            # report a mixture. Older shards predate the key -> treat as "segs".
            prev_rq = (str(prev["right_queries"]) if "right_queries" in prev.files
                       else "segs")
            same = (list(prev["clip_ids_2d"]) == list(out["clip_ids_2d"])
                    and list(prev["clip_ids_3d"]) == list(out["clip_ids_3d"])
                    and prev_rq == str(cfg.right_queries)
                    and (str(prev["fix_config"]) if "fix_config" in prev.files
                         else "") == str(out["fix_config"]))
            if same:
                kept = []
                for k in prev.files:
                    if "__i" in k and k not in out:
                        out[k] = prev[k]
                        kept.append(k.split("__i")[1])
                if kept:
                    prev_iters = sorted({int(i) for i in kept})
                    out["iters_list"] = np.array(sorted(set(iters_list) | set(prev_iters)))
                    print(f"[{cfg.tag}] merged in earlier iters {prev_iters}")
                    if os.path.exists(json_path):
                        old_s = json.load(open(json_path))
                        for it_s, m in old_s.get("iters", {}).items():
                            summary["iters"].setdefault(it_s, m)
            else:
                print(f"[{cfg.tag}] WARNING: existing {npz_path} used a different "
                      f"clip set or right_queries mode (was {prev_rq!r}, now "
                      f"{str(cfg.right_queries)!r}) -- overwriting, not merging")
        except Exception as e:
            print(f"[{cfg.tag}] could not merge existing npz ({e}); overwriting")

    np.savez_compressed(npz_path, **out)
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    if summary_match:
        m = summary_match
        print(f"\n[{cfg.tag}] frame-0 match_right() vs getsegsstereo, "
              f"{m['n_points']} points:")
        print(f"    gate pass {100*m['gate_pass_rate']:.1f}%  median NCC {m['ncc_median']:.3f}")
        print(f"    |dx| median {m['dx_median_px']:.2f}px  p90 {m['dx_p90_px']:.2f}px  "
              f"within 1px {100*m['dx_within_1px']:.1f}%  within 3px {100*m['dx_within_3px']:.1f}%")
    print(f"\n[{cfg.tag}] wrote {npz_path} and {json_path}")
    for it in iters_list:
        s = summary["iters"][str(it)]
        print(f"  iters={it}: 2D delta_avg {s.get('delta_avg_2d', float('nan')):.4f} "
              f"EPE {s.get('epe_mean_px', float('nan')):.2f}px | "
              f"3D avg {s.get('acc_avg_3d', float('nan')):.4f} "
              f"{s.get('err_mean_mm', float('nan')):.2f}mm | "
              f"cycle {s.get('cycle_mean_px', float('nan')):.2f}px | "
              f"invis {100*s.get('invisible_rate', float('nan')):.1f}%")


if __name__ == "__main__":
    main()
