"""STIR 2026 stereo (3D) submission: DUAL TRACKING with our student. No MFT.

This is the pipeline our 3D evaluation actually measured, and the design that
won STIR 2025: track the query points in the LEFT view and their stereo
correspondents in the RIGHT view with the SAME checkpoint, then triangulate
from the per-frame horizontal offset.

    disparity_t = u_left_t - u_right_t
    Z = fx * baseline / disparity     X = (u-cx) Z / fx     Y = (v-cy) Z / fy

THE ONE THING THAT DIFFERS FROM THE OFFLINE EVAL -- read before changing
anything. eval_sweep.py took the right-view start points from STIRLoader's
`getsegsstereo`, which contour-matches the IR TATTOO SEGMENTATION IN BOTH VIEWS.
That is annotation-derived; it exists only because the STIR release ships the
segmentation for both cameras. The 2026 harness gives both IMAGES every frame
but `queries.json` carries `left_x`/`left_y` ONLY -- an arbitrary 64px grid. So
the right-view start points must be recovered from the images, which is what
match_right() does. It is the sole part of this pipeline the offline 3D numbers
never exercised, and therefore the only place new error can enter: treat those
numbers as optimistic until this is measured.

WHY A SCANLINE SEARCH SUFFICES: the 2026 pair is rectified to a COMMON principal
point -- val/*/calib.json has cx_left == cx_right exactly (the older STIR release
differs by ~90px, which is what STIRLoader's `disparitypad` compensates for).
Rectified + common cx means the correspondent of (u,v) lies on row v at u-d for
d>0, so the search is 1-D and normalised cross-correlation solves it.

NO cv2, DELIBERATELY: opencv is not in the `lt` dependency group (only `rlt`
pulls it in, via MFT). NCC over a 1-D search is a dozen lines of numpy, so this
image needs exactly the same deps as the mono one -- `uv sync --group lt`.

VISIBILITY: left AND right visibility AND a valid disparity AND a frame-0
match-quality gate. A point whose initial match is wrong has wrong depth on
EVERY frame; reported visible it costs both a false positive and a false
negative under the 2026 AJ, reported invisible only the false negative.

UNITS: metres. baseline comes from CameraCalib.t_left_to_right, which is metres,
so Z is metres -- exactly what STEREO_THRESHOLDS (0.002..0.032 m) expects. Do
NOT scale by 1000.

    export STUDENT_LT_WEIGHTS=/workspace/weights/student.pth
    export STUDENT_LT_ITERS=4
    uv run stereo.py --data_dir <dataset> --output_dir results_3d \
        --model models.stereo.student_stereo.StudentStereoWrapper
"""

import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# NOTE: StudentLTWrapper (torch + lite_tracker) is imported lazily inside
# __init__ so match_right() and the self-test below run on numpy alone.

# Disparity search window, pixels. At the 2026 val geometry (fx 567.5, baseline
# 4.14mm, ~640px wide) a 20mm working distance is ~118px and 300mm is ~8px, so
# [4, 128] spans the plausible surgical range with margin.
MIN_DISP = float(os.environ.get("STUDENT_STEREO_MIN_DISP", "4"))
MAX_DISP = float(os.environ.get("STUDENT_STEREO_MAX_DISP", "128"))
# Half-size of the square NCC patch: 7 -> 15x15. Large enough to be
# discriminative on low-texture tissue, small enough not to straddle depth edges.
PATCH = int(os.environ.get("STUDENT_STEREO_PATCH", "7"))
# Below this NCC the frame-0 match is not trusted and the point is gated out.
MIN_NCC = float(os.environ.get("STUDENT_STEREO_MIN_NCC", "0.3"))
# OFF BY DEFAULT, DELIBERATELY. The measured 3D number (0.6863 on STIRTest_2025)
# was produced WITHOUT these two refinements, and an unmeasured default is how
# you end up submitting something nobody has scored. Enable + re-run
# eval_sweep.py with right_queries=match before trusting either.
#
# Left-right consistency: re-match the found point back R->L; if it does not
# return within this many px the match is ambiguous. <=0 disables the check.
LR_TOL = float(os.environ.get("STUDENT_STEREO_LR_TOL", "0"))
# Points whose NCC is below this (or which fail the L-R check) have their
# DISPARITY REPLACED by the median of their repair_k nearest confident
# neighbours. Tissue surfaces are smooth and the 2026 queries are a grid, so a
# local median is a far better estimate than a confidently wrong correlation
# peak -- and unlike gating it actually recovers the point, which matters
# because ATA scores position regardless of predicted visibility.
REPAIR_NCC = float(os.environ.get("STUDENT_STEREO_REPAIR_NCC", "0"))
REPAIR_K = int(os.environ.get("STUDENT_STEREO_REPAIR_K", "5"))

# --- per-frame geometric consistency -----------------------------------------
# Disparity is a DIFFERENCE of two independently tracked points, so the two
# trackers' errors add and nothing notices when they drift apart. One bad
# triangulation is then emitted as a confident 3D point -- which is exactly the
# measured failure: 3D mean 31.09mm against a median of 3.18mm.
#
# The fix uses the structure of the problem: the queries are a grid on tissue,
# tissue is a smooth surface, so disparity varies smoothly across neighbouring
# points. A point whose disparity departs far from its neighbours' is wrong, and
# the neighbours' median is a better estimate than the bad measurement. Note it
# REPAIRS rather than rejects -- ATA scores position regardless of predicted
# visibility, so dropping a point recovers nothing.
#   0 disables. Tolerance is max(CONSIST_ABS px, CONSIST_MAD * robust sigma).
#
# MEASURED on the 32 held-out STIRTest_2025 clips (tag agg_e44_match_fix4), and
# these values are tuned, not guessed:
#   off          3D acc_avg 0.6863, mean 31.09mm
#   k8 a8 m3.0   3D acc_avg 0.7031, mean 19.31mm   <- shipped
# Every one of the five thresholds improves (2mm +0.0088 ... 32mm +0.0220).
#
# Why these numbers. 1px of disparity is ~2.7mm of depth at a 109mm working
# distance, and the scored thresholds are 2/4/8/16/32mm -- so a pixel is roughly
# a threshold bin, and the tolerance has to sit OUTSIDE the good population or
# it overwrites correct depth with a coarser neighbourhood median. ABS=3 did
# exactly that (2mm fell 0.3304 -> 0.3084). ABS=8 (~22mm) is clear of it.
# MAD is the knob that actually bites: at MAD=6 the term 6*sigma swamps ABS
# everywhere, so ABS=7 and ABS=12 gave byte-identical results and caught almost
# no outliers. MAD=3 lets the check fire.
CONSIST_K = int(os.environ.get("STUDENT_STEREO_CONSIST_K", "8"))
CONSIST_ABS = float(os.environ.get("STUDENT_STEREO_CONSIST_ABS", "8.0"))
CONSIST_MAD = float(os.environ.get("STUDENT_STEREO_CONSIST_MAD", "3.0"))


def repair_disparity(disparity, coords, k=CONSIST_K, abs_tol=CONSIST_ABS,
                     mad_tol=CONSIST_MAD):
    """Replace disparities that disagree with their spatial neighbours.

    disparity [N], coords [N,2] left-image positions. Returns
    (disparity_out [N], repaired [N] bool). k<=0 or too few points -> no-op.

    Robust by construction: the neighbour median and MAD are unaffected by the
    outliers being detected, so one bad point cannot drag its neighbourhood.
    """
    d = np.asarray(disparity, dtype=np.float32).copy()
    n = d.size
    if k <= 0 or n < k + 2:
        return d, np.zeros(n, dtype=bool)
    xy = np.asarray(coords, dtype=np.float32)[:, :2]
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)                    # exclude self
    nbr = np.argsort(d2, axis=1)[:, :k]
    dn = d[nbr]                                     # [N,k]
    med = np.median(dn, axis=1)
    mad = np.median(np.abs(dn - med[:, None]), axis=1) * 1.4826
    tol = np.maximum(abs_tol, mad_tol * mad)
    bad = np.abs(d - med) > tol
    d[bad] = med[bad]
    return d, bad


def _gray(im):
    """(H,W,3) uint8 RGB -> (H,W) float32 luma."""
    a = np.asarray(im, dtype=np.float32)
    if a.ndim == 2:
        return a
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def match_right(frame_left, frame_right, queries, min_disp=MIN_DISP,
                max_disp=MAX_DISP, patch=PATCH, lr_tol=LR_TOL, repair_ncc=REPAIR_NCC,
                repair_k=REPAIR_K):
    """Find each left query's correspondent on the right image by NCC along its
    epipolar row (rectified pair, so the row is the same v).

    Equivalent to cv2.matchTemplate(..., TM_CCOEFF_NORMED) over a 1-D strip,
    with a parabolic subpixel refinement of the correlation peak.

    Returns (right_xy [N,2] float32, ncc [N] float32). A point whose search
    window leaves the image keeps a fallback position and ncc = -1 so the caller
    can gate it out.
    """
    gl, gr = _gray(frame_left), _gray(frame_right)
    H, W = gl.shape
    lo, hi = int(round(min_disp)), int(round(max_disp))
    q = np.asarray(queries, dtype=np.float32)

    out = np.stack([q[:, 0] - lo, q[:, 1]], axis=1).astype(np.float32)
    ncc = np.full(len(q), -1.0, dtype=np.float32)

    for i, (u, v) in enumerate(q):
        ui, vi = int(round(u)), int(round(v))
        y0, y1 = vi - patch, vi + patch + 1
        x0, x1 = ui - patch, ui + patch + 1
        # the correspondent sits at u-d for d in [lo,hi], so the strip spans
        # [u-hi-patch, u-lo+patch]
        sx0, sx1 = ui - hi - patch, ui - lo + patch + 1
        if y0 < 0 or y1 > H or x0 < 0 or x1 > W or sx0 < 0 or sx1 > W:
            continue

        P = gl[y0:y1, x0:x1]
        S = gr[y0:y1, sx0:sx1]
        if S.shape[1] < P.shape[1]:
            continue
        # every horizontal placement of the patch inside the strip
        Wv = sliding_window_view(S, P.shape)[0]          # (D+1, h, w)

        Pc = P - P.mean()
        Wc = Wv - Wv.mean(axis=(-2, -1), keepdims=True)
        den = np.sqrt((Pc * Pc).sum() * (Wc * Wc).sum(axis=(-2, -1)))
        r = np.where(den > 1e-12, (Pc * Wc).sum(axis=(-2, -1)) / np.maximum(den, 1e-12), -1.0)
        if r.size == 0:
            continue

        k = int(np.argmax(r))
        # window k places the patch centre at sx0+k+patch, i.e. disparity hi-k
        d = float(hi - k)
        if 0 < k < len(r) - 1:                            # parabolic subpixel
            a, b, c = float(r[k - 1]), float(r[k]), float(r[k + 1])
            dd = a - 2.0 * b + c
            if abs(dd) > 1e-12:
                d -= float(np.clip(0.5 * (a - c) / dd, -1.0, 1.0))
        out[i] = (u - d, v)
        ncc[i] = float(r[k])

    if lr_tol > 0:
        # L->R->L consistency: a correct match maps back onto itself. This
        # catches confidently-wrong peaks on repetitive tissue texture, which
        # NCC alone rates highly.
        back, _ = _match_1d(gr, gl, out, -hi, -lo, patch)
        bad = ~np.isfinite(back) | (np.abs(back - q[:, 0]) > lr_tol)
        ncc[bad] = np.minimum(ncc[bad], 0.0)

    if repair_ncc > 0 and repair_k > 0:
        conf = ncc >= repair_ncc
        if conf.sum() >= 3 and (~conf).any():
            d_all = q[:, 0] - out[:, 0]
            src = q[conf]
            for i in np.flatnonzero(~conf):
                w = np.hypot(src[:, 0] - q[i, 0], src[:, 1] - q[i, 1])
                nn = np.argsort(w)[:repair_k]
                out[i, 0] = q[i, 0] - float(np.median(d_all[conf][nn]))
    return out, ncc


def _match_1d(g_src, g_dst, pts, lo, hi, patch):
    """Helper for the L->R->L check: for each pt in g_src find its x in g_dst
    searching disparities [lo, hi]. Returns (x_found [N], ncc [N]); NaN where
    the search window leaves the image."""
    H, W = g_src.shape
    lo, hi = int(round(lo)), int(round(hi))
    xs = np.full(len(pts), np.nan, dtype=np.float32)
    nc = np.full(len(pts), -1.0, dtype=np.float32)
    for i, (u, v) in enumerate(np.asarray(pts, dtype=np.float32)):
        ui, vi = int(round(u)), int(round(v))
        y0, y1, x0, x1 = vi - patch, vi + patch + 1, ui - patch, ui + patch + 1
        sx0, sx1 = ui - hi - patch, ui - lo + patch + 1
        if y0 < 0 or y1 > H or x0 < 0 or x1 > W or sx0 < 0 or sx1 > W:
            continue
        P, S = g_src[y0:y1, x0:x1], g_dst[y0:y1, sx0:sx1]
        if S.shape[1] < P.shape[1]:
            continue
        Wv = sliding_window_view(S, P.shape)[0]
        Pc = P - P.mean()
        Wc = Wv - Wv.mean(axis=(-2, -1), keepdims=True)
        den = np.sqrt((Pc * Pc).sum() * (Wc * Wc).sum(axis=(-2, -1)))
        r = np.where(den > 1e-12, (Pc * Wc).sum(axis=(-2, -1)) / np.maximum(den, 1e-12), -1.0)
        if r.size == 0:
            continue
        k = int(np.argmax(r))
        xs[i] = u - float(hi - k)
        nc[i] = float(r[k])
    return xs, nc


class StudentStereoWrapper:
    """Dual-tracking stereo wrapper.

    Interface required by stereo.py: constructed as tracker_cls(seq.calib), then
        track(frame_left, frame_right, queries=None)
            -> (coords_3d [N,3] float32 metres, visibilities [N] bool)
    A fresh instance per sequence is what resets the streaming state -- each
    StudentLTWrapper calls init_video_online_processing in its own constructor.
    """

    # Pinned to a CLASS, not read from the environment: run_docker_inference.sh
    # passes no -e flags, so an env-dependent operating point is a silent
    # failure. iters=4 because EVERY match_right() measurement was made at
    # iters=4 -- the shipped stereo pipeline has never been scored at iters=1,
    # and the 3D track carries no latency penalty.
    LT_CLASS = "StudentLTWrapperAccurate"

    def __init__(self, calib):
        import models.mono.student_lt as _lt

        self.modeltype = "StudentStereo"
        lt_cls = getattr(_lt, self.LT_CLASS)
        # two independent tracker instances: LiteTracker keeps per-video
        # streaming buffers, so one instance cannot carry two videos at once
        self.lt_left = lt_cls()
        self.lt_right = lt_cls()

        self.fx = float(calib.K_left[0, 0])
        self.fy = float(calib.K_left[1, 1])
        self.cx = float(calib.K_left[0, 2])
        self.cy = float(calib.K_left[1, 2])
        # negated because t_left_to_right[0] is negative for this rig; matches
        # RLTWrapper and stir-challenge-2026-metrics/utils.py _intrinsics()
        self.baseline = float(calib.t_left_to_right[0]) * -1.0

        self.first = True
        self.gate = None            # frame-0 match quality, applied every frame

    def _backproject(self, disparity, coords):
        d = np.maximum(disparity, 1e-6)
        Z = self.fx * self.baseline / d
        X = (coords[:, 0] - self.cx) * Z / self.fx
        Y = (coords[:, 1] - self.cy) * Z / self.fy
        return np.stack([X, Y, Z], axis=1).astype(np.float32)

    def track(self, frame_left, frame_right, queries=None):
        if self.first:
            if queries is None:
                raise ValueError("queries must be provided on the first track() call")
            q = np.asarray(queries, dtype=np.float32)
            qr, ncc = match_right(frame_left, frame_right, q)
            self.gate = ncc >= MIN_NCC
            cl, vl = self.lt_left.track(frame_left, q)
            cr, vr = self.lt_right.track(frame_right, qr)
            self.first = False
        else:
            cl, vl = self.lt_left.track(frame_left)
            cr, vr = self.lt_right.track(frame_right)

        disparity = cl[:, 0] - cr[:, 0]
        disparity, _repaired = repair_disparity(disparity, cl)
        coords_3d = self._backproject(disparity, cl)
        visible = (vl.astype(bool) & vr.astype(bool) & self.gate
                   & (disparity > MIN_DISP * 0.5) & (disparity < MAX_DISP * 1.5))
        return coords_3d, visible.astype(bool)


# --------------------------------------------------------------------------- #
# Self-test. Run BEFORE building the image:
#     python models/stereo/student_stereo.py            # synthetic only
#     python models/stereo/student_stereo.py ../val/00  # + real frame 0
# --------------------------------------------------------------------------- #
def _shift(img, d):
    """Shift left by d px with linear interpolation (test helper)."""
    H, W = img.shape[:2]
    xs = np.arange(W, dtype=np.float32) + d
    x0 = np.floor(xs).astype(int); w = (xs - x0).astype(np.float32)
    x0c = np.clip(x0, 0, W - 1); x1c = np.clip(x0 + 1, 0, W - 1)
    a = img[:, x0c].astype(np.float32); b = img[:, x1c].astype(np.float32)
    return (a * (1 - w)[None, :, None] + b * w[None, :, None]).astype(np.uint8)


def _selftest(seq_dir=None):
    ok = True
    print("1) synthetic rectified pair, known disparity")
    rng = np.random.default_rng(0)
    H, W = 200, 640
    left = np.stack([(rng.random((H, W)) * 255).astype(np.uint8)] * 3, -1)
    q = np.array([[u, v] for u in (200, 320, 450) for v in (60, 100, 140)],
                 dtype=np.float32)
    for true_d in (8.0, 17.0, 35.5, 64.0, 100.0):
        right = _shift(left, true_d)
        qr, ncc = match_right(left, right, q)
        d = q[:, 0] - qr[:, 0]
        err = float(np.abs(d - true_d).max())
        good = err < 0.6 and np.allclose(qr[:, 1], q[:, 1])
        ok &= good
        print(f"   d={true_d:6.1f}  recovered {d.mean():7.3f}  max_err {err:.3f}  "
              f"min_ncc {ncc.min():.3f}  {'OK' if good else 'FAIL'}")

    if seq_dir:
        import json
        from pathlib import Path
        import imageio.v3 as iio
        p = Path(seq_dir)
        print(f"\n2) real frame 0 from {p}")
        calib = json.load(open(p / "calib.json"))
        fx = calib["K_left"][0][0]
        base = -calib["t_left_to_right"][0]
        raw = json.load(open(p / "queries.json"))["0/0"]
        q = np.stack([np.array(raw["left_x"], np.float32),
                      np.array(raw["left_y"], np.float32)], 1)
        fl = iio.imread(p / "left.mp4", index=0)
        fr = iio.imread(p / "right.mp4", index=0)
        qr, ncc = match_right(fl, fr, q)
        d = q[:, 0] - qr[:, 0]
        keep = ncc >= MIN_NCC
        print(f"   image {fl.shape}, {len(q)} queries, "
              f"{int(keep.sum())} matched (NCC>={MIN_NCC})")
        if keep.any():
            Z = fx * base / np.maximum(d[keep], 1e-6) * 1000.0     # mm, readable
            print(f"   disparity min/med/max: {d[keep].min():.1f} / "
                  f"{np.median(d[keep]):.1f} / {d[keep].max():.1f} px")
            print(f"   implied Z min/med/max: {Z.min():.0f} / {np.median(Z):.0f} / "
                  f"{Z.max():.0f} mm")
            sane = 5.0 < float(np.median(Z)) < 500.0 and keep.mean() > 0.5
        else:
            sane = False
        ok &= sane
        print(f"   plausible working distance and >50% matched: "
              f"{'OK' if sane else 'FAIL -- inspect before submitting'}")
    print("\nSELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(_selftest(sys.argv[1] if len(sys.argv) > 1 else None))
