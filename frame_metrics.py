#!/usr/bin/env python3
"""Phase1b §5 acceptance tests for event-accumulated frames.

The Phase 1 lesson this encodes: event reconstructions were judged by eye twice
and were wrong both times. Nothing here is a visual check. Ordered so the
cheapest disqualifier runs first, exactly as docs/Phase1b-MotionCompensation.md §5
lists them:

    determinism .. same events, different preceding history -> BIT-IDENTICAL PNG.
                   A hard gate, not a metric. This is the test E2VID failed
                   (MAE 15-31/255, correlation 0.05-0.58, 0-6 SIFT matches).
    sharpness .... mean image variance; with --motion-comp it must not fall.
    flow ......... SIFT matches vs time separation, and the number that actually
                   predicted the E2VID failure: MEDIAN FEATURE FLOW MUST SCALE
                   WITH THE GAP. E2VID sat at ~0.2 px for every gap, because
                   SIFT was locking onto the LSTM state's imprint rather than
                   the scene. Anything scene-locked cannot do that.
    inlier ratio . homography/fundamental RANSAC inliers below ~0.9, i.e. real
                   parallax rather than a degenerate (planar/pure-rotation) pair.
    colmap ....... THE gate: >=90% of frames registered, mean track length > 3.
                   E2VID scored exactly 2.0 -- features that never survive a
                   third view.
    fragments .... how many DISCONNECTED reconstructions the mapper produced,
                   and how many seconds of the take each one reaches. Once
                   track length clears 3 this is the binding constraint, and
                   `colmap` above cannot see it: it scores only the largest
                   model, so twelve healthy fragments read there as one badly
                   failed registration.

    mc-probe ..... Phase1b §4.1 VERIFY-FIRST: does ONE global motion suffice, or
                   does depth-dependent parallax need per-tile motion? Answer
                   this before building anything on the 2-DOF model.

Usage:
    python frame_metrics.py all        --frames <dir> [--input <recording>]
    python frame_metrics.py determinism --input <recording> [-n 50000]
    python frame_metrics.py flow       --frames <dir>
    python frame_metrics.py sharpness  --frames <dir> [--frames-b <dir>]
    python frame_metrics.py mc-probe   --input <recording> [-n 50000]
    python frame_metrics.py colmap     --scene <dir>
    python frame_metrics.py fragments  --scene <dir> [--frames <dir>]
"""

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv-python is required: pip install opencv-python")

import accumulate_frames as acc
import view_sparse
from convert_to_inceventgs import _resolve_input
from export_e2vid_input import read_meta, resolve_geometry

PASS, FAIL, INFO = "PASS", "FAIL", "  --"


def _fmt(status, name, detail):
    return f"  [{status:>4}] {name:<26} {detail}"


# --------------------------------------------------------------------------
# frame-set helpers
# --------------------------------------------------------------------------

def load_frames(frames_dir):
    """(manifest, [(path, t_mid_s), ...]). Falls back to filename order when a
    frames.json is missing, so third-party frame sets can be scored too."""
    man_path = os.path.join(frames_dir, "frames.json")
    if os.path.exists(man_path):
        with open(man_path) as f:
            man = json.load(f)
        # A scene dir keeps its PNGs in input/ (that is what convert.py wants)
        # while the manifest sits at the scene root; accept either layout.
        png_dir = frames_dir
        if not os.path.exists(os.path.join(frames_dir, man["frames"][0]["file"])):
            png_dir = os.path.join(frames_dir, "input")
        items = [(os.path.join(png_dir, e["file"]), e["t_mid_s"])
                 for e in man["frames"]]
        return man, items
    pngs = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not pngs:
        sys.exit(f"No PNGs in {frames_dir}")
    return None, [(p, float(i)) for i, p in enumerate(pngs)]


def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f"Unreadable: {path}")
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), img.shape[2]
    return img, 1


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------

def cmd_determinism(args):
    """Same events, two streams with different preceding history -> same bytes.

    Run A starts at the take start, so by the probe region it carries seconds of
    prior events. Run B starts inside the take, cold. Both are pinned to the
    same --norm-hi, because the global white point is a declared per-take
    constant (Phase1b §3) -- letting it differ would test the constant, not the
    reconstruction. Windows are matched by ABSOLUTE EVENT INDEX from frames.json,
    so the comparison is over provably identical event sets.
    """
    h5_path, meta_path = _resolve_input(args.input)
    width, height, n_total = resolve_geometry(h5_path, read_meta(meta_path))
    tmp = args.workdir or tempfile.mkdtemp(prefix="det_")
    a_dir, b_dir = os.path.join(tmp, "hist"), os.path.join(tmp, "fresh")

    common = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "accumulate_frames.py"),
              "--input", args.input,
              "--events-per-frame", str(args.events_per_frame),
              "--norm-hi", f"{args.norm_hi:.6f}"]
    if args.motion_comp:
        common += ["--motion-comp"]
        if args.mc_tiles != "1x1":
            common += ["--mc-tiles", args.mc_tiles]

    def run(cmd):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"accumulate_frames failed:\n{r.stdout}\n{r.stderr}")

    # A: reads from the take start, so by the probe region it has streamed
    # seconds of prior events through the same buffer.
    run(common + ["--out", a_dir,
                  "--duration-s", str(args.history_s + args.probe_s)])
    man_a, _ = load_frames(a_dir)
    after = [e for e in man_a["frames"] if e["t_start_s"] >= args.history_s]
    if not after:
        sys.exit("No frame after --history-s; lower it or raise --probe-s.")

    # B: cold, and pinned to A's window grid by absolute event index --
    # microsecond timestamps tie, so a matching --start-s would not align them.
    run(common + ["--out", b_dir, "--start-s", str(args.history_s),
                  "--duration-s", str(args.probe_s),
                  "--start-event", str(after[0]["i_start"])])
    man_b, _ = load_frames(b_dir)
    by_range_a = {(e["i_start"], e["i_end"]): e for e in man_a["frames"]}

    compared, mismatched, examples = 0, 0, []
    for eb in man_b["frames"]:
        ea = by_range_a.get((eb["i_start"], eb["i_end"]))
        if ea is None:
            continue
        pa = os.path.join(a_dir, ea["file"])
        pb = os.path.join(b_dir, eb["file"])
        ha = hashlib.sha256(open(pa, "rb").read()).hexdigest()
        hb = hashlib.sha256(open(pb, "rb").read()).hexdigest()
        compared += 1
        if ha != hb:
            mismatched += 1
            if len(examples) < 3:
                ga, _ = read_gray(pa)
                gb, _ = read_gray(pb)
                mae = float(np.abs(ga.astype(np.int16) - gb.astype(np.int16)).mean())
                examples.append(f"{ea['file']}~{eb['file']} MAE={mae:.2f}")

    if args.workdir is None:
        shutil.rmtree(tmp, ignore_errors=True)

    if not compared:
        print(_fmt(FAIL, "determinism", "no window matched between the two runs "
                                        "(raise --probe-s)"))
        return False
    ok = mismatched == 0
    detail = f"{compared} shared windows, {mismatched} differ"
    if examples:
        detail += "  " + "; ".join(examples)
    print(_fmt(PASS if ok else FAIL, "determinism (bit-exact)", detail))
    return ok


# --------------------------------------------------------------------------
# sharpness
# --------------------------------------------------------------------------

def frame_variances(items, cap):
    step = max(1, len(items) // cap)
    return np.array([read_gray(p)[0].astype(np.float64).var()
                     for p, _t in items[::step]])


def cmd_sharpness(args):
    _man, items = load_frames(args.frames)
    v = frame_variances(items, args.sample)
    print(_fmt(INFO, "sharpness (variance)",
               f"mean={v.mean():.1f}  median={np.median(v):.1f}  "
               f"min={v.min():.1f}  max={v.max():.1f}  n={v.size}"))
    if not args.frames_b:
        return True
    _mb, items_b = load_frames(args.frames_b)
    vb = frame_variances(items_b, args.sample)
    ratio = vb.mean() / v.mean() if v.mean() else float("nan")
    ok = ratio >= 1.0
    print(_fmt(PASS if ok else FAIL, "sharpness vs baseline",
               f"{vb.mean():.1f} vs {v.mean():.1f}  (x{ratio:.3f}) "
               f"-- must be >= 1.0, else the motion model is wrong"))
    return ok


# --------------------------------------------------------------------------
# multi-view consistency
# --------------------------------------------------------------------------

def sift_pair(g1, g2, sift, ratio=0.8):
    """(n_matches, median flow px, F-inlier ratio, H-inlier ratio)."""
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return 0, float("nan"), float("nan"), float("nan"), len(k1 or []), len(k2 or [])
    bf = cv2.BFMatcher(cv2.NORM_L2)
    good = [m for m, n in bf.knnMatch(d1, d2, k=2) if m.distance < ratio * n.distance]
    if len(good) < 8:
        return len(good), float("nan"), float("nan"), float("nan"), len(k1), len(k2)
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    flow = float(np.median(np.linalg.norm(p2 - p1, axis=1)))
    _F, mF = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 2.0, 0.999)
    _H, mH = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
    rF = float(mF.mean()) if mF is not None else float("nan")
    rH = float(mH.mean()) if mH is not None else float("nan")
    return len(good), flow, rF, rH, len(k1), len(k2)


def cmd_flow(args):
    _man, items = load_frames(args.frames)
    sift = cv2.SIFT_create(nfeatures=args.max_features)
    times = np.array([t for _p, t in items])

    rows = []
    for gap in args.gaps:
        pairs = []
        # Spread probe pairs over the take so one bad stretch cannot decide it.
        for i in np.linspace(0, len(items) - 1, args.pairs).astype(int):
            j = int(np.argmin(np.abs(times - (times[i] + gap))))
            if j == i or abs((times[j] - times[i]) - gap) > 0.5 * gap:
                continue
            pairs.append((i, j))
        if not pairs:
            continue
        stats = []
        for i, j in pairs:
            g1, _c = read_gray(items[i][0])
            g2, _c = read_gray(items[j][0])
            stats.append(sift_pair(g1, g2, sift, args.ratio))
        arr = np.array(stats, dtype=np.float64)
        rows.append({
            "gap_s": gap,
            "pairs": len(pairs),
            "kp": float(np.nanmedian(arr[:, 4])),
            "matches": float(np.nanmedian(arr[:, 0])),
            "flow_px": float(np.nanmedian(arr[:, 1])),
            "F_inlier": float(np.nanmedian(arr[:, 2])),
            "H_inlier": float(np.nanmedian(arr[:, 3])),
        })

    if not rows:
        print(_fmt(FAIL, "multi-view consistency", "no usable frame pairs"))
        return False

    print("        gap_s   pairs   keypts   matches   flow_px   F_inl   H_inl")
    for r in rows:
        print(f"        {r['gap_s']:<6.2f}  {r['pairs']:<6d}  {r['kp']:<7.0f}  "
              f"{r['matches']:<8.0f}  {r['flow_px']:<8.2f}  "
              f"{r['F_inlier']:<6.2f}  {r['H_inlier']:<.2f}")

    flows = np.array([r["flow_px"] for r in rows])
    gaps = np.array([r["gap_s"] for r in rows])
    matches = np.array([r["matches"] for r in rows])

    nonzero = np.nanmin(flows) > 1.0
    # "Scales with time separation": the E2VID failure signature was a FLAT
    # ~0.2 px across every gap. Require the widest gap to move meaningfully
    # more than the narrowest -- correlation alone passes on noise.
    scales = len(rows) < 2 or (flows[-1] > 1.5 * flows[0])
    ok_flow = bool(nonzero and scales)
    print(_fmt(PASS if ok_flow else FAIL, "median flow scales w/ gap",
               f"{flows[0]:.2f} px @ {gaps[0]:.2f}s -> {flows[-1]:.2f} px @ "
               f"{gaps[-1]:.2f}s   (E2VID: ~0.2 px, flat)"))

    ok_match = np.nanmedian(matches) >= args.min_matches
    print(_fmt(PASS if ok_match else FAIL, "SIFT matches",
               f"median {np.nanmedian(matches):.0f} per pair "
               f"(need >= {args.min_matches})"))

    inl = np.nanmax([r["H_inlier"] for r in rows])
    ok_inl = not np.isnan(inl) and inl < 0.9
    print(_fmt(PASS if ok_inl else FAIL, "parallax (H inlier < 0.9)",
               f"max H inlier ratio {inl:.2f} -- >=0.9 means degenerate geometry"))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
    return bool(ok_flow and ok_match and ok_inl)


# --------------------------------------------------------------------------
# Phase1b §4.1 -- is one global motion enough?
# --------------------------------------------------------------------------

def cmd_mc_probe(args):
    """Global 2-DOF vs per-tile motion on real windows, before trusting either.

    Reported per sampled window: the variance gain of a single global v over no
    compensation, the extra gain from letting each tile solve its own v, and the
    spread of the per-tile solutions. A large tile spread with a large extra
    gain is depth-dependent parallax -- one v cannot fit the frame and the model
    must go piecewise (or the window must get shorter).
    """
    h5_path, meta_path = _resolve_input(args.input)
    width, height, n_total = resolve_geometry(h5_path, read_meta(meta_path))
    rows, cols = acc.parse_tiles(args.tiles)

    i_hi = n_total
    starts = np.linspace(0, max(0, i_hi - args.events_per_frame),
                         args.samples).astype(int)
    print(f"  {rows}x{cols} tiles, {args.events_per_frame:,} ev windows, "
          f"{args.samples} samples across {n_total:,} events\n")
    print("        t_mid_s   v_global(px/s)     var0     var_g   gain    "
          "var_tile   gain   tile_v_std")

    gains_g, gains_t, spreads = [], [], []
    with h5py.File(h5_path, "r") as f:
        g = f["events"]
        t_first = float(g["t"][0])
        for s in starts:
            e = min(s + args.events_per_frame, i_hi)
            t = g["t"][s:e].astype(np.float64)
            x = g["x"][s:e].astype(np.float64)
            y = g["y"][s:e].astype(np.float64)
            t_ref = 0.5 * (t[0] + t[-1])
            dt = (t - t_ref) / 1e6

            sl = acc.subsample(x.size, args.search_events)
            vx, vy, _s, _z = acc.estimate_motion(
                x[sl], y[sl], dt[sl], width, height, v_init=None,
                vmax=args.vmax, grid=args.grid, levels=args.levels)
            # Scored on ALL the window's events, not the search subsample.
            var_g = float(acc.warp_accumulate(x, y, dt, vx, vy, width, height).var())
            var_0 = float(acc.accumulate(x.astype(np.int64), y.astype(np.int64),
                                         width, height).var())

            tile_id, motions = acc.tile_motions(
                x, y, dt, width, height, rows, cols, (vx, vy),
                args.warm_half, args.grid, args.levels, args.search_events)
            counts = np.zeros((height, width), dtype=np.int64)
            for tid, (tvx, tvy, _n) in motions.items():
                m = tile_id == tid
                if m.any():
                    counts += acc.warp_accumulate(x[m], y[m], dt[m], tvx, tvy,
                                                  width, height)
            var_t = float(counts.var())
            vs = np.array([[v[0], v[1]] for v in motions.values()])
            spread = float(np.hypot(vs[:, 0].std(), vs[:, 1].std()))

            gains_g.append(var_g / var_0)
            gains_t.append(var_t / var_g)
            spreads.append(spread)
            print(f"        {(t_ref - t_first) / 1e6:<8.2f}  "
                  f"({vx:>6.0f},{vy:>6.0f})  {var_0:>8.2f}  {var_g:>8.2f}  "
                  f"x{var_g / var_0:<5.2f}  {var_t:>8.2f}  x{var_t / var_g:<5.2f}  "
                  f"{spread:>6.0f} px/s")

    gg, gt, sp = np.mean(gains_g), np.mean(gains_t), np.mean(spreads)
    print()
    print(_fmt(INFO, "global 2-DOF gain", f"x{gg:.2f} over no compensation"))
    print(_fmt(INFO, f"extra from {rows}x{cols} tiles", f"x{gt:.2f}  "
               f"(tile v spread {sp:.0f} px/s)"))
    verdict = ("one global v is adequate at this window length"
               if gt < 1.15 else
               "PARALLAX: tiles beat global materially -- use --mc-tiles "
               f"{args.tiles} or shorten the window")
    print(_fmt(INFO, "§4.1 verdict", verdict))
    return True


# --------------------------------------------------------------------------
# COLMAP -- the gate
# --------------------------------------------------------------------------

def _model_dirs(base):
    """Model dirs directly under `base`, plus `base` itself, mapper order.

    The mapper writes sparse/0, sparse/1, ... so sorting must be numeric:
    lexicographic order puts sparse/10 between sparse/1 and sparse/2 and the
    fragment table then reads as if the mapper emitted them out of order.
    """
    def key(p):
        b = os.path.basename(p)
        return (0, int(b)) if b.isdigit() else (1, b)

    found = []
    for p in [base] + sorted(glob.glob(os.path.join(base, "*")), key=key):
        for name in ("images.bin", "images.txt"):
            f = os.path.join(p, name)
            if os.path.exists(f):
                found.append((p, os.path.getsize(f)))
                break
    return found


def find_models(scene):
    """[(model_dir, images-file size)] for EVERY reconstruction under the scene."""
    out = []
    for root in ("sparse", "distorted/sparse"):
        out += _model_dirs(os.path.join(scene, root))
    return out


def find_sparse(scene):
    """Largest reconstruction under the scene.

    The mapper writes sparse/0, sparse/1, ... when it cannot join everything
    into one model. This picks the biggest; that is deliberately NOT the whole
    story, and analyze_fragments() below is what tells the rest of it -- a scene
    of twelve healthy fragments reads here as one failed registration.
    """
    cands = find_models(scene)
    return max(cands, key=lambda c: c[1])[0] if cands else None


def model_analyzer(model):
    """Parsed `colmap model_analyzer`, or (None, raw text) when it says nothing."""
    if not shutil.which("colmap"):
        sys.exit("colmap is not on PATH.")
    r = subprocess.run(["colmap", "model_analyzer", "--path", model],
                       capture_output=True, text=True)
    text = r.stdout + r.stderr

    def grab(pat, cast=float):
        m = re.search(pat, text)
        return cast(m.group(1)) if m else None

    stats = {
        "registered": grab(r"Registered images:\s*(\d+)", int),
        "points": grab(r"Points:\s*(\d+)", int),
        "observations": grab(r"Observations:\s*(\d+)", int),
        "track_len_mean": grab(r"Mean track length:\s*([\d.]+)"),
        "obs_per_image": grab(r"Mean observations per image:\s*([\d.]+)"),
        "reproj_err": grab(r"Mean reprojection error:\s*([\d.]+)"),
    }
    return (stats if stats["registered"] is not None else None), text


def model_image_names(model):
    """Registered image file names, via view_sparse.load_model.

    That reader converts a .bin model through `colmap model_converter` rather
    than parsing COLMAP's binary format by hand. It exits on a model with no 3D
    points, which a degenerate fragment can be -- caught here, because one dead
    fragment must not abort the survey of the other eleven.
    """
    try:
        _pts, cams = view_sparse.load_model(model)
    except SystemExit:
        return []
    return [name for _c, _r, name in cams]


def frame_times(frames_dir):
    """({basename: t_mid_s}, n_input, take_span_s) or ({}, 0, None)."""
    if not frames_dir:
        return {}, 0, None
    _man, items = load_frames(frames_dir)
    t_of = {os.path.basename(p): t for p, t in items}
    times = sorted(t for _p, t in items)
    return t_of, len(items), (times[-1] - times[0]) if len(times) > 1 else 0.0


def analyze_fragments(scene, frames_dir=None):
    """Every reconstruction under `scene`, with the time span each one covers.

    THE metric this exists for. cmd_colmap reports the largest model's
    registration rate, so twelve disconnected models with healthy track lengths
    read as one badly-failed registration -- when the real failure is
    CONNECTIVITY: how far a single reconstruction reaches before it breaks.
    Frame names are mapped back to t_mid_s through frames.json, so each
    fragment's reach is reported in seconds of the take rather than in frames.

    Returns {"fragments": [...], "n_input": int, "take_span_s": float|None,
             "largest": dict|None, "union_registered": int}.
    """
    t_of, n_input, take_span = frame_times(frames_dir)
    if not n_input:
        n_input = (len(glob.glob(os.path.join(scene, "input", "*.png"))) or
                   len(glob.glob(os.path.join(scene, "images", "*.png"))))

    frags, union = [], set()
    for model, _size in find_models(scene):
        stats, text = model_analyzer(model)
        if stats is None:
            frags.append({"model": model, "error": text.strip()[-200:]})
            continue
        names = model_image_names(model)
        union |= set(names)
        ts = sorted(t_of[n] for n in names if n in t_of)
        frags.append({
            "model": model,
            "name": os.path.basename(model),
            "registered": stats["registered"],
            "points": stats["points"],
            "track_len_mean": stats["track_len_mean"],
            "obs_per_image": stats["obs_per_image"],
            "reproj_err": stats["reproj_err"],
            "t_start_s": ts[0] if ts else None,
            "t_end_s": ts[-1] if ts else None,
            "span_s": (ts[-1] - ts[0]) if len(ts) > 1 else (0.0 if ts else None),
        })

    good = [f for f in frags if "error" not in f]
    largest = max(good, key=lambda f: f["registered"]) if good else None
    return {"fragments": frags, "n_input": n_input, "take_span_s": take_span,
            "largest": largest, "union_registered": len(union)}


def report_fragments(res, table=True, gates=True):
    """Print the per-fragment table and/or the gates it implies.

    capture_probe.py wants the table but owns the gate rows itself (it prints
    and serialises them from one list), so it passes gates=False rather than
    having every gate appear twice under two slightly different wordings.
    Returns True when all gates pass, whether or not they were printed.
    """
    frags, n_input = res["fragments"], res["n_input"]
    if not frags:
        if gates:
            print(_fmt(FAIL, "fragments", "no sparse model found"))
        return False

    if table:
        print("        model      registered   track   pts      t_start   t_end    span_s")
        for f in frags:
            if "error" in f:
                print(f"        {os.path.basename(f['model']):<10} "
                      f"model_analyzer failed: {f['error']}")
                continue
            ts = "  --  " if f["t_start_s"] is None else f"{f['t_start_s']:>6.2f}"
            te = "  --  " if f["t_end_s"] is None else f"{f['t_end_s']:>6.2f}"
            sp = "  --  " if f["span_s"] is None else f"{f['span_s']:>6.2f}"
            print(f"        {f['name']:<10} {f['registered']:>10}   "
                  f"{f['track_len_mean'] or 0:>5.2f}   {f['points'] or 0:<7} "
                  f"{ts}   {te}   {sp}")
        print()

    n_frag = len(frags)
    ok_one = n_frag == 1
    big = res["largest"]
    if big is None:
        if gates:
            print(_fmt(FAIL, "fragments", f"{n_frag} model(s), none readable"))
        return False
    rate = big["registered"] / n_input if n_input else float("nan")
    ok_big = bool(n_input) and rate >= 0.9
    ok_track = big["track_len_mean"] is not None and big["track_len_mean"] > 3.0

    if gates:
        print(_fmt(PASS if ok_one else FAIL, "fragments", f"{n_frag} "
                   f"reconstruction(s) -- must be 1; more means the mapper "
                   f"could not connect the take"))
        print(_fmt(PASS if ok_big else FAIL, "largest model >= 90%",
                   f"{big['registered']}/{n_input} = {100 * rate:.0f}%  "
                   f"({big['name']})"))
        span, take = big["span_s"], res["take_span_s"]
        if span is not None:
            detail = f"{span:.2f} s"
            if take:
                # Against the FRAME SET's span, not the recording's -- a probe
                # is usually run over a segment, and dividing by the full take
                # length would flatter or damn the mapper for the operator's
                # choice of one.
                detail += (f" of the {take:.2f} s frame set "
                           f"= {100 * span / take:.0f}%")
            print(_fmt(INFO, "span of largest model", detail))
        print(_fmt(PASS if ok_track else FAIL, "mean track length > 3",
                   f"{big['track_len_mean']}   (largest model; E2VID: exactly 2.0)"))
        cov = res["union_registered"] / n_input if n_input else float("nan")
        print(_fmt(INFO, "union over all fragments",
                   f"{res['union_registered']}/{n_input} = {100 * cov:.0f}% of "
                   f"frames appear in SOME model"))
    return bool(ok_one and ok_big and ok_track)


def cmd_fragments(args):
    # A scene dir carries frames.json at its root (accumulate_frames wrote the
    # PNGs into input/), so --scene is the right default for --frames.
    frames = args.frames or args.scene
    if not os.path.exists(os.path.join(frames, "frames.json")):
        frames = None   # spans stay unreported rather than guessed from filenames
    res = analyze_fragments(args.scene, frames)
    print(f"\n  fragments -- {args.scene}\n")
    ok = report_fragments(res)
    print()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)
    return ok


def cmd_colmap(args):
    sparse = args.model or find_sparse(args.scene)
    if not sparse:
        print(_fmt(FAIL, "colmap", f"no sparse model under {args.scene}"))
        return False
    stats, text = model_analyzer(sparse)
    if stats is None:
        print(_fmt(FAIL, "colmap model_analyzer", text.strip()[-300:]))
        return False

    n_input = len(glob.glob(os.path.join(args.scene, "input", "*.png"))) or \
        len(glob.glob(os.path.join(args.scene, "images", "*.png")))
    registered, track = stats["registered"], stats["track_len_mean"]

    rate = registered / n_input if n_input else float("nan")
    print(_fmt(INFO, "colmap model", f"{sparse}  points={stats['points']}  "
               f"obs/img={stats['obs_per_image']}  "
               f"reproj_err={stats['reproj_err']}"))
    # Say how many models exist, always. Reporting only the largest is what let
    # a connectivity failure masquerade as a registration failure.
    n_models = len(find_models(args.scene))
    print(_fmt(INFO if n_models == 1 else FAIL, "reconstructions",
               f"{n_models} under {args.scene}/sparse"
               + ("" if n_models == 1 else
                  "  -- fragmented; run `frame_metrics.py fragments`")))
    ok_reg = n_input and rate >= 0.9
    print(_fmt(PASS if ok_reg else FAIL, "registered >= 90%",
               f"{registered}/{n_input} = {100 * rate:.0f}%  (largest model only)"))
    ok_track = track is not None and track > 3.0
    print(_fmt(PASS if ok_track else FAIL, "mean track length > 3",
               f"{track}   (E2VID: exactly 2.0)"))
    return bool(ok_reg and ok_track)

# --------------------------------------------------------------------------

def cmd_all(args):
    print(f"\n  Phase1b §5 gates -- {args.frames}\n")
    man, items = load_frames(args.frames)
    if man:
        print(_fmt(INFO, "frame set",
                   f"{len(items)} frames  {man['width']}x{man['height']}  "
                   f"{man['events_per_frame']:,} ev/frame  stride {man['stride']:,}  "
                   f"motion_comp={man['motion_comp']}  norm_hi={man['norm_hi']:.3f}"))
    g, ch = read_gray(items[0][0])
    print(_fmt(PASS if ch == 3 else FAIL, "3-channel PNG (Phase1 §3.3)",
               f"{ch} channel(s) -- 1-channel broadcasts silently in l1_loss"))

    results = [("channels", ch == 3)]
    if args.input:
        d = argparse.Namespace(
            input=args.input, events_per_frame=man["events_per_frame"] if man else 50_000,
            norm_hi=man["norm_hi"] if man else 1.0, history_s=args.history_s,
            probe_s=args.probe_s, workdir=None,
            motion_comp=bool(man and man["motion_comp"]),
            mc_tiles="x".join(str(v) for v in (man or {}).get("mc_tiles") or [1, 1]))
        results.append(("determinism", cmd_determinism(d)))
    else:
        print(_fmt(INFO, "determinism", "skipped -- pass --input <recording>"))

    results.append(("sharpness", cmd_sharpness(argparse.Namespace(
        frames=args.frames, frames_b=args.frames_b, sample=args.sample))))
    results.append(("multi-view", cmd_flow(argparse.Namespace(
        frames=args.frames, gaps=args.gaps, pairs=args.pairs, ratio=args.ratio,
        max_features=args.max_features, min_matches=args.min_matches,
        json=args.json))))
    if args.scene:
        results.append(("colmap", cmd_colmap(argparse.Namespace(
            scene=args.scene, model=None))))
    else:
        print(_fmt(INFO, "colmap", "skipped -- pass --scene <dir> after convert.py"))

    hard = [n for n, ok in results if not ok]
    print(f"\n  {'ALL GATES PASSED' if not hard else 'FAILED: ' + ', '.join(hard)}\n")
    return not hard


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("determinism", help="same events, different history")
    d.add_argument("--input", required=True)
    d.add_argument("-n", "--events-per-frame", type=int, default=50_000)
    d.add_argument("--norm-hi", type=float, default=2.0,
                   help="pinned white point; the two runs must share it")
    d.add_argument("--history-s", type=float, default=2.0,
                   help="seconds of prior events run A carries (default 2.0)")
    d.add_argument("--probe-s", type=float, default=2.0,
                   help="seconds of overlap to compare (default 2.0)")
    d.add_argument("--workdir", default=None, help="keep the two runs here")
    d.add_argument("--motion-comp", action="store_true")
    d.add_argument("--mc-tiles", default="1x1")
    d.set_defaults(func=cmd_determinism)

    s = sub.add_parser("sharpness", help="image variance")
    s.add_argument("--frames", required=True)
    s.add_argument("--frames-b", default=None, help="compare this set against --frames")
    s.add_argument("--sample", type=int, default=60)
    s.set_defaults(func=cmd_sharpness)

    f = sub.add_parser("flow", help="SIFT matches, flow vs gap, inlier ratio")
    f.add_argument("--frames", required=True)
    f.add_argument("--gaps", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.8])
    f.add_argument("--pairs", type=int, default=12, help="probe pairs per gap")
    f.add_argument("--ratio", type=float, default=0.8, help="Lowe ratio")
    f.add_argument("--max-features", type=int, default=8192)
    f.add_argument("--min-matches", type=int, default=30)
    f.add_argument("--json", default=None)
    f.set_defaults(func=cmd_flow)

    m = sub.add_parser("mc-probe", help="§4.1 -- global vs per-tile motion")
    m.add_argument("--input", required=True)
    m.add_argument("-n", "--events-per-frame", type=int, default=50_000)
    m.add_argument("--samples", type=int, default=5)
    m.add_argument("--tiles", default="3x3")
    m.add_argument("--vmax", type=float, default=2000.0)
    m.add_argument("--warm-half", type=float, default=400.0)
    m.add_argument("--grid", type=int, default=9)
    m.add_argument("--levels", type=int, default=6)
    m.add_argument("--search-events", type=int, default=30_000)
    m.set_defaults(func=cmd_mc_probe)

    c = sub.add_parser("colmap", help="registration rate + mean track length")
    c.add_argument("--scene", required=True)
    c.add_argument("--model", default=None, help="explicit sparse/0 path")
    c.set_defaults(func=cmd_colmap)

    g = sub.add_parser("fragments", help="every reconstruction + its time span")
    g.add_argument("--scene", required=True)
    g.add_argument("--frames", default=None,
                   help="frame dir with frames.json; without it spans cannot be "
                        "reported in seconds (defaults to --scene)")
    g.add_argument("--json", default=None)
    g.set_defaults(func=cmd_fragments)

    a = sub.add_parser("all", help="every gate, cheapest first")
    a.add_argument("--frames", required=True)
    a.add_argument("--input", default=None, help="recording, enables determinism")
    a.add_argument("--scene", default=None, help="scene dir, enables the COLMAP gate")
    a.add_argument("--frames-b", default=None)
    a.add_argument("--sample", type=int, default=60)
    a.add_argument("--gaps", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.8])
    a.add_argument("--pairs", type=int, default=12)
    a.add_argument("--ratio", type=float, default=0.8)
    a.add_argument("--max-features", type=int, default=8192)
    a.add_argument("--min-matches", type=int, default=30)
    a.add_argument("--history-s", type=float, default=2.0)
    a.add_argument("--probe-s", type=float, default=2.0)
    a.add_argument("--json", default=None)
    a.set_defaults(func=cmd_all)

    args = ap.parse_args()
    ok = args.func(args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
