#!/usr/bin/env python3
"""Convert a ThinkCam raw-event recording -> intensity frames, with no learned model.

This is docs/Phase1b-MotionCompensation.md, and it REPLACES the E2VID stage of
docs/Phase1.md §5 Step 3. Its output is drop-in for that plan's Step 4 onward:
3-channel PNGs in <scene>/input/, then gaussian-splatting/convert.py.

Why not E2VID: its checkpoint is E2VIDRecurrent, whose ConvLSTM hidden state
(external/rpg_e2vid/model/model.py:82-99) carries across windows. At our event
density (0.87 ev/px/s, 12x below E2VID's own reference data) the state dominates
the events -- the same events fed twice, with different preceding history, gave
correlations of 0.05-0.58 and 0-6 SIFT matches. Frames that are not a function of
the scene cannot support multi-view geometry, and COLMAP's mean track length of
exactly 2.0 said so. See Phase1b §0.

The two properties that fix this, both structural rather than tuned:

  * DETERMINISTIC. A frame depends only on the events inside its own window and
    on constants fixed once per take. Same viewpoint -> same appearance -> SIFT
    features persist across views. Verified by the frame_metrics determinism
    gate.
  * SCALE-FREE. No training distribution, so 1280x720 is not out-of-regime and
    no --downsample is needed.

Three decisions are load-bearing (Phase1b §3); none is obvious, all are easy to
undo by accident:

  * COUNT, NOT SIGNED POLARITY. An edge crossed left-to-right fires the opposite
    polarity to the same edge crossed right-to-left, so on an orbit a signed
    accumulation makes one physical edge bright in one view and dark in another
    -- destroying the very multi-view consistency this exists to provide. We
    accumulate |events| per pixel and ignore the p column entirely. (ThinkCam
    stores p in {0,1}, thinkcam/raw_recorder.py:96, encoding '0_neg_1_pos'.)
  * GLOBAL NORMALISATION. Per-frame min/max stretching makes appearance depend
    on window content, which reintroduces view-dependence through the back door.
    One percentile pair is computed once over the take (--norm-sample-frames
    windows spread across it) and applied unchanged to every frame. Printed at
    runtime, and re-injectable with --norm-hi for a bit-exact rerun.
  * WINDOW BY EVENT COUNT, NOT TIME. Event rate varies 4x over an orbit; a fixed
    time window would vary in density and therefore in appearance.

Windows are independent, so unlike E2VID they may overlap: --stride below
--events-per-frame gives COLMAP more frames and longer tracks for compute only.

Motion compensation (--motion-comp, Phase1b §2 Step 2) is contrast maximization
(Gallego et al., A Unifying Contrast Maximization Framework for Event Cameras):
warp the events by a candidate motion, accumulate, score sharpness as image
variance, keep the motion that maximises it. It is OFF by default -- Phase1b §4.2
requires measuring plain accumulation first and only paying for the optimiser if
the gates fail.

Memory: streams via export_e2vid_input.event_chunks(). The 64 s take is 51.2 M
events / 676 MB and must never be materialised whole -- do not switch this to
convert_to_inceventgs.load_events().
"""

import argparse
import json
import os
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv-python is required: pip install opencv-python")

try:
    import h5py
except ImportError:
    sys.exit("h5py is required: pip install h5py")

from pipeline.convert_to_inceventgs import _resolve_input
from pipeline.export_e2vid_input import (
    READ_CHUNK,
    event_chunks,
    read_meta,
    resolve_geometry,
)

# Percentile of the NON-ZERO count values used as the white point. Zeros are
# excluded because at 0.87 ev/px/s a short window touches ~2% of the sensor --
# a percentile over all pixels would be pinned to 0 and every frame would clip.
NORM_PERCENTILE = 99.0


# --------------------------------------------------------------------------
# accumulation
# --------------------------------------------------------------------------

def accumulate(x, y, width, height):
    """Unsigned event count per pixel. p is deliberately not read (docstring)."""
    idx = y.astype(np.int64) * width + x.astype(np.int64)
    counts = np.bincount(idx, minlength=width * height)
    return counts.reshape(height, width)


def density_stats(x, y, width, height):
    """Events per LIT pixel, and how much of the sensor is lit.

    The quantity Phase1b-Results.md §9.1 sets its >= 5 target on, which had no
    implementation anywhere -- the 0.87 ev/px/s and 1.95 ev/active-px figures in
    the wiki were computed ad hoc and left no code behind.

    "Lit" means hit at least once in THIS window, not "active over the take":
    dividing by the whole sensor gives events/pixel, which on a 2.5%-lit window
    is 40x smaller and answers a different question. The reading depends
    strongly on window length (1.07 at 25 k events, 2.27 at 2 s on the reference
    take) because a moving camera paints new pixels rather than revisiting old
    ones -- so a value is meaningless without the window it was measured over.

    multi_hit_share counts EVENTS landing on a pixel that fired more than once,
    as a share of all events -- not the share of pixels -- which is the
    definition the numbers in the wiki and in plans/01_CaptureDashboard.md use.
    """
    x = np.asarray(x).astype(np.int64, copy=False)
    y = np.asarray(y).astype(np.int64, copy=False)
    # Live camera batches are float32 straight off the wire and are not
    # guaranteed in range; a stray coordinate would silently lengthen the
    # bincount and blow up the reshape.
    ok = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not ok.all():
        x, y = x[ok], y[ok]
    n_px = width * height
    if x.size == 0:
        return {"events": 0, "lit_pixels": 0, "ev_per_lit_px": 0.0,
                "lit_coverage": 0.0, "multi_hit_share": 0.0}
    counts = accumulate(x, y, width, height)
    total = int(counts.sum())
    lit = int(np.count_nonzero(counts))
    multi = int(counts[counts > 1].sum())
    return {
        "events": total,
        "lit_pixels": lit,
        "ev_per_lit_px": total / lit if lit else 0.0,
        "lit_coverage": lit / n_px,
        "multi_hit_share": multi / total if total else 0.0,
    }


def warp_accumulate(x, y, dt_s, vx, vy, width, height, bin_px=1):
    """Accumulate after warping each event back to the window's reference time.

    x' = x - (t - t_ref) * vx, with v in px/s and dt_s = (t - t_ref) in seconds.
    Events warped outside the sensor are dropped, which costs image mass and so
    costs variance -- the objective is self-limiting against runaway motions.

    bin_px > 1 accumulates into a coarser grid. That is not a downsample of the
    output; it is the search's spatial scale (see estimate_motion).
    """
    w = (width + bin_px - 1) // bin_px
    h = (height + bin_px - 1) // bin_px
    xw = np.rint((x - dt_s * vx) / bin_px)
    yw = np.rint((y - dt_s * vy) / bin_px)
    keep = (xw >= 0) & (xw < w) & (yw >= 0) & (yw < h)
    idx = yw[keep].astype(np.int64) * w + xw[keep].astype(np.int64)
    counts = np.bincount(idx, minlength=w * h)
    return counts.reshape(h, w)


def normalise(counts, norm_hi):
    """Fixed 0..norm_hi -> 0..255 mapping. Identical for every frame of a take."""
    scaled = np.clip(counts * (255.0 / norm_hi), 0, 255)
    return scaled.astype(np.uint8)


def to_bgr(gray_u8):
    """3-channel, per Phase1 §3.3: a 1-channel ground truth broadcasts silently
    against a 3-channel render in l1_loss -- no error, just a wrong loss."""
    return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)


# --------------------------------------------------------------------------
# contrast maximization (Step 2)
# --------------------------------------------------------------------------

def contrast_score(x, y, dt_s, vx, vy, width, height, bin_px=1):
    return float(warp_accumulate(x, y, dt_s, vx, vy, width, height, bin_px).var())


def estimate_motion(x, y, dt_s, width, height, v_init=None,
                    vmax=2000.0, warm_half=400.0, grid=9, levels=6,
                    bin_cap=32):
    """Coarse-to-fine grid search for the 2-DOF global translation (vx, vy).

    THE SPATIAL BINNING IS NOT OPTIONAL, and this is the one thing about
    contrast maximization that a plain grid search gets wrong. The objective's
    peak is only as wide in velocity as an edge is in pixels: an edge ~1 px wide
    over a half-window of 0.12 s is a peak ~8 px/s across. A coarse grid
    stepping 500 px/s therefore steps straight over it, every time, and the
    search returns v ~ 0 with a variance gain of ~1.00 -- measured, and it is
    exactly what a scene with no motion would return, so it is not
    self-announcing. Each level accumulates into bins as wide as that level's
    step displaces an event at the window edge, which makes the peak as wide as
    the grid is coarse. With binning the same windows solve to 500-2200 px/s at
    gains up to x2.2.

    Two parameters do not justify a differentiable warp: a grid is robust to the
    objective's local maxima (event alignment is periodic on repeated texture)
    and needs no gradient of a scatter-add.

    v_init warm-starts the search from the previous window's solution, which is
    a large speedup and is what Phase1b §2 Step 2 suggests -- but it MAKES THE
    OUTPUT DEPEND ON PRECEDING WINDOWS, and it was measured failing the §5
    determinism gate at MAE 27/255. That is E2VID's defect in miniature, so the
    caller must opt in (--mc-warm-start) and accept the gate failure.

    Returns (vx, vy, score, score_at_zero), both scores at full resolution.
    """
    dt_max = float(np.abs(dt_s).max()) if dt_s.size else 0.0
    if dt_max <= 0:
        return 0.0, 0.0, contrast_score(x, y, dt_s, 0, 0, width, height), 1.0

    if v_init is None:
        bx, by, half = 0.0, 0.0, float(vmax)
    else:
        bx, by, half = float(v_init[0]), float(v_init[1]), float(warm_half)

    for _ in range(levels):
        step = 2.0 * half / (grid - 1)
        bin_px = int(np.clip(np.ceil(step * dt_max), 1, bin_cap))
        axis = np.linspace(-half, half, grid)
        best_s, best_v = -1.0, (bx, by)
        for dvx in axis:
            for dvy in axis:
                v = (bx + dvx, by + dvy)
                s = contrast_score(x, y, dt_s, v[0], v[1], width, height, bin_px)
                if s > best_s:
                    best_s, best_v = s, v
        bx, by = best_v
        half = step  # next level searches one cell of this one
        if step * dt_max < 0.5:  # finer than a pixel at the window edge
            break

    return (bx, by,
            contrast_score(x, y, dt_s, bx, by, width, height),
            contrast_score(x, y, dt_s, 0.0, 0.0, width, height))


def subsample(n, cap):
    """Deterministic stride subsample of indices, for cheap objective evals."""
    if cap <= 0 or n <= cap:
        return slice(None)
    return slice(0, n, int(np.ceil(n / cap)))


def tile_motions(x, y, dt_s, width, height, rows, cols, v_global,
                 warm_half, grid, levels, search_cap):
    """Per-tile (vx, vy), seeded from THIS window's global fit. Phase1b §4.1.

    The seed comes from the same window, never from the previous one, so tiles
    do not reintroduce the cross-window dependence that --mc-warm-start does.

    An orbit produces depth-dependent parallax, so one motion sharpens the
    dominant depth and smears the rest. Tiles approximate a piecewise-constant
    flow field; the tile a event belongs to is decided by its ORIGINAL position,
    so the partition itself does not depend on the motion being solved for.
    """
    tw, th = width / cols, height / rows
    col = np.clip((x / tw).astype(np.int32), 0, cols - 1)
    row = np.clip((y / th).astype(np.int32), 0, rows - 1)
    tile_id = row * cols + col
    out = {}
    for tid in range(rows * cols):
        m = tile_id == tid
        n = int(m.sum())
        if n < 2000:  # too few events to fit a motion; fall back to global
            out[tid] = (v_global[0], v_global[1], n)
            continue
        xs, ys, ds = x[m], y[m], dt_s[m]
        sl = subsample(n, search_cap)
        vx, vy, _, _ = estimate_motion(
            xs[sl], ys[sl], ds[sl], width, height, v_init=v_global,
            warm_half=warm_half, grid=grid, levels=levels)
        out[tid] = (vx, vy, n)
    return tile_id, out


def compensated_frame(x, y, dt_s, width, height, args, v_prev):
    """One motion-compensated frame. Returns (counts, info-dict)."""
    n = x.size
    sl = subsample(n, args.mc_search_events)
    vx, vy, _s, _z = estimate_motion(
        x[sl], y[sl], dt_s[sl], width, height, v_init=v_prev,
        vmax=args.mc_vmax, warm_half=args.mc_warm_half,
        grid=args.mc_grid, levels=args.mc_levels)

    # Score on ALL the window's events, never on the search subsample: a
    # subsample is sparser, so its variance ratio is not the delivered frame's
    # and reading it as one understates the gain (measured 0.99 vs 1.06).
    counts = warp_accumulate(x, y, dt_s, vx, vy, width, height)
    var_0 = float(accumulate(x.astype(np.int64), y.astype(np.int64),
                             width, height).var())
    info = {"vx": round(vx, 2), "vy": round(vy, 2),
            "var_gain": round(float(counts.var()) / var_0, 4) if var_0 else None}

    if args.mc_tiles == (1, 1):
        return counts, info

    rows, cols = args.mc_tiles
    tile_id, motions = tile_motions(
        x, y, dt_s, width, height, rows, cols, (vx, vy),
        args.mc_warm_half, args.mc_grid, args.mc_levels, args.mc_search_events)
    counts = np.zeros((height, width), dtype=np.int64)
    for tid, (tvx, tvy, _n) in motions.items():
        m = tile_id == tid
        if not m.any():
            continue
        counts += warp_accumulate(x[m], y[m], dt_s[m], tvx, tvy, width, height)
    vs = np.array([[v[0], v[1]] for v in motions.values()])
    info["tile_v_std"] = [round(float(vs[:, 0].std()), 1),
                          round(float(vs[:, 1].std()), 1)]
    return counts, info


# --------------------------------------------------------------------------
# windowing
# --------------------------------------------------------------------------

def index_bounds(h5_path, t0, t1):
    """Event-index range for a time range, by binary search on /events/t.

    Reads O(log n) single elements rather than the 409 MB t column of the 64 s
    take. t may have a handful of non-monotonic steps (export_e2vid_input warns
    about them); a few misplaced events at the boundary do not matter here.
    """
    with h5py.File(h5_path, "r") as f:
        t = f["events"]["t"]
        n = t.shape[0]

        def search(target):
            lo, hi = 0, n
            while lo < hi:
                mid = (lo + hi) // 2
                if float(t[mid]) < target:
                    lo = mid + 1
                else:
                    hi = mid
            return lo

        return (search(t0) if t0 is not None else 0,
                search(t1) if t1 is not None else n)


def sample_windows(h5_path, i_lo, i_hi, n_events, k):
    """k windows spread evenly over [i_lo, i_hi), read by direct slicing.

    Used only for the global normalisation constant, so it must not depend on
    which subset of the take is being exported -- it spans the whole range.
    """
    span = i_hi - i_lo - n_events
    if span <= 0:
        starts = [i_lo]
    else:
        starts = [int(i_lo + round(f)) for f in np.linspace(0, span, k)]
    with h5py.File(h5_path, "r") as f:
        g = f["events"]
        for s in starts:
            e = min(s + n_events, i_hi)
            yield g["x"][s:e], g["y"][s:e]


def compute_norm_hi(h5_path, i_lo, i_hi, width, height, n_events, k, pct):
    """The take-wide white point: the pct-th percentile of non-zero counts."""
    vals = []
    for x, y in sample_windows(h5_path, i_lo, i_hi, n_events, k):
        c = accumulate(x, y, width, height)
        nz = c[c > 0]
        if nz.size:
            vals.append(np.percentile(nz, pct))
    if not vals:
        sys.exit("Normalisation pass found no events -- check --start-s/--duration-s.")
    return float(max(1.0, np.median(vals)))


def windows(h5_path, chunk, n_events, stride, t0, t1, i_lo=0, i_first=None):
    """Yield (i_start, t, x, y) per window, streaming. Buffers one window + one chunk.

    Windows are cut by event COUNT and may overlap (stride < n_events): unlike
    E2VID's sequential hidden state there is nothing carried between windows, so
    overlap costs compute and nothing else.

    i_start is the window's absolute index into /events, so two runs over
    different sub-ranges of the same take can be compared window-for-window --
    that is what the Phase1b §5 determinism gate needs. i_first pins the FIRST
    window to an absolute index, which is how two runs are made to share a
    window grid despite starting at different times (timestamps tie in
    microseconds, so a time offset alone cannot align the grids).
    """
    bt = bx = by = None
    consumed = i_lo  # absolute index of buffer element 0
    pending = 0      # events still to skip, beyond what the buffer holds
    if i_first is not None:
        pending = max(0, i_first - i_lo)
    for t, x, y, _p in event_chunks(h5_path, chunk, t0, t1):
        if bt is None:
            bt, bx, by = t, x, y
        else:
            bt = np.concatenate((bt, t))
            bx = np.concatenate((bx, x))
            by = np.concatenate((by, y))
        if pending:
            # A stride longer than the window, or an i_first past this chunk,
            # skips events that have not been read yet -- carry the remainder
            # rather than advancing `consumed` past the buffer.
            drop = min(bt.size, pending)
            bt, bx, by = bt[drop:], bx[drop:], by[drop:]
            consumed += drop
            pending -= drop
            if pending:
                continue
        cursor = 0
        while bt.size - cursor >= n_events:
            s = slice(cursor, cursor + n_events)
            yield consumed + cursor, bt[s], bx[s], by[s]
            cursor += stride
        if cursor:
            drop = min(cursor, bt.size)
            bt, bx, by = bt[drop:], bx[drop:], by[drop:]
            consumed += drop
            pending = cursor - drop


# --------------------------------------------------------------------------

def parse_tiles(s):
    if s in (None, "", "1x1"):
        return (1, 1)
    try:
        r, c = s.lower().split("x")
        return (int(r), int(c))
    except Exception:
        raise argparse.ArgumentTypeError("--mc-tiles wants RxC, e.g. 3x3")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="recordings/<session>/events.h5 or the session dir")
    ap.add_argument("--out", required=True,
                    help="output dir; PNGs land in <out>/ (point convert.py at "
                         "its parent when this is a scene's input/)")
    ap.add_argument("--events-per-frame", type=int, default=50_000, metavar="N",
                    help="window length in events (default 50,000). The central "
                         "trade-off: short is noisy, long is motion-blurred. "
                         "Sweep 20k/50k/100k/200k and pick on frame_metrics "
                         "numbers, not by eye.")
    ap.add_argument("--stride", type=int, default=None, metavar="N",
                    help="events between window starts (default: no overlap)")
    ap.add_argument("--start-s", type=float, default=None,
                    help="drop events before this offset, in seconds from take start")
    ap.add_argument("--duration-s", type=float, default=None,
                    help="keep only this many seconds after --start-s")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="stop after this many frames")
    ap.add_argument("--start-event", type=int, default=None, metavar="I",
                    help="pin the first window to this absolute event index. "
                         "Only needed to make two runs share a window grid for "
                         "the §5 determinism gate; frame_metrics sets it.")
    ap.add_argument("--norm-hi", type=float, default=None,
                    help="white point in events/pixel. Default: measured over "
                         "the take. Pass the printed value for a bit-exact rerun.")
    ap.add_argument("--norm-sample-frames", type=int, default=40, metavar="K",
                    help="windows sampled across the take for the white point")
    ap.add_argument("--norm-percentile", type=float, default=NORM_PERCENTILE,
                    help=f"percentile of non-zero counts (default {NORM_PERCENTILE})")
    ap.add_argument("--chunk", type=int, default=READ_CHUNK,
                    help=f"events per streaming read (default {READ_CHUNK:,})")
    ap.add_argument("--prefix", default="f", help="PNG name prefix (default 'f')")

    mc = ap.add_argument_group("motion compensation (Phase1b §2 Step 2)")
    mc.add_argument("--motion-comp", action="store_true",
                    help="contrast-maximise each window before accumulating. "
                         "Off by default: §4.2 says measure Step 1 first.")
    mc.add_argument("--mc-vmax", type=float, default=2000.0, metavar="PX_S",
                    help="cold-start search half-range in px/s (default 2000; "
                         "this take solves to 500-2200 px/s)")
    mc.add_argument("--mc-warm-start", action="store_true",
                    help="seed each window's search from the previous window's "
                         "solution. ~2x faster and BREAKS THE §5 DETERMINISM "
                         "GATE (measured: MAE 27/255 between runs with "
                         "different history) -- the frame stops being a function "
                         "of its own events alone, which is what sank E2VID. "
                         "Off by default; for previews only.")
    mc.add_argument("--mc-warm-half", type=float, default=400.0, metavar="PX_S",
                    help="warm-started search half-range in px/s (default 400)")
    mc.add_argument("--mc-grid", type=int, default=9,
                    help="samples per axis per level (default 9)")
    mc.add_argument("--mc-levels", type=int, default=6,
                    help="coarse-to-fine levels (default 6; the search stops "
                         "early once a step moves under half a pixel)")
    mc.add_argument("--mc-search-events", type=int, default=30_000, metavar="N",
                    help="cap on events used to SCORE a candidate motion; the "
                         "final frame always uses all of them (default 30,000)")
    mc.add_argument("--mc-tiles", type=parse_tiles, default=(1, 1), metavar="RxC",
                    help="piecewise-constant motion on an RxC tile grid instead "
                         "of one global v. Phase1b §4.1: an orbit has "
                         "depth-dependent parallax, which one v cannot fit. "
                         "Confirm the need with `frame_metrics mc-probe`.")
    return ap


def main():
    args = build_parser().parse_args()
    stride = args.stride or args.events_per_frame
    if stride <= 0 or args.events_per_frame <= 0:
        sys.exit("--events-per-frame and --stride must be positive.")

    h5_path, meta_path = _resolve_input(args.input)
    meta = read_meta(meta_path)
    width, height, n_total = resolve_geometry(h5_path, meta)

    unit = meta.get("timestamp_unit")
    if unit and unit != "microseconds":
        sys.exit(f"Refusing: metadata timestamp_unit is {unit!r}, not 'microseconds'. "
                 "See docs/Phase1.md §6 -- this take predates hardware verification.")

    with h5py.File(h5_path, "r") as f:
        t_first = float(f["events"]["t"][0])
    t0 = t_first + args.start_s * 1e6 if args.start_s is not None else None
    t1 = None
    if args.duration_s is not None:
        t1 = (t0 if t0 is not None else t_first) + args.duration_s * 1e6

    os.makedirs(args.out, exist_ok=True)

    i_lo, i_hi = index_bounds(h5_path, t0, t1)
    if args.norm_hi is not None:
        norm_hi, norm_src = args.norm_hi, "given"
    else:
        norm_hi = compute_norm_hi(h5_path, i_lo, i_hi, width, height,
                                  args.events_per_frame, args.norm_sample_frames,
                                  args.norm_percentile)
        norm_src = (f"measured over {args.norm_sample_frames} windows, "
                    f"p{args.norm_percentile:g} of non-zero counts")

    print(f"\n  {os.path.basename(os.path.dirname(h5_path))}  "
          f"{n_total:,} events  {width}x{height}")
    print(f"  window={args.events_per_frame:,} ev  stride={stride:,} ev  "
          f"overlap={100 * (1 - stride / args.events_per_frame):.0f}%")
    print(f"  accumulating UNSIGNED counts (p column ignored)")
    print(f"  norm_hi={norm_hi:.3f} ev/px ({norm_src})  -> global, per-take")
    if args.motion_comp:
        r, c = args.mc_tiles
        print(f"  motion-comp ON  model={'global 2-DOF' if (r, c) == (1, 1) else f'{r}x{c} tiles'}"
              f"  grid={args.mc_grid}^2 x {args.mc_levels} levels"
              + ("  WARM START (breaks the §5 determinism gate)"
                 if args.mc_warm_start else "  cold search (deterministic)"))
    print()

    index = []
    v_prev = None
    t_started = time.time()
    n_frames = 0
    for i_start, t, x, y in windows(h5_path, args.chunk, args.events_per_frame,
                                    stride, t0, t1, i_lo, args.start_event):
        t_ref = 0.5 * (float(t[0]) + float(t[-1]))  # midpoint: halves max warp
        entry = {
            "frame": n_frames,
            "file": f"{args.prefix}{n_frames:06d}.png",
            "i_start": int(i_start),
            "i_end": int(i_start + t.size),
            "t_start_s": (float(t[0]) - t_first) / 1e6,
            "t_end_s": (float(t[-1]) - t_first) / 1e6,
            "t_mid_s": (t_ref - t_first) / 1e6,
            "n_events": int(t.size),
        }
        if args.motion_comp:
            counts, info = compensated_frame(
                x, y, (t - t_ref) / 1e6, width, height, args,
                v_prev if args.mc_warm_start else None)
            v_prev = (info["vx"], info["vy"])
            entry.update(info)
        else:
            counts = accumulate(x, y, width, height)

        img = to_bgr(normalise(counts, norm_hi))
        cv2.imwrite(os.path.join(args.out, entry["file"]), img)
        index.append(entry)
        n_frames += 1
        if n_frames % 25 == 0:
            rate = n_frames / (time.time() - t_started)
            print(f"    {n_frames} frames  ({rate:.1f} fps)"
                  + (f"  v=({v_prev[0]:.0f},{v_prev[1]:.0f}) px/s" if v_prev else ""))
        if args.max_frames and n_frames >= args.max_frames:
            break

    if not n_frames:
        sys.exit("No frames written -- window longer than the selected event range?")

    manifest = {
        "source": os.path.abspath(h5_path),
        "width": width, "height": height,
        "events_per_frame": args.events_per_frame, "stride": stride,
        "norm_hi": norm_hi, "norm_percentile": args.norm_percentile,
        "polarity": "ignored (unsigned counts)",
        "motion_comp": bool(args.motion_comp),
        "mc_tiles": list(args.mc_tiles) if args.motion_comp else None,
        "start_s": args.start_s, "duration_s": args.duration_s,
        "frames": index,
    }
    with open(os.path.join(args.out, "frames.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    span = index[-1]["t_end_s"] - index[0]["t_start_s"]
    print(f"\n  Wrote {n_frames} frames -> {args.out}  "
          f"({n_frames / span:.2f} fps over {span:.2f} s)")
    print(f"  3-channel PNGs (Phase1 §3.3), {width}x{height}, "
          f"global norm_hi={norm_hi:.3f}")
    print(f"  Reproduce bit-exactly with:  --norm-hi {norm_hi:.6f}")
    print(f"  elapsed {time.time() - t_started:.1f} s\n")
    print("  Next: python -m pipeline.frame_metrics all --frames "
          f"{args.out}  (Phase1b §5 gates)\n")


if __name__ == "__main__":
    main()
