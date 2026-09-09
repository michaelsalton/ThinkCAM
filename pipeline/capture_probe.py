#!/usr/bin/env python3
"""Score a finished take end to end: frames -> LK tracks -> COLMAP -> gates.

This closes the capture loop that plans/01_CaptureDashboard.md §0 describes as
missing. Phase1b-Results.md §9 ends with one instruction -- re-capture with bias
changes, then re-run the gates -- and nothing in the repo ran the gates as one
command. Every number below existed only as a hand-assembled sequence of tool
invocations, or (events per lit pixel) not at all.

    python -m pipeline.capture_probe --input recordings/<session> \
        [--out work/probe/<take>]

WHAT IS PINNED, AND WHY. The window is 25,000 events with motion compensation
ON, from thinkcam/constants.py. Those are not defaults chosen here; they are
work/phase1b/tracked_mc25k, the best configuration on record. This is a
correction to Phase1b-Results.md, which used 200 k and concluded compensation
was not worth 1.7x the cost -- true for the SIFT path it measured, false for the
LK path in track_frames that replaced it. Two probes are only comparable if
they share this configuration, so --events-per-frame exists but moves the run
off the reference.

WHAT THE GATE TABLE IS ACTUALLY WATCHING. It is no longer track length. On
tracked_mc25k the LK path reaches mean track length 3.07-4.37, clearing the
Phase1b §5 gate -- and the mapper still emits TWELVE disconnected models, the
largest spanning 1.04 s of a 64 s capture. The binding constraint is how far one
reconstruction reaches before it breaks, so `fragments` and `span of largest
model` lead the table and `mean track length` trails it.

TWO WINDOWS FOR EVENTS PER LIT PIXEL, measured on this repo's reference take:

    25 k events (the pipeline window)  0.026 s   2.5% lit   1.07 ev/lit-px
    200 k events (the SIFT-era window) 0.244 s  17.9% lit   1.21 ev/lit-px
    2.0 s (the reference window)       2.000 s  83.6% lit   2.27 ev/lit-px

A 76x longer window buys 2x the density, because a moving camera paints a wider
swath rather than building up the same pixels (Phase1b-Results.md §7's closing
finding, now reproducible in one command). So Phase1b-Results.md §9.1's ">= 5"
target is unreachable BY CONSTRUCTION at the pipeline window and says nothing
there; it is hung on the 2 s reference window instead. §9.1 states no window of
its own -- if 2 s is not the intent, the target needs revisiting, not this code.

LEDGER of what was measured while building this, against
recordings/20260602_102004_demo_scene_orbit_1 --start-s 12 --duration-s 6:

  * The wiki's 1.95 "ev/active-px" and the plan's 2.15 both correspond to the
    1.834 s / 1.6 M-event window, not to a 2.0 s one. At a true 2.0 s window the
    same position reads 2.27. Expect 2.27, not 2.15.
  * multi_hit_share is a share of EVENTS on multiply-hit pixels, not of pixels
    (6.0% of pixels vs 12.1% of events at the 25 k window). The wiki's figures
    are the event share.
  * /usr/bin/colmap 3.12.6 is built WITHOUT CUDA, so the mapper is CPU-only
    regardless of the RTX 5080. The plan budgets ~13 minutes for that; measured
    here, a 200-frame probe is 180 s end to end (40 s accumulate, ~15 s LK
    tracking, the rest COLMAP). The 13-minute figure was the SIFT path's, and
    this path never runs feature_extractor at all. Still slow enough to want
    the progress line, nowhere near slow enough to abandon the GUI over.
  * work/phase1b/tracked_mc25k WAS NOT RUN AT track_frames' DEFAULTS. It used
    --max-gap 12; the default is 10. Bit-identical frames and bit-identical LK
    tracks (81,823 keypoints either way, and pair counts that agree exactly for
    every gap 1-10) then diverge at the mapper: gap 10 gives 7 fragments,
    largest 24 frames / 0.63 s, 55% union coverage, while gap 12 gives 12
    fragments, largest 37 frames / 1.04 s, 84% union coverage. PROBE_MAX_GAP
    pins 12. Nothing recorded this anywhere; it was recovered by decoding
    pair_ids out of that run's db.db.

Runs on the analysis interpreter (numpy/h5py/cv2 + the colmap binary), never the
GUI venv; thinkcam/constants.py PROBE_PYTHON is the path the GUI shells out to.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py is required: pip install h5py")

from pipeline import accumulate_frames as acc
from pipeline import child_env
from pipeline import frame_metrics as fm
from pipeline.convert_to_inceventgs import _resolve_input
from pipeline.export_e2vid_input import read_meta, resolve_geometry
from thinkcam.constants import (
    PROBE_EV_PER_LIT_TARGET,
    PROBE_EVENTS_PER_FRAME,
    PROBE_MAX_GAP,
    PROBE_MOTION_COMP,
    PROBE_MOTION_SEARCH_EVENTS,
    PROBE_REFERENCE_S,
    PROBE_STRIDE,
    PROBE_WORK_DIR,
)

PROBE_VERSION = 1


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------

class Gates:
    """The report table and probe.json's `gates` array, built once.

    Printing and serialising from the same list is the point: the GUI panel
    renders whatever rows are here without knowing what any of them mean, so a
    new metric shows up in both places or in neither.
    """

    def __init__(self):
        self.rows = []

    def add(self, name, value, detail, target=None, ok=None):
        status = fm.INFO if ok is None else (fm.PASS if ok else fm.FAIL)
        self.rows.append({"name": name, "value": value, "detail": detail,
                          "target": target,
                          "status": "INFO" if ok is None else status})
        print(fm._fmt(status, name, detail))

    @property
    def passed(self):
        return not any(r["status"] == fm.FAIL for r in self.rows)


# --------------------------------------------------------------------------
# density -- the new metric (§3)
# --------------------------------------------------------------------------

def _index_at(t_dset, target_us, lo, hi):
    """First index in [lo, hi) whose timestamp reaches target_us.

    Binary search on the open dataset rather than acc.index_bounds, which
    reopens the file per call -- this runs once per sampled window.
    """
    while lo < hi:
        mid = (lo + hi) // 2
        if float(t_dset[mid]) < target_us:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _aggregate(rows):
    """Median of each numeric field across sampled windows."""
    if not rows:
        return {}
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    out = {k: float(np.median([r[k] for r in rows])) for k in keys}
    out["windows"] = len(rows)
    return out


def measure_density(h5_path, i_lo, i_hi, width, height,
                    pipeline_n, reference_s, samples):
    """ev/lit-px at BOTH windows, from window starts spread over the segment.

    Sampled rather than read from the first window alone: event rate varies ~4x
    over an orbit (accumulate_frames' docstring), so a single position measures
    that position. The reference window is cut by TIME and the pipeline window
    by EVENT COUNT, because that is what each one is defined as -- the pipeline
    window has to be the same window the frames were built from, and the
    reference window has to be comparable between takes at different rates.
    """
    pipe, ref = [], []
    with h5py.File(h5_path, "r") as f:
        g = f["events"]
        t = g["t"]
        i_hi = min(i_hi, t.shape[0])
        t_lo, t_hi = float(t[i_lo]), float(t[i_hi - 1])
        span_s = (t_hi - t_lo) / 1e6
        ref_s = min(reference_s, span_s)
        truncated = ref_s < reference_s

        # Last start at which a full reference window still fits; the pipeline
        # window rides the same starts so both readings describe one position.
        last = max(i_lo, _index_at(t, t_hi - ref_s * 1e6, i_lo, i_hi))
        starts = np.unique(np.linspace(i_lo, last, samples).astype(np.int64))

        for s in starts:
            e = min(int(s) + pipeline_n, i_hi)
            if e - s < pipeline_n // 2:
                continue
            d = acc.density_stats(g["x"][s:e], g["y"][s:e], width, height)
            d["window_s"] = (float(t[e - 1]) - float(t[s])) / 1e6
            pipe.append(d)

            e = min(_index_at(t, float(t[s]) + ref_s * 1e6, int(s), i_hi), i_hi)
            d = acc.density_stats(g["x"][s:e], g["y"][s:e], width, height)
            d["window_s"] = (float(t[e - 1]) - float(t[s])) / 1e6
            ref.append(d)

    return {"pipeline": _aggregate(pipe), "reference": _aggregate(ref),
            "reference_s": ref_s, "reference_truncated": truncated,
            "segment_s": span_s}


def measure_speed(h5_path, i_lo, i_hi, width, height, n_events, samples):
    """Median camera speed in px/s, for a run with compensation turned off.

    With --motion-comp on, accumulate_frames already solves (vx, vy) for every
    frame and writes them to frames.json -- every frame beats a sample, and it
    is free. This path exists only for --no-motion-comp.
    """
    speeds = []
    with h5py.File(h5_path, "r") as f:
        g = f["events"]
        i_hi = min(i_hi, g["t"].shape[0])
        for s in np.linspace(i_lo, max(i_lo, i_hi - n_events), samples).astype(np.int64):
            e = min(int(s) + n_events, i_hi)
            if e - s < 1000:
                continue
            t = g["t"][s:e].astype(np.float64)
            x = g["x"][s:e].astype(np.float64)
            y = g["y"][s:e].astype(np.float64)
            dt = (t - 0.5 * (t[0] + t[-1])) / 1e6
            sl = acc.subsample(x.size, PROBE_MOTION_SEARCH_EVENTS)
            vx, vy, _s, _z = acc.estimate_motion(
                x[sl], y[sl], dt[sl], width, height, v_init=None)
            speeds.append(float(np.hypot(vx, vy)))
    return float(np.median(speeds)) if speeds else float("nan")


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------

def run_step(cmd, log_path, label):
    """Run a tool, tee stdout+stderr to a log, and stream it. Exits on failure.

    Streamed line by line rather than captured: the GUI reads this process's
    stdout to drive its progress line, and a ~13 minute COLMAP pass that printed
    nothing until it finished would be indistinguishable from a hang.
    """
    print(f"\n  $ {' '.join(os.path.basename(c) for c in cmd[:3])} ...   "
          f"-> {log_path}", flush=True)
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                env=child_env())
        for line in proc.stdout:
            log.write(line)
            sys.stdout.write("    " + line)
            sys.stdout.flush()
        code = proc.wait()
    if code:
        sys.exit(f"\n  {label} FAILED (exit {code}) -- see {log_path}\n")


def step_frames(args, out):
    """accumulate_frames -> <out>/input/*.png, manifest lifted to <out>/."""
    cmd = [sys.executable, "-m", "pipeline.accumulate_frames",
           "--input", args.input, "--out", os.path.join(out, "input"),
           "--events-per-frame", str(args.events_per_frame),
           "--stride", str(args.stride)]
    if args.motion_comp:
        cmd += ["--motion-comp"]
    if args.start_s is not None:
        cmd += ["--start-s", str(args.start_s)]
    if args.duration_s is not None:
        cmd += ["--duration-s", str(args.duration_s)]
    if args.norm_hi is not None:
        cmd += ["--norm-hi", f"{args.norm_hi:.6f}"]
    run_step(cmd, os.path.join(out, "accumulate_frames.log"), "accumulate_frames")

    # A scene dir keeps PNGs in input/ and the manifest at its root -- the
    # layout frame_metrics.load_frames and gaussian-splatting/convert.py both
    # expect, and the one work/phase1b/tracked_mc25k is in.
    src = os.path.join(out, "input", "frames.json")
    dst = os.path.join(out, "frames.json")
    if os.path.exists(src):
        os.replace(src, dst)
    if not os.path.exists(dst):
        sys.exit(f"  accumulate_frames wrote no manifest to {src}")
    with open(dst) as f:
        return json.load(f)


def step_tracks(args, out):
    """track_frames -> COLMAP database, mapper, tracks.json."""
    cmd = [sys.executable, "-m", "pipeline.track_frames",
           "--frames", out, "--out", out,
           "--max-gap", str(args.max_gap)]
    if not args.skip_colmap:
        cmd += ["--run-colmap"]
        print("\n  COLMAP next. /usr/bin/colmap is built WITHOUT CUDA, so the "
              "mapper is CPU-only:\n  ~2 min of mapper for a 200-frame take, "
              "measured. It is not hung.", flush=True)
    run_step(cmd, os.path.join(out, "track_frames.log"), "track_frames")
    path = os.path.join(out, "tracks.json")
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="recordings/<session> or its events.h5")
    ap.add_argument("--out", default=None,
                    help=f"work dir (default {PROBE_WORK_DIR}/<take>)")
    ap.add_argument("--events-per-frame", type=int, default=PROBE_EVENTS_PER_FRAME,
                    metavar="N", help="pipeline window, in events (default "
                                      f"{PROBE_EVENTS_PER_FRAME:,}; moving it "
                                      "makes this probe incomparable with the "
                                      "reference run)")
    ap.add_argument("--stride", type=int, default=PROBE_STRIDE, metavar="N")
    ap.add_argument("--max-gap", type=int, default=PROBE_MAX_GAP, metavar="N",
                    help="widest frame gap track_frames emits a pair for "
                         f"(default {PROBE_MAX_GAP}, the reference run's value; "
                         "track_frames' own default of 10 gives materially "
                         "worse connectivity -- see the ledger above)")
    ap.add_argument("--start-s", type=float, default=None)
    ap.add_argument("--duration-s", type=float, default=None)
    ap.add_argument("--reference-s", type=float, default=PROBE_REFERENCE_S,
                    help="long window the ev/lit-px gate is measured over "
                         f"(default {PROBE_REFERENCE_S} s)")
    ap.add_argument("--density-samples", type=int, default=5, metavar="K",
                    help="window positions sampled across the segment (default 5)")
    ap.add_argument("--norm-hi", type=float, default=None,
                    help="pin the white point for a bit-exact rerun")
    ap.add_argument("--no-motion-comp", dest="motion_comp", action="store_false",
                    help="escape hatch; compensation is ON by default because "
                         "the reference run used it")
    ap.add_argument("--skip-colmap", action="store_true",
                    help="mechanics-only run: frames and tracks, no mapper")
    ap.add_argument("--json", default=None,
                    help="machine-readable result (default <out>/probe.json)")
    ap.set_defaults(motion_comp=PROBE_MOTION_COMP)
    return ap


def main():
    args = build_parser().parse_args()
    t_started = time.time()

    h5_path, meta_path = _resolve_input(args.input)
    meta = read_meta(meta_path)
    width, height, n_total = resolve_geometry(h5_path, meta)
    take = os.path.basename(os.path.dirname(os.path.abspath(h5_path)))
    out = args.out or os.path.join(PROBE_WORK_DIR, take)
    os.makedirs(out, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        t_first = float(f["events"]["t"][0])
    t0 = t_first + args.start_s * 1e6 if args.start_s is not None else None
    t1 = None
    if args.duration_s is not None:
        t1 = (t0 if t0 is not None else t_first) + args.duration_s * 1e6
    i_lo, i_hi = acc.index_bounds(h5_path, t0, t1)

    print(f"\n  probe -- {take}  {width}x{height}  {n_total:,} events")
    print(f"  segment: events [{i_lo:,}, {i_hi:,})"
          + (f"  start={args.start_s}s" if args.start_s is not None else "")
          + (f"  duration={args.duration_s}s" if args.duration_s is not None else ""))
    print(f"  window={args.events_per_frame:,} ev  stride={args.stride:,} ev  "
          f"motion_comp={args.motion_comp}  max_gap={args.max_gap}  "
          f"reference={args.reference_s} s")
    print(f"  out -> {out}")

    # ---- density: the only stage that reads events directly -----------------
    density = measure_density(h5_path, i_lo, i_hi, width, height,
                              args.events_per_frame, args.reference_s,
                              args.density_samples)

    # ---- the pipeline -------------------------------------------------------
    man = step_frames(args, out)
    tracks = step_tracks(args, out)
    frags = (None if args.skip_colmap
             else fm.analyze_fragments(out, out))

    # ---- frame cadence, for the feature-lifetime gate -----------------------
    times = [e["t_mid_s"] for e in man["frames"]]
    frame_dt = float(np.median(np.diff(times))) if len(times) > 1 else float("nan")

    if args.motion_comp:
        vs = [np.hypot(e.get("vx", 0.0), e.get("vy", 0.0)) for e in man["frames"]]
        speed, speed_src = float(np.median(vs)), "frames.json, every frame"
    else:
        speed = measure_speed(h5_path, i_lo, i_hi, width, height,
                              args.events_per_frame, args.density_samples)
        speed_src = f"estimate_motion over {args.density_samples} windows"

    # ---- Report (restate the conventions actually used, then the gates) -----
    print(f"\n\n  Probe -- {take}\n")
    print(fm._fmt(fm.INFO, "configuration",
                  f"{man['events_per_frame']:,} ev/frame  stride "
                  f"{man['stride']:,}  motion_comp={man['motion_comp']}  "
                  f"max_gap={args.max_gap}  norm_hi={man['norm_hi']:.3f}"))
    print(fm._fmt(fm.INFO, "frame set",
                  f"{len(man['frames'])} frames over {density['segment_s']:.2f} s  "
                  f"({1 / frame_dt:.1f} fps, frame_dt {1000 * frame_dt:.1f} ms)"))
    print()

    g = Gates()
    pipe, ref = density["pipeline"], density["reference"]
    g.add("ev/lit-px @ pipeline", round(pipe["ev_per_lit_px"], 3),
          f"{pipe['ev_per_lit_px']:.2f} over {pipe['window_s']:.3f} s  "
          f"({args.events_per_frame:,} ev; informational -- a >= "
          f"{PROBE_EV_PER_LIT_TARGET:g} gate here is unreachable by construction)")
    ok_ref = ref["ev_per_lit_px"] >= PROBE_EV_PER_LIT_TARGET
    g.add("ev/lit-px @ reference", round(ref["ev_per_lit_px"], 3),
          f"{ref['ev_per_lit_px']:.2f} over {density['reference_s']:.2f} s"
          + ("  [SEGMENT SHORTER THAN THE REFERENCE WINDOW]"
             if density["reference_truncated"] else "")
          + f"   (need >= {PROBE_EV_PER_LIT_TARGET:g}, Phase1b §9.1)",
          target=PROBE_EV_PER_LIT_TARGET, ok=ok_ref)
    g.add("lit coverage", round(100 * ref["lit_coverage"], 2),
          f"{100 * pipe['lit_coverage']:.1f}% @ pipeline, "
          f"{100 * ref['lit_coverage']:.1f}% @ reference  "
          f"(multi-hit {100 * ref['multi_hit_share']:.0f}% of events)")
    g.add("camera speed", round(speed, 1),
          f"{speed:.0f} px/s median  ({speed_src})")

    life = tracks.get("track_lifetime_s_mean")
    need = 3.0 * frame_dt
    ok_life = life is not None and life >= need
    g.add("feature lifetime", round(life, 4) if life is not None else None,
          f"{life:.3f} s mean, {tracks.get('track_lifetime_s_median', 0):.3f} s "
          f"median  (need >= 3 x frame_dt = {need:.3f} s; measured limit ~0.4 s)",
          target=round(need, 4), ok=ok_life)
    g.add("mean track length", round(tracks["track_len_mean"], 3),
          f"{tracks['track_len_mean']:.2f} frames over "
          f"{tracks['tracks_kept']:,} tracks  (LK, before the mapper)")

    if frags is None:
        print(fm._fmt(fm.INFO, "colmap", "skipped (--skip-colmap)"))
        result_frags = None
    else:
        print()
        # Table only: the gate rows below are this module's, so that they land
        # in probe.json as well as on stdout.
        fm.report_fragments(frags, table=True, gates=False)
        big = frags["largest"]
        n_in = frags["n_input"]
        n_frag = len(frags["fragments"])
        g.add("fragments", n_frag, f"{n_frag} disconnected reconstruction(s)",
              target=1, ok=n_frag == 1)
        if big:
            rate = big["registered"] / n_in if n_in else float("nan")
            g.add("frames in largest", big["registered"],
                  f"{big['registered']}/{n_in} = {100 * rate:.0f}%",
                  target=f">= 90% of {n_in}", ok=rate >= 0.9)
            g.add("span of largest", round(big["span_s"] or 0.0, 3),
                  f"{big['span_s']:.2f} s of the {density['segment_s']:.2f} s "
                  f"segment = {100 * (big['span_s'] or 0) / density['segment_s']:.0f}%")
            g.add("track length (largest)", round(big["track_len_mean"] or 0, 3),
                  f"{big['track_len_mean']}  (COLMAP; E2VID scored exactly 2.0)",
                  target=3.0, ok=(big["track_len_mean"] or 0) > 3.0)
            cov = frags["union_registered"] / n_in if n_in else float("nan")
            g.add("union over fragments", frags["union_registered"],
                  f"{frags['union_registered']}/{n_in} = {100 * cov:.0f}% of "
                  f"frames appear in SOME model")
        result_frags = frags

    elapsed = time.time() - t_started
    print(f"\n  {'ALL GATES PASSED' if g.passed else 'GATES FAILED'}"
          f"   ({elapsed:.0f} s)\n")

    result = {
        "probe_version": PROBE_VERSION,
        "take": take,
        "input": os.path.abspath(h5_path),
        "out": os.path.abspath(out),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_s": round(elapsed, 1),
        "config": {
            "events_per_frame": man["events_per_frame"],
            "stride": man["stride"],
            "motion_comp": man["motion_comp"],
            "max_gap": args.max_gap,
            "norm_hi": man["norm_hi"],
            "reference_s": density["reference_s"],
            "start_s": args.start_s,
            "duration_s": args.duration_s,
            "skip_colmap": args.skip_colmap,
        },
        "biases": meta.get("biases"),
        "frames": {"n": len(man["frames"]), "frame_dt_s": frame_dt,
                   "segment_s": density["segment_s"]},
        "density": density,
        "speed_px_s": speed,
        "tracks": tracks,
        "fragments": result_frags,
        "gates": g.rows,
        "passed": g.passed,
    }
    json_path = args.json or os.path.join(out, "probe.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"  probe.json -> {json_path}\n")
    sys.exit(0 if g.passed else 1)


if __name__ == "__main__":
    main()
