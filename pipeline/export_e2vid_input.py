#!/usr/bin/env python3
"""Convert a ThinkCam raw-event recording -> E2VID event-file format.

Target: rpg_e2vid (github.com/uzh-rpg/rpg_e2vid), events -> intensity frames.
This is Step 2 of docs/Phase1.md; its output feeds run_reconstruction.py, whose
frames then feed COLMAP and vanilla 3DGS.

The schema below was VERIFIED by reading rpg_e2vid's own readers, not the README
(which documents no format at all):

    utils/event_readers.py:14-20  FixedSizeEventReader ..... names=['t','x','y','pol']
                                                             delim_whitespace, header=None
    utils/event_readers.py:19     skiprows=start_index + 1 .. line 0 is a HEADER
    run_reconstruction.py:45      width, height = header.values[0]
                                                             .. header is "<W> <H>"
    utils/event_readers.py:79-83  t,x,y,pol = line.split(' ')
                                  t > last_stamp + duration_s
                                                             .. t is in SECONDS
    utils/inference_utils.py:460  pols[pols == 0] = -1 ...... {0,1} accepted as-is

Consequences, baked in as the defaults here:

  * Column order is [t, x, y, p] -- timestamp FIRST. This is the opposite of the
    [x, y, t, p] that convert_to_inceventgs writes; do not copy that layout.
  * Timestamps are SECONDS, not microseconds. ThinkCam records microseconds
    (docs/ImplementationPlan.md §2), so we divide by 1e6 and re-zero to the first
    event. %.6f keeps full microsecond resolution.
  * Polarity stays {0,1}. E2VID remaps 0 -> -1 itself, so the raw sensor encoding
    passes through untouched -- no remap needed (unlike the IncEventGS path).
  * Fields are separated by a SINGLE space. FixedSizeEventReader tolerates any
    whitespace, but FixedDurationEventReader does a literal line.split(' ').

Memory: reads the HDF5 in chunks and streams straight to the output file. The
target take (recordings/20260602_102004_demo_scene_orbit_1) is 51.2M events /
676 MB, which must never be materialised whole -- do not switch this to
convert_to_inceventgs.load_events(), which slurps all four arrays.
"""

import argparse
import json
import os
import sys
import zipfile

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py is required: pip install h5py")

from pipeline.convert_to_inceventgs import _resolve_input

# Chunk length for the streaming read. Matches RAW_HDF5_CHUNK in
# thinkcam/constants.py so we read whole HDF5 chunks and never straddle a
# compression boundary. Takes before 20260601_203611 are LZF, later ones are
# uncompressed; h5py handles both transparently, so never assume a filter.
READ_CHUNK = 1_000_000


def read_meta(meta_path):
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        return json.load(f)


def resolve_geometry(h5_path, meta):
    """Sensor size, preferring the HDF5 attrs and falling back to metadata.json."""
    with h5py.File(h5_path, "r") as f:
        g = f["events"]
        width = int(g.attrs.get("width", 0))
        height = int(g.attrs.get("height", 0))
        n = g["x"].shape[0]
        for name in ("y", "t", "p"):
            if g[name].shape[0] != n:
                sys.exit("Corrupt recording: x/y/t/p length mismatch.")
    if (not width or not height) and meta.get("resolution"):
        width, height = meta["resolution"]
    if not width or not height:
        sys.exit("Resolution unknown (not in H5 attrs or metadata.json).")
    return width, height, n


def event_chunks(h5_path, chunk, t0=None, t1=None, downsample=1):
    """Yield (t, x, y, p) slices in order. t is raw microseconds, unshifted.

    downsample bins x,y by integer division, which is the standard way to
    rescale an event stream (events are points, so binning == downsampling).
    This trades spatial resolution for events-per-pixel, which is what lets a
    motion-limited (short) window still reach E2VID's ~0.35 ev/px design point.
    """
    with h5py.File(h5_path, "r") as f:
        g = f["events"]
        n = g["x"].shape[0]
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            t = g["t"][start:stop].astype(np.float64)
            x = g["x"][start:stop]
            y = g["y"][start:stop]
            p = g["p"][start:stop]
            if downsample > 1:
                x = x // downsample
                y = y // downsample
            if t0 is not None or t1 is not None:
                keep = np.ones(t.shape, dtype=bool)
                if t0 is not None:
                    keep &= t >= t0
                if t1 is not None:
                    keep &= t <= t1
                if not keep.any():
                    continue
                t, x, y, p = t[keep], x[keep], y[keep], p[keep]
            yield t, x, y, p


def first_timestamp(h5_path):
    with h5py.File(h5_path, "r") as f:
        t = f["events"]["t"]
        return float(t[0]) if t.shape[0] else 0.0


def write_events(out_path, h5_path, width, height, t_origin, chunk,
                 t0=None, t1=None, as_zip=False, downsample=1):
    """Stream the take out as E2VID's text format. Returns (n_written, t_span_s)."""
    inner_name = os.path.basename(out_path)
    if as_zip:
        inner_name = os.path.splitext(inner_name)[0] + ".txt"

    written = 0
    t_min = None
    t_max = None
    nonmono = 0
    prev_last = None

    zf = zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) if as_zip else None
    # Both the .txt and the in-archive member start with the sensor-size header.
    sink = zf.open(inner_name, "w") if as_zip else open(out_path, "wb")

    try:
        sink.write(f"{width} {height}\n".encode())
        for t, x, y, p in event_chunks(h5_path, chunk, t0, t1, downsample):
            ts = (t - t_origin) / 1e6  # microseconds -> seconds, re-zeroed

            if prev_last is not None and ts.size and ts[0] < prev_last:
                nonmono += 1
            if ts.size:
                nonmono += int((np.diff(ts) < 0).sum())
                prev_last = ts[-1]
                t_min = ts[0] if t_min is None else min(t_min, float(ts.min()))
                t_max = float(ts.max()) if t_max is None else max(t_max, float(ts.max()))

            block = np.empty((ts.size, 4), dtype=np.float64)
            block[:, 0] = ts
            block[:, 1] = x
            block[:, 2] = y
            block[:, 3] = p
            np.savetxt(sink, block, fmt="%.6f %d %d %d")
            written += int(ts.size)
    finally:
        sink.close()
        if zf is not None:
            zf.close()

    span = (t_max - t_min) if (t_min is not None and t_max is not None) else 0.0
    return written, span, nonmono


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="recordings/<session>/events.h5 or the session dir")
    ap.add_argument("--out", required=True,
                    help="output .txt (or .zip with --zip)")
    ap.add_argument("--zip", action="store_true",
                    help="write a .zip containing one .txt (E2VID accepts both)")
    ap.add_argument("--downsample", type=int, default=1, metavar="F",
                    help="bin x,y by F (1280x720 -> 1280//F x 720//F). Raises "
                         "events-per-pixel so a short, motion-limited window can "
                         "still reach E2VID's ~0.35 ev/px design point.")
    ap.add_argument("--chunk", type=int, default=READ_CHUNK,
                    help=f"events per streaming read (default {READ_CHUNK:,})")
    ap.add_argument("--start-s", type=float, default=None,
                    help="drop events before this offset, in seconds from take start")
    ap.add_argument("--duration-s", type=float, default=None,
                    help="keep only this many seconds after --start-s (for smoke tests)")
    args = ap.parse_args()

    h5_path, meta_path = _resolve_input(args.input)
    meta = read_meta(meta_path)
    width, height, n_total = resolve_geometry(h5_path, meta)

    # A Gen-1 take (20260601_175434) predates hardware verification and its
    # timestamp unit is undocumented -- refuse rather than silently mis-scale.
    unit = meta.get("timestamp_unit")
    if unit and unit != "microseconds":
        sys.exit(f"Refusing: metadata timestamp_unit is {unit!r}, not 'microseconds'. "
                 "See docs/Phase1.md §6 -- this take predates hardware verification.")

    t_first = first_timestamp(h5_path)
    t0 = t_first + args.start_s * 1e6 if args.start_s is not None else None
    t1 = None
    if args.duration_s is not None:
        base = t0 if t0 is not None else t_first
        t1 = base + args.duration_s * 1e6

    # Re-zero to the first *emitted* event so the output always starts at 0.0.
    t_origin = t0 if t0 is not None else t_first

    F = max(1, args.downsample)
    # Binned extent must be ceil(W/F), not W//F: the largest source coordinate is
    # W-1, which bins to (W-1)//F. When F does not divide W those differ, and the
    # extra column overflows the declared width -- E2VID then scatters out of
    # bounds and dies with a CUDA device-side assert.
    out_w, out_h = ((width - 1) // F) + 1, ((height - 1) // F) + 1

    written, span, nonmono = write_events(
        args.out, h5_path, out_w, out_h, t_origin, args.chunk,
        t0=t0, t1=t1, as_zip=args.zip, downsample=F)

    size = os.path.getsize(args.out)
    print(f"\n  Wrote {written:,} events  -> {args.out}")
    print(f"  columns=[t, x, y, p]  t_unit=seconds  polarity={{0,1}} (passthrough)")
    print(f"  header line: '{out_w} {out_h}'   separator: single space")
    if F > 1:
        px = out_w * out_h
        print(f"  downsample={F}x  ({width}x{height} -> {out_w}x{out_h}, {px:,} px)")
        print(f"  E2VID 0.35 ev/px window here = {int(0.35 * px):,} events "
              f"(-N {int(0.35 * px)})")
    print(f"  resolution={out_w}x{out_h}  t_span={span:.3f} s  "
          f"rate={written / span:,.0f} ev/s" if span else "")
    print(f"  file size {size / 1e6:.1f} MB   (source take: {n_total:,} events)")
    if nonmono:
        print(f"  ⚠ {nonmono} out-of-order timestamp step(s) left as-is. E2VID's "
              "FixedDuration reader assumes non-decreasing t; if reconstruction "
              "stalls, sort the take first.")
    if written == 0:
        sys.exit("No events written -- check --start-s / --duration-s.")

    print("\n  Next: run rpg_e2vid/run_reconstruction.py on this file "
          "(docs/Phase1.md §5 Step 3).\n")


if __name__ == "__main__":
    main()
