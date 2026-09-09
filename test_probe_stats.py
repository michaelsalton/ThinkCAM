#!/usr/bin/env python3
"""Offline check of the live probe tier -- no camera, no Qt, no GUI venv.

plans/01_CaptureDashboard.md §7 step 3: the live tier needs hardware to
validate as a whole, but its ARITHMETIC does not. This replays a slice of a
recorded take as the acquisition loop would deliver it -- (N, 4) float32
(x, y, t, p) batches of ~470 events, the size the camera emits at ~1.7 k
buffers/s -- and checks thinkcam.probe_worker.compute_stats against figures
measured independently out of the same file.

Every expected value below is pinned to ONE position in ONE take
(recordings/20260602_102004_demo_scene_orbit_1 at event 10,167,048, which is
where work/phase1b/tracked_mc25k starts). That is the point: these are the
numbers Phase1b-Results.md and the plan quote, so a drift here means the live
panel and the wiki have stopped describing the same quantity. Do not "fix" a
failure by moving a tolerance.

    PYTHONPATH=. ~/envs/phase1/bin/python test_probe_stats.py
"""

import sys
import types

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py is required: pip install h5py")

# probe_worker imports PySide6 for QThread/QMutex only; compute_stats is
# module-level and touches none of it. Stubbing lets this run on the analysis
# interpreter, which has numpy/h5py/cv2 but no GUI packages.
if "PySide6" not in sys.modules:
    qtcore = types.ModuleType("PySide6.QtCore")

    class _Stub:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    for _n in ("QMutex", "QMutexLocker", "QThread", "Signal"):
        setattr(qtcore, _n, _Stub)
    pyside = types.ModuleType("PySide6")
    pyside.QtCore = qtcore
    sys.modules["PySide6"] = pyside
    sys.modules["PySide6.QtCore"] = qtcore

from thinkcam.probe_worker import compute_stats     # noqa: E402

REC = "recordings/20260602_102004_demo_scene_orbit_1/events.h5"
I0 = 10_167_048        # tracked_mc25k's first window
BATCH = 470            # ~798 Kev/s over ~1.7 k buffers/s
WIDTH, HEIGHT = 1280, 720

FAILS = []


def batches(n_events):
    with h5py.File(REC, "r") as f:
        g = f["events"]
        cols = [g[c][I0:I0 + n_events].astype(np.float32) for c in "xytp"]
    arr = np.stack(cols, axis=1)
    return [arr[i:i + BATCH] for i in range(0, len(arr), BATCH)]


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<34} {got:.3f}   "
          f"(want {want} +-{tol})")
    if not ok:
        FAILS.append(name)


def main():
    # 2.0 s at this position is 1,750,213 events -- the full reference window.
    full = batches(1_750_213)
    print(f"\n  {len(full):,} batches of {BATCH} events "
          f"({BATCH * len(full):,} events)\n")
    s = compute_stats(full, WIDTH, HEIGHT)
    check("reference ev/lit-px @ 2 s", s["reference_ev_per_lit_px"], 2.27, 0.02)
    check("reference lit coverage %", 100 * s["reference_lit_coverage"], 83.6, 0.5)
    check("reference window s", s["reference_window_s"], 2.00, 0.01)
    check("reference multi-hit share %",
          100 * s["reference_multi_hit_share"], 85.0, 1.0)
    check("event rate Kev/s", s["event_rate"] / 1e3, 875.0, 5.0)

    # The pipeline window is the TAIL of the buffer, so it is a different 25 k
    # events from the head window every wiki figure was quoted at -- it should
    # land in the same band, not on the same number.
    check("pipeline window events", s["pipeline_events"], 25_000, 0)
    check("pipeline ev/lit-px in band", s["pipeline_ev_per_lit_px"], 1.10, 0.15)
    check("speed in the recorded band", s["speed_px_s"], 1350.0, 850.0)

    # A buffer shorter than one pipeline window must answer, not crash: that is
    # the state the panel is in for the first two seconds of every stream.
    short = compute_stats(batches(5_000), WIDTH, HEIGHT)
    check("short buffer, pipeline events", short["pipeline_events"], 5_000, 0)
    check("short buffer, ev/lit-px", short["pipeline_ev_per_lit_px"], 1.02, 0.10)

    # Fed exactly the head 25 k, it must reproduce the wiki's 1.07 and the
    # (vx, vy) = (-125, 62.5) that tracked_mc25k's frames.json records for
    # frame 0 -- |v| = 139.75 px/s.
    head = compute_stats(batches(25_000), WIDTH, HEIGHT)
    check("head 25 k ev/lit-px", head["pipeline_ev_per_lit_px"], 1.070, 0.005)
    check("head 25 k speed px/s", head["speed_px_s"], 139.75, 1.0)

    print(f"\n  {'ALL PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}\n")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
