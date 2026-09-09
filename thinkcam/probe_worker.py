"""Live capture statistics on their own thread: ev/lit-px, coverage, speed.

The readout a bias change is judged by, at 2 Hz, while the take is rolling.
Without it the loop in plans/01_CaptureDashboard.md is open: biases could only be
changed by editing constants.py and restarting, and the only verdict on whether
a change helped came from a ~13 minute COLMAP pass after the fact.

WHY A THREAD. Measured on this repo's reference take, one update costs ~220 ms
for estimate_motion (grid=9, levels=6 -> ~486 objective evaluations over 25 k
events) plus ~24 ms for density_stats over a 2 s buffer of 1.75 M events. At
2 Hz that is ~50% of one core. The acquisition loop must sustain ~1.7 k
buffers/s and every one of them must reach raw_recorder losslessly, so none of
this may happen on it.

TWO WINDOWS, ONE BUFFER (plans §1). ev/lit-px depends strongly on window length
-- 1.07 at the 25 k pipeline window against 2.27 at 2 s on the reference take --
because a moving camera paints new pixels rather than revisiting old ones. The
pipeline window is what the frames COLMAP sees are built from; the 2 s reference
window is the one Phase1b-Results.md §9.1's ">= 5" target is hung on, and the
one comparable between takes at different event rates. The deque holds the
reference window and the pipeline window is sliced off its tail, so both
readings describe the same instant.

DEVIATION FROM THE PLAN, deliberate. plans §3 specifies a "single-slot mailbox
-- newest batch wins, older ones dropped". That cannot coexist with the rolling
2 s window the same paragraph asks for: at ~1.7 k buffers/s a single slot drained
at 2 Hz would keep 2 batches per second out of 1700, and ev/lit-px -- a COUNT
per pixel -- would read ~1/850 of the truth while looking perfectly plausible.
The plan's own sizing (~26 MB at 798 Kev/s, i.e. 2 s of the WHOLE stream) says
the deque is meant to see every batch. So the deque itself is the mailbox:
submit() appends under the mutex and trims by time, which is a few microseconds
on the acquisition thread, and "newest wins" is applied where it belongs -- to
the computation, which is simply skipped if the previous one is still running.
"""

import time
from collections import deque

import numpy as np
from PySide6.QtCore import QMutex, QMutexLocker, QThread, Signal

import accumulate_frames as acc
from thinkcam.constants import (
    PROBE_EV_PER_LIT_TARGET,
    PROBE_EVENTS_PER_FRAME,
    PROBE_MOTION_SEARCH_EVENTS,
    PROBE_REFERENCE_S,
    PROBE_UPDATE_HZ,
)

# Hard ceiling on the rolling buffer, independent of the time trim. The sensor
# is ERC-limited to 10 Mev/s, so 2 s cannot legitimately exceed 20 M events;
# this only bounds memory if timestamps ever go wrong (they have before --
# export_e2vid_input warns about non-monotonic steps).
MAX_BUFFERED_EVENTS = 24_000_000


class ProbeWorker(QThread):
    """Rolling live statistics over the last PROBE_REFERENCE_S of events."""

    probe_ready = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lock = QMutex()
        self._running = False
        self._batches = deque()      # (N, 4) float32 (x, y, t, p), oldest first
        self._n_buffered = 0
        self._width = 0
        self._height = 0

    def set_geometry(self, width: int, height: int):
        with QMutexLocker(self._lock):
            self._width, self._height = width, height

    def stop(self):
        with QMutexLocker(self._lock):
            self._running = False

    def clear(self):
        with QMutexLocker(self._lock):
            self._batches.clear()
            self._n_buffered = 0

    # ------------------------------------------------------------------
    # acquisition thread
    # ------------------------------------------------------------------

    def submit(self, events: np.ndarray) -> None:
        """Append a batch and drop whatever has aged out. Never blocks.

        Called from the acquisition loop beside raw_recorder.submit(). Trimming
        here rather than in run() keeps the buffer bounded even while the worker
        is busy inside a 220 ms motion search.
        """
        if events.shape[0] == 0:
            return
        with QMutexLocker(self._lock):
            if not self._running:
                return
            self._batches.append(events)
            self._n_buffered += events.shape[0]
            t_new = float(events[-1, 2])
            horizon = t_new - PROBE_REFERENCE_S * 1e6
            # A batch goes only when it is ENTIRELY older than the horizon, so
            # the window is never short. len > 1 keeps the newest batch even if
            # it alone busts the ceiling.
            while len(self._batches) > 1 and (
                    float(self._batches[0][-1, 2]) < horizon
                    or self._n_buffered > MAX_BUFFERED_EVENTS):
                self._n_buffered -= self._batches.popleft().shape[0]

    # ------------------------------------------------------------------
    # probe thread
    # ------------------------------------------------------------------

    def run(self):
        with QMutexLocker(self._lock):
            self._running = True
        period_ms = int(1000.0 / PROBE_UPDATE_HZ)
        while True:
            t0 = time.monotonic()
            with QMutexLocker(self._lock):
                if not self._running:
                    break
                # Copy the list of array references, not the arrays: the
                # acquisition thread may append or popleft while we compute,
                # and the batches themselves are never mutated in place.
                batches = list(self._batches)
                width, height = self._width, self._height
            if batches and width and height:
                stats = self._compute(batches, width, height)
                if stats is not None:
                    self.probe_ready.emit(stats)
            # Period, not gap: an update costs ~250 ms, and sleeping the full
            # period on top of it would quietly halve PROBE_UPDATE_HZ.
            spent = int(1000 * (time.monotonic() - t0))
            self.msleep(max(20, period_ms - spent))

    def _compute(self, batches, width, height):
        return compute_stats(batches, width, height)


def compute_stats(batches, width, height):
    """The whole statistics tier, as a plain function of (batches, geometry).

    Module level and Qt-free on purpose: this is the part with arithmetic in it,
    and plans/01_CaptureDashboard.md §7 asks for it to be testable against an
    events.h5 slice with no camera attached. ProbeWorker adds only threading.

    `batches` is a list of (N, 4) float32 (x, y, t, p) arrays, oldest first, as
    the camera delivers them; t is in microseconds.
    """
    x = np.concatenate([b[:, 0] for b in batches])
    y = np.concatenate([b[:, 1] for b in batches])
    t = np.concatenate([b[:, 2] for b in batches]).astype(np.float64)
    n = x.size
    if n < 2:
        return None
    ref_s = (float(t[-1]) - float(t[0])) / 1e6

    ref = acc.density_stats(x, y, width, height)
    # The pipeline window is the TAIL of the buffer, so it is the most
    # recent 25 k events rather than a window from 2 s ago.
    k = min(PROBE_EVENTS_PER_FRAME, n)
    px, py, pt = x[n - k:], y[n - k:], t[n - k:]
    pipe = acc.density_stats(px, py, width, height)
    pipe_s = (float(pt[-1]) - float(pt[0])) / 1e6

    speed = float("nan")
    if k >= 1000 and pipe_s > 0:
        dt = (pt - 0.5 * (pt[0] + pt[-1])) / 1e6
        sl = acc.subsample(k, PROBE_MOTION_SEARCH_EVENTS)
        # estimate_motion carries the spatial-binning fix from
        # Phase1b-Results.md §4; without it this returns v ~ 0 on this data
        # and reads exactly like a stationary camera. Do not reimplement.
        vx, vy, _s, _z = acc.estimate_motion(
            px[sl].astype(np.float64), py[sl].astype(np.float64), dt[sl],
            width, height, v_init=None)
        speed = float(np.hypot(vx, vy))

    return {
        "pipeline_ev_per_lit_px": pipe["ev_per_lit_px"],
        "pipeline_lit_coverage": pipe["lit_coverage"],
        "pipeline_window_s": pipe_s,
        "pipeline_events": int(k),
        "reference_ev_per_lit_px": ref["ev_per_lit_px"],
        "reference_lit_coverage": ref["lit_coverage"],
        "reference_multi_hit_share": ref["multi_hit_share"],
        "reference_window_s": ref_s,
        "reference_events": int(n),
        "reference_target": PROBE_EV_PER_LIT_TARGET,
        "event_rate": n / ref_s if ref_s > 0 else 0.0,
        "speed_px_s": speed,
    }
