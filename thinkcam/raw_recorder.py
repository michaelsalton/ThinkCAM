import json
import os
import queue
import threading
import time
from datetime import datetime, timezone

import h5py
import numpy as np

from thinkcam.constants import (
    BIAS_REFRACTORY,
    BIAS_THRESHOLD_NEG,
    BIAS_THRESHOLD_POS,
    BURST_FILTER_ENABLE,
    RAW_HDF5_CHUNK,
    RAW_HDF5_COMPRESSION,
    RAW_QUEUE_MAXSIZE,
    RAW_RECORD_DIR,
)

# Sentinel pushed onto the queue to tell the writer thread to drain and exit.
_STOP = object()

# We store t as float64 rather than int64 because the camera delivers t as
# float32; float64 is a lossless passthrough. Hardware confirmed on 2026-06-01
# (docs/ImplementationPlan.md §2): t is in MICROSECONDS (integer-valued, smallest
# observed tick 2 us), and p is {0,1} (0 = negative, 1 = positive). x,y are
# integer-valued floats -> uint16. p is kept as the raw sensor {0,1} here; the
# IncEventGS converter remaps it to signed {-1,+1} (which that pipeline requires).


class RawEventRecorder:
    """Lossless raw-event recorder.

    Acquisition runs in a QThread; this recorder owns a bounded queue and a
    dedicated writer thread so file I/O never stalls the camera loop. The
    camera worker calls submit() with each (N, 4) float32 event batch; the
    writer thread appends to chunked HDF5 datasets. If the queue ever saturates,
    batches are dropped and counted (the counter must stay 0 at normal rates).
    """

    def __init__(self, output_root: str = RAW_RECORD_DIR):
        self._output_root = output_root

        self._queue: queue.Queue = queue.Queue(maxsize=RAW_QUEUE_MAXSIZE)
        self._thread: threading.Thread | None = None
        self._h5: h5py.File | None = None
        self._h5_path: str | None = None
        self._session_dir: str | None = None

        self._is_recording = False
        self._width = 0
        self._height = 0
        self._label = ""
        self._biases: dict = {}
        self._start_monotonic = 0.0
        self._start_utc = ""

        # Updated by the writer thread, read by the GUI thread -> guard.
        self._stats_lock = threading.Lock()
        self._events_written = 0
        self._dropped_events = 0

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, width: int, height: int, label: str = "",
              biases: dict | None = None) -> str:
        """`biases` is the set actually in effect, read back from the camera.

        Before biases were adjustable, _write_metadata hardcoded the four
        constants — which is why all nine existing takes record 10/10/10/true:
        that is all the code could say, not necessarily what the sensor was
        doing. Passing them in makes the sidecar a record instead of an echo.
        Falls back to the constants when the caller has nothing better.
        """
        if self._is_recording:
            return self._session_dir or ""

        self._width = width
        self._height = height
        self._label = label.strip()
        self._biases = dict(biases) if biases else {
            "BiasEventThresholdPositive": BIAS_THRESHOLD_POS,
            "BiasEventThresholdNegative": BIAS_THRESHOLD_NEG,
            "BiasRefractoryPeriod": BIAS_REFRACTORY,
            "EventBurstFilterEnable": BURST_FILTER_ENABLE,
        }

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{ts}_{self._label}" if self._label else ts
        self._session_dir = os.path.join(self._output_root, name)
        os.makedirs(self._session_dir, exist_ok=True)
        self._h5_path = os.path.join(self._session_dir, "events.h5")

        self._h5 = h5py.File(self._h5_path, "w")
        grp = self._h5.create_group("events")
        common = dict(shape=(0,), maxshape=(None,), chunks=(RAW_HDF5_CHUNK,),
                      compression=RAW_HDF5_COMPRESSION)
        grp.create_dataset("x", dtype=np.uint16, **common)
        grp.create_dataset("y", dtype=np.uint16, **common)
        grp.create_dataset("t", dtype=np.float64, **common)
        grp.create_dataset("p", dtype=np.int8, **common)
        grp.attrs["width"] = width
        grp.attrs["height"] = height
        grp.attrs["polarity_encoding"] = "0_neg_1_pos"
        grp.attrs["t_unit"] = "microseconds"
        grp.attrs["sensor"] = "IMX636"

        with self._stats_lock:
            self._events_written = 0
            self._dropped_events = 0

        self._start_monotonic = time.monotonic()
        self._start_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        self._is_recording = True
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
        return self._session_dir

    def submit(self, events: np.ndarray) -> None:
        """Hand a (N, 4) float32 event batch to the writer thread (non-blocking)."""
        if not self._is_recording or events.shape[0] == 0:
            return
        try:
            self._queue.put_nowait(events)
        except queue.Full:
            with self._stats_lock:
                self._dropped_events += int(events.shape[0])

    def stop(self) -> str | None:
        if not self._is_recording:
            return None

        # Stop accepting new batches, then flush whatever is queued.
        self._is_recording = False
        self._queue.put(_STOP)
        if self._thread is not None:
            self._thread.join()
            self._thread = None

        if self._h5 is not None:
            self._h5.flush()
            self._h5.close()
            self._h5 = None

        self._write_metadata()

        session_dir = self._session_dir
        self._session_dir = None
        self._h5_path = None
        return session_dir

    # ------------------------------------------------------------------
    # Writer thread
    # ------------------------------------------------------------------

    def _writer_loop(self):
        stop = False
        while not stop:
            item = self._queue.get()
            if item is _STOP:
                break

            # Coalesce every batch already waiting in the queue into a single
            # append. The camera delivers buffers very frequently (~1.7k/s), and
            # at that cadence the per-append overhead (resize + compress 4
            # datasets) — not the event volume — is what makes the writer fall
            # behind and the queue start dropping. One concatenated write per
            # drain cycle cuts appends to a handful per second.
            batch = [item]
            try:
                while True:
                    nxt = self._queue.get_nowait()
                    if nxt is _STOP:
                        stop = True
                        break
                    batch.append(nxt)
            except queue.Empty:
                pass

            try:
                self._append(np.concatenate(batch) if len(batch) > 1 else batch[0])
            except Exception:
                # Never let a write error kill the thread mid-take; keep draining
                # so stop() can still close a valid (if truncated) file.
                pass

    def _append(self, events: np.ndarray):
        if self._h5 is None:
            return
        n = events.shape[0]
        grp = self._h5["events"]
        cols = {
            "x": events[:, 0].astype(np.uint16),
            "y": events[:, 1].astype(np.uint16),
            "t": events[:, 2].astype(np.float64),
            "p": events[:, 3].astype(np.int8),
        }
        for name, data in cols.items():
            dset = grp[name]
            old = dset.shape[0]
            dset.resize((old + n,))
            dset[old:] = data

        with self._stats_lock:
            self._events_written += n

    # ------------------------------------------------------------------
    # Stats + metadata
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        with self._stats_lock:
            written = self._events_written
            dropped = self._dropped_events
        elapsed = time.monotonic() - self._start_monotonic if self._is_recording else 0.0
        rate = written / elapsed if elapsed > 0 else 0.0
        size = 0
        if self._h5_path and os.path.exists(self._h5_path):
            try:
                size = os.path.getsize(self._h5_path)
            except OSError:
                size = 0
        return {
            "raw_elapsed_s": elapsed,
            "raw_events_written": written,
            "raw_event_rate": rate,
            "raw_file_size": size,
            "raw_dropped_events": dropped,
        }

    def _write_metadata(self):
        if self._session_dir is None:
            return
        with self._stats_lock:
            written = self._events_written
            dropped = self._dropped_events
        duration = time.monotonic() - self._start_monotonic
        metadata = {
            "sensor": "Sony IMX636 (LUCID TRT009S-E)",
            "resolution": [self._width, self._height],
            "timestamp_unit": "microseconds",
            "polarity_encoding": "0_neg_1_pos",
            "recording_start_utc": self._start_utc,
            "duration_s": round(duration, 3),
            "event_count": written,
            "dropped_events": dropped,
            "lossless": dropped == 0,
            "biases": {
                "threshold_positive":
                    self._biases.get("BiasEventThresholdPositive"),
                "threshold_negative":
                    self._biases.get("BiasEventThresholdNegative"),
                "refractory_period":
                    self._biases.get("BiasRefractoryPeriod"),
                "burst_filter":
                    self._biases.get("EventBurstFilterEnable"),
            },
            "noise_filter": {
                "background_activity_filter": None,
                "duration_us": None,
            },
            "intrinsics": {
                "fx": None, "fy": None, "cx": None, "cy": None,
                "distortion": None,
                "calibrated": False,
            },
            "operator_notes": "",
            "take_label": self._label,
        }
        path = os.path.join(self._session_dir, "metadata.json")
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)
