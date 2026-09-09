import ctypes
import time

import numpy as np
from PySide6.QtCore import QMutex, QMutexLocker, QThread, Signal

from arena_api.system import system

from thinkcam.constants import (
    BIAS_REFRACTORY,
    BIAS_THRESHOLD_NEG,
    BIAS_THRESHOLD_POS,
    BURST_FILTER_ENABLE,
    CAMERA_IP,
    DISPLAY_FPS,
    ERC_RATE_LIMIT_MEV,
    EVS_OUTPUT_FORMAT,
    IMAGE_TIMEOUT_MS,
    NUM_BUFFERS,
)
from thinkcam.visualizer import render_events

# The four nodes the Biases group in controls.py drives. Names are the ArenaSDK
# nodemap's, and the order is the order they are written in.
BIAS_NODES = ("BiasEventThresholdPositive", "BiasEventThresholdNegative",
              "BiasRefractoryPeriod", "EventBurstFilterEnable")


class CameraWorker(QThread):
    frame_ready = Signal(np.ndarray, dict)
    status_message = Signal(str)
    error = Signal(str)
    connected = Signal(int, int)
    # (settings, ranges, writable_while_streaming) once the stream is up, so the
    # UI can seed its spinboxes from the hardware rather than from constants.py.
    bias_state = Signal(dict, dict, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lock = QMutex()
        self._running = False
        # Set by MainWindow. The recorder is thread-safe (its own queue/lock),
        # so the acquisition loop can submit batches directly without going
        # through Qt signals — keeps high-rate event data off the GUI thread.
        self.raw_recorder = None
        # Same contract for the live probe: it owns its own mutex and never
        # blocks, so the acquisition loop hands it batches directly.
        self.probe_worker = None
        # Pending bias writes, drained at the top of the acquisition loop.
        # Nodemap writes must happen on the thread that owns the device, so the
        # GUI thread can only leave them here.
        self._pending_biases = None
        self._biases = {
            "BiasEventThresholdPositive": BIAS_THRESHOLD_POS,
            "BiasEventThresholdNegative": BIAS_THRESHOLD_NEG,
            "BiasRefractoryPeriod": BIAS_REFRACTORY,
            "EventBurstFilterEnable": BURST_FILTER_ENABLE,
        }

    def set_biases(self, settings: dict):
        """Queue a bias change from the GUI thread. Applied on the next loop."""
        with QMutexLocker(self._lock):
            self._pending_biases = dict(settings)

    def biases(self) -> dict:
        """The values currently in effect, for the recording's metadata."""
        with QMutexLocker(self._lock):
            return dict(self._biases)

    def stop(self):
        with QMutexLocker(self._lock):
            self._running = False

    def _connect_device(self, max_tries=6, wait_secs=10):
        system.DEVICE_INFOS_TIMEOUT_MILLISEC = 1000
        system.add_unicast_discovery_device(CAMERA_IP)
        for attempt in range(1, max_tries + 1):
            self.status_message.emit(
                f"Searching for camera… (attempt {attempt}/{max_tries})"
            )
            devices = system.create_device()
            if devices:
                return devices
            time.sleep(wait_secs)
        return None

    def _configure_evs(self, device) -> dict:
        nm = device.nodemap
        tl = device.tl_stream_nodemap

        saved = {
            "AcquisitionMode": nm["AcquisitionMode"].value,
            "EventFormat": nm["EventFormat"].value,
            "ErcEnable": nm["ErcEnable"].value,
            "ErcRateLimit": nm["ErcRateLimit"].value,
        }

        nm["AcquisitionMode"].value = "Continuous"
        tl["StreamBufferHandlingMode"].value = "NewestOnly"
        nm["EventFormat"].value = "EVT3_0"
        nm["ErcEnable"].value = True
        nm["ErcRateLimit"].value = ERC_RATE_LIMIT_MEV
        tl["StreamEvsOutputFormat"].value = EVS_OUTPUT_FORMAT

        fps = tl["StreamFrameGeneratorFPS"].value
        tl["StreamFrameGeneratorAccumTime"].value = int(1_000_000 / fps)

        return saved

    def _apply_biases(self, nm, settings: dict) -> dict:
        """Write the bias nodes; return whatever each one held beforehand.

        Shared by the pre-stream configuration and the live mailbox drain, so a
        bias set the same way in both places cannot drift between them. Every
        access stays inside try/except, matching the rest of this file: a node
        the firmware does not expose must not take the stream down.
        """
        saved = {}
        applied = {}
        for node_name, new_val in settings.items():
            try:
                node = nm[node_name]
                saved[node_name] = node.value
                node.value = new_val
                applied[node_name] = node.value
            except Exception:
                pass
        if applied:
            with QMutexLocker(self._lock):
                self._biases.update(applied)
        return saved

    def _configure_noise_filters(self, device) -> dict:
        return self._apply_biases(device.nodemap, self.biases())

    def _probe_bias_nodes(self, nm) -> tuple[dict, dict, bool]:
        """VERIFY-FIRST (plan §5): are these nodes writable WHILE streaming?

        Every write in this file before this change happened before
        start_stream(), so the answer was never established. IMX636 biases are
        generally live-writable, but ArenaSDK may mark the nodes read-only while
        acquiring. This asks the nodemap directly, after the stream is up, and
        the UI is built around the answer rather than around an assumption.

        Also reads each node's legal range, which is the second unknown -- the
        spinboxes are seeded from node.min/node.max where they exist and fall
        back to the constants where they do not.
        """
        settings, ranges = {}, {}
        writable = True
        for name in BIAS_NODES:
            try:
                node = nm[name]
                settings[name] = node.value
                if not bool(getattr(node, "is_writable", True)):
                    writable = False
                lo, hi = getattr(node, "min", None), getattr(node, "max", None)
                if lo is not None and hi is not None:
                    ranges[name] = (lo, hi)
            except Exception:
                continue
        return settings, ranges, writable

    @staticmethod
    def _decode_xytp(buffer) -> np.ndarray:
        """Copy a LUCID_LucidXYTP128f buffer into an (N, 4) float32 array.

        Each fired pixel is 4x float32 (x, y, t, p); only pixels that had an
        event are present (size_filled, not width*height). The copy is required
        because the buffer is requeued to the camera immediately after.
        """
        bytes_per_event = max(1, buffer.bits_per_pixel // 8)  # 16 for 128f
        n = buffer.size_filled // bytes_per_event
        if n == 0:
            return np.empty((0, 4), dtype=np.float32)
        raw = (ctypes.c_float * (n * 4)).from_address(
            ctypes.addressof(buffer.pbytes)
        )
        return np.frombuffer(raw, dtype=np.float32).reshape(n, 4).copy()

    def _restore_settings(self, device, *saved_dicts):
        nm = device.nodemap
        for saved in saved_dicts:
            for key, val in saved.items():
                try:
                    nm[key].value = val
                except Exception:
                    pass

    def run(self):
        self._running = True
        devices = self._connect_device()
        if not devices:
            self.error.emit("No camera found. Check connection and retry.")
            return

        # Camera enumerates once per transport (GVCP + TCP); pick the first
        # and don't fall through to system.select_device(), which prompts on stdin.
        device = devices[0]
        nm = device.nodemap
        tl = device.tl_stream_nodemap
        width = nm["Width"].value
        height = nm["Height"].value

        self.connected.emit(width, height)
        self.status_message.emit(f"Connected: {width}x{height}")

        saved_evs = self._configure_evs(device)
        saved_noise = self._configure_noise_filters(device)
        device.start_stream(NUM_BUFFERS)

        # Answered on hardware, after start_stream, and reported to the UI --
        # see _probe_bias_nodes.
        bias_now, bias_ranges, bias_writable = self._probe_bias_nodes(nm)
        self.bias_state.emit(bias_now, bias_ranges, bias_writable)
        self.status_message.emit(
            "Biases writable while streaming" if bias_writable else
            "Biases are READ-ONLY while streaming — stop the stream to change them")

        # The camera delivers buffers far faster than the GUI can paint, so we
        # consume every buffer but only render/emit one accumulated frame per
        # display interval. This keeps raw recording lossless while preventing
        # the Qt signal queue from backing up (which causes ever-growing lag).
        display_interval = 1.0 / DISPLAY_FPS
        last_emit = time.perf_counter()
        accum: list[np.ndarray] = []
        last_frame_id = 0

        try:
            while True:
                with QMutexLocker(self._lock):
                    if not self._running:
                        break

                with QMutexLocker(self._lock):
                    pending, self._pending_biases = self._pending_biases, None
                if pending:
                    self._apply_biases(nm, pending)

                try:
                    buffer = device.get_buffer(timeout=IMAGE_TIMEOUT_MS)
                except Exception:
                    continue

                if buffer.is_incomplete:
                    device.requeue_buffer(buffer)
                    continue

                last_frame_id = buffer.frame_id
                events = self._decode_xytp(buffer)
                device.requeue_buffer(buffer)

                # Lossless: every buffer is recorded, regardless of display rate.
                if self.raw_recorder is not None and self.raw_recorder.is_recording:
                    self.raw_recorder.submit(events)

                # The probe needs EVENTS, not the rendered BGR image _on_frame
                # receives, and it needs every batch: ev/lit-px is a count per
                # pixel, so a sampled stream would read a fraction of the truth.
                if self.probe_worker is not None:
                    self.probe_worker.submit(events)

                accum.append(events)

                now = time.perf_counter()
                if now - last_emit < display_interval:
                    continue
                last_emit = now

                t0 = time.perf_counter()
                batch = (
                    np.concatenate(accum)
                    if accum
                    else np.empty((0, 4), dtype=np.float32)
                )
                accum.clear()
                bgr, pos_count, neg_count = render_events(batch, width, height)
                render_ms = (time.perf_counter() - t0) * 1000.0

                stats = {
                    "frame_id": last_frame_id,
                    "event_rate": tl["StreamEvsEventRate"].value,
                    "gvsp_fps": tl["StreamEvsGvspFrameRate"].value,
                    "throughput": tl["StreamEvsLinkThroughput"].value,
                    "render_ms": render_ms,
                    "pos_count": pos_count,
                    "neg_count": neg_count,
                    "biases": self.biases(),
                }

                self.frame_ready.emit(bgr, stats)

        finally:
            try:
                device.stop_stream()
            except Exception:
                pass
            self._restore_settings(device, saved_evs, saved_noise)
            system.destroy_device()
