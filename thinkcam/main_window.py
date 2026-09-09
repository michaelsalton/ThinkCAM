import os
import time
from datetime import datetime

import cv2
import numpy as np
from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from thinkcam.camera_worker import CameraWorker
from thinkcam.constants import PROBE_PYTHON, PROBE_WORK_DIR
from thinkcam.controls import ControlPanel
from thinkcam.dashboard import DashboardPanel
from thinkcam.derivative_plot import DerivativePlotWindow
from thinkcam.probe_worker import ProbeWorker
from thinkcam.raw_recorder import RawEventRecorder
from thinkcam.recorder import VideoRecorder
from thinkcam.status_bar import StatsStatusBar

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ThinkCam  \u2014  LUCID TRT009S-E EVS")

        self._cam_width = 0
        self._cam_height = 0
        self._last_bgr: np.ndarray | None = None
        self._save_dir = "evs_captures"
        self._save_idx = 0

        self._flash_filter_enabled = False
        self._flash_threshold = 0.70
        # Need at least this fraction of pixels active before the imbalance
        # ratio is meaningful — otherwise random noise on a few pixels can
        # look highly imbalanced.
        self._flash_min_activity_frac = 0.01

        self._recorder = VideoRecorder(self._save_dir)
        self._raw_recorder = RawEventRecorder()
        self._probe_worker = ProbeWorker()
        self._worker = CameraWorker()
        # The worker submits raw event batches straight to the recorder from the
        # acquisition thread (the recorder's queue is thread-safe). The live
        # probe takes the same treatment for the same reason.
        self._worker.raw_recorder = self._raw_recorder
        self._worker.probe_worker = self._probe_worker
        self._plot_window = DerivativePlotWindow()
        # The full probe runs under QProcess: a mapper pass is minutes long and
        # must not take the GUI with it.
        self._probe_process: QProcess | None = None
        self._probe_out = ""
        self._probe_started_at = 0.0

        self._build_ui()
        self._connect_signals()
        self._setup_shortcuts()

        # Start camera worker
        self._probe_worker.start()
        self._worker.start()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        # Viewport
        self._viewport = QLabel()
        self._viewport.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._viewport.setMinimumSize(640, 480)
        self._viewport.setStyleSheet("background-color: #1a1a1a;")
        self._viewport.setText("Connecting to camera…")
        self._viewport.setStyleSheet(
            "background-color: #1a1a1a; color: #666; font-size: 16px;"
        )
        layout.addWidget(self._viewport, stretch=1)

        # Sidebar: controls on top, dashboard filling what is left.
        self._controls = ControlPanel()
        self._dashboard = DashboardPanel()
        sidebar = QWidget()
        sidebar.setFixedWidth(260)
        side_col = QVBoxLayout(sidebar)
        side_col.setContentsMargins(0, 0, 0, 0)
        side_col.setSpacing(0)
        side_col.addWidget(self._controls)
        side_col.addWidget(self._dashboard, stretch=1)
        layout.addWidget(sidebar)

        # Status bar
        self._status_bar = StatsStatusBar()
        self.setStatusBar(self._status_bar)

    def _connect_signals(self):
        # Worker -> UI
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.connected.connect(self._on_connected)
        self._worker.error.connect(self._on_error)
        self._worker.status_message.connect(self._on_status)
        self._worker.bias_state.connect(self._controls.set_bias_state)
        self._probe_worker.probe_ready.connect(self._dashboard.update_live)

        # Controls -> UI
        self._controls.save_requested.connect(self._save_frame)
        self._controls.record_toggled.connect(self._toggle_recording)
        self._controls.raw_record_toggled.connect(self._toggle_raw_recording)
        self._controls.plots_requested.connect(self._show_plots)
        self._controls.flash_filter_toggled.connect(self._on_flash_filter_toggled)
        self._controls.flash_threshold_changed.connect(self._on_flash_threshold_changed)
        self._controls.bias_changed.connect(self._worker.set_biases)
        self._dashboard.probe_requested.connect(self._run_probe)

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("S"), self, self._save_frame)
        QShortcut(QKeySequence("P"), self, self._show_plots)
        QShortcut(QKeySequence("R"), self, self._toggle_raw_recording_shortcut)
        QShortcut(QKeySequence("D"), self, self._run_probe)
        QShortcut(QKeySequence("Q"), self, self.close)
        QShortcut(QKeySequence("Escape"), self, self.close)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_connected(self, width: int, height: int):
        self._cam_width = width
        self._cam_height = height
        self._probe_worker.set_geometry(width, height)
        self._viewport.setStyleSheet("background-color: #1a1a1a;")
        self._viewport.setText("")

    def _on_error(self, msg: str):
        QMessageBox.critical(self, "Camera Error", msg)

    def _on_status(self, msg: str):
        self._status_bar.showMessage(msg, 3000)

    def _on_frame(self, bgr: np.ndarray, stats: dict):
        if self._is_global_flash(stats):
            bgr = np.full_like(bgr, 128)
            stats = {**stats, "flash_filtered": True}

        self._last_bgr = bgr.copy()

        if self._recorder.is_recording:
            self._recorder.write_frame(bgr)

        # Raw event batches are fed to the recorder inside the camera worker;
        # here we just surface its stats.
        if self._raw_recorder.is_recording:
            self._status_bar.update_raw_stats(self._raw_recorder.stats())

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        scaled = pixmap.scaled(
            self._viewport.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self._viewport.setPixmap(scaled)

        self._status_bar.update_stats(stats)

        if self._plot_window.isVisible():
            self._plot_window.push(
                stats.get("frame_id", 0),
                stats.get("pos_count", 0),
                stats.get("neg_count", 0),
            )

    def _is_global_flash(self, stats: dict) -> bool:
        if not self._flash_filter_enabled:
            return False
        pos = stats.get("pos_count", 0)
        neg = stats.get("neg_count", 0)
        total = pos + neg
        min_activity = self._cam_width * self._cam_height * self._flash_min_activity_frac
        if total < min_activity:
            return False
        imbalance = abs(pos - neg) / total
        return imbalance >= self._flash_threshold

    def _on_flash_filter_toggled(self, enabled: bool):
        self._flash_filter_enabled = enabled

    def _on_flash_threshold_changed(self, value: float):
        self._flash_threshold = value

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _save_frame(self):
        if self._last_bgr is None:
            return
        os.makedirs(self._save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self._save_dir, f"evs_{ts}_{self._save_idx:04d}.png")
        cv2.imwrite(path, self._last_bgr)
        self._save_idx += 1
        self._status_bar.showMessage(f"Saved {path}", 3000)

    def _show_plots(self):
        self._plot_window.show()
        self._plot_window.raise_()
        self._plot_window.activateWindow()

    def _toggle_recording(self, start: bool):
        if start:
            if self._cam_width == 0:
                return
            self._recorder.start(self._cam_width, self._cam_height)
            self._status_bar.showMessage("Recording started…", 2000)
        else:
            path = self._recorder.stop()
            if path:
                self._status_bar.showMessage(f"Saved recording: {path}", 5000)

    def _toggle_raw_recording(self, start: bool):
        if start:
            if self._cam_width == 0:
                # Camera not connected yet — revert the button.
                self._controls.set_raw_recording(False)
                return
            session = self._raw_recorder.start(
                self._cam_width, self._cam_height, self._controls.take_label(),
                # What the camera actually has in effect, not what constants.py
                # says — the two can now differ.
                biases=self._worker.biases(),
            )
            self._status_bar.showMessage(f"Raw recording → {session}", 2000)
        else:
            session = self._raw_recorder.stop()
            self._status_bar.clear_raw_stats()
            if session:
                # Arm Run Probe for the take that just finished.
                self._dashboard.set_take(session)
                self._status_bar.showMessage(f"Saved raw take: {session}", 5000)

    def _toggle_raw_recording_shortcut(self):
        # Flip the control button; it emits raw_record_toggled -> _toggle_raw_recording.
        self._controls.set_raw_recording(not self._raw_recorder.is_recording)

    # ------------------------------------------------------------------
    # Probe
    # ------------------------------------------------------------------

    def _run_probe(self):
        """Score the last finished take. Runs capture_probe.py under QProcess.

        Out of process and on a different interpreter: the analysis tools need
        numpy/h5py/cv2 and the colmap binary, which are not in the GUI venv, and
        a mapper pass is minutes long.
        """
        if self._probe_process is not None:
            self._status_bar.showMessage("A probe is already running.", 3000)
            return
        take = self._dashboard.take_dir()
        if not take:
            self._status_bar.showMessage(
                "No finished take yet — record one with R first.", 4000)
            return
        if not os.path.exists(PROBE_PYTHON):
            self._status_bar.showMessage(
                f"No analysis interpreter at {PROBE_PYTHON} "
                "(set THINKCAM_PROBE_PYTHON).", 8000)
            return

        out = os.path.join(PROBE_WORK_DIR, os.path.basename(take))
        proc = QProcess(self)
        proc.setWorkingDirectory(REPO_ROOT)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_probe_output)
        proc.finished.connect(self._on_probe_finished)
        self._probe_process = proc
        self._probe_out = out
        self._probe_started_at = time.time()
        self._dashboard.set_probe_running(True)
        self._dashboard.set_progress("starting…")
        proc.start(PROBE_PYTHON,
                   [os.path.join(REPO_ROOT, "capture_probe.py"),
                    "--input", take, "--out", out])
        self._status_bar.showMessage(f"Probing {os.path.basename(take)}…", 4000)

    def _on_probe_output(self):
        if self._probe_process is None:
            return
        chunk = bytes(self._probe_process.readAllStandardOutput()).decode(
            "utf-8", "replace")
        # Only the last line that said anything; the mapper emits thousands.
        lines = [l for l in chunk.splitlines() if l.strip()]
        if lines:
            self._dashboard.set_progress(lines[-1])

    def _on_probe_finished(self, code: int, _status):
        self._probe_process = None
        self._dashboard.set_probe_running(False)
        path = os.path.join(self._probe_out, "probe.json")
        # Exit 1 means gates FAILED, which is a result, not an error — the only
        # real failure is not producing a probe.json for THIS run. The mtime
        # check is what makes that distinction: a probe that died in the mapper
        # leaves the previous run's probe.json sitting there, and reporting it
        # as this run's result would be the one lie the panel must not tell.
        fresh = (os.path.exists(path)
                 and os.path.getmtime(path) >= self._probe_started_at - 1)
        if fresh and self._dashboard.load_probe(path):
            self._dashboard.set_progress(
                "done" + ("" if code == 0 else "  (gates failed)"))
            self._status_bar.showMessage(
                f"Probe finished: {'passed' if code == 0 else 'gates failed'}"
                f" — {path}", 8000)
        else:
            self._dashboard.set_progress(f"probe failed (exit {code})")
            self._status_bar.showMessage(
                f"Probe failed (exit {code}); see {self._probe_out}/*.log", 8000)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._worker.stop()
        self._worker.wait(5000)
        # After the camera worker, so no more batches arrive mid-computation.
        self._probe_worker.stop()
        self._probe_worker.wait(5000)
        if self._probe_process is not None:
            # A mapper pass would otherwise outlive the window it reports to.
            # Disconnect first: kill() fires finished(), and the handler would
            # be repainting widgets that are on their way out.
            proc, self._probe_process = self._probe_process, None
            proc.finished.disconnect(self._on_probe_finished)
            proc.kill()
            proc.waitForFinished(3000)
        self._recorder.stop()
        # Stop after the worker so no more batches arrive mid-flush; this also
        # writes the metadata sidecar.
        self._raw_recorder.stop()
        self._plot_window.close()
        event.accept()
