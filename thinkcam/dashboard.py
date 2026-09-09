"""The capture dashboard: live statistics above, last probe's gate table below.

Closes the loop plans/01_CaptureDashboard.md opens -- shoot, probe, adjust,
shoot again -- by putting both halves of the verdict in front of the operator
while the camera is still pointed at the scene.

The two halves answer different questions and are deliberately not merged. LIVE
is what a bias change moves within a second: events per lit pixel, how much of
the sensor is lit, how fast the scene is crossing it. LAST PROBE is what only
COLMAP can say, ~13 minutes after the take ends, and its headline is no longer
track length but FRAGMENTS -- the reference run reaches mean track length 3.53
and still breaks into twelve disconnected models, the largest covering 1.04 s.

The gate table is rendered from probe.json's `gates` array without interpreting
any of it: capture_probe prints and serialises from one list, so a metric
added there appears here with no change to this file.
"""

import json
import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from thinkcam.constants import (
    PROBE_EV_PER_LIT_TARGET,
    PROBE_EVENTS_PER_FRAME,
    PROBE_PYTHON,
    PROBE_REFERENCE_S,
    PROBE_WORK_DIR,
)

OK, WARN, BAD, MUTED = "#4caf50", "#e0a52a", "#e05252", "#8a8a8a"

# The band Phase1b-Results.md §5 measured on a hand-held orbit. Outside it the
# reading is not wrong, it is just not the regime any of the recorded numbers
# describe -- so it is amber, not red.
SPEED_BAND = (500.0, 2200.0)


def _swatch(text_color):
    return f"color: {text_color}; font-family: monospace; font-size: 12px;"


class DashboardPanel(QWidget):
    """Live probe readouts + the last full probe's gates."""

    probe_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._take_dir = None
        self._probe_running = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self._build_live())
        layout.addWidget(self._build_probe(), stretch=1)

        self._set_live_placeholder()
        self._refresh_run_button()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _build_live(self):
        box = QGroupBox("Live")
        grid = QGridLayout(box)
        grid.setContentsMargins(8, 6, 8, 8)
        grid.setVerticalSpacing(3)
        grid.setColumnStretch(1, 1)

        self._live = {}
        rows = [
            ("pipeline", f"ev/lit-px @ {PROBE_EVENTS_PER_FRAME // 1000}k",
             "Events per lit pixel over the pipeline window -- the same "
             f"{PROBE_EVENTS_PER_FRAME:,}-event window the frames COLMAP sees are "
             "built from. Informational: a >= 5 target is unreachable at this "
             "window by construction (the reference take reads 1.07)."),
            ("reference", f"ev/lit-px @ {PROBE_REFERENCE_S:g}s",
             f"Events per lit pixel over a fixed {PROBE_REFERENCE_S:g} s window. "
             f"Phase1b-Results.md §9.1's >= {PROBE_EV_PER_LIT_TARGET:g} target is "
             "hung on this one, because it barely moves with window length and "
             "is therefore close to a pure sensor property -- which is what "
             "makes it the right live readout for a bias change."),
            ("coverage", "lit coverage",
             "Share of the sensor hit at least once in the reference window."),
            ("speed", "camera speed",
             "Global 2-DOF motion from contrast maximization, in px/s. The "
             f"recorded hand-held orbit band is {SPEED_BAND[0]:.0f}-"
             f"{SPEED_BAND[1]:.0f} px/s."),
            ("rate", "event rate",
             "Events per second across the reference window."),
        ]
        for r, (key, label, tip) in enumerate(rows):
            name = QLabel(label)
            name.setStyleSheet("color: #aaa; font-size: 11px;")
            name.setToolTip(tip)
            value = QLabel("--")
            value.setStyleSheet(_swatch(MUTED))
            value.setAlignment(Qt.AlignmentFlag.AlignRight)
            value.setToolTip(tip)
            grid.addWidget(name, r, 0)
            grid.addWidget(value, r, 1)
            self._live[key] = value
        return box

    def _build_probe(self):
        box = QGroupBox("Last probe")
        col = QVBoxLayout(box)
        col.setContentsMargins(8, 6, 8, 8)
        col.setSpacing(6)

        self._take_label = QLabel("no take yet")
        self._take_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._take_label.setWordWrap(True)
        col.addWidget(self._take_label)

        self._run_btn = QPushButton("Run Probe  (D)")
        self._run_btn.clicked.connect(self.probe_requested.emit)
        col.addWidget(self._run_btn)

        self._progress = QLabel("")
        self._progress.setStyleSheet("color: #888; font-size: 10px;")
        self._progress.setWordWrap(True)
        # Two lines' worth, fixed, so a long COLMAP line cannot reflow the panel.
        self._progress.setMinimumHeight(28)
        self._progress.setAlignment(Qt.AlignmentFlag.AlignTop)
        col.addWidget(self._progress)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        col.addWidget(sep)

        self._gates = QTreeWidget()
        self._gates.setColumnCount(2)
        self._gates.setHeaderLabels(["gate", "value"])
        self._gates.setRootIsDecorated(False)
        self._gates.setUniformRowHeights(True)
        self._gates.setAlternatingRowColors(True)
        self._gates.header().setStretchLastSection(False)
        self._gates.setColumnWidth(0, 150)
        self._gates.setSizePolicy(QSizePolicy.Policy.Preferred,
                                  QSizePolicy.Policy.Expanding)
        self._gates.setEnabled(False)
        col.addWidget(self._gates, stretch=1)

        self._verdict = QLabel("run a probe on a finished take")
        self._verdict.setStyleSheet("color: #888; font-size: 11px;")
        self._verdict.setWordWrap(True)
        col.addWidget(self._verdict)
        return box

    # ------------------------------------------------------------------
    # live tier
    # ------------------------------------------------------------------

    def _set_live_placeholder(self):
        for v in self._live.values():
            v.setText("--")
            v.setStyleSheet(_swatch(MUTED))

    def update_live(self, s: dict):
        """Slot for ProbeWorker.probe_ready."""
        pipe = s.get("pipeline_ev_per_lit_px", 0.0)
        self._live["pipeline"].setText(f"{pipe:.2f}")
        self._live["pipeline"].setStyleSheet(_swatch(MUTED))

        ref = s.get("reference_ev_per_lit_px", 0.0)
        target = s.get("reference_target", PROBE_EV_PER_LIT_TARGET)
        colour = OK if ref >= target else (WARN if ref >= 0.5 * target else BAD)
        span = s.get("reference_window_s", 0.0)
        self._live["reference"].setText(f"{ref:.2f}")
        self._live["reference"].setStyleSheet(_swatch(colour))
        # The window can be short right after a start or a clear, and a reading
        # over 0.3 s is not the reading the gate is about.
        self._live["reference"].setToolTip(
            f"{ref:.3f} ev/lit-px over {span:.2f} s of buffer "
            f"({s.get('reference_events', 0):,} events); target >= {target:g}")

        cov = s.get("reference_lit_coverage", 0.0)
        self._live["coverage"].setText(f"{100 * cov:.1f}%")
        self._live["coverage"].setStyleSheet(_swatch(MUTED))

        speed = s.get("speed_px_s", float("nan"))
        if speed != speed:  # NaN
            self._live["speed"].setText("--")
            self._live["speed"].setStyleSheet(_swatch(MUTED))
        else:
            in_band = SPEED_BAND[0] <= speed <= SPEED_BAND[1]
            self._live["speed"].setText(f"{speed:.0f} px/s")
            self._live["speed"].setStyleSheet(_swatch(OK if in_band else WARN))

        rate = s.get("event_rate", 0.0)
        self._live["rate"].setText(f"{rate / 1e3:.0f} Kev/s")
        self._live["rate"].setStyleSheet(_swatch(MUTED))

    def clear_live(self):
        self._set_live_placeholder()

    # ------------------------------------------------------------------
    # probe tier
    # ------------------------------------------------------------------

    def set_take(self, session_dir: str | None):
        """Arm Run Probe for a take that has finished recording."""
        self._take_dir = session_dir
        self._take_label.setText(
            os.path.basename(session_dir) if session_dir else "no take yet")
        self._refresh_run_button()
        # A probe already sitting next to that take is the honest thing to show.
        if session_dir:
            self.load_probe(self.probe_json_path(session_dir), quiet=True)

    def take_dir(self):
        return self._take_dir

    @staticmethod
    def probe_json_path(session_dir: str) -> str:
        return os.path.join(PROBE_WORK_DIR, os.path.basename(session_dir),
                            "probe.json")

    def set_probe_running(self, running: bool):
        self._probe_running = running
        self._refresh_run_button()

    def _refresh_run_button(self):
        if not os.path.exists(PROBE_PYTHON):
            # Say why here rather than failing at subprocess time: the analysis
            # interpreter is a separate environment from the GUI's and its
            # absence is a setup problem, not a capture problem.
            self._run_btn.setEnabled(False)
            self._run_btn.setToolTip(
                f"No analysis interpreter at {PROBE_PYTHON}. The GUI venv "
                "carries arena_api + PySide6; the probe needs numpy/h5py/cv2 "
                "and the colmap binary. Set THINKCAM_PROBE_PYTHON.")
            self._run_btn.setText("Run Probe  (no interpreter)")
            return
        self._run_btn.setToolTip(
            "Accumulate frames, LK-track them, run the COLMAP mapper and score "
            "the gates. CPU-only colmap: ~13 min for a 200-frame take.")
        if self._probe_running:
            self._run_btn.setEnabled(False)
            self._run_btn.setText("Probing…")
        else:
            self._run_btn.setEnabled(self._take_dir is not None)
            self._run_btn.setText("Run Probe  (D)")

    def set_progress(self, text: str):
        self._progress.setText(text.strip()[-160:])

    def load_probe(self, path: str, quiet: bool = False) -> bool:
        """Render probe.json's gate table. False if there is nothing to show."""
        self._gates.clear()
        if not path or not os.path.exists(path):
            self._gates.setEnabled(False)
            if not quiet:
                self._verdict.setText(f"no probe.json at {path}")
                self._verdict.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
            else:
                self._verdict.setText("not probed yet")
                self._verdict.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
            return False
        try:
            with open(path) as f:
                res = json.load(f)
        except (OSError, ValueError) as exc:
            self._verdict.setText(f"unreadable probe.json: {exc}")
            self._verdict.setStyleSheet(f"color: {BAD}; font-size: 11px;")
            return False

        for row in res.get("gates", []):
            item = QTreeWidgetItem([row.get("name", "?"),
                                    _fmt_value(row.get("value"))])
            colour = {"PASS": OK, "FAIL": BAD}.get(row.get("status"), MUTED)
            item.setForeground(1, _brush(colour))
            item.setToolTip(0, row.get("detail", ""))
            item.setToolTip(1, row.get("detail", ""))
            self._gates.addTopLevelItem(item)
        self._gates.setEnabled(True)

        passed = res.get("passed")
        failed = [r["name"] for r in res.get("gates", [])
                  if r.get("status") == "FAIL"]
        self._verdict.setText(
            "ALL GATES PASSED" if passed else "FAILED: " + ", ".join(failed))
        self._verdict.setStyleSheet(
            f"color: {OK if passed else BAD}; font-size: 11px; font-weight: bold;")
        cfg = res.get("config", {})
        n_ev = cfg.get("events_per_frame")
        n_ev = f"{n_ev:,}" if isinstance(n_ev, int) else "?"
        self._take_label.setText(
            f"{res.get('take', '?')}\n{n_ev} ev/frame  "
            f"mc={cfg.get('motion_comp')}  gap={cfg.get('max_gap', '?')}  "
            f"took {res.get('elapsed_s', 0):.0f} s")
        return True


def _fmt_value(v):
    if v is None:
        return "--"
    if isinstance(v, float):
        return f"{v:.2f}" if abs(v) < 1000 else f"{v:.0f}"
    return str(v)


def _brush(colour):
    return QBrush(QColor(colour))
