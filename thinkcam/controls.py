"""Sidebar controls.

LEDGER -- are the bias nodes writable while streaming? (plan §5 VERIFY-FIRST)

    Before this change every nodemap write in the project happened BEFORE
    device.start_stream(); nothing wrote mid-stream, so the question had never
    been asked of the hardware. It is now asked at runtime instead of assumed:
    CameraWorker._probe_bias_nodes() reads node.is_writable for each of the four
    bias nodes AFTER the stream is up and emits the answer as bias_state. This
    panel builds itself around whatever comes back --

      * writable      -> the group is live; edits go to the acquisition thread's
                         pending-bias mailbox and take effect on the next loop.
      * NOT writable  -> the group is disabled with the reason in its tooltip.
                         The fallback the plan names (stop / reconfigure / start)
                         is deliberately NOT offered from here: it would have to
                         be refused outright while raw recording, and a control
                         that silently means two different things depending on
                         recording state is worse than one that means nothing.

    IMX636 biases are generally live-writable, so `writable` is the expected
    answer; this note records how it was established rather than what it was.
    Whoever runs it on hardware first should replace this paragraph with the
    observed result.

The group is also disabled for the duration of a raw take. metadata.json records
ONE bias set per take, so a mid-take change would make that record a lie.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from thinkcam.constants import (
    BIAS_REFRACTORY,
    BIAS_THRESHOLD_NEG,
    BIAS_THRESHOLD_POS,
    BURST_FILTER_ENABLE,
)


class ControlPanel(QWidget):
    save_requested = Signal()
    record_toggled = Signal(bool)
    raw_record_toggled = Signal(bool)
    plots_requested = Signal()
    flash_filter_toggled = Signal(bool)
    flash_threshold_changed = Signal(float)
    bias_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(260)
        self._bias_writable = True

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._save_btn = QPushButton("Save PNG")
        self._save_btn.clicked.connect(self.save_requested.emit)
        layout.addWidget(self._save_btn)

        self._record_btn = QPushButton("Record Video")
        self._record_btn.setCheckable(True)
        self._record_btn.toggled.connect(self._on_record_toggled)
        layout.addWidget(self._record_btn)

        self._take_label_edit = QLineEdit()
        self._take_label_edit.setPlaceholderText("Take label (optional)")
        layout.addWidget(self._take_label_edit)

        self._raw_record_btn = QPushButton("Record RAW")
        self._raw_record_btn.setCheckable(True)
        self._raw_record_btn.setToolTip(
            "Record the lossless raw event stream (x, y, t, p) to "
            "recordings/<timestamp>_<label>/. Independent of video recording."
        )
        self._raw_record_btn.toggled.connect(self._on_raw_record_toggled)
        layout.addWidget(self._raw_record_btn)

        self._plots_btn = QPushButton("Show Plots")
        self._plots_btn.clicked.connect(self.plots_requested.emit)
        layout.addWidget(self._plots_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        self._flash_check = QCheckBox("Suppress flashes")
        self._flash_check.setToolTip(
            "Replace frames dominated by one polarity (lights on/off) "
            "with neutral gray."
        )
        self._flash_check.toggled.connect(self.flash_filter_toggled.emit)
        layout.addWidget(self._flash_check)

        self._threshold_label = QLabel("Imbalance ≥ 70%")
        self._threshold_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._threshold_label)

        self._threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._threshold_slider.setRange(50, 95)
        self._threshold_slider.setValue(70)
        self._threshold_slider.valueChanged.connect(self._on_threshold_changed)
        layout.addWidget(self._threshold_slider)

        layout.addWidget(self._build_biases())
        layout.addStretch()

    def _on_record_toggled(self, checked: bool):
        self._record_btn.setText("Stop Recording" if checked else "Record Video")
        style = "background-color: #cc3333; color: white;" if checked else ""
        self._record_btn.setStyleSheet(style)
        self.record_toggled.emit(checked)

    def _on_raw_record_toggled(self, checked: bool):
        self._raw_record_btn.setText("Stop RAW" if checked else "Record RAW")
        style = "background-color: #cc3333; color: white;" if checked else ""
        self._raw_record_btn.setStyleSheet(style)
        # Lock the label while a take is rolling so it can't change mid-recording.
        self._take_label_edit.setEnabled(not checked)
        self._refresh_bias_enabled()
        self.raw_record_toggled.emit(checked)

    def take_label(self) -> str:
        return self._take_label_edit.text()

    def set_raw_recording(self, recording: bool):
        """Reflect external state (e.g. keyboard shortcut) on the button."""
        if self._raw_record_btn.isChecked() != recording:
            self._raw_record_btn.setChecked(recording)

    def _on_threshold_changed(self, value: int):
        self._threshold_label.setText(f"Imbalance ≥ {value}%")
        self.flash_threshold_changed.emit(value / 100.0)

    # ------------------------------------------------------------------
    # Biases
    # ------------------------------------------------------------------

    def _build_biases(self):
        box = QGroupBox("Biases")
        self._bias_tip = (
            "Sensor contrast thresholds and refractory period. Raising the "
            "thresholds cuts the event rate; watch ev/lit-px on the dashboard "
            "move with it — that pairing is the point of the capture loop.")
        box.setToolTip(self._bias_tip)
        form = QFormLayout(box)
        form.setContentsMargins(8, 6, 8, 8)
        form.setVerticalSpacing(4)

        self._bias_spins = {}
        # Seeded from the constants and re-seeded from node.min/node.max the
        # moment the camera reports them (bias_state).
        spec = [
            ("BiasEventThresholdPositive", "threshold +", BIAS_THRESHOLD_POS),
            ("BiasEventThresholdNegative", "threshold −", BIAS_THRESHOLD_NEG),
            ("BiasRefractoryPeriod", "refractory", BIAS_REFRACTORY),
        ]
        for node, label, default in spec:
            spin = QSpinBox()
            spin.setRange(0, 255)
            spin.setValue(int(default))
            spin.setKeyboardTracking(False)   # one signal per committed edit
            spin.valueChanged.connect(self._emit_biases)
            form.addRow(QLabel(label), spin)
            self._bias_spins[node] = spin

        self._burst_check = QCheckBox("burst filter")
        self._burst_check.setChecked(bool(BURST_FILTER_ENABLE))
        self._burst_check.toggled.connect(self._emit_biases)
        form.addRow(self._burst_check)

        self._bias_note = QLabel("")
        self._bias_note.setStyleSheet("color: #888; font-size: 10px;")
        self._bias_note.setWordWrap(True)
        form.addRow(self._bias_note)

        self._bias_box = box
        return box

    def biases(self) -> dict:
        d = {n: int(s.value()) for n, s in self._bias_spins.items()}
        d["EventBurstFilterEnable"] = bool(self._burst_check.isChecked())
        return d

    def _emit_biases(self, _value=None):
        self.bias_changed.emit(self.biases())

    def set_bias_state(self, settings: dict, ranges: dict, writable: bool):
        """Seed from the hardware: current values, legal ranges, writability."""
        self._bias_writable = writable
        for node, spin in self._bias_spins.items():
            lo, hi = ranges.get(node, (0, 255))
            val = settings.get(node, spin.value())
            spin.blockSignals(True)      # seeding is not an operator edit
            try:
                spin.setRange(int(lo), int(hi))
                spin.setValue(int(val))
            except (TypeError, ValueError):
                pass
            spin.blockSignals(False)
        if "EventBurstFilterEnable" in settings:
            self._burst_check.blockSignals(True)
            self._burst_check.setChecked(bool(settings["EventBurstFilterEnable"]))
            self._burst_check.blockSignals(False)
        if not writable:
            self._bias_note.setText(
                "read-only while streaming — the nodemap refused; restart the "
                "stream to change these")
        else:
            self._bias_note.setText("")
        self._refresh_bias_enabled()

    def _refresh_bias_enabled(self):
        recording = self._raw_record_btn.isChecked()
        self._bias_box.setEnabled(self._bias_writable and not recording)
        self._bias_box.setToolTip(
            "Locked for the take: metadata.json records ONE bias set per "
            "recording, so a mid-take change would make it a lie."
            if recording else self._bias_tip)
