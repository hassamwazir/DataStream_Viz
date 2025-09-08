# main_window.py
import time
from collections import deque
from typing import List, Optional

import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import serial
import serial.tools.list_ports

from config import (
    DEFAULT_BAUD, TARGET_HZ, DEFAULT_POLL_HZ, Y_LIMITS,
    MAX_POINTS_PER_CHANNEL,   # optional safety cap
)
from serial_reader import SerialReader


class MultiPlot(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Serial Streaming → PyQtGraph (polling '1')")
        pg.setConfigOptions(antialias=False, useOpenGL=False)

        # --- Central plot ---
        central = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(8, 8, 8, 8)
        self.setCentralWidget(central)

        self.y_limit = Y_LIMITS

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.addLegend()
        self.plot.setLabel("bottom", "Sample")
        self.plot.setLabel("left", "Value")
        self.plot.setYRange(*self.y_limit)
        vbox.addWidget(self.plot, 1)

        # --- Top control bar ---
        ctrl = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(ctrl)
        h.setContentsMargins(0, 0, 0, 4)

        self.port_combo = QtWidgets.QComboBox()
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.baud_combo = QtWidgets.QComboBox()
        self.connect_btn = QtWidgets.QPushButton("Connect")

        # polling controls
        self.poll_chk = QtWidgets.QCheckBox("Auto-send '1'")
        self.poll_chk.setChecked(True)
        self.poll_hz = QtWidgets.QDoubleSpinBox()
        self.poll_hz.setDecimals(1)
        self.poll_hz.setRange(0.1, 1000.0)
        self.poll_hz.setSingleStep(0.5)
        self.poll_hz.setValue(DEFAULT_POLL_HZ)

        for b in [9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600]:
            self.baud_combo.addItem(str(b))
        self.baud_combo.setCurrentText(str(DEFAULT_BAUD))

        h.addWidget(QtWidgets.QLabel("Port:"))
        h.addWidget(self.port_combo, 2)
        h.addWidget(self.refresh_btn)
        h.addSpacing(12)
        h.addWidget(QtWidgets.QLabel("Baud:"))
        h.addWidget(self.baud_combo)
        h.addSpacing(12)
        h.addWidget(self.poll_chk)
        h.addWidget(QtWidgets.QLabel("Hz:"))
        h.addWidget(self.poll_hz)
        h.addSpacing(12)
        h.addWidget(self.connect_btn)
        vbox.insertWidget(0, ctrl)  # top

        # --- Status bar ---
        self.sb = self.statusBar()

        # --- Runtime state (full history only) ---
        self.reader: Optional[SerialReader] = None
        self.connected = False
        self.paused = False

        self.curves = []
        self.num_ch: Optional[int] = None
        self.buffers: List[deque] = []

        self._samples_seen = 0
        self._t0 = time.perf_counter()
        self._estimated_hz: Optional[float] = None

        # Flicker-reduction: throttle X-range changes and debounce Y bumps
        self._last_x_extend_len = 0
        self._x_extend_step = 1     # extend the x-view every N new points
        self._last_y_top = self.y_limit[1]
        self._y_bump_margin = 0.05    # bump y-top if >5% higher than current top
        self._y_update_ms = 200       # y-top can update at most 5x/sec
        self._last_y_update_t = 0.0

        # --- Signals ---
        self.refresh_btn.clicked.connect(self.populate_ports)
        self.connect_btn.clicked.connect(self.toggle_connection)
        self.poll_chk.toggled.connect(self._apply_poll_settings)
        self.poll_hz.valueChanged.connect(self._apply_poll_settings)

        QtWidgets.QShortcut(QtCore.Qt.Key_P, self, activated=self._toggle_pause)
        QtWidgets.QShortcut(QtCore.Qt.Key_C, self, activated=self._clear_buffers)

        # --- Timer for UI redraws ---
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._redraw)
        self.timer.start(int(1000 / TARGET_HZ))

        # Initial port scan
        self.populate_ports()

    # ---------- Controls ----------
    def populate_ports(self):
        current = self.port_combo.currentText()
        self.port_combo.clear()
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            label = f"{p.device} – {p.description}"
            self.port_combo.addItem(label, userData=p.device)
        if not ports:
            self.port_combo.addItem("(no ports found)", userData=None)
        idx = self.port_combo.findText(current)
        if idx >= 0:
            self.port_combo.setCurrentIndex(idx)
        self.sb.showMessage(f"Found {len(ports)} port(s)")

    def toggle_connection(self):
        if not self.connected:
            device = self.port_combo.currentData()
            if not device:
                self.sb.showMessage("No port selected")
                return
            baud = int(self.baud_combo.currentText())
            self.start_reader(device, baud)
        else:
            self.stop_reader()

    def start_reader(self, device, baud):
        self.reset_plot_buffers()
        self.reader = SerialReader(
            device,
            baud,
            poll_enabled=self.poll_chk.isChecked(),
            poll_hz=float(self.poll_hz.value()),
        )
        self.reader.sample.connect(self._on_sample)
        self.reader.status.connect(self._on_status)
        self.reader.connected.connect(self._on_connected)
        self.reader.disconnected.connect(self._on_disconnected)
        self.reader.start()
        self.connect_btn.setEnabled(False)  # will re-enable once connected

    def stop_reader(self):
        if self.reader and self.reader.isRunning():
            self.reader.stop()
            self.reader.wait(1000)
        self.reader = None
        self.connected = False
        self.connect_btn.setText("Connect")
        self.connect_btn.setEnabled(True)
        self.enable_controls(True)
        self.sb.showMessage("Disconnected")

    def enable_controls(self, enable: bool):
        self.port_combo.setEnabled(enable)
        self.refresh_btn.setEnabled(enable)
        self.baud_combo.setEnabled(enable)
        self.poll_chk.setEnabled(enable)
        self.poll_hz.setEnabled(enable)

    # live-apply polling settings to the running reader
    def _apply_poll_settings(self):
        if self.reader and self.reader.isRunning():
            self.reader.set_polling(self.poll_chk.isChecked(), float(self.poll_hz.value()))

    # ---------- Reader slots ----------
    def _on_status(self, msg: str):
        self.sb.showMessage(msg, 5000)

    def _on_connected(self):
        self.connected = True
        self.connect_btn.setText("Disconnect")
        self.connect_btn.setEnabled(True)
        self.enable_controls(False)  # lock controls while connected
        # make sure current polling UI is applied to the thread
        self._apply_poll_settings()

    def _on_disconnected(self):
        self.connected = False
        self.connect_btn.setText("Connect")
        self.connect_btn.setEnabled(True)
        self.enable_controls(True)

    # ---------- Data & drawing ----------
    def _on_sample(self, vals: np.ndarray):
        if self.paused:
            return

        if self.num_ch is None:
            # First packet: set up channel count and full-history buffers
            self.num_ch = int(vals.size)
            self.buffers = [deque() for _ in range(self.num_ch)]

            # Color palette (extend if needed)
            colors = [
                (255, 0, 0),     # red
                (0, 200, 0),     # green
                (0, 0, 255),     # blue
                (200, 100, 0),   # orange
                (200, 0, 200),   # magenta
                (0, 200, 200),   # cyan
                (150, 150, 150), # gray
            ]

            # Create one curve per channel with performance opts
            for i in range(self.num_ch):
                name = f"ch{i+1}"
                color = colors[i % len(colors)]
                curve = self.plot.plot(pen=pg.mkPen(color=color, width=2), name=name)
                curve.setClipToView(True)            # render only what's visible
                curve.setDownsampling(auto=True)     # auto downsample for speed
                self.curves.append(curve)

            self._on_status(f"Detected {self.num_ch} channel(s)")
            self._samples_seen = 0
            self._t0 = time.perf_counter()
            self._estimated_hz = None

        # Append new values (full history)
        n = min(self.num_ch, vals.size)
        for i in range(n):
            self.buffers[i].append(vals[i])

        # Optional safety cap for memory
        if MAX_POINTS_PER_CHANNEL:
            for i in range(self.num_ch):
                extra = len(self.buffers[i]) - MAX_POINTS_PER_CHANNEL
                if extra > 0:
                    for _ in range(extra):
                        self.buffers[i].popleft()

        # Keep an estimate of input rate for the status bar
        self._samples_seen += 1
        if self._samples_seen % 50 == 0:
            dt = time.perf_counter() - self._t0
            if dt > 0.2:
                self._estimated_hz = self._samples_seen / dt

    def _redraw(self):
        if not self.buffers:
            return
        length = max((len(b) for b in self.buffers), default=0)
        if length == 0:
            return

        # Build x once per frame
        x = np.arange(length)

        # Optional coarse decimation for huge histories (on top of pg's downsampling)
        decim = 1
        if length > 500_000:
            decim = 16
        elif length > 200_000:
            decim = 8
        elif length > 100_000:
            decim = 4

        x_plot = x[::decim] if decim > 1 else x

        # Update curves
        for i, curve in enumerate(self.curves):
            y = np.fromiter(self.buffers[i], dtype=float, count=length)
            y_plot = y[::decim] if decim > 1 else y
            curve.setData(x_plot, y_plot)

        # ---- Throttle x-range updates (reduce label flicker) ----
        if length - self._last_x_extend_len >= self._x_extend_step:
            self.plot.getViewBox().enableAutoRange(axis='x', enable=False)
            self.plot.setXRange(0, max(1, length - 1), padding=0)
            self._last_x_extend_len = length

        # ---- Debounce y-range bumps (only when meaningfully higher) ----
        if any(len(b) for b in self.buffers):
            max_val = max((max(b) for b in self.buffers if len(b) > 0), default=0.0)
        else:
            max_val = 0.0

        now = time.perf_counter()
        need_bump = max_val > (self._last_y_top * (1 + self._y_bump_margin))
        if need_bump and (now - self._last_y_update_t) >= (self._y_update_ms / 1000.0):
            self._last_y_top = max_val
            self.plot.enableAutoRange(axis='y', enable=False)
            self.plot.setYRange(self.y_limit[0], self._last_y_top, padding=0)
            self._last_y_update_t = now

        # ---- Status bar ----
        if self._estimated_hz:
            self.sb.showMessage(
                f"{'Connected' if self.connected else 'Idle'} | "
                f"{self.num_ch or 0} ch | ~{self._estimated_hz:.1f} Hz | "
                f"{length} samples | P=Pause, C=Clear"
            )

    # ---------- Helpers ----------
    def _toggle_pause(self):
        self.paused = not self.paused
        self.sb.showMessage("Paused" if self.paused else "Resumed", 2000)

    def _clear_buffers(self):
        for dq in self.buffers:
            dq.clear()
        self._last_x_extend_len = 0
        self._last_y_top = self.y_limit[1]
        self._last_y_update_t = 0.0
        self.sb.showMessage("Cleared buffers", 2000)

    def reset_plot_buffers(self):
        self.plot.clear()
        self.plot.addLegend()
        self.curves = []
        self.buffers = []
        self.num_ch = None
        self._samples_seen = 0
        self._t0 = time.perf_counter()
        self._estimated_hz = None
        self._last_x_extend_len = 0
        self._last_y_top = self.y_limit[1]
        self._last_y_update_t = 0.0

    def closeEvent(self, event):
        try:
            if self.reader and self.reader.isRunning():
                self.reader.stop()
                self.reader.wait(1000)
        except Exception:
            pass
        return super().closeEvent(event)
