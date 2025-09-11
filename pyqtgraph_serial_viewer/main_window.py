# main_window.py
import time
import queue
from collections import deque
from logger import CsvWriter
from typing import List, Optional

from ring import Ring2D
import numpy as np

import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import serial
import serial.tools.list_ports

from config import (
    DEFAULT_BAUD, TARGET_HZ, Y_LIMITS,
    MAX_POINTS_PER_CHANNEL, LINE_WIDTH, LINE_COLORS
)
from serial_reader import SerialReader


class MultiPlot(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Serial Streaming → PyQtGraph (request-on-receive)")
        pg.setConfigOptions(antialias=False, useOpenGL=False, background='w')

        # --- Logging ---
        self.log_q = queue.Queue(maxsize=10_000) # buffer up to 10k rows
        self.logger = None  # created on connect when we know channel count

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
        h.addWidget(self.connect_btn)
        vbox.insertWidget(0, ctrl)  # top

        # --- Status bar ---
        self.sb = self.statusBar()
        self._last_status_t = 0.0
        self._status_every = 0.25  # update the status bar at ~4 Hz

        # --- Runtime state (full history only) ---
        self.reader: Optional[SerialReader] = None
        self.connected = False
        self.paused = False

        self.curves: List[pg.PlotDataItem] = []
        self.num_ch: Optional[int] = None
        self.ring = None

        # Flicker-reduction: debounce Y bumps
        self._last_y_top = self.y_limit[1]
        self._y_bump_margin = 0.05    # bump y-top if >5% above current top
        self._y_update_ms = 200       # y-top can update at most 5x/sec
        self._last_y_update_t = 0.0

        # --- Signals ---
        self.refresh_btn.clicked.connect(self.populate_ports)
        self.connect_btn.clicked.connect(self.toggle_connection)

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
        self.reader = SerialReader(device, baud)
        self.reader.sample.connect(self._on_sample)
        self.reader.status.connect(self._on_status)
        self.reader.connected.connect(self._on_connected)
        self.reader.disconnected.connect(self._on_disconnected)
        self.reader.start()
        self.connect_btn.setEnabled(False)

    def stop_reader(self):
        if self.reader and self.reader.isRunning():
            self.reader.stop()
            self.reader.wait(1000)
        self.reader = None

        # stop logger last
        if self.logger:
            self.logger.stop()
            self.logger.join(timeout=1.0)
            self.logger = None

        self.connected = False
        pg.setConfigOptions(antialias=True)
        self.connect_btn.setText("Connect")
        self.connect_btn.setEnabled(True)
        self.enable_controls(True)
        self.sb.showMessage("Disconnected")

    def enable_controls(self, enable: bool):
        self.port_combo.setEnabled(enable)
        self.refresh_btn.setEnabled(enable)
        self.baud_combo.setEnabled(enable)

    # ---------- Reader slots ----------
    def _on_status(self, msg: str):
        self.sb.showMessage(msg, 5000)

    def _on_connected(self):
        self.connected = True
        pg.setConfigOptions(antialias=False)  # fast while streaming
        self.connect_btn.setText("Disconnect")
        self.connect_btn.setEnabled(True)
        self.enable_controls(False)

    def _on_disconnected(self):
        self.connected = False
        pg.setConfigOptions(antialias=True)   # pretty when idle
        self.connect_btn.setText("Connect")
        self.connect_btn.setEnabled(True)
        self.enable_controls(True)

    # ---------- Data & drawing ----------
    def _on_sample(self, vals: np.ndarray):
        if self.paused:
            return

        if self.num_ch is None:
            self.num_ch = int(vals.size)
            # Allocate ring with your existing cap (72_000)
            cap = MAX_POINTS_PER_CHANNEL or 72_000
            self.ring = Ring2D(self.num_ch, cap, dtype=np.float32)

            # Color palette (extend if needed)
            colors = list(LINE_COLORS)
            if self.num_ch > len(colors):
                import random
                random.seed(1234)
                while len(colors) < self.num_ch:
                    colors.append(tuple(random.choices(range(256), k=3)))

            # Create one curve per channel with performance opts
            for i in range(self.num_ch):
                name = f"ch{i+1}"
                color = colors[i % len(colors)]
                curve = self.plot.plot(pen=pg.mkPen(color=color, width=LINE_WIDTH), name=name)
                curve.setClipToView(True)
                # Version-safe downsampling:
                try:
                    curve.setDownsampling(method="peak")   # newer pyqtgraph
                except TypeError:
                    try:
                        curve.setDownsampling(mode="peak") # some versions
                    except TypeError:
                        curve.setDownsampling(auto=True)   # oldest fallback
                self.curves.append(curve)

            self._on_status(f"Detected {self.num_ch} channel(s)")

            # Start CSV logger now that we know channel count
            header = ["timestamp"] + [f"ch{i+1}" for i in range(self.num_ch)]
            self.logger = CsvWriter(self.log_q, rotate_minutes=60, header=header)
            self.logger.start()

        # Append new values (full history for plotting)
        n = min(self.num_ch, vals.size)
        for i in range(n):
            self.buffers[i].append(vals[i])

        # Enqueue for logging (lossless capture)
        try:
            self.log_q.put_nowait((time.time(), *vals[:n].tolist()))
        except queue.Full:
            # Optional: keep a drop counter and show an occasional warning
            pass

    def _redraw(self):
        if not self.ring or self.ring.n == 0:
            return

        Y = self.ring.as_2d()          # shape: (C, n), chronological
        length = Y.shape[1]
        x = np.arange(length)

        # Optional coarse decimation (rarely needed at 72k, but keep your logic)
        decim = 1
        if length > 500_000:
            decim = 16
        elif length > 200_000:
            decim = 8
        elif length > 100_000:
            decim = 4
        x_plot = x[::decim] if decim > 1 else x

        # Update curves (no allocations besides slicing)
        for i, curve in enumerate(self.curves):
            y = Y[i]
            y_plot = y[::decim] if decim > 1 else y
            curve.setData(x_plot, y_plot)

        # Debounced y-range bump using the array directly
        max_val = float(Y.max()) if length else 0.0
        now = time.perf_counter()
        need_bump = max_val > (self._last_y_top * (1 + self._y_bump_margin))
        if need_bump and (now - self._last_y_update_t) >= (self._y_update_ms / 1000.0):
            self._last_y_top = max_val
            self.plot.enableAutoRange(axis='y', enable=False)
            self.plot.setYRange(self.y_limit[0], self._last_y_top, padding=0)
            self._last_y_update_t = now

        # ---- Status bar (throttled) ----
        if (now - self._last_status_t) >= self._status_every:
            self._last_status_t = now
            self.sb.showMessage(
                f"{'Connected' if self.connected else 'Idle'} | "
                f"{self.num_ch or 0} ch | {length} samples | P=Pause, C=Clear"
            )

    # ---------- Helpers ----------
    def _toggle_pause(self):
        self.paused = not self.paused
        pg.setConfigOptions(antialias=self.paused)
        self.plot.repaint()
        self.sb.showMessage("Paused" if self.paused else "Resumed", 2000)

    def _clear_buffers(self):
        if self.ring:
            # Just re-init the ring with same shape
            self.ring = Ring2D(self.num_ch, self.ring.N, dtype=np.float32)
        self._last_y_top = self.y_limit[1]
        self._last_y_update_t = 0.0
        self.sb.showMessage("Cleared buffers", 2000)

    def reset_plot_buffers(self):
        self.plot.clear()
        self.plot.addLegend()
        self.curves = []
        self.ring = None
        self.num_ch = None
        self._last_y_top = self.y_limit[1]
        self._last_y_update_t = 0.0

    def closeEvent(self, event):
        try:
            if self.reader and self.reader.isRunning():
                self.reader.stop()
                self.reader.wait(1000)
        except Exception:
            pass
        # Ensure logger is stopped even if window closed without pressing "Disconnect"
        try:
            if self.logger:
                self.logger.stop()
                self.logger.join(timeout=1.0)
                self.logger = None
        except Exception:
            pass
        return super().closeEvent(event)
