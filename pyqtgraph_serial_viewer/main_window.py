# main_window.py
import time
import queue
from logger import CsvWriter
from typing import List, Optional

from ring import Ring2D
import numpy as np

# from PyQt5 import QtWidgets, QtCore, QtGui
from PySide6 import QtWidgets, QtCore, QtGui

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
        self.log_q = queue.Queue(maxsize=10_000)  # buffer up to 10k rows
        self.logger = None  # created on connect when we know channel count

        # Time origin for elapsed-x axis
        self.t0 = None

        # --- Central plot ---
        central = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(8, 8, 8, 8)
        self.setCentralWidget(central)

        self.y_limit = Y_LIMITS

                # ---- Runtime-configurable copies of config.py values ----
        self.default_baud = DEFAULT_BAUD
        self.target_hz = TARGET_HZ
        self.y_limit = list(Y_LIMITS)  # mutable
        self.max_points_per_channel = MAX_POINTS_PER_CHANNEL
        self.line_width = LINE_WIDTH
        self.line_colors = list(LINE_COLORS)

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.addLegend()
        self.plot.setLabel("bottom", "Time Elapsed (s)")
        self.plot.setLabel("left", "Value")
        self.plot.setYRange(*self.y_limit)
        vbox.addWidget(self.plot, 1)

        # Click-to-select: scatter that renders all picked points
        self.pick_marker = pg.ScatterPlotItem(
            size=10,
            pen=pg.mkPen('k', width=1),
            brush=pg.mkBrush(255, 0, 0, 180),
        )
        self.pick_marker.setZValue(10)
        self.plot.addItem(self.pick_marker)

        # Store selected points; each entry: {"x": float, "y": float, "ch": int, "label": TextItem}
        self.picks: List[dict] = []

        # Reusable label styling + factory
        self._label_font = QtGui.QFont()
        self._label_font.setPointSize(12)
        self._label_font.setBold(True)

        def _make_pick_label(text: str) -> pg.TextItem:
            ti = pg.TextItem(text, anchor=(0, 1))  # bottom-left anchored at point
            ti.setZValue(11)
            ti.setFont(self._label_font)
            ti.setColor(pg.mkColor(0, 0, 0))
            try:
                ti.setFill(pg.mkBrush(255, 255, 255, 230))   # mostly opaque background
                ti.setBorder(pg.mkPen(0, 0, 0, 200))         # thin black border
            except Exception:
                pass
            return ti

        self._make_pick_label = _make_pick_label  # keep factory on the instance

        # Mouse click handler on the plot (uses toggle logic in _on_plot_click)
        self.plot.scene().sigMouseClicked.connect(self._on_plot_click)

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
        self.baud_combo.setCurrentText(str(self.default_baud))

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

        # --- Runtime state ---
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

        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_P), self, activated=self._toggle_pause)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_C), self, activated=self._clear_buffers)

        # --- Timer for UI redraws ---
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._redraw)
        self.timer.start(int(1000 / self.target_hz))

        # Initial port scan
        self.populate_ports()

        # --- File menu: Open CSV ---
        file_menu = self.menuBar().addMenu("&File")

        self.open_csv_act = QtGui.QAction("Open CSV…", self)
        self.open_csv_act.setShortcut(QtGui.QKeySequence.Open)  # Ctrl/Cmd+O
        self.open_csv_act.triggered.connect(self._open_csv_dialog)
        file_menu.addAction(self.open_csv_act)

        # --- Settings menu ---
        settings_menu = self.menuBar().addMenu("&Settings")
        self.settings_act = QtGui.QAction("Preferences…", self)
        # Standard shortcut if available
        try:
            self.settings_act.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Preferences))
        except Exception:
            self.settings_act.setShortcut(QtGui.QKeySequence("Ctrl+,"))
        self.settings_act.triggered.connect(self._open_settings_dialog)
        settings_menu.addAction(self.settings_act)




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
        self.reset_plot_buffers()              # wipes curves, ring, num_ch, picks, etc.
        self.log_q = queue.Queue(maxsize=10_000)  # fresh queue for this session
        self.logger = None                     # force new CsvWriter on first sample

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
            cap = self.max_points_per_channel or 72_000
            self.ring = Ring2D(self.num_ch, cap, dtype=np.float32)

            # Color palette (extend if needed)
            colors = list(self.line_colors)
            if self.num_ch > len(colors):
                import random
                random.seed(1234)
                while len(colors) < self.num_ch:
                    colors.append(tuple(random.choices(range(256), k=3)))

            # One curve per channel + version-safe downsampling
            for i in range(self.num_ch):
                name = f"ch{i+1}"
                color = colors[i % len(colors)]
                curve = self.plot.plot(pen=pg.mkPen(color=color, width=self.line_width), name=name)
                curve.setClipToView(True)
                try:
                    curve.setDownsampling(method="peak")    # newer pyqtgraph
                except TypeError:
                    try:
                        curve.setDownsampling(mode="peak")  # some versions
                    except TypeError:
                        curve.setDownsampling(auto=True)    # oldest fallback
                self.curves.append(curve)

            self._on_status(f"Detected {self.num_ch} channel(s)")

            # Start CSV logger now that we know channel count
            header = ["timestamp"] + [f"ch{i+1}" for i in range(self.num_ch)]
            self.logger = CsvWriter(self.log_q, rotate_minutes=60, header=header)
            self.logger.start()

        # Append new values with a timestamp and log using the SAME timestamp
        n = min(self.num_ch, vals.size)
        t = time.time()
        if getattr(self, "t0", None) is None:
            self.t0 = t  # first-sample origin for elapsed-time x-axis

        self.ring.append(vals[:n], t=t)

        # Enqueue for logging
        try:
            self.log_q.put_nowait((t, *vals[:n].tolist()))
        except queue.Full:
            # Optional: track drops if you care
            pass


    def _redraw(self):
        # Need data and a time origin
        if not self.ring or self.ring.n == 0 or getattr(self, "t0", None) is None:
            return

        t, Y = self.ring.view()   # t: (n,), Y: (C, n) in chronological order
        length = Y.shape[1]
        if length == 0:
            return

        # Elapsed seconds since start
        x = t - self.t0

        # Optional coarse decimation on very large histories
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
            y = Y[i]
            y_plot = y[::decim] if decim > 1 else y
            curve.setData(x_plot, y_plot)

        # Debounced y-range bump (preserve current bottom, expand top as needed)
        max_val = float(Y.max()) if length else 0.0
        now = time.perf_counter()
        need_bump = max_val > (self._last_y_top * (1 + self._y_bump_margin))
        if need_bump and (now - self._last_y_update_t) >= (self._y_update_ms / 1000.0):
            self._last_y_top = max_val
            self.plot.enableAutoRange(axis='y', enable=False)
            self.plot.setYRange(self.y_limit[0], self._last_y_top, padding=0)
            self._last_y_update_t = now

        # Status bar (throttled)
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
        # 1) Clear selected markers & labels
        # -- multi-label version --
        if hasattr(self, "picks") and self.picks:
            for p in self.picks:
                lbl = p.get("label")
                if lbl is not None:
                    try:
                        self.plot.removeItem(lbl)
                    except Exception:
                        pass
            self.picks.clear()

        # -- single-label fallback (if you still have it) --
        if hasattr(self, "pick_label") and self.pick_label is not None:
            try:
                self.pick_label.setText("")
            except Exception:
                pass

        # Clear the scatter points
        if hasattr(self, "pick_marker") and self.pick_marker is not None:
            try:
                self.pick_marker.setData(x=[], y=[])
            except Exception:
                pass

        # 2) Reset the data buffers and time origin
        if self.ring:
            self.ring = Ring2D(self.num_ch, self.ring.N, dtype=np.float32)
        self.t0 = None

        # 3) Reset y-range bump state
        self._last_y_top = self.y_limit[1]
        self._last_y_update_t = 0.0

        # 4) Status
        self.sb.showMessage("Cleared buffers & picks", 2000)


    def reset_plot_buffers(self):
        # remove existing labels from the scene
        for p in getattr(self, "picks", []):
            if p.get("label"):
                try:
                    self.plot.removeItem(p["label"])
                except Exception:
                    pass

        self.plot.clear()
        self.plot.addLegend()

        # re-add scatter after clear
        try:
            self.plot.addItem(self.pick_marker)
        except Exception:
            pass

        self.picks.clear()
        self.curves = []
        self.ring = None
        self.num_ch = None
        self.t0 = None
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
    
    def _on_plot_click(self, ev):
        """Left: toggle nearest point (add/remove). Right: clear all picks."""
        if not self.plot.sceneBoundingRect().contains(ev.scenePos()):
            return

        # Version-safe buttons (PyQt5/6)
        if hasattr(QtCore.Qt, "MouseButton"):
            LEFT = QtCore.Qt.MouseButton.LeftButton
            RIGHT = QtCore.Qt.MouseButton.RightButton
        else:
            LEFT = QtCore.Qt.LeftButton
            RIGHT = QtCore.Qt.RightButton

        # Clear all picks on right-click
        if ev.button() == RIGHT:
            # remove labels from scene
            for p in self.picks:
                if p.get("label"):
                    try:
                        self.plot.removeItem(p["label"])
                    except Exception:
                        pass
            self.picks.clear()
            self._update_picks()
            return

        if ev.button() != LEFT:
            return
        if not self.ring or self.ring.n == 0 or getattr(self, "t0", None) is None:
            return

        # Map click to data coords
        mouse_pt = self.plot.plotItem.vb.mapSceneToView(ev.scenePos())
        x_click = float(mouse_pt.x())   # elapsed seconds
        y_click = float(mouse_pt.y())

        # Snap to nearest sample in time, then nearest channel by Y
        t_abs, Y = self.ring.view()     # t_abs: (n,), Y: (C, n)
        x = t_abs - self.t0
        n = x.size
        if n == 0:
            return

        idx = int(np.searchsorted(x, x_click))
        if idx <= 0:
            i_near = 0
        elif idx >= n:
            i_near = n - 1
        else:
            i_near = idx - 1 if (x_click - x[idx - 1]) <= (x[idx] - x_click) else idx

        y_at = Y[:, i_near]
        ch = int(np.argmin(np.abs(y_at - y_click)))
        x_sel = float(x[i_near])
        y_sel = float(y_at[ch])

        # Toggle: if a pick already near this (x_sel,y_sel), remove it
        # Get the current x-axis range (min_x, max_x)
        # Get the ViewBox of the plot
        vb = self.plot.plotItem.vb
        x_range = vb.viewRange()[0]
        # make dt_tol change with x-range

        dt_tol = 0.1 * (x_range[1] - x_range[0])  # 10% of x-range
        dy_tol = (self.y_limit[1] - self.y_limit[0]) * 0.02  # 2% of Y range

        for k, p in enumerate(self.picks):
            if abs(p["x"] - x_sel) <= dt_tol and abs(p["y"] - y_sel) <= dy_tol and p["ch"] == ch:
                # remove label from scene
                if p.get("label"):
                    try:
                        self.plot.removeItem(p["label"])
                    except Exception:
                        pass
                del self.picks[k]
                self._update_picks()
                ev.accept()
                return

        # Otherwise add a new pick + its own label
        label = self._make_pick_label(f"ch{ch+1}:{y_sel:.3f}\nt:{x_sel:.3f}s  ")
        label.setPos(x_sel, y_sel)
        self.plot.addItem(label)

        self.picks.append({"x": x_sel, "y": y_sel, "ch": ch, "label": label})
        self._update_picks()
        ev.accept()



    def _update_picks(self):
        if self.picks:
            xs = [p["x"] for p in self.picks]
            ys = [p["y"] for p in self.picks]
            self.pick_marker.setData(x=xs, y=ys)
        else:
            self.pick_marker.setData(x=[], y=[])

    def _update_picks(self):
        if self.picks:
            xs = [p["x"] for p in self.picks]
            ys = [p["y"] for p in self.picks]
            self.pick_marker.setData(x=xs, y=ys)
        else:
            self.pick_marker.setData(x=[], y=[])


    def _open_csv_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open CSV",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        self.load_csv(path)

    def load_csv(self, path: str):
        """Load a CSV with header: timestamp,ch1,ch2,... and plot it."""
        # If serial is running, stop it first so we’re not mixing live & file data
        if self.reader and self.reader.isRunning():
            self.stop_reader()

        # Clear existing buffers/labels/markers
        self._clear_buffers()

        # Read header to count channels (and basic validation)
        import csv
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        if not header or len(header) < 2 or header[0].strip().lower() != "timestamp":
            self._on_status("CSV must start with header: timestamp,ch1,...")
            return

        num_ch = len(header) - 1

        # Load data (fast path with numpy)
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=float)
        except Exception as e:
            self._on_status(f"Failed to load CSV: {e}")
            return

        # Handle single-row CSV (loadtxt returns 1D in that case)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] != (1 + num_ch):
            self._on_status("Column count mismatch between header and rows.")
            return

        t_abs = data[:, 0].astype(float)               # shape: (N,)
        Y = data[:, 1:].astype(np.float32).T           # shape: (C, N)

        # Ensure timestamps are strictly increasing for plotting
        # (If not, we sort by time)
        if not np.all(np.diff(t_abs) >= 0):
            order = np.argsort(t_abs)
            t_abs = t_abs[order]
            Y = Y[:, order]

        # Initialize curves if this is our first data
        if self.num_ch is None:
            self.num_ch = num_ch
            cap = max(Y.shape[1], MAX_POINTS_PER_CHANNEL or 72_000)
            self.ring = Ring2D(self.num_ch, cap, dtype=np.float32)

            # Colors & curves (same logic you use on live connect)
            colors = list(LINE_COLORS)
            if self.num_ch > len(colors):
                import random
                random.seed(1234)
                while len(colors) < self.num_ch:
                    colors.append(tuple(random.choices(range(256), k=3)))
            for i in range(self.num_ch):
                name = f"ch{i+1}"
                color = colors[i % len(colors)]
                curve = self.plot.plot(pen=pg.mkPen(color=color, width=LINE_WIDTH), name=name)
                curve.setClipToView(True)
                try:
                    curve.setDownsampling(method="peak")
                except TypeError:
                    try:
                        curve.setDownsampling(mode="peak")
                    except TypeError:
                        curve.setDownsampling(auto=True)
                self.curves.append(curve)
        else:
            # If curves already exist but count changed, rebuild
            if self.num_ch != num_ch:
                self.reset_plot_buffers()
                self.num_ch = num_ch
                cap = max(Y.shape[1], MAX_POINTS_PER_CHANNEL or 72_000)
                self.ring = Ring2D(self.num_ch, cap, dtype=np.float32)

                colors = list(LINE_COLORS)
                if self.num_ch > len(colors):
                    import random
                    random.seed(1234)
                    while len(colors) < self.num_ch:
                        colors.append(tuple(random.choices(range(256), k=3)))
                for i in range(self.num_ch):
                    name = f"ch{i+1}"
                    color = colors[i % len(colors)]
                    curve = self.plot.plot(pen=pg.mkPen(color=color, width=LINE_WIDTH), name=name)
                    curve.setClipToView(True)
                    try:
                        curve.setDownsampling(method="peak")
                    except TypeError:
                        try:
                            curve.setDownsampling(mode="peak")
                        except TypeError:
                            curve.setDownsampling(auto=True)
                    self.curves.append(curve)

        # Feed the samples into the ring buffer
        self.t0 = float(t_abs[0]) if t_abs.size else None
        if self.t0 is None:
            self._on_status("CSV file has no data rows.")
            return

        # Append sample-by-sample (Ring2D append API matches your live path)
        # For very large files, this can take time; it’s simple & safe.
        N = t_abs.shape[0]
        for i in range(N):
            self.ring.append(Y[:, i], t=float(t_abs[i]))

        # Adjust Y range to data (respect your lower bound)
        self._last_y_top = max(self.y_limit[1], float(np.nanmax(Y)) if Y.size else self.y_limit[1])
        self.plot.enableAutoRange(axis='y', enable=False)
        self.plot.setYRange(self.y_limit[0], self._last_y_top, padding=0)

        # Refresh UI
        self.connected = False
        self.enable_controls(True)
        self.connect_btn.setText("Connect")
        self.sb.showMessage(f"Loaded CSV: {path} | {self.num_ch} ch | {N} samples", 5000)

        # Trigger a draw now
        self._redraw()


    def _open_settings_dialog(self):
        dlg = SettingsDialog(
            parent=self,
            default_baud=self.default_baud,
            target_hz=self.target_hz,
            y_min=float(self.y_limit[0]),
            y_max=float(self.y_limit[1]),
            max_points=self.max_points_per_channel,
            line_width=self.line_width,
            colors=self.line_colors,
        )
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            vals = dlg.values()

            # 1) Baud default
            self.default_baud = vals["default_baud"]
            # Update dropdown default selection
            self.baud_combo.setCurrentText(str(self.default_baud))

            # 2) Refresh Hz
            if vals["target_hz"] != self.target_hz and vals["target_hz"] > 0:
                self.target_hz = vals["target_hz"]
                self.timer.setInterval(int(1000 / self.target_hz))

            # 3) Y limits
            new_ymin, new_ymax = vals["y_limits"]
            if new_ymin >= new_ymax:
                self._on_status("Y min must be < Y max")
            else:
                self.y_limit = [new_ymin, new_ymax]
                self._last_y_top = new_ymax  # reset bump baseline
                self.plot.enableAutoRange(axis='y', enable=False)
                self.plot.setYRange(new_ymin, new_ymax, padding=0)

            # 4) Max points per channel (cap)
            new_cap = vals["max_points_per_channel"]  # None or int
            if new_cap != self.max_points_per_channel:
                self.max_points_per_channel = new_cap
                if self.ring and self.num_ch:
                    # Rebuild ring with new capacity and keep as much data as possible
                    t_old, Y_old = self.ring.view()
                    keep = Y_old.shape[1]
                    if new_cap is not None:
                        keep = min(keep, int(new_cap))
                    new_cap_final = new_cap or max(keep, 1)
                    new_ring = Ring2D(self.num_ch, new_cap_final, dtype=np.float32)
                    # Append the last 'keep' samples in chronological order
                    if keep > 0:
                        Y_keep = Y_old[:, -keep:] if keep < Y_old.shape[1] else Y_old
                        t_keep = t_old[-keep:] if keep < t_old.shape[0] else t_old
                        for i in range(t_keep.shape[0]):
                            new_ring.append(Y_keep[:, i], t=float(t_keep[i]))
                    self.ring = new_ring
                    # Force a redraw on next tick
                    self.t0 = t_old[0] if t_old.size else None

            # 5) Line width
            if vals["line_width"] != self.line_width:
                self.line_width = vals["line_width"]
                for i, curve in enumerate(self.curves):
                    color = self.line_colors[i % len(self.line_colors)] if self.line_colors else (0, 0, 0)
                    curve.setPen(pg.mkPen(color=color, width=self.line_width))

            # 6) Line colors
            new_colors = vals["line_colors"]
            if new_colors and new_colors != self.line_colors:
                self.line_colors = new_colors
                # Extend palette if fewer than channels
                colors = list(self.line_colors)
                if self.num_ch and self.num_ch > len(colors):
                    import random
                    random.seed(1234)
                    while len(colors) < self.num_ch:
                        colors.append(tuple(random.choices(range(256), k=3)))
                for i, curve in enumerate(self.curves):
                    curve.setPen(pg.mkPen(color=colors[i % len(colors)], width=self.line_width))

            self._on_status("Settings updated.")

class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, default_baud=115200, target_hz=50, y_min=-0.1, y_max=1.0,
                 max_points=72_000, line_width=2, colors=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setModal(True)

        if colors is None:
            colors = []

        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        # Default baud
        self.baud_cb = QtWidgets.QComboBox()
        self.baud_cb.setEditable(True)
        for b in [9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600]:
            self.baud_cb.addItem(str(b))
        self.baud_cb.setCurrentText(str(default_baud))
        form.addRow("Default baud:", self.baud_cb)

        # Target Hz
        self.hz_spin = QtWidgets.QSpinBox()
        self.hz_spin.setRange(1, 240)
        self.hz_spin.setValue(int(target_hz))
        form.addRow("UI refresh (Hz):", self.hz_spin)

        # Y limits
        self.ymin_spin = QtWidgets.QDoubleSpinBox()
        self.ymin_spin.setRange(-1e9, 1e9)
        self.ymin_spin.setDecimals(6)
        self.ymin_spin.setValue(float(y_min))
        self.ymax_spin = QtWidgets.QDoubleSpinBox()
        self.ymax_spin.setRange(-1e9, 1e9)
        self.ymax_spin.setDecimals(6)
        self.ymax_spin.setValue(float(y_max))
        y_box = QtWidgets.QHBoxLayout()
        y_box.addWidget(QtWidgets.QLabel("Min:"))
        y_box.addWidget(self.ymin_spin)
        y_box.addSpacing(12)
        y_box.addWidget(QtWidgets.QLabel("Max:"))
        y_box.addWidget(self.ymax_spin)
        form.addRow("Y axis:", QtWidgets.QWidget())
        form.itemAt(form.rowCount()-1, QtWidgets.QFormLayout.FieldRole).widget().setLayout(y_box)

        # Max points per channel (0 = unlimited)
        self.max_points_spin = QtWidgets.QSpinBox()
        self.max_points_spin.setRange(0, 5_000_000)
        self.max_points_spin.setValue(int(max_points or 0))
        form.addRow("Max points / channel (0 = unlimited):", self.max_points_spin)

        # Line width
        self.width_spin = QtWidgets.QDoubleSpinBox()
        self.width_spin.setRange(0.5, 10.0)
        self.width_spin.setSingleStep(0.5)
        self.width_spin.setValue(float(line_width))
        form.addRow("Line width:", self.width_spin)

        # Line colors table
        self.colors_tbl = QtWidgets.QTableWidget(len(colors), 2)
        self.colors_tbl.setHorizontalHeaderLabels(["Preview", "RGB"])
        self.colors_tbl.verticalHeader().setVisible(False)
        self.colors_tbl.horizontalHeader().setStretchLastSection(True)
        for r, c in enumerate(colors):
            self._set_color_row(r, c)

        btns_line = QtWidgets.QHBoxLayout()
        self.add_color_btn = QtWidgets.QPushButton("Add")
        self.edit_color_btn = QtWidgets.QPushButton("Edit…")
        self.remove_color_btn = QtWidgets.QPushButton("Remove")
        btns_line.addWidget(self.add_color_btn)
        btns_line.addWidget(self.edit_color_btn)
        btns_line.addWidget(self.remove_color_btn)

        layout.addWidget(QtWidgets.QLabel("Line colors:"))
        layout.addWidget(self.colors_tbl)
        layout.addLayout(btns_line)

        # Dialog buttons
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        layout.addWidget(btn_box)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

        # Wire up color controls
        self.add_color_btn.clicked.connect(self._on_add_color)
        self.edit_color_btn.clicked.connect(self._on_edit_color)
        self.remove_color_btn.clicked.connect(self._on_remove_color)

    def _set_color_row(self, row, rgb_tuple):
        r, g, b = [int(x) for x in rgb_tuple]
        # Preview cell
        preview = QtWidgets.QLabel()
        preview.setAutoFillBackground(True)
        pal = preview.palette()
        pal.setColor(preview.backgroundRole(), QtGui.QColor(r, g, b))
        preview.setPalette(pal)
        preview.setMinimumHeight(18)
        preview.setFrameShape(QtWidgets.QFrame.Panel)
        preview.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.colors_tbl.setCellWidget(row, 0, preview)
        # RGB text
        item = QtWidgets.QTableWidgetItem(f"({r}, {g}, {b})")
        item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        self.colors_tbl.setItem(row, 1, item)

    def _on_add_color(self):
        col = QtWidgets.QColorDialog.getColor(QtGui.QColor(0, 0, 0), self, "Pick color")
        if not col.isValid():
            return
        row = self.colors_tbl.rowCount()
        self.colors_tbl.insertRow(row)
        self._set_color_row(row, (col.red(), col.green(), col.blue()))

    def _on_edit_color(self):
        row = self.colors_tbl.currentRow()
        if row < 0:
            return
        cur = self._row_rgb(row)
        col = QtWidgets.QColorDialog.getColor(QtGui.QColor(*cur), self, "Pick color")
        if not col.isValid():
            return
        self._set_color_row(row, (col.red(), col.green(), col.blue()))

    def _on_remove_color(self):
        row = self.colors_tbl.currentRow()
        if row >= 0:
            self.colors_tbl.removeRow(row)

    def _row_rgb(self, row):
        text = self.colors_tbl.item(row, 1).text()
        nums = text.strip("() ").split(",")
        return tuple(int(float(x)) for x in nums)

    def values(self):
        # Collect settings
        default_baud = int(self.baud_cb.currentText())
        target_hz = int(self.hz_spin.value())
        y_min = float(self.ymin_spin.value())
        y_max = float(self.ymax_spin.value())
        max_points_val = int(self.max_points_spin.value())
        max_points = None if max_points_val == 0 else max_points_val
        line_width = float(self.width_spin.value())

        # Colors
        colors = []
        for row in range(self.colors_tbl.rowCount()):
            colors.append(self._row_rgb(row))

        return {
            "default_baud": default_baud,
            "target_hz": target_hz,
            "y_limits": (y_min, y_max),
            "max_points_per_channel": max_points,
            "line_width": line_width,
            "line_colors": colors,
        }

