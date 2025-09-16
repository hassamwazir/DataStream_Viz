# main_window.py
import time
import queue
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
import serial
import serial.tools.list_ports
from PySide6 import QtWidgets, QtCore, QtGui

from config import (
    DEFAULT_BAUD, TARGET_HZ, Y_LIMITS,
    MAX_POINTS_PER_CHANNEL, LINE_WIDTH, TAIL_WINDOW_S,  LINE_COLORS
)
from logger import CsvWriter
from ring import Ring2D
from serial_reader import SerialReader

try:
    from pyqtgraph.graphicsItems.PlotItem.ButtonItem import ButtonItem as _PgButtonItem
except Exception:
    _PgButtonItem = None


# split-out modules
from settings_dialog import SettingsDialog
from io_csv import load_csv_file, write_csv


import re  # at top of file (with other imports)

def _is_usb_serial(p) -> bool:
    """Heuristically detect USB-backed serial adapters across platforms."""
    # Easiest/strongest signal (pyserial fills these for USB devices)
    if getattr(p, "vid", None) is not None or getattr(p, "pid", None) is not None:
        return True

    dev  = (p.device or "").lower()
    desc = (getattr(p, "description", "") or "").lower()
    hwid = (getattr(p, "hwid", "") or "").lower()

    # Common identifiers across OSes/adapters
    tokens = ("usb", "usbserial", "usb-modem", "usbmodem", "acm", "cdc", "cp210", "ch340", "ftdi")
    if any(t in dev for t in tokens) or any(t in desc for t in tokens) or any(t in hwid for t in tokens):
        return True

    # Extra path hints (POSIX/macOS)
    if dev.startswith(("/dev/tty.usb", "/dev/cu.usb", "/dev/ttyusb", "/dev/ttyacm")):
        return True

    # Windows hint: COMx with an USB string in description
    if dev.startswith("com") and "usb" in desc:
        return True

    return False


class MultiPlot(QtWidgets.QMainWindow):
    """
    Two-plot layout:
      • Top plot: zoomed view (what you analyze), crosshair + click-to-pick.
      • Bottom plot: full-history overview with a LinearRegionItem that controls the top plot's X range.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Serial Streaming → PyQtGraph (overview + zoom)")
        pg.setConfigOptions(antialias=False, useOpenGL=False, background='w')

        # --- Central widget & layout ---
        central = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)
        self.setCentralWidget(central)

        # ---- Runtime-configurable copies of config.py values ----
        self.default_baud = DEFAULT_BAUD
        self.target_hz = TARGET_HZ
        self.y_limit = list(Y_LIMITS)  # mutable
        self.max_points_per_channel = MAX_POINTS_PER_CHANNEL
        self.line_width = LINE_WIDTH
        self.line_colors = list(LINE_COLORS)

        # --- Logging ---
        self.log_q = queue.Queue(maxsize=10_000)  # buffer up to 10k rows
        self.logger = None  # created on connect when we know channel count

        # Time origin for elapsed-x axis
        self.t0 = None

        # --- Top plot (zoomed) ---
        self.plot_top = pg.PlotWidget()
        self.plot_top.showGrid(x=True, y=True, alpha=0.25)
        self.plot_top.addLegend()
        self.plot_top.setLabel("bottom", "Elapsed (s)")
        self.plot_top.setLabel("left", "Value")
        self.plot_top.setYRange(*self.y_limit)

        # Disable panning/zooming on TOP plot
        vb_top = self.plot_top.getViewBox()
        vb_top.setMouseEnabled(x=True, y=True)
        try:
            vb_top.setWheelEnabled(True)
        except Exception:
            self.plot_top.wheelEvent = lambda ev: ev.ignore()
        try:
            vb_top.setMenuEnabled(False)
        except Exception:
            vb_top.enableMenu(False)
        vb_top.mouseDoubleClickEvent = lambda ev: ev.ignore()

        # Crosshair on TOP plot
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((80, 80, 80), width=1))
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen((80, 80, 80), width=1))
        self.plot_top.addItem(self.vline, ignoreBounds=True)
        self.plot_top.addItem(self.hline, ignoreBounds=True)

        # Cursor readout label
        self.crosshair_label = pg.TextItem("", anchor=(0, 1))
        self.crosshair_label.setZValue(12)
        self.plot_top.addItem(self.crosshair_label, ignoreBounds=True)


        # Click-to-select: scatter that renders all picked points (TOP plot)
        self.pick_marker = pg.ScatterPlotItem(
            size=10,
            pen=pg.mkPen('k', width=1),
            brush=pg.mkBrush(255, 0, 0, 180),
        )
        self.pick_marker.setZValue(10)
        self.plot_top.addItem(self.pick_marker, ignoreBounds=True)


        # Store selected points; each entry: {"x": float, "y": float, "ch": int, "label": TextItem}
        self.picks: List[dict] = []

        # Reusable label styling
        self._label_font = QtGui.QFont()
        self._label_font.setPointSize(12)
        self._label_font.setBold(True)

        # --- Bottom plot (overview) ---
        self.plot_overview = pg.PlotWidget()
        self.plot_overview.showGrid(x=True, y=True, alpha=0.15)
        self.plot_overview.setLabel("bottom", "Elapsed (s)")
        self.plot_overview.setLabel("left", "Overview")
        self.plot_overview.setMouseEnabled(x=False, y=False)

        # Disable panning/zooming on the BOTTOM (overview) plot
        vb = self.plot_overview.getViewBox()
        vb.setMouseEnabled(x=False, y=False)
        try:
            vb.setWheelEnabled(False)
        except Exception:
            self.plot_overview.wheelEvent = lambda ev: ev.ignore()
        try:
            vb.setMenuEnabled(False)
        except Exception:
            vb.enableMenu(False)

        # LinearRegionItem lives on the OVERVIEW plot and controls TOP x-range
        self.sel_region = pg.LinearRegionItem()
        self.sel_region.setZValue(10)
        self.sel_region.setMovable(True)
        self.sel_region.hide()  # show once we have data
        self.plot_overview.addItem(self.sel_region, ignoreBounds=True)

        # --- Tail follow (pin-to-right) settings ---
        # If you added TAIL_WINDOW_S to config, make sure it's imported; else default to 10.0.
        try:
            self.tail_window_s = TAIL_WINDOW_S  # from config
        except NameError:
            self.tail_window_s = 10.0
        self.follow_tail = True
        self._region_width = None  # remember current window width

        # Layout: controls row, then TOP (stretch 3), then OVERVIEW (stretch 1)
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

        vbox.addWidget(ctrl)
        vbox.addWidget(self.plot_top, 3)
        vbox.addWidget(self.plot_overview, 1)

        # --- Status bar ---
        self.sb = self.statusBar()
        self._last_status_t = 0.0
        self._status_every = 0.25  # update the status bar at ~4 Hz

        # --- Runtime state ---
        self.reader: Optional[SerialReader] = None
        self.connected = False
        self.paused = False

        self.curves_top: List[pg.PlotDataItem] = []
        self.curves_overview: List[pg.PlotDataItem] = []
        self.num_ch: Optional[int] = None
        self.ring = None

        # Flicker-reduction: debounce Y bumps
        self._last_y_top = self.y_limit[1]
        self._y_bump_margin = 0.05    # bump y-top if >5% above current top
        self._y_update_ms = 200       # y-top can update at most 5x/sec
        self._last_y_update_t = 0.0

        # Region init flag
        self._region_ready = False

        # --- Signals ---
        self.refresh_btn.clicked.connect(self.populate_ports)
        self.connect_btn.clicked.connect(self.toggle_connection)

        # Mouse / crosshair on TOP plot
        self.plot_top.scene().sigMouseMoved.connect(self._on_scene_mouse_moved)
        self.plot_top.scene().sigMouseClicked.connect(self._on_plot_click)

        # Region <-> top range coupling (like the example)
        self.sel_region.sigRegionChanged.connect(self._on_region_changed)
        self.sel_region.sigRegionChangeFinished.connect(self._on_region_change_finished)
        self.plot_top.sigRangeChanged.connect(self._on_top_range_changed)

        # Shortcuts
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_P), self, activated=self._toggle_pause)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_C), self, activated=self._clear_buffers)

        # --- Timer for UI redraws ---
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._redraw)
        self.timer.start(int(1000 / self.target_hz))

        # Initial port scan
        self.populate_ports()

        # --- Menus ---
        self._build_menus()


    # ---------- Menu construction ----------
    def _build_menus(self):
        # File
        file_menu = self.menuBar().addMenu("&File")
        self.open_csv_act = QtGui.QAction("Open CSV…", self)
        self.open_csv_act.setShortcut(QtGui.QKeySequence.Open)  # Ctrl/Cmd+O
        self.open_csv_act.triggered.connect(self._open_csv_dialog)
        file_menu.addAction(self.open_csv_act)

        # Settings
        settings_menu = self.menuBar().addMenu("&Settings")
        self.settings_act = QtGui.QAction("Preferences…", self)
        try:
            self.settings_act.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Preferences))
        except Exception:
            self.settings_act.setShortcut(QtGui.QKeySequence("Ctrl+,"))
        self.settings_act.triggered.connect(self._open_settings_dialog)
        settings_menu.addAction(self.settings_act)

        # Tools
        tools_menu = self.menuBar().addMenu("&Tools")

        self.follow_tail_act = QtGui.QAction("Follow New Data (Pin Right)", self, checkable=True)
        self.follow_tail_act.setChecked(True)
        self.follow_tail_act.toggled.connect(self._on_follow_tail_toggled)
        tools_menu.addAction(self.follow_tail_act)

        self.export_sel_act = QtGui.QAction("Export Selection…", self)
        self.export_sel_act.triggered.connect(self._export_selection_csv)
        self.export_sel_act.setEnabled(False)
        tools_menu.addAction(self.export_sel_act)

        self.reset_sel_act = QtGui.QAction("Reset Selection to Full", self)
        self.reset_sel_act.triggered.connect(self._reset_selection_full)
        self.reset_sel_act.setEnabled(False)
        tools_menu.addAction(self.reset_sel_act)

    def _on_follow_tail_toggled(self, enabled: bool):
        self.follow_tail = enabled
        # If enabling while data exists, snap the region to the right immediately.
        if enabled:
            self._snap_region_to_right()

    def _on_region_change_finished(self):
        """User finished dragging the region; if it's not near the right edge, disable follow."""
        if not self.sel_region.isVisible():
            return
        # Remember current width
        r0, r1 = self.sel_region.getRegion()
        self._region_width = max(0.001, float(r1 - r0))
        # If user left the right edge, turn off follow so the app won't fight them
        if not self._region_is_pinned_right():
            if self.follow_tail:
                self.follow_tail = False
                self.follow_tail_act.setChecked(False)

    def _region_is_pinned_right(self) -> bool:
        if not (self.ring and self.ring.n and getattr(self, "t0", None) is not None):
            return False
        t_abs, _ = self.ring.view()
        x = t_abs - self.t0
        if x.size == 0:
            return False
        _, r1 = self.sel_region.getRegion()
        xmax = float(x[-1])
        # within 1 sample or 0.2 s (whichever bigger) counts as pinned
        eps = max(0.2, (x[-1] - x[0]) / max(1000, x.size))
        return (xmax - r1) <= eps
        

    def _snap_region_to_right(self):
        if not (self.ring and self.ring.n and getattr(self, "t0", None) is not None):
            return
        t_abs, _ = self.ring.view()
        x = t_abs - self.t0
        if x.size == 0:
            return
        xmin_data = float(x[0]); xmax_data = float(x[-1])
        width = self._region_width if (self._region_width and self._region_width > 0) else self.tail_window_s
        width = max(0.001, min(width, xmax_data - xmin_data))  # clamp
        new0 = max(xmin_data, xmax_data - width)
        new1 = xmax_data
        self.sel_region.blockSignals(True)
        self.sel_region.setRegion([new0, new1])
        self.sel_region.blockSignals(False)
        self.plot_top.setXRange(new0, new1, padding=0)



    # ---------- Controls ----------
    def populate_ports(self):
        current = self.port_combo.currentText()
        self.port_combo.clear()

        all_ports = list(serial.tools.list_ports.comports())
        ports = [p for p in all_ports if _is_usb_serial(p)]

        for p in ports:
            label = f"{p.device} – {p.description}"
            self.port_combo.addItem(label, userData=p.device)

        if not ports:
            self.port_combo.addItem("(no USB ports found)", userData=None)

        # Try to restore previous selection if still present
        idx = self.port_combo.findText(current)
        if idx >= 0:
            self.port_combo.setCurrentIndex(idx)

        self.sb.showMessage(f"Found {len(ports)} USB port(s)")


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
        self._clear_buffers()
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

    def _make_pick_label(self, text: str) -> pg.TextItem:
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
    def _ensure_curves(self, num_ch: int):
        """Create curves for BOTH plots when channel count is known or changes."""
        self.reset_plot_buffers()  # clears any existing curves/plots
        self.num_ch = num_ch

        cap = self.max_points_per_channel or 72_000
        self.ring = Ring2D(self.num_ch, cap, dtype=np.float32)

        colors = list(self.line_colors)
        if self.num_ch > len(colors):
            import random
            random.seed(1234)
            while len(colors) < self.num_ch:
                colors.append(tuple(random.choices(range(256), k=3)))

        for i in range(self.num_ch):
            name = f"ch{i+1}"
            color = colors[i % len(colors)]

            # TOP curve
            c_top = self.plot_top.plot(pen=pg.mkPen(color=color, width=self.line_width), name=name)
            c_top.setClipToView(True)
            try:
                c_top.setDownsampling(method="peak")
            except TypeError:
                try:
                    c_top.setDownsampling(mode="peak")
                except TypeError:
                    c_top.setDownsampling(auto=True)
            self.curves_top.append(c_top)

            # OVERVIEW curve (no legend; lighter weight)
            c_ov = self.plot_overview.plot(pen=pg.mkPen(color=color, width=max(1, int(self.line_width * 0.8))))
            try:
                c_ov.setDownsampling(method="peak")
            except TypeError:
                try:
                    c_ov.setDownsampling(mode="peak")
                except TypeError:
                    c_ov.setDownsampling(auto=True)
            self.curves_overview.append(c_ov)

        # region will be initialized on first redraw once we have data
        self._region_ready = False
        self.sel_region.show()
        self.export_sel_act.setEnabled(True)
        self.reset_sel_act.setEnabled(True)

    def _on_sample(self, vals: np.ndarray):
        if self.paused:
            return

        if self.num_ch is None:
            self._ensure_curves(int(vals.size))

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

        # Update TOP curves
        for i, curve in enumerate(self.curves_top):
            y = Y[i]
            y_plot = y[::decim] if decim > 1 else y
            curve.setData(x_plot, y_plot)

        # Update OVERVIEW curves (can use stronger decimation if desired)
        decim_ov = decim if decim > 1 else 1
        for i, curve in enumerate(self.curves_overview):
            y = Y[i]
            y_plot = y[::decim_ov] if decim_ov > 1 else y
            curve.setData(x_plot, y_plot)

        # Initialize region to full range on first draw with data
        if not self._region_ready:
            # set an initial width (use tail_window_s or whole span if shorter)
            xmin = float(x_plot[0]); xmax = float(x_plot[-1])
            span = max(0.001, xmax - xmin)
            init_width = min(self.tail_window_s, span)
            self._region_width = init_width
            self.sel_region.blockSignals(True)
            self.sel_region.setRegion([xmax - init_width, xmax])
            self.sel_region.blockSignals(False)
            self.plot_top.setXRange(xmax - init_width, xmax, padding=0)
            self._region_ready = True
            self.sel_region.show()
            self.export_sel_act.setEnabled(True)
            self.reset_sel_act.setEnabled(True)

        # Debounced y-range bump (preserve current bottom, expand top as needed) on TOP plot
        max_val = float(Y.max()) if length else 0.0
        now = time.perf_counter()
        need_bump = max_val > (self._last_y_top * (1 + self._y_bump_margin))
        if need_bump and (now - self._last_y_update_t) >= (self._y_update_ms / 1000.0):
            self._last_y_top = max_val
            self.plot_top.enableAutoRange(axis='y', enable=False)
            self.plot_top.setYRange(self.y_limit[0], self._last_y_top, padding=0)
            self._last_y_update_t = now

        # Keep OVERVIEW Y range aligned with config (optional)
        self.plot_overview.enableAutoRange(axis='y', enable=False)
        self.plot_overview.setYRange(self.y_limit[0], max(self._last_y_top, self.y_limit[1]), padding=0)

        # Keep region pinned to right while following
        if self.follow_tail and self.sel_region.isVisible():
            # preserve current width if user changed it
            if self._region_width is None:
                r0, r1 = self.sel_region.getRegion()
                self._region_width = max(0.001, float(r1 - r0))
            self._snap_region_to_right()


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
        self.plot_top.repaint()
        self.plot_overview.repaint()
        self.sb.showMessage("Paused" if self.paused else "Resumed", 2000)

    def _clear_buffers(self):
        # 1) Clear selected markers & labels
        if hasattr(self, "picks") and getattr(self, "picks", None):
            for p in self.picks:
                lbl = p.get("label")
                if lbl is not None:
                    try:
                        self.plot_top.removeItem(lbl)
                    except Exception:
                        pass
            self.picks.clear()

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

        # 3) Reset y-range bump state & region
        self._last_y_top = self.y_limit[1]
        self._last_y_update_t = 0.0
        self._region_ready = False
        self.sel_region.hide()
        self.export_sel_act.setEnabled(False)
        self.reset_sel_act.setEnabled(False)

        # 4) Status
        self.sb.showMessage("Cleared buffers & picks", 2000)

    def reset_plot_buffers(self):
        # remove existing labels from the scene (TOP)
        for p in getattr(self, "picks", []):
            if p.get("label"):
                try:
                    self.plot_top.removeItem(p["label"])
                except Exception:
                    pass

        # clear plots
        self.plot_top.clear()
        self.plot_top.addLegend()
        self.plot_overview.clear()

        # re-add persistent items
        try:
            self.plot_top.addItem(self.pick_marker, ignoreBounds=True)
        except Exception:
            pass
        try:
            self.plot_top.addItem(self.vline, ignoreBounds=True)
            self.plot_top.addItem(self.hline, ignoreBounds=True)
            self.plot_top.addItem(self.crosshair_label, ignoreBounds=True)
        except Exception:
            pass
        try:
            self.plot_overview.addItem(self.sel_region, ignoreBounds=True)
            self.sel_region.hide()
        except Exception:
            pass

        self.picks = []
        self.curves_top = []
        self.curves_overview = []
        self.ring = None
        self.num_ch = None
        self.t0 = None
        self._last_y_top = self.y_limit[1]
        self._last_y_update_t = 0.0
        self._region_ready = False

        # restore axes labels/ranges
        self.plot_top.setLabel("bottom", "Elapsed (s)")
        self.plot_top.setLabel("left", "Value")
        self.plot_top.setYRange(*self.y_limit)
        self.plot_overview.setLabel("bottom", "Elapsed (s)")
        self.plot_overview.setLabel("left", "Overview")

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

    # ---------- Mouse interactions (TOP plot) ----------
    def _on_plot_click(self, ev):
        vb = self.plot_top.getViewBox()

        # Only handle clicks inside the data area (ViewBox), not axes/labels/toolbar
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return

        # Mouse buttons (PySide6 / PyQt-safe)
        if hasattr(QtCore.Qt, "MouseButton"):
            LEFT = QtCore.Qt.MouseButton.LeftButton
            RIGHT = QtCore.Qt.MouseButton.RightButton
        else:
            LEFT = QtCore.Qt.LeftButton
            RIGHT = QtCore.Qt.RightButton

        # Topmost item under cursor
        items_under = self.plot_top.scene().items(ev.scenePos())
        top = items_under[0] if items_under else None

        # Ignore clicks on overlays (auto button, axes, legend, our crosshair/label)
        try:
            from pyqtgraph.graphicsItems.PlotItem.ButtonItem import ButtonItem as _PgButtonItem
        except Exception:
            _PgButtonItem = tuple()  # no-op

        if (isinstance(top, _PgButtonItem) or
            isinstance(top, (pg.AxisItem, pg.LegendItem)) or
            top in (self.vline, self.hline, self.crosshair_label)):
            return

        # Right-click: clear picks
        if ev.button() == RIGHT:
            for p in list(self.picks):
                if p.get("label"):
                    try:
                        self.plot_top.removeItem(p["label"])
                    except Exception:
                        pass
            self.picks.clear()
            self._update_picks()
            return

        # Only left-click creates/removes a pick
        if ev.button() != LEFT:
            return
        if not (self.ring and self.ring.n and getattr(self, "t0", None) is not None):
            return

        # Map click to data coords
        mouse_pt = self.plot_top.plotItem.vb.mapSceneToView(ev.scenePos())
        x_click = float(mouse_pt.x())
        y_click = float(mouse_pt.y())

        # Snap to nearest sample/time and channel by Y
        t_abs, Y = self.ring.view()
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

        # Toggle near-duplicate pick
        dt_tol = 0.02
        dy_tol = (self.y_limit[1] - self.y_limit[0]) * 0.02
        for k, p in enumerate(list(self.picks)):
            if abs(p["x"] - x_sel) <= dt_tol and abs(p["y"] - y_sel) <= dy_tol and p["ch"] == ch:
                if p.get("label"):
                    try:
                        self.plot_top.removeItem(p["label"])
                    except Exception:
                        pass
                del self.picks[k]
                self._update_picks()
                ev.accept()
                return

        # Add new pick + label (don’t affect autorange)
        label = self._make_pick_label(f"ch{ch+1}:{y_sel:.3f}\nt:{x_sel:.3f}s  ")
        label.setPos(x_sel, y_sel)
        self.plot_top.addItem(label, ignoreBounds=True)

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

    def _on_scene_mouse_moved(self, pos):
        vb = self.plot_top.getViewBox()

        # Only react if the cursor is inside the ViewBox (data area), not axes/labels
        if not vb.sceneBoundingRect().contains(pos):
            self.vline.setVisible(False)
            self.hline.setVisible(False)
            self.crosshair_label.setText("")
            return

        self.vline.setVisible(True)
        self.hline.setVisible(True)

        # Map to data coords and clamp to current view range
        mouse_pt = vb.mapSceneToView(pos)
        (xmin, xmax), (ymin, ymax) = vb.viewRange()
        x = float(min(max(mouse_pt.x(), xmin), xmax))
        y = float(min(max(mouse_pt.y(), ymin), ymax))

        # Move crosshair
        self.vline.setPos(x)
        self.hline.setPos(y)

        # No data -> clear readout
        if not (self.ring and self.ring.n and getattr(self, "t0", None) is not None):
            self.crosshair_label.setText("")
            return

        # Snap readout to nearest time sample
        t_abs, Y = self.ring.view()      # t_abs: (n,), Y: (C, n)
        x_rel = t_abs - self.t0
        n = x_rel.size
        if n == 0:
            self.crosshair_label.setText("")
            return

        idx = int(np.searchsorted(x_rel, x))
        if idx <= 0:
            i_near = 0
        elif idx >= n:
            i_near = n - 1
        else:
            i_near = idx - 1 if (x - x_rel[idx - 1]) <= (x_rel[idx] - x) else idx

        y_at = Y[:, i_near]
        preview_ch = min(4, y_at.size)
        parts = [f"t={x_rel[i_near]:.3f}s"] + [f"ch{i+1}={y_at[i]:.3f}" for i in range(preview_ch)]
        if y_at.size > preview_ch:
            parts.append("…")
        text = "  |  ".join(parts)

        # Place label near top-left of the current view
        br = vb.viewRect()
        self.crosshair_label.setText(text)
        self.crosshair_label.setPos(br.left() + 6, br.top())  # anchor is (0,1)
        self.sb.showMessage(text, 1000)


    # ---------- Region <-> Top range sync ----------
    def _on_region_changed(self):
        if not self.sel_region.isVisible():
            return
        if self.follow_tail:
            # When following, region is managed by code; don't chase ourselves
            return
        r0, r1 = self.sel_region.getRegion()
        if r0 > r1:
            r0, r1 = r1, r0
        self._region_width = max(0.001, float(r1 - r0))
        self.plot_top.setXRange(r0, r1, padding=0)


    def _on_top_range_changed(self, window, viewRange):
        if not self.sel_region.isVisible():
            return
        if self.follow_tail:
            # When following, we drive the region from _redraw; ignore top zoom.
            return
        rgn = viewRange[0]
        self._region_width = max(0.001, float(rgn[1] - rgn[0]))
        self.sel_region.blockSignals(True)
        self.sel_region.setRegion(rgn)
        self.sel_region.blockSignals(False)


    def _reset_selection_full(self):
        if not (self.ring and self.ring.n and getattr(self, "t0", None) is not None):
            return
        t_abs, _Y = self.ring.view()
        x = t_abs - self.t0
        if x.size < 2:
            return
        new0, new1 = float(x[0]), float(x[-1])
        self._region_width = max(0.001, new1 - new0)
        self.sel_region.setRegion([new0, new1])
        self.plot_top.setXRange(new0, new1, padding=0)
        # Optionally re-enable follow and then snap to right
        self.follow_tail_act.setChecked(True)


    # ---------- CSV open/load ----------
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

        try:
            _header, t_abs, Y = load_csv_file(path)
        except Exception as e:
            self._on_status(str(e))
            return

        num_ch = Y.shape[0]
        self._ensure_curves(num_ch)

        # Feed the samples into the ring buffer
        self.t0 = float(t_abs[0]) if t_abs.size else None
        if self.t0 is None:
            self._on_status("CSV file has no data rows.")
            return

        for i in range(t_abs.shape[0]):
            self.ring.append(Y[:, i], t=float(t_abs[i]))

        # Adjust Y range to data (respect your lower bound)
        self._last_y_top = max(self.y_limit[1], float(np.nanmax(Y)) if Y.size else self.y_limit[1])
        self.plot_top.enableAutoRange(axis='y', enable=False)
        self.plot_top.setYRange(self.y_limit[0], self._last_y_top, padding=0)
        self.plot_overview.enableAutoRange(axis='y', enable=False)
        self.plot_overview.setYRange(self.y_limit[0], max(self._last_y_top, self.y_limit[1]), padding=0)

        # Initialize region to full range
        self._region_ready = False  # will be configured on next _redraw
        self.sel_region.show()
        self.export_sel_act.setEnabled(True)
        self.reset_sel_act.setEnabled(True)

        # UI
        self.connected = False
        self.enable_controls(True)
        self.connect_btn.setText("Connect")
        self.sb.showMessage(f"Loaded CSV: {path} | {self.num_ch} ch | {t_abs.shape[0]} samples", 5000)

        # Trigger a draw now
        self._redraw()

    # ---------- Selection export ----------
    def _export_selection_csv(self):
        if not (self.ring and self.ring.n and getattr(self, "t0", None) is not None):
            self._on_status("No data to export.")
            return
        if not self.sel_region.isVisible():
            self._on_status("No selection region.")
            return

        r0, r1 = self.sel_region.getRegion()
        if r0 > r1:
            r0, r1 = r1, r0

        # Map selection (elapsed seconds) back to absolute timestamps
        t_abs, Y = self.ring.view()
        x_rel = t_abs - self.t0
        if x_rel.size == 0:
            self._on_status("No data in buffer.")
            return

        lo = int(np.searchsorted(x_rel, r0, side="left"))
        hi = int(np.searchsorted(x_rel, r1, side="right"))
        if hi <= lo:
            self._on_status("Selection is too small (no samples).")
            return

        t_sel = t_abs[lo:hi]
        Y_sel = Y[:, lo:hi]

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Selection", "selection.csv", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return

        try:
            write_csv(path, t_sel, Y_sel)
            self._on_status(f"Exported {hi - lo} samples → {path}")
        except Exception as e:
            self._on_status(f"Export failed: {e}")

    # ---------- Settings dialog ----------
    def _open_settings_dialog(self):
        dlg = SettingsDialog(
            parent=self,
            default_baud=self.default_baud,
            target_hz=self.target_hz,
            y_min=float(self.y_limit[0]),
            y_max=float(self.y_limit[1]),
            max_points=self.max_points_per_channel,
            line_width=self.line_width,
            tail_window_s=self.tail_window_s,
            colors=self.line_colors,
        )
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            vals = dlg.values()

            # 1) Baud default
            self.default_baud = vals["default_baud"]
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
                self.plot_top.enableAutoRange(axis='y', enable=False)
                self.plot_top.setYRange(new_ymin, new_ymax, padding=0)
                self.plot_overview.enableAutoRange(axis='y', enable=False)
                self.plot_overview.setYRange(new_ymin, new_ymax, padding=0)

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
                    if keep > 0:
                        Y_keep = Y_old[:, -keep:] if keep < Y_old.shape[1] else Y_old
                        t_keep = t_old[-keep:] if keep < t_old.shape[0] else t_old
                        for i in range(t_keep.shape[0]):
                            new_ring.append(Y_keep[:, i], t=float(t_keep[i]))
                    self.ring = new_ring
                    self.t0 = t_old[0] if t_old.size else None

            # 5) Line width
            if vals["line_width"] != self.line_width:
                self.line_width = vals["line_width"]
                for i, c in enumerate(self.curves_top):
                    color = self.line_colors[i % len(self.line_colors)] if self.line_colors else (0, 0, 0)
                    c.setPen(pg.mkPen(color=color, width=self.line_width))
                for i, c in enumerate(self.curves_overview):
                    color = self.line_colors[i % len(self.line_colors)] if self.line_colors else (0, 0, 0)
                    c.setPen(pg.mkPen(color=color, width=max(1, int(self.line_width * 0.8))))

            # 6) Line colors
            new_colors = vals["line_colors"]
            if new_colors and new_colors != self.line_colors:
                self.line_colors = new_colors
                colors = list(self.line_colors)
                if self.num_ch and self.num_ch > len(colors):
                    import random
                    random.seed(1234)
                    while len(colors) < self.num_ch:
                        colors.append(tuple(random.choices(range(256), k=3)))
                for i, c in enumerate(self.curves_top):
                    c.setPen(pg.mkPen(color=colors[i % len(colors)], width=self.line_width))
                for i, c in enumerate(self.curves_overview):
                    c.setPen(pg.mkPen(color=colors[i % len(colors)], width=max(1, int(self.line_width * 0.8))))

            if "tail_window_s" in vals and vals["tail_window_s"] > 0:
                self.tail_window_s = vals["tail_window_s"]
                # make it the default width for following
                self._region_width = self.tail_window_s
                if self.follow_tail:
                    self._snap_region_to_right()
            self._on_status("Settings updated.")
