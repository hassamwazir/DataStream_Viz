import sys
import time
import numpy as np
from collections import deque

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import serial
import serial.tools.list_ports


# ------------------ Defaults ------------------
DEFAULT_BAUD = 115200
ROLLING_SECONDS = 10     # visible window
TARGET_HZ = 50           # UI refresh Hz (not serial rate)
DEFAULT_POLL_HZ = 200.0   # how often to send '1' while connected
# ---------------------------------------------------


class SerialReader(QtCore.QThread):
    """Reads newline-delimited, space-separated numeric samples from a serial port.
       Emits a numpy array for each complete line of values.
       Also periodically writes '1' to request a datapoint (polling)."""
    sample = QtCore.pyqtSignal(np.ndarray)
    status = QtCore.pyqtSignal(str)
    connected = QtCore.pyqtSignal()
    disconnected = QtCore.pyqtSignal()

    # Allow UI to update polling behavior safely
    @QtCore.pyqtSlot(bool, float)
    def set_polling(self, enabled: bool, hz: float):
        self.poll_enabled = bool(enabled)
        self.poll_hz = max(float(hz), 0.0)
        self.poll_period = (1.0 / self.poll_hz) if self.poll_enabled and self.poll_hz > 0 else None
        self._next_poll = time.perf_counter()  # resync schedule

    def __init__(self, port, baud, poll_enabled=True, poll_hz=DEFAULT_POLL_HZ, parent=None):
        super().__init__(parent)
        self.port_name = port
        self.baud = baud
        self._stop = False
        self.ser = None
        self._buf = bytearray()

        # polling config
        self.poll_enabled = bool(poll_enabled)
        self.poll_hz = float(poll_hz)
        self.poll_period = (1.0 / self.poll_hz) if self.poll_enabled and self.poll_hz > 0 else None
        self._next_poll = time.perf_counter()

    def run(self):
        try:
            self.ser = serial.Serial(self.port_name, self.baud, timeout=0.01)
            self.status.emit(f"Opened {self.port_name} @ {self.baud} baud")
            self.connected.emit()
        except Exception as e:
            self.status.emit(f"ERROR opening {self.port_name}: {e}")
            return

        while not self._stop:
            now = time.perf_counter()

            # ---- POLLING: send '1' at the specified rate ----
            if self.poll_period is not None and now >= self._next_poll:
                try:
                    # If your device requires newline/CR, change to b"1\n" or b"1\r\n"
                    self.ser.write(b"1")
                except Exception as e:
                    self.status.emit(f"Serial write error: {e}")
                # schedule next poll
                self._next_poll += self.poll_period
                # in case of drift or long delays, prevent runaway backlog
                if now - self._next_poll > 2 * self.poll_period:
                    self._next_poll = now + self.poll_period

            # ---- READ any available bytes ----
            try:
                data = self.ser.read(4096)
                if data:
                    self._buf.extend(data)
                    *lines, remainder = self._buf.split(b'\n')
                    self._buf = bytearray(remainder)
                    for raw in lines:
                        line = raw.strip().replace(b'\r', b'')
                        if not line:
                            continue
                        parts = line.split()
                        try:
                            vals = np.array([float(p) for p in parts], dtype=float)
                            if vals.size > 0:
                                self.sample.emit(vals)
                        except ValueError:
                            continue
                else:
                    # small sleep to avoid busy-wait if nothing arrived
                    self.msleep(1)
            except Exception as e:
                self.status.emit(f"Serial read error: {e}")
                self.msleep(50)

        # ---- Cleanup ----
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
                self.status.emit("Serial port closed")
        except Exception:
            pass
        self.disconnected.emit()

    def stop(self):
        self._stop = True


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

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.addLegend()
        self.plot.setLabel("bottom", "Sample")
        self.plot.setLabel("left", "Value")
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

        # --- Runtime state ---
        self.reader = None
        self.connected = False
        self.paused = False

        self.curves = []
        self.num_ch = None
        self.buffers = []
        self.maxlen = 10000  # will be adjusted once rate is estimated

        self._samples_seen = 0
        self._t0 = time.perf_counter()
        self._estimated_hz = None

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
            self.num_ch = int(vals.size)
            self.buffers = [deque(maxlen=self.maxlen) for _ in range(self.num_ch)]

            # color palette (extend if you have many channels)
            colors = [
                (255, 0, 0),     # red
                (0, 200, 0),     # green
                (0, 0, 255),     # blue
                (200, 100, 0),   # orange
                (200, 0, 200),   # magenta
                (0, 200, 200),   # cyan
                (150, 150, 150), # gray
            ]

            for i in range(self.num_ch):
                name = f"ch{i+1}"
                color = colors[i % len(colors)]
                curve = self.plot.plot(pen=pg.mkPen(color=color, width=2), name=name)
                self.curves.append(curve)

            self._on_status(f"Detected {self.num_ch} channel(s)")
            self._samples_seen = 0
            self._t0 = time.perf_counter()
            self._estimated_hz = None

        n = min(self.num_ch, vals.size)
        for i in range(n):
            self.buffers[i].append(vals[i])

        self._samples_seen += 1
        if self._samples_seen % 50 == 0:
            dt = time.perf_counter() - self._t0
            if dt > 0.2:
                self._estimated_hz = self._samples_seen / dt
                if self._estimated_hz > 0:
                    new_maxlen = max(int(self._estimated_hz * ROLLING_SECONDS), 200)
                    if new_maxlen != self.maxlen:
                        self.maxlen = new_maxlen
                        for i in range(self.num_ch):
                            old = list(self.buffers[i])[-self.maxlen:]
                            self.buffers[i] = deque(old, maxlen=self.maxlen)

    def _redraw(self):
        if not self.buffers:
            return
        length = max((len(b) for b in self.buffers), default=0)
        if length == 0:
            return
        x = np.arange(length)
        for i, curve in enumerate(self.curves):
            y = np.fromiter(self.buffers[i], dtype=float, count=length)
            curve.setData(x, y)
        if self._estimated_hz:
            self.sb.showMessage(
                f"{'Connected' if self.connected else 'Idle'} | "
                f"{self.num_ch or 0} ch | ~{self._estimated_hz:.1f} Hz | "
                f"window ~{ROLLING_SECONDS}s | P=Pause, C=Clear"
            )

    # ---------- Helpers ----------
    def _toggle_pause(self):
        self.paused = not self.paused
        self.sb.showMessage("Paused" if self.paused else "Resumed", 2000)

    def _clear_buffers(self):
        for dq in self.buffers:
            dq.clear()
        self.sb.showMessage("Cleared buffers", 2000)

    def reset_plot_buffers(self):
        self.plot.clear()
        self.plot.addLegend()
        self.curves = []
        self.buffers = []
        self.num_ch = None
        self.maxlen = 10000
        self._samples_seen = 0
        self._t0 = time.perf_counter()
        self._estimated_hz = None

    def closeEvent(self, event):
        try:
            if self.reader and self.reader.isRunning():
                self.reader.stop()
                self.reader.wait(1000)
        except Exception:
            pass
        return super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MultiPlot()
    win.resize(1100, 650)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
