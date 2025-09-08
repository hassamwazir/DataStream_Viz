import time
from typing import Optional
import numpy as np
import serial
from PyQt5 import QtCore

from config import DEFAULT_POLL_HZ

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

    def __init__(self, port: str, baud: int, poll_enabled: bool = True, poll_hz: float = DEFAULT_POLL_HZ, parent=None):
        super().__init__(parent)
        self.port_name = port
        self.baud = baud
        self._stop = False
        self.ser: Optional[serial.Serial] = None
        self._buf = bytearray()

        # polling config
        self.poll_enabled = bool(poll_enabled)
        self.poll_hz = float(poll_hz)
        self.poll_period = (1.0 / self.poll_hz) if self.poll_enabled and self.poll_hz > 0 else None
        self._next_poll = time.perf_counter()

    def run(self):
        import serial.tools.list_ports  # noqa
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
                    # WRITE HERE if your device requires newline/CR
                    self.ser.write(b"1")
                except Exception as e:
                    self.status.emit(f"Serial write error: {e}")
                # schedule next poll
                self._next_poll += self.poll_period
                # prevent runaway backlog after delays
                if self.poll_period and (now - self._next_poll > 2 * self.poll_period):
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
                            # ignore malformed line
                            continue
                else:
                    self.msleep(1)  # small sleep if nothing arrived
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
