# serial_reader.py
import time
from typing import Optional
import numpy as np
import serial
from PyQt5 import QtCore

# ---- Tunables ----
REQUEST_BYTE = b"1"        # change to b"1\n" or b"1\r\n" if your device expects newline/CR
REQUEST_TIMEOUT_S = 0.300  # resend if no reply within this time
REQUEST_MAX_RETRIES = 3    # retries before a short backoff
REQUEST_BACKOFF_S = 0.500  # pause after hitting max retries


class SerialReader(QtCore.QThread):
    """
    Reads newline-delimited, space-separated numeric samples from a serial port.
    - Sends one request byte per sample ("request-on-receive").
    - Emits a numpy array for each complete, parsed line.
    - Has a watchdog to recover from dropped/missing responses.
    """
    sample = QtCore.pyqtSignal(np.ndarray)
    status = QtCore.pyqtSignal(str)
    connected = QtCore.pyqtSignal()
    disconnected = QtCore.pyqtSignal()

    def __init__(self, port: str, baud: int, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self.port_name = port
        self.baud = baud
        self._stop = False
        self.ser: Optional[serial.Serial] = None
        self._buf = bytearray()

        # request-on-receive state
        self._awaiting_reply = False
        self._last_request_t = 0.0
        self._retry_count = 0

    # ---------- helpers ----------
    def _send_request(self):
        try:
            self.ser.write(REQUEST_BYTE)
            self._last_request_t = time.perf_counter()
            self._awaiting_reply = True
            # self.status.emit("→ requested sample")  # noisy; uncomment if useful
        except Exception as e:
            self.status.emit(f"Serial write error: {e}")

    def _watchdog(self, now: float):
        """If waiting for a reply and it's late, retry with limited backoff."""
        if not self._awaiting_reply:
            return
        if now - self._last_request_t < REQUEST_TIMEOUT_S:
            return

        if self._retry_count < REQUEST_MAX_RETRIES:
            self._retry_count += 1
            self.status.emit(f"Watchdog retry {self._retry_count}/{REQUEST_MAX_RETRIES}")
            self._send_request()
        else:
            self.status.emit("Max retries hit; backing off…")
            time.sleep(REQUEST_BACKOFF_S)
            self._retry_count = 0
            self._send_request()

    # ---------- thread ----------
    def run(self):
        try:
            self.ser = serial.Serial(self.port_name, self.baud, timeout=0.01)
            self.status.emit(f"Opened {self.port_name} @ {self.baud} baud")
            self.connected.emit()
        except Exception as e:
            self.status.emit(f"ERROR opening {self.port_name}: {e}")
            return

        # Kick the first request
        self._send_request()

        while not self._stop:
            now = time.perf_counter()

            # Read any available bytes
            try:
                data = self.ser.read(4096)
                if data:
                    self._buf.extend(data)
                    *lines, remainder = self._buf.split(b"\n")
                    self._buf = bytearray(remainder)

                    for raw in lines:
                        line = raw.strip().replace(b"\r", b"")
                        if not line:
                            continue
                        parts = line.split()
                        try:
                            vals = np.array([float(p) for p in parts], dtype=float)
                        except ValueError:
                            # malformed; request another sample and continue
                            self.status.emit("Parse error; re-requesting…")
                            self._send_request()
                            continue

                        if vals.size > 0:
                            # Got a valid sample: clear awaiting flag and emit
                            self._awaiting_reply = False
                            self._retry_count = 0
                            self.sample.emit(vals)

                            # Immediately request next sample
                            self._send_request()
                else:
                    self.msleep(1)  # avoid busy-wait
            except Exception as e:
                self.status.emit(f"Serial read error: {e}")
                self.msleep(50)

            # Watchdog for late/missing replies
            self._watchdog(now)

        # Cleanup
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
                self.status.emit("Serial port closed")
        except Exception:
            pass
        self.disconnected.emit()

    def stop(self):
        self._stop = True
