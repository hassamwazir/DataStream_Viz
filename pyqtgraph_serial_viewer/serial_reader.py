import time
from typing import Optional
import numpy as np
import serial
from PySide6.QtCore import Signal, QThread, QObject

# ---- Tunables ----
START_STREAM_BYTE = b"1"   # command to start streaming on the device
STOP_STREAM_BYTE  = b"0"   # command to stop streaming on the device


class SerialReader(QThread):
    """
    Reads newline-delimited, space-separated numeric samples from a serial port.

    - On connect: sends a single 'start' command (START_STREAM_BYTE) to the device.
    - The device then streams samples continuously: "t ch1 ch2 ch3 ..."
    - Emits a numpy array for each complete, parsed line.
    - On stop/cleanup: sends a single 'stop' command (STOP_STREAM_BYTE).
    """

    sample = Signal(np.ndarray)
    status = Signal(str)
    connected = Signal()
    disconnected = Signal()

    def __init__(self, port: str, baud: int, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.port_name = port
        self.baud = baud
        self._stop = False
        self.ser: Optional[serial.Serial] = None
        self._buf = bytearray()

    # ---------- helpers ----------
    def _send_start(self):
        """Tell the device to start streaming."""
        if not self.ser:
            return
        try:
            self.ser.write(START_STREAM_BYTE)
            # self.status.emit("→ sent START command")
        except Exception as e:
            self.status.emit(f"Serial write error (start): {e}")

    def _send_stop(self):
        """Tell the device to stop streaming."""
        if not self.ser:
            return
        try:
            self.ser.write(STOP_STREAM_BYTE)
            # small delay to let the command flush out
            self.ser.flush()
            # self.status.emit("→ sent STOP command")
        except Exception as e:
            self.status.emit(f"Serial write error (stop): {e}")

    # ---------- thread ----------
    def run(self):
        try:
            self.ser = serial.Serial(self.port_name, self.baud, timeout=0.01)
            self.status.emit(f"Opened {self.port_name} @ {self.baud} baud")
            self.connected.emit()
        except Exception as e:
            self.status.emit(f"ERROR opening {self.port_name}: {e}")
            return

        # Send one-time start command to begin continuous streaming
        self._send_start()

        while not self._stop:
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
                            # malformed; skip
                            self.status.emit("Parse error; skipping line…")
                            continue

                        if vals.size > 0:
                            # Emit the parsed sample (t, ch1, ch2, ...)
                            self.sample.emit(vals)
                else:
                    self.msleep(1)  # avoid busy-wait
            except Exception as e:
                self.status.emit(f"Serial read error: {e}")
                self.msleep(50)

        # Cleanup
        try:
            if self.ser and self.ser.is_open:
                # Tell device to stop streaming before closing
                self._send_stop()
                time.sleep(0.05)
                self.ser.close()
                self.status.emit("Serial port closed")
        except Exception:
            pass

        self.disconnected.emit()

    def stop(self):
        """Request the reader thread to stop and (in run) send '0' before closing."""
        self._stop = True
