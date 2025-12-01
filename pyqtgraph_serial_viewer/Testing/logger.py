# logger.py
import csv, os, time, datetime, threading, queue

class CsvWriter(threading.Thread):
    """
    Line-buffered CSV writer that consumes rows from a queue.
    Each row should be a tuple like: (timestamp, ch1, ch2, ...)
    Rotates files every `rotate_minutes` to keep sizes manageable.
    """
    def __init__(self, q: "queue.Queue", base_dir="logs", rotate_minutes=60, header=None):
        super().__init__(daemon=True)
        self.q = q
        self.stop_flag = threading.Event()
        self.base_dir = base_dir
        self.rotate_minutes = rotate_minutes
        self.header = header
        self._f = None
        self._w = None
        self._next_roll = 0.0

    def _open_new(self):
        os.makedirs(self.base_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(self.base_dir, f"log_{ts}.csv")
        self._f = open(path, "a", newline="", buffering=1)  # line buffered
        self._w = csv.writer(self._f)
        if self.header:
            self._w.writerow(self.header)
        self._next_roll = time.time() + 60 * self.rotate_minutes

    def run(self):
        self._open_new()
        while not self.stop_flag.is_set():
            try:
                row = self.q.get(timeout=0.25)
            except queue.Empty:
                continue
            if time.time() >= self._next_roll:
                try:
                    self._f.close()
                except Exception:
                    pass
                self._open_new()
            self._w.writerow(row)
        try:
            if self._f:
                self._f.close()
        except Exception:
            pass

    def stop(self):
        self.stop_flag.set()
