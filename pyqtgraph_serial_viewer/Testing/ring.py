# ring.py
import numpy as np
import time as _time

class Ring2D:
    """
    Fixed-size ring for C channels × N samples, with timestamps.
    """
    def __init__(self, channels: int, capacity: int, dtype=np.float32,
                 t_dtype=np.float64):
        self.C = channels
        self.N = capacity
        self.data = np.empty((channels, capacity), dtype=dtype)
        self.t = np.empty(capacity, dtype=t_dtype)  # Unix seconds
        self.i = 0
        self.n = 0

    def append(self, vals, t=None):
        if t is None:
            t = _time.time()
        c = min(self.C, len(vals))
        self.data[:c, self.i] = vals[:c]
        self.t[self.i] = t
        self.i = (self.i + 1) % self.N
        if self.n < self.N:
            self.n += 1

    def view(self):
        """
        Returns (t, Y):
          t: (n,) float64 seconds since Unix epoch
          Y: (C, n) float32 values
        Both in chronological order.
        """
        if self.n < self.N:
            return self.t[:self.n], self.data[:, :self.n]
        i = self.i
        return (np.concatenate((self.t[i:], self.t[:i])),
                np.concatenate((self.data[:, i:], self.data[:, :i]), axis=1))
