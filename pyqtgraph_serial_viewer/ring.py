# ring.py
import numpy as np

class Ring2D:
    """
    Fixed-size ring for C channels × N samples.
    Appends one multi-channel sample at a time (vals shape ~ (C,)).
    """
    def __init__(self, channels: int, capacity: int, dtype=np.float32):
        self.C = channels
        self.N = capacity
        self.data = np.empty((channels, capacity), dtype=dtype)
        self.i = 0              # next write index
        self.n = 0              # number of valid samples (<= N)

    def append(self, vals):
        # vals: 1D array-like, length >= C (extra values ignored)
        c = min(self.C, len(vals))
        self.data[:c, self.i] = vals[:c]
        self.i = (self.i + 1) % self.N
        if self.n < self.N:
            self.n += 1

    def as_2d(self) -> np.ndarray:
        """
        Return a (C, n) view in chronological order (no copies when not wrapped;
        when wrapped, concatenates two slices).
        """
        if self.n < self.N:
            return self.data[:, :self.n]
        i = self.i
        return np.concatenate((self.data[:, i:], self.data[:, :i]), axis=1)
