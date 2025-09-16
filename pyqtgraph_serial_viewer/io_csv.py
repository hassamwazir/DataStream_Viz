from __future__ import annotations
import numpy as np
import csv

def load_csv_file(path: str):
    """
    Returns (header: list[str], t_abs: (N,), Y: (C,N) float32)
    Expects header[0] == 'timestamp'.
    Raises ValueError on validation problems.
    """
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    if not header or header[0].strip().lower() != "timestamp" or len(header) < 2:
        raise ValueError("CSV must start with header: timestamp,ch1,...")

    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=float)
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {e}") from e

    if data.ndim == 1:
        data = data.reshape(1, -1)

    num_ch = len(header) - 1
    if data.shape[1] != (1 + num_ch):
        raise ValueError("Column count mismatch between header and rows.")

    t_abs = data[:, 0].astype(float)          # (N,)
    Y = data[:, 1:].astype(np.float32).T      # (C,N)

    # sort by time if needed
    if t_abs.size and not np.all(np.diff(t_abs) >= 0):
        order = np.argsort(t_abs)
        t_abs = t_abs[order]
        Y = Y[:, order]

    return header, t_abs, Y

def write_csv(path: str, t_abs: np.ndarray, Y: np.ndarray):
    """
    Writes header: timestamp, ch1..chC; rows in chronological order.
    Y shape: (C,N) matching t_abs (N,)
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["timestamp"] + [f"ch{i+1}" for i in range(Y.shape[0])]
        w.writerow(header)
        for i in range(t_abs.shape[0]):
            row = [f"{float(t_abs[i]):.10f}"] + [f"{float(v):.10f}" for v in Y[:, i]]
            w.writerow(row)
