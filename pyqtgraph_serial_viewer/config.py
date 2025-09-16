# ------------------ Defaults ------------------
DEFAULT_BAUD = 115200
TARGET_HZ = 50           # UI refresh Hz (not serial rate)
Y_LIMITS = (-0.1, 1.0)   # default y-axis limits. Max Y will auto-expand if needed

# optional safety caps for full-history mode (purely defensive)
MAX_POINTS_PER_CHANNEL = 72_000  # set None to disable

TAIL_WINDOW_S = 30.0
# ---------------- Plot Options ----------------
LINE_WIDTH = 2
LINE_COLORS = [
    (31, 119, 180),  # blue
    (255, 127, 14),  # orange
    (44, 160, 44),   # green
    (214, 39, 40),   # red
    (148, 103, 189), # purple
    (140, 86, 75),   # brown
    (227, 119, 194), # pink
    (127, 127, 127), # gray
    (188, 189, 34),  # yellow-green
    (23, 190, 207),  # cyan
]
