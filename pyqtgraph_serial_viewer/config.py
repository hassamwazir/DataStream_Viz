# ------------------ Defaults ------------------
DEFAULT_BAUD = 115200
ROLLING_SECONDS = 10     # visible window
TARGET_HZ = 50           # UI refresh Hz (not serial rate)
DEFAULT_POLL_HZ = 200.0  # how often to send '1' while connected
Y_LIMITS = (-0.5, 1.0)  # default y-axis limits. Max Y will auto-expand if needed
# ------------------------------------------------
# plotting history mode
FULL_HISTORY = True  # True = show everything; False = rolling window

# optional safety caps for full-history mode (purely defensive)
MAX_POINTS_PER_CHANNEL = 2_000_000  # set None to disable