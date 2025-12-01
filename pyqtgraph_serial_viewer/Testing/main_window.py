# main_window.py
import time
import queue
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
import serial
import serial.tools.list_ports
from PySide6 import QtWidgets, QtCore, QtGui

from config import (
    DEFAULT_BAUD, TARGET_HZ, Y_LIMITS,
    MAX_POINTS_PER_CHANNEL, LINE_WIDTH, TAIL_WINDOW_S,  LINE_COLORS
)
from logger import CsvWriter
from ring import Ring2D
from serial_reader import SerialReader

try:
    from pyqtgraph.graphicsItems.PlotItem.ButtonItem import ButtonItem as _PgButtonItem
except Exception:
    _PgButtonItem = None


# split-out modules
from settings_dialog import SettingsDialog
from io_csv import load_csv_file, write_csv


class MultiPlot(QtWidgets.QMainWindow):
    """
    Two-plot layout:
      • Top plot: zoomed view (what you analyze), crosshair + click-to-pick.
      • Bottom plot: full-history overview with a LinearRegionItem that controls the top plot's X range.
    """
    def __init__(self):
        super().__init__()

