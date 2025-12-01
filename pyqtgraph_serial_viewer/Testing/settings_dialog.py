from PySide6 import QtWidgets, QtCore, QtGui

class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, default_baud=115200, target_hz=50, y_min=-0.1, y_max=1.0,
                 max_points=72_000, line_width=2, tail_window_s=30.0, colors=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setModal(True)

        if colors is None:
            colors = []

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        # Default baud
        self.baud_cb = QtWidgets.QComboBox()
        self.baud_cb.setEditable(True)
        for b in [9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600]:
            self.baud_cb.addItem(str(b))
        self.baud_cb.setCurrentText(str(default_baud))
        form.addRow("Default baud:", self.baud_cb)

        # Target Hz
        self.hz_spin = QtWidgets.QSpinBox()
        self.hz_spin.setRange(1, 240)
        self.hz_spin.setValue(int(target_hz))
        form.addRow("UI refresh (Hz):", self.hz_spin)

        # Y limits
        self.ymin_spin = QtWidgets.QDoubleSpinBox()
        self.ymin_spin.setRange(-1e9, 1e9); self.ymin_spin.setDecimals(6); self.ymin_spin.setValue(float(y_min))
        self.ymax_spin = QtWidgets.QDoubleSpinBox()
        self.ymax_spin.setRange(-1e9, 1e9); self.ymax_spin.setDecimals(6); self.ymax_spin.setValue(float(y_max))
        y_box = QtWidgets.QHBoxLayout()
        y_box.addWidget(QtWidgets.QLabel("Min:")); y_box.addWidget(self.ymin_spin); y_box.addSpacing(12)
        y_box.addWidget(QtWidgets.QLabel("Max:")); y_box.addWidget(self.ymax_spin)
        form.addRow("Y axis:", QtWidgets.QWidget())
        form.itemAt(form.rowCount()-1, QtWidgets.QFormLayout.FieldRole).widget().setLayout(y_box)

        # Max points per channel (0 = unlimited)
        self.max_points_spin = QtWidgets.QSpinBox()
        self.max_points_spin.setRange(0, 5_000_000)
        self.max_points_spin.setValue(int(max_points or 0))
        form.addRow("Max points / channel (0 = unlimited):", self.max_points_spin)

        # Line width
        self.width_spin = QtWidgets.QDoubleSpinBox()
        self.width_spin.setRange(0.5, 10.0); self.width_spin.setSingleStep(0.5)
        self.width_spin.setValue(float(line_width))
        form.addRow("Line width:", self.width_spin)

        # Line colors table
        self.colors_tbl = QtWidgets.QTableWidget(len(colors), 2)
        self.colors_tbl.setHorizontalHeaderLabels(["Preview", "RGB"])
        self.colors_tbl.verticalHeader().setVisible(False)
        self.colors_tbl.horizontalHeader().setStretchLastSection(True)
        for r, c in enumerate(colors):
            self._set_color_row(r, c)

        btns_line = QtWidgets.QHBoxLayout()
        self.add_color_btn = QtWidgets.QPushButton("Add")
        self.edit_color_btn = QtWidgets.QPushButton("Edit…")
        self.remove_color_btn = QtWidgets.QPushButton("Remove")
        btns_line.addWidget(self.add_color_btn); btns_line.addWidget(self.edit_color_btn); btns_line.addWidget(self.remove_color_btn)

        layout.addWidget(QtWidgets.QLabel("Line colors:"))
        layout.addWidget(self.colors_tbl)
        layout.addLayout(btns_line)

        # Dialog buttons
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(btn_box)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

        # Tail window (seconds)
        self.tail_spin = QtWidgets.QDoubleSpinBox()
        self.tail_spin.setRange(0.001, 1e6)
        self.tail_spin.setDecimals(3)
        self.tail_spin.setSingleStep(0.5)
        self.tail_spin.setValue(float(tail_window_s))
        form.addRow("Tail window (s):", self.tail_spin)

        # Wire up color controls
        self.add_color_btn.clicked.connect(self._on_add_color)
        self.edit_color_btn.clicked.connect(self._on_edit_color)
        self.remove_color_btn.clicked.connect(self._on_remove_color)

    def _set_color_row(self, row, rgb_tuple):
        r, g, b = [int(x) for x in rgb_tuple]
        preview = QtWidgets.QLabel()
        preview.setAutoFillBackground(True)
        pal = preview.palette()
        pal.setColor(preview.backgroundRole(), QtGui.QColor(r, g, b))
        preview.setPalette(pal)
        preview.setMinimumHeight(18)
        preview.setFrameShape(QtWidgets.QFrame.Panel)
        preview.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.colors_tbl.setCellWidget(row, 0, preview)

        item = QtWidgets.QTableWidgetItem(f"({r}, {g}, {b})")
        item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        self.colors_tbl.setItem(row, 1, item)

    def _on_add_color(self):
        col = QtWidgets.QColorDialog.getColor(QtGui.QColor(0, 0, 0), self, "Pick color")
        if not col.isValid():
            return
        row = self.colors_tbl.rowCount()
        self.colors_tbl.insertRow(row)
        self._set_color_row(row, (col.red(), col.green(), col.blue()))

    def _on_edit_color(self):
        row = self.colors_tbl.currentRow()
        if row < 0:
            return
        cur = self._row_rgb(row)
        col = QtWidgets.QColorDialog.getColor(QtGui.QColor(*cur), self, "Pick color")
        if not col.isValid():
            return
        self._set_color_row(row, (col.red(), col.green(), col.blue()))

    def _on_remove_color(self):
        row = self.colors_tbl.currentRow()
        if row >= 0:
            self.colors_tbl.removeRow(row)

    def _row_rgb(self, row):
        text = self.colors_tbl.item(row, 1).text()
        nums = text.strip("() ").split(",")
        return tuple(int(float(x)) for x in nums)

    def values(self):
        default_baud = int(self.baud_cb.currentText())
        target_hz = int(self.hz_spin.value())
        y_min = float(self.ymin_spin.value())
        y_max = float(self.ymax_spin.value())
        max_points_val = int(self.max_points_spin.value())
        max_points = None if max_points_val == 0 else max_points_val
        line_width = float(self.width_spin.value())
        colors = [self._row_rgb(r) for r in range(self.colors_tbl.rowCount())]
        return {
            "default_baud": default_baud,
            "target_hz": target_hz,
            "y_limits": (y_min, y_max),
            "max_points_per_channel": max_points,
            "line_width": line_width,
            "tail_window_s": float(self.tail_spin.value()),
            "line_colors": colors,
        }
