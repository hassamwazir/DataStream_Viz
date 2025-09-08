import sys
from PyQt5 import QtWidgets
from main_window import MultiPlot

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MultiPlot()
    win.resize(1100, 650)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
