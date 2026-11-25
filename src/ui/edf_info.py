from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5 import uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import sys
import re
import os
from pathlib import Path
ROOT_DIR = Path(__file__).parent


class EDFInformation(QtWidgets.QDialog):
    def __init__(self):
        super(EDFInformation, self).__init__()
        self.ui = uic.loadUi(os.path.join(ROOT_DIR, 'src/ui/edf_info.ui'), self)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = EDFInformation()
    mainWindow.show()
    sys.exit(app.exec_())
