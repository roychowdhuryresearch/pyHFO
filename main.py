import os
import re
import sys
from src.ui.main_window import MainWindow
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
import multiprocessing as mp
import torch
import warnings
warnings.filterwarnings("ignore")

# Enable DPI scaling
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


def closeAllWindows():
    QApplication.instance().closeAllWindows()


if __name__ == '__main__':
    mp.freeze_support()
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    app.aboutToQuit.connect(closeAllWindows)
    sys.exit(app.exec_())
    
