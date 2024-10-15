import os
import re
import sys
import traceback
from queue import Queue
from PyQt5.QtWidgets import QMessageBox
from src.hfo_app import HFO_App
from src.controllers.main_window_controller import MainWindowController
from src.models.main_window_model import MainWindowModel
from src.views.main_window_view import MainWindowView
from src.utils.utils_gui import *
from PyQt5.QtCore import pyqtSignal


class MainWindow(QMainWindow):
    close_signal = pyqtSignal()

    def __init__(self):
        super(MainWindow, self).__init__()

        self.backend = None
        self.model = MainWindowModel(self, self.backend)
        self.view = MainWindowView(self)
        self.main_window_controller = MainWindowController(self.view, self.model)

        # initialize general UI
        self.main_window_controller.init_general_window()

        # initialize biomarker type
        self.main_window_controller.init_biomarker_type()

        # initialize biomarker specific UI
        biomarker = self.main_window_controller.get_biomarker_type()
        self.main_window_controller.init_biomarker_window(biomarker)
