from src.models.main_waveform_plot_model import MainWaveformPlotModel
from src.views.main_window_view import MainWindowView
import pyqtgraph as pg
import numpy as np


class MainWindowController:
    def __init__(self, main_window):
        self.model = None
        self.view = MainWindowView(main_window)