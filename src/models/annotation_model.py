import numpy as np
from src.hfo_app import HFO_App
from src.ui.annotation_plot import AnnotationPlot, FFTPlot


class AnnotationModel:
    def __init__(self, backend):
        self.backend = backend

    def create_waveform_plot(self):
        self.waveform_plot = AnnotationPlot(hfo_app=self.backend)

    def create_fft_plot(self):
        self.fft_plot = FFTPlot(hfo_app=self.backend)
