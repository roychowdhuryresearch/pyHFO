import numpy as np
from src.models.annotation_model import AnnotationModel
from src.views.annotation_view import AnnotationView


class AnnotationController:
    def __init__(self, annotation_widget, backend=None):
        self.model = AnnotationModel(backend)
        self.view = AnnotationView(annotation_widget)

    def create_waveform_plot(self):
        self.model.create_waveform_plot()
        self.view.add_widget('VisulaizationVerticalLayout', self.model.waveform_plot)

    def create_fft_plot(self):
        self.model.create_fft_plot()
        self.view.add_widget('FFT_layout', self.model.fft_plot)
