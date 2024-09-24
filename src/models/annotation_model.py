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

    def get_current_event(self):
        channel, start, end = self.backend.event_features.get_current()
        return channel, start, end

    def get_previous_event(self):
        channel, start, end = self.backend.event_features.get_prev()
        return channel, start, end

    def get_next_event(self):
        channel, start, end = self.backend.event_features.get_next()
        return channel, start, end

    def get_jumped_event(self, index):
        channel, start, end = self.backend.event_features.get_jump(index)
        return channel, start, end

    def set_doctor_annotation(self, ann):
        self.backend.event_features.doctor_annotation(ann)
        # Update the text of the selected item in the dropdown menu
        selected_index = self.backend.event_features.index
        item_text = self.backend.event_features.get_annotation_text(selected_index)
        return selected_index, item_text
