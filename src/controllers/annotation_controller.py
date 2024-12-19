import numpy as np
from src.models.annotation_model import AnnotationModel
from src.views.annotation_view import AnnotationView


class AnnotationController:
    def __init__(self, annotation_widget, backend=None):
        self.model = AnnotationModel(backend)
        self.view = AnnotationView(annotation_widget)

        # define window length
        if self.model.backend.biomarker_type == "HFO":
            self.interval = 1.0
        elif self.model.backend.biomarker_type == "Spindle":
            self.interval = 4.0

    def create_waveform_plot(self):
        self.model.create_waveform_plot()
        self.view.add_widget('VisulaizationVerticalLayout', self.model.waveform_plot)
        channel, start, end = self.get_current_event()
        self.model.waveform_plot.plot(start, end, channel, interval=self.interval)  # Default interval

    def create_fft_plot(self):
        self.model.create_fft_plot()
        self.view.add_widget('FFT_layout', self.model.fft_plot)
        channel, start, end = self.get_current_event()
        self.model.fft_plot.plot(start, end, channel, interval=self.interval)  # Default interval

    def update_plots(self, start, end, channel, interval):
        self.model.waveform_plot.plot(start, end, channel, interval=interval)
        self.model.fft_plot.plot(start, end, channel, interval=interval)

    def get_current_event(self):
        channel, start, end = self.model.get_current_event()
        return channel, start, end

    def get_previous_event(self):
        channel, start, end = self.model.get_previous_event()
        return channel, start, end

    def get_next_event(self):
        channel, start, end = self.model.get_next_event()
        return channel, start, end

    def get_jumped_event(self, index):
        channel, start, end = self.model.get_jumped_event(index)
        return channel, start, end

    def set_doctor_annotation(self, ann):
        selected_index, item_text = self.model.set_doctor_annotation(ann)
        return selected_index, item_text
