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
        self.model.waveform_plot.plot_all_axes(start, end, channel)  # Default interval

    def create_fft_plot(self):
        self.model.create_fft_plot()
        self.view.add_widget('FFT_layout', self.model.fft_plot)
        channel, start, end = self.get_current_event()
        self.model.fft_plot.plot(start, end, channel)  # Default interval

    def update_plots(self, start, end, channel, reset_intervals=False, default_interval=None):
        if reset_intervals and default_interval is not None:
            self.model.reset_intervals_to_default(default_interval)
        self.model.waveform_plot.plot_all_axes(start, end, channel)
        self.model.fft_plot.plot(start, end, channel)

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

    def get_next_unannotated_event(self):
        return self.model.get_next_unannotated_event()

    def get_prev_unannotated_event(self):
        return self.model.get_prev_unannotated_event()

    def set_doctor_annotation(self, ann):
        selected_index, item_text = self.model.set_doctor_annotation(ann)
        return selected_index, item_text

    def clear_current_annotation(self):
        selected_index, item_text = self.model.clear_current_annotation()
        return selected_index, item_text

    def get_review_progress(self):
        return self.model.get_review_progress()

    def get_annotation_counts(self):
        return self.model.get_annotation_counts()

    def has_reviewable_events(self):
        return self.model.has_reviewable_events()

    def set_current_freq_limit(self, min_freq, max_freq):
        self.model.set_current_freq_limit(min_freq, max_freq)

    def set_current_interval(self, interval):
        self.model.set_current_interval(interval)

    def reset_view_to_default(self, default_interval):
        """Reset all plot windows to default view."""
        self.model.reset_intervals_to_default(default_interval)

    def set_sync_views(self, enabled):
        """Enable or disable syncing of view movements across all subplots."""
        self.model.set_sync_views(enabled)
    
    def reset_to_default_view(self):
        """Reset all plot views to default auto-zoom without replotting."""
        self.model.reset_to_default_view()

    def go_back_view(self):
        return self.model.go_back_view()

    def go_forward_view(self):
        return self.model.go_forward_view()

    def can_go_back(self):
        return self.model.can_go_back()

    def can_go_forward(self):
        return self.model.can_go_forward()

    def set_fft_window(self, time_window):
        self.model.set_fft_window(time_window)

    def clear_fft_window(self):
        self.model.clear_fft_window()
