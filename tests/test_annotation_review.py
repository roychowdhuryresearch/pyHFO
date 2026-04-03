import numpy as np
from PyQt5 import QtCore

from src.hfo_feature import HFO_Feature
from src.spindle_feature import SpindleFeature
from src.ui.annotation import Annotation
from src.ui.annotation_plot import FFTPlot


class DummyReviewBackend:
    def __init__(self, biomarker_type, event_features, eeg_data=None):
        self.biomarker_type = biomarker_type
        self.sample_freq = 2000
        self.param_filter = None
        self.filter_data = None
        self.channel_names = np.array(["A1", "A2"])
        self.eeg_data = eeg_data if eeg_data is not None else np.vstack(
            [
                np.sin(np.linspace(0, 20, 8000)),
                np.cos(np.linspace(0, 20, 8000)),
            ]
        ) * 100
        self.event_features = event_features

    def get_eeg_data(self, start=None, end=None, filtered=False):
        data = self.eeg_data if not filtered or self.filter_data is None else self.filter_data
        if start is None and end is None:
            return data, self.channel_names
        if start is None:
            return data[:, :end], self.channel_names
        if end is None:
            return data[:, start:], self.channel_names
        return data[:, start:end], self.channel_names

    def get_eeg_data_shape(self):
        return self.eeg_data.shape

    def filter_eeg_data(self, *args, **kwargs):
        self.filter_data = self.eeg_data * 0.5

    def sync_active_run(self):
        pass


def test_hfo_relabel_clears_previous_choice():
    feature = HFO_Feature(np.array(["A1"]), np.array([[10, 20]]), sample_freq=2000)

    feature.doctor_annotation("Artifact")
    feature.doctor_annotation("Pathological")
    feature.doctor_annotation("Physiological")

    assert feature.artifact_annotations.tolist() == [0.0]
    assert feature.pathological_annotations.tolist() == [0.0]
    assert feature.physiological_annotations.tolist() == [1.0]
    assert feature.get_annotation() == "Physiological"


def test_spindle_relabel_to_artifact_clears_spike_flag():
    feature = SpindleFeature(np.array(["A1"]), np.array([10]), np.array([20]), sample_freq=2000)

    feature.doctor_annotation("Spike")
    feature.doctor_annotation("Artifact")

    assert feature.spike_annotations.tolist() == [0.0]
    assert feature.get_annotation() == "Artifact"
    assert feature.get_annotation_counts()["Spike"] == 0


def test_unannotated_navigation_skips_reviewed_events():
    feature = HFO_Feature(
        np.array(["A1", "A1", "A2"]),
        np.array([[10, 20], [30, 40], [50, 60]]),
        sample_freq=2000,
    )
    feature.doctor_annotation("Artifact")

    next_event = feature.get_next_unannotated()
    assert next_event == ("A1", 30, 40)
    assert feature.index == 1

    feature.doctor_annotation("Pathological")
    next_event = feature.get_next_unannotated()
    assert next_event == ("A2", 50, 60)
    assert feature.index == 2


def test_annotation_window_uses_saved_annotation_in_review_dropdown(qapp):
    feature = HFO_Feature(
        np.array(["A1", "A2"]),
        np.array([[1000, 1100], [3000, 3120]]),
        sample_freq=2000,
    )
    feature.update_pred(np.array([1, 1]), np.array([0, 0]), np.array([0, 0]))
    backend = DummyReviewBackend("HFO", feature)
    carrier = QtCore.QObject()
    window = Annotation(backend=backend, close_signal=carrier.destroyed)

    window.select_annotation_option("Artifact")
    window.update_button_clicked()

    assert backend.event_features.index == 1
    assert window.EventDropdown_Box.currentText() == "--- Event Type ---"
    assert window.reviewed_textbox.text() == "Unannotated"

    window.close()


def test_fft_plot_handles_zero_power_windows_without_nan(qapp):
    feature = HFO_Feature(np.array(["A1"]), np.array([[100, 200]]), sample_freq=2000)
    backend = DummyReviewBackend("HFO", feature, eeg_data=np.zeros((2, 8000)))
    fft_plot = FFTPlot(backend=backend)

    fft_plot.plot(100, 200, "A1")

    xdata, ydata = fft_plot.axs.lines[0].get_data()
    assert len(xdata) == len(ydata)
    assert np.isfinite(ydata).all()
