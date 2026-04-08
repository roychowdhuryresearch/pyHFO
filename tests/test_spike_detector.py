import numpy as np
from PyQt5 import QtCore

from src.param.param_detector import ParamSpikeRMSLL
from src.spike_feature import SpikeFeature
from src.ui.annotation import Annotation
from src.utils.utils_detector import SpikeRMSLLDetector


def test_spike_feature_relabel_updates_counts_and_overlay_state():
    feature = SpikeFeature(np.array(["A1"]), np.array([10]), np.array([20]), sample_freq=2000)

    starts, ends, artifacts, spikes = feature.get_biomarkers_for_channel("A1", 0, 100)
    assert starts.tolist() == [10]
    assert ends.tolist() == [20]
    assert artifacts.tolist() == [1]
    assert spikes.tolist() == [True]

    feature.doctor_annotation("Accepted")
    assert feature.get_annotation() == "Accepted"
    starts, ends, artifacts, spikes = feature.get_biomarkers_for_channel("A1", 0, 100)
    assert artifacts.tolist() == [1]
    assert spikes.tolist() == [False]

    feature.doctor_annotation("Artifact")
    assert feature.accepted_annotations.tolist() == [0.0]
    assert feature.get_annotation_counts() == {"Artifact": 1, "Accepted": 0}
    starts, ends, artifacts, spikes = feature.get_biomarkers_for_channel("A1", 0, 100)
    assert artifacts.tolist() == [0]
    assert spikes.tolist() == [False]


def test_spike_rms_ll_detector_finds_synthetic_spike():
    sample_freq = 2000
    duration_seconds = 2.0
    times = np.arange(int(sample_freq * duration_seconds)) / sample_freq
    data = np.random.default_rng(0).normal(scale=0.02, size=times.shape)

    spike_center = 1.0
    data += 4.5 * np.exp(-((times - spike_center) / 0.004) ** 2)
    data -= 1.5 * np.exp(-((times - (spike_center + 0.006)) / 0.006) ** 2)

    detector = SpikeRMSLLDetector(
        sample_freq=sample_freq,
        filter_freq=[4, 80],
        rms_window=0.01,
        ll_window=0.008,
        rms_thres=1.5,
        ll_thres=1.5,
        peak_thres=2.0,
        min_window=0.003,
        max_window=0.05,
        min_gap=0.02,
        n_jobs=1,
        front_num=1,
    )

    events = detector.detect_channel(data)

    assert len(events) >= 1
    first_start, first_end = events[0]
    assert first_start < int(spike_center * sample_freq) < first_end


def test_spike_default_parameters_use_conservative_review_preset():
    params = ParamSpikeRMSLL()

    assert params.rms_window == 0.02
    assert params.ll_window == 0.01
    assert params.rms_thres == 5.0
    assert params.ll_thres == 5.0
    assert params.peak_thres == 6.0
    assert params.min_window == 0.015
    assert params.max_window == 0.05
    assert params.min_gap == 0.08


class _DummySpikeReviewBackend:
    def __init__(self, feature):
        self.biomarker_type = "Spike"
        self.sample_freq = 2000
        self.param_filter = None
        self.filter_data = None
        self.event_features = feature
        self.channel_names = np.array(["A1"])
        self.eeg_data = np.zeros((1, 4000))

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
        self.filter_data = self.eeg_data

    def sync_active_run(self):
        pass


def test_spike_annotation_window_exposes_spike_review_labels(qapp):
    feature = SpikeFeature(np.array(["A1"]), np.array([100]), np.array([120]), sample_freq=2000)
    backend = _DummySpikeReviewBackend(feature)
    carrier = QtCore.QObject()
    window = Annotation(backend=backend, close_signal=carrier.destroyed)

    options = [window.EventDropdown_Box.itemText(index) for index in range(window.EventDropdown_Box.count())]
    assert options == ["--- Event Type ---", "Accepted", "Artifact"]

    window.select_annotation_option("Accepted")
    window.update_button_clicked()
    assert backend.event_features.accepted_annotations.tolist() == [1.0]

    window.close()
