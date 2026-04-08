import numpy as np
from PyQt5 import QtCore, QtTest

from src.hfo_feature import HFO_Feature
from src.spindle_feature import SpindleFeature
from src.ui.annotation import Annotation
from src.ui.annotation_plot import AnnotationPlot, FFTPlot


class DummyReviewBackend:
    def __init__(self, biomarker_type, event_features, eeg_data=None, sample_freq=2000):
        self.biomarker_type = biomarker_type
        self.sample_freq = sample_freq
        self.param_filter = None
        self.filter_data = None
        event_channels = getattr(event_features, "channel_names", np.array([]))
        unique_channels = np.unique(event_channels) if len(event_channels) else np.array(["A1", "A2"])
        self.channel_names = unique_channels
        if eeg_data is not None:
            if eeg_data.shape[0] != len(self.channel_names):
                if eeg_data.shape[0] > len(self.channel_names):
                    extra_count = eeg_data.shape[0] - len(self.channel_names)
                    extras = np.array([f"AUX{idx + 1}" for idx in range(extra_count)])
                    self.channel_names = np.concatenate([self.channel_names, extras])
                else:
                    self.channel_names = self.channel_names[: eeg_data.shape[0]]
            self.eeg_data = eeg_data
        else:
            total_samples = int(self.sample_freq * 4)
            timeline = np.linspace(0, total_samples / self.sample_freq, total_samples, endpoint=False)
            waves = []
            for idx, _channel in enumerate(self.channel_names):
                phase = idx * (np.pi / 4)
                waves.append(np.sin(timeline + phase) * 100)
            self.eeg_data = np.vstack(waves)
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


def test_hfo_prediction_scope_navigation_tracks_match_groups():
    feature = HFO_Feature(
        np.array(["A1", "A1", "A2", "A2"]),
        np.array([[10, 20], [30, 40], [50, 60], [70, 80]]),
        sample_freq=2000,
    )
    feature.update_pred(
        np.array([0, 1, 1, 1]),
        np.array([0, 1, 0, 1]),
        np.array([0, 0, 1, 1]),
    )

    assert feature.get_prediction_scope_options() == ["All", "Artifact", "Non-artifact", "spkHFO", "eHFO"]
    assert feature.get_matching_indexes("Artifact").tolist() == [0]
    assert feature.get_matching_indexes("spkHFO").tolist() == [1, 3]
    assert feature.get_matching_indexes("eHFO").tolist() == [2, 3]

    next_event = feature.get_next_matching("eHFO")
    assert next_event == ("A2", 50, 60)
    assert feature.index == 2

    prev_event = feature.get_prev_matching("spkHFO")
    assert prev_event == ("A1", 30, 40)
    assert feature.index == 1


def test_hfo_prediction_scope_can_intersect_with_unannotated_only():
    feature = HFO_Feature(
        np.array(["A1", "A1", "A2", "A2"]),
        np.array([[10, 20], [30, 40], [50, 60], [70, 80]]),
        sample_freq=2000,
    )
    feature.update_pred(
        np.array([1, 1, 1, 1]),
        np.array([0, 1, 0, 1]),
        np.array([0, 1, 1, 1]),
    )
    feature.get_jump(1)
    feature.doctor_annotation("Pathological")

    assert feature.get_matching_indexes("eHFO").tolist() == [1, 2, 3]
    assert feature.get_matching_indexes("eHFO", unannotated_only=True).tolist() == [2, 3]

    next_event = feature.get_next_matching("eHFO", unannotated_only=True)
    assert next_event == ("A2", 50, 60)
    assert feature.index == 2


def test_spindle_prediction_scope_navigation_tracks_spike_matches():
    feature = SpindleFeature(
        np.array(["A1", "A2", "A3"]),
        np.array([10, 30, 50]),
        np.array([20, 40, 60]),
        sample_freq=2000,
    )
    feature.update_pred(np.array([0, 1, 1]), np.array([0, 1, 0]))

    assert feature.get_prediction_scope_options() == ["All", "Artifact", "Non-artifact", "Spike"]
    assert feature.get_matching_indexes("Artifact").tolist() == [0]
    assert feature.get_matching_indexes("Spike").tolist() == [1]

    next_event = feature.get_next_matching("Spike")
    assert next_event == ("A2", 30, 40)
    assert feature.index == 1


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


def test_annotation_window_prediction_scope_navigates_matching_events(qapp):
    feature = HFO_Feature(
        np.array(["A1", "A2", "A3"]),
        np.array([[1000, 1100], [3000, 3120], [5000, 5160]]),
        sample_freq=2000,
    )
    feature.update_pred(np.array([0, 1, 1]), np.array([0, 1, 0]), np.array([0, 0, 1]))
    backend = DummyReviewBackend("HFO", feature)
    carrier = QtCore.QObject()
    window = Annotation(backend=backend, close_signal=carrier.destroyed)

    options = [window.PredictionScopeBox.itemText(i) for i in range(window.PredictionScopeBox.count())]
    assert options == ["All", "Artifact", "Non-artifact", "spkHFO", "eHFO"]

    window.PredictionScopeBox.setCurrentText("eHFO")
    assert window.match_summary_label.text() == "Scope: eHFO | 1 matches | current event outside scope"

    window.plot_next_match()
    assert backend.event_features.index == 2
    assert window.match_summary_label.text() == "Scope: eHFO | 1/1 matches"

    window.close()


def test_annotation_window_prediction_scope_can_limit_to_unannotated(qapp):
    feature = HFO_Feature(
        np.array(["A1", "A2", "A3"]),
        np.array([[1000, 1100], [3000, 3120], [5000, 5160]]),
        sample_freq=2000,
    )
    feature.update_pred(np.array([1, 1, 1]), np.array([0, 0, 0]), np.array([0, 1, 1]))
    feature.get_jump(1)
    feature.doctor_annotation("Pathological")
    backend = DummyReviewBackend("HFO", feature)
    carrier = QtCore.QObject()
    window = Annotation(backend=backend, close_signal=carrier.destroyed)

    window.PredictionScopeBox.setCurrentText("eHFO")
    assert window.match_summary_label.text() == "Scope: eHFO | 1/2 matches"

    window.UnannotatedOnlyCheckBox.setChecked(True)
    assert window.match_summary_label.text() == "Scope: eHFO + Unannotated | 1 matches | current event outside scope"

    window.plot_next_match()
    assert backend.event_features.index == 2
    assert window.match_summary_label.text() == "Scope: eHFO + Unannotated | 1/1 matches"

    window.close()


def test_annotation_window_places_progress_at_top_and_tunes_frequency_controls(qapp):
    feature = HFO_Feature(
        np.array(["A1", "A2", "A3"]),
        np.array([[1000, 1100], [3000, 3120], [5000, 5160]]),
        sample_freq=2000,
    )
    backend = DummyReviewBackend("HFO", feature)
    carrier = QtCore.QObject()
    window = Annotation(backend=backend, close_signal=carrier.destroyed)

    assert window.progress_textbox.parent() is window.progress_card
    assert window.progress_card.parent() is window.centralwidget
    assert window.progress_textbox.text() == "0 reviewed / 3 total | 3 remaining"
    progress_row = None
    for index in range(window.left_panel.count()):
        item = window.left_panel.itemAt(index)
        if item.widget() is window.progress_card:
            progress_row = window.left_panel.getItemPosition(index)[0]
            break
    assert progress_row == 0
    assert window.label.text() == "Min Hz"
    assert window.label_2.text() == "Max Hz"
    assert window.SetFreqLimit.text() == "Apply"
    layout_positions = {}
    for index in range(window.gridLayout.count()):
        item = window.gridLayout.itemAt(index)
        widget = item.widget()
        if widget in {
            window.label,
            window.spinBox_minFreq,
            window.label_2,
            window.spinBox_maxFreq,
            window.SetFreqLimit,
        }:
            layout_positions[widget.objectName()] = window.gridLayout.getItemPosition(index)
    assert layout_positions["label"][0] == 0
    assert layout_positions["spinBox_minFreq"][0] == 0
    assert layout_positions["label_2"][0] == 0
    assert layout_positions["spinBox_maxFreq"][0] == 0
    assert layout_positions["SetFreqLimit"][0] == 0

    window.close()


def test_annotation_frequency_fields_submit_on_edit_commit(qapp):
    feature = HFO_Feature(
        np.array(["A1", "A2"]),
        np.array([[1000, 1100], [3000, 3120]]),
        sample_freq=2000,
    )
    backend = DummyReviewBackend("HFO", feature)
    window = Annotation(backend=backend)
    try:
        clicked = []
        window.SetFreqLimit.clicked.connect(lambda: clicked.append(True))

        line_edit = window.spinBox_minFreq.lineEdit()
        assert line_edit is not None
        line_edit.setFocus(QtCore.Qt.OtherFocusReason)
        line_edit.selectAll()
        QtTest.QTest.keyClicks(line_edit, "20")
        QtTest.QTest.keyClick(line_edit, QtCore.Qt.Key_Return)
        qapp.processEvents()

        assert clicked
    finally:
        window.close()


def test_fft_plot_handles_zero_power_windows_without_nan(qapp):
    feature = HFO_Feature(np.array(["A1"]), np.array([[100, 200]]), sample_freq=2000)
    backend = DummyReviewBackend("HFO", feature, eeg_data=np.zeros((2, 8000)))
    fft_plot = FFTPlot(backend=backend)

    fft_plot.plot(100, 200, "A1")

    xdata, ydata = fft_plot.get_curve_data()
    assert len(xdata) == len(ydata)
    assert np.isfinite(ydata).all()


def test_annotation_plot_places_high_frequency_at_top(qapp):
    feature = HFO_Feature(np.array(["A1"]), np.array([[1000, 1100]]), sample_freq=2000)
    t = np.arange(8000) / 2000
    high_freq_signal = np.sin(2 * np.pi * 400 * t) * 100
    eeg_data = np.vstack([high_freq_signal, high_freq_signal])
    backend = DummyReviewBackend("HFO", feature, eeg_data=eeg_data)
    plot = AnnotationPlot(backend=backend)

    plot.plot_full_data(2, 1000, 1100, "A1")

    metadata = plot.get_spectrogram_metadata()
    assert metadata["high_frequency_at_top"] is True
    assert metadata["cmap"] == "magma"
    assert plot.get_axis_ranges()[2][1][0] < plot.get_axis_ranges()[2][1][1]


def test_annotation_plot_maps_time_frequency_image_as_row_major(qapp):
    feature = HFO_Feature(np.array(["A1"]), np.array([[1000, 1100]]), sample_freq=2000)
    t = np.arange(8000) / 2000
    burst = np.zeros_like(t)
    mask = (t >= 0.5) & (t <= 0.56)
    burst[mask] = np.sin(2 * np.pi * 120 * t[mask]) * 100
    eeg_data = np.vstack([burst, burst])
    backend = DummyReviewBackend("HFO", feature, eeg_data=eeg_data)
    plot = AnnotationPlot(backend=backend)

    plot.plot_full_data(2, 1000, 1100, "A1")

    image_item = plot.tf_image_items[2]
    assert image_item is not None
    assert image_item.axisOrder == "row-major"


def test_annotation_frequency_controls_start_in_sync_with_tf_plot(qapp):
    feature = HFO_Feature(np.array(["A1"]), np.array([[1000, 1100]]), sample_freq=2000)
    backend = DummyReviewBackend("HFO", feature)
    window = Annotation(backend=backend)
    try:
        tf_limits = tuple(int(round(value)) for value in window.annotation_controller.model.waveform_plot.get_axis_ranges()[2][1])
        fft_limits = (
            window.annotation_controller.model.fft_plot.min_freq,
            window.annotation_controller.model.fft_plot.max_freq,
        )
        assert (window.spinBox_minFreq.value(), window.spinBox_maxFreq.value()) == tf_limits
        assert fft_limits == tf_limits
    finally:
        window.close()


def test_annotation_frequency_controls_clamp_to_nyquist(qapp):
    feature = HFO_Feature(np.array(["A1"]), np.array([[100, 140]]), sample_freq=200)
    backend = DummyReviewBackend("HFO", feature, sample_freq=200)
    window = Annotation(backend=backend)
    try:
        tf_limits = tuple(int(round(value)) for value in window.annotation_controller.model.waveform_plot.get_axis_ranges()[2][1])
        assert tf_limits == (1, 100)
        assert (window.spinBox_minFreq.value(), window.spinBox_maxFreq.value()) == tf_limits
        assert window.annotation_controller.model.fft_plot.max_freq == 100
        assert window.spinBox_minFreq.maximum() == 99
        assert window.spinBox_maxFreq.maximum() == 100
    finally:
        window.close()


def test_fft_plot_displays_peak_frequency_badge(qapp):
    feature = HFO_Feature(np.array(["A1"]), np.array([[1000, 1100]]), sample_freq=2000)
    t = np.arange(8000) / 2000
    peak_signal = np.sin(2 * np.pi * 90 * t) * 100
    eeg_data = np.vstack([peak_signal, peak_signal])
    backend = DummyReviewBackend("HFO", feature, eeg_data=eeg_data)
    fft_plot = FFTPlot(backend=backend)

    fft_plot.plot(1000, 1100, "A1")

    assert fft_plot.has_peak_badge() is True
    assert fft_plot.has_peak_marker() is True


def test_fft_plot_keeps_frequency_axis_label_visible(qapp):
    feature = HFO_Feature(np.array(["A1"]), np.array([[1000, 1100]]), sample_freq=2000)
    t = np.arange(8000) / 2000
    peak_signal = np.sin(2 * np.pi * 90 * t) * 100
    eeg_data = np.vstack([peak_signal, peak_signal])
    backend = DummyReviewBackend("HFO", feature, eeg_data=eeg_data)
    fft_plot = FFTPlot(backend=backend)
    fft_plot.resize(520, 220)
    fft_plot.show()
    qapp.processEvents()

    fft_plot.plot(1000, 1100, "A1")
    qapp.processEvents()

    label = fft_plot.getPlotItem().getAxis("bottom").label
    scene_rect = label.mapRectToScene(label.boundingRect())
    top_left = fft_plot.mapFromScene(scene_rect.topLeft())
    bottom_right = fft_plot.mapFromScene(scene_rect.bottomRight())

    assert label.isVisible() is True
    assert bottom_right.y() <= fft_plot.height()


def test_annotation_window_shows_overlap_tag_scope_without_classifier(qapp):
    feature = HFO_Feature(
        np.array(["A1", "A2", "B1"]),
        np.array([[1000, 1120], [1050, 1150], [5000, 5160]]),
        sample_freq=2000,
    )
    feature.apply_cross_channel_overlap_settings(
        {
            "action": HFO_Feature.OVERLAP_ACTION_TAG,
            "min_overlap_ms": 0.0,
            "min_channels": 2,
            "tag_name": "Cross-channel overlap",
        }
    )
    backend = DummyReviewBackend("HFO", feature)
    carrier = QtCore.QObject()
    window = Annotation(backend=backend, close_signal=carrier.destroyed)

    options = [window.PredictionScopeBox.itemText(i) for i in range(window.PredictionScopeBox.count())]
    assert "Overlap tagged" in options
    assert "Cross-channel overlap" not in window.AnotationDropdownBox.itemText(0)
    assert "Cross-channel overlap" in window.AnotationDropdownBox.itemText(1)
    assert "Cross-channel overlap" not in window.model_textbox.text()
    window.PredictionScopeBox.setCurrentText("Overlap tagged")
    assert window.match_summary_label.text() == "Scope: Overlap tagged | 1 matches | current event outside scope"

    window.plot_next_match()
    assert backend.event_features.get_current_info()["channel_name"] == "A2"
    assert "Cross-channel overlap" in window.model_textbox.text()
    assert window.match_summary_label.text() == "Scope: Overlap tagged | 1/1 matches"

    window.close()


def test_annotation_window_hides_overlap_filtered_events_from_dropdown(qapp):
    feature = HFO_Feature(
        np.array(["A1", "A2", "B1"]),
        np.array([[1000, 1120], [1050, 1150], [5000, 5160]]),
        sample_freq=2000,
    )
    feature.apply_cross_channel_overlap_settings(
        {
            "action": HFO_Feature.OVERLAP_ACTION_HIDE,
            "min_overlap_ms": 0.0,
            "min_channels": 2,
            "tag_name": "Cross-channel overlap",
        }
    )
    backend = DummyReviewBackend("HFO", feature)
    carrier = QtCore.QObject()
    window = Annotation(backend=backend, close_signal=carrier.destroyed)

    assert window.AnotationDropdownBox.count() == 2
    dropdown_items = [window.AnotationDropdownBox.itemText(i) for i in range(window.AnotationDropdownBox.count())]
    assert any("A1" in item for item in dropdown_items)
    assert any("B1" in item for item in dropdown_items)
    assert window.progress_textbox.text() == "0 reviewed / 2 total | 2 remaining"
    assert backend.event_features.get_current_info()["channel_name"] == "A1"

    window.close()
