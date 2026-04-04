from pathlib import Path

import numpy as np
import pytest
import src.ui.quick_detection as quick_detection_module
from PyQt5 import QtCore, QtTest, QtWidgets

from src.hfo_feature import HFO_Feature
from src.param.param_classifier import ParamClassifier
from src.ui.annotation import Annotation
from src.ui.bipolar_channel_selection import BipolarChannelSelectionWindow
from src.ui.channels_selection import ChannelSelectionWindow
from src.ui.quick_detection import HFOQuickDetector
from src.ui.waveform_shortcut_settings import WaveformShortcutSettingsDialog


def _process_events(qapp, cycles=6):
    for _ in range(cycles):
        qapp.processEvents()


def _prepare_quick_detection_dialog(dialog, tmp_path, qapp, filename="sample_raw.fif"):
    recording_path = tmp_path / filename
    recording_path.write_text("placeholder")
    dialog.fname = str(recording_path)
    dialog.filename = recording_path.name
    dialog.update_edf_info([recording_path.name, "2000", "4", "12000"])
    _process_events(qapp)
    return recording_path


class _DummyQuickDetectionBackend:
    def __init__(self):
        self.n_jobs = 1
        self._classifier_param = ParamClassifier(
            artifact_path="",
            spike_path="",
            ehfo_path="",
            use_spike=True,
            use_ehfo=True,
            device="cpu",
            batch_size=32,
            model_type="default_cpu",
            source_preference="auto",
        )

    def get_classifier_param(self):
        return self._classifier_param

    def set_default_cpu_classifier(self):
        self._classifier_param = ParamClassifier(
            artifact_path="",
            spike_path="",
            ehfo_path="",
            use_spike=True,
            use_ehfo=True,
            device="cpu",
            batch_size=32,
            model_type="default_cpu",
            source_preference="auto",
        )

    def set_default_gpu_classifier(self):
        self._classifier_param = ParamClassifier(
            artifact_path="",
            spike_path="",
            ehfo_path="",
            use_spike=True,
            use_ehfo=True,
            device="cuda",
            batch_size=32,
            model_type="default_gpu",
            source_preference="auto",
        )


class _QuickDetectionEventFeatures:
    def __init__(self, count):
        self._count = count

    def get_num_biomarker(self):
        return self._count


class _DummyReviewBackend:
    def __init__(self):
        self.biomarker_type = "HFO"
        self.sample_freq = 2000
        self.param_filter = None
        self.filter_data = None
        self.event_features = HFO_Feature(
            np.array(["A1", "A2"]),
            np.array([[1000, 1100], [3000, 3120]]),
            sample_freq=2000,
        )
        self.channel_names = np.array(["A1", "A2"])
        timeline = np.linspace(0, 20, 8000)
        self.eeg_data = np.vstack(
            [
                np.sin(timeline) * 100,
                np.sin(timeline + (np.pi / 4)) * 100,
            ]
        )

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


class _DummyChannelManager:
    def __init__(self):
        self.channel_updates = []
        self.update_count = 0

    def get_channel_indices_to_plot(self):
        return [0, 1, 2]

    def set_channels_to_plot(self, channels, display_all=True):
        self.channel_updates.append((list(channels), display_all))

    def channel_selection_update(self):
        self.update_count += 1


class _DummyChannelBackend:
    def __init__(self):
        self.channel_names = np.array(["B10Ref", "A2Ref", "A1Ref"])
        self.added_pairs = []

    def get_eeg_data(self):
        return np.zeros((3, 20)), self.channel_names

    def add_bipolar_channel(self, channel_1, channel_2):
        self.added_pairs.append((channel_1, channel_2))
        self.channel_names = np.append(self.channel_names, f"{channel_1}#-#{channel_2}")


class _DummyWaveformPlot:
    def __init__(self):
        self.last_channel_names = None

    def update_channel_names(self, channel_names):
        self.last_channel_names = list(channel_names)


def test_quick_detection_classifier_controls_follow_master_toggle(qapp):
    dialog = HFOQuickDetector(backend=_DummyQuickDetectionBackend())
    try:
        dialog.show()
        _process_events(qapp)

        dependent_controls = (
            dialog.default_cpu_button,
            dialog.qd_use_spikes_checkbox,
            dialog.qd_use_ehfo_checkbox,
            dialog.qd_choose_artifact_model_button,
            dialog.qd_classifier_artifact_filename_display,
            dialog.qd_ignore_sec_before_input,
            dialog.qd_classifier_device_input,
        )

        assert dialog.run_button.isEnabled() is False
        assert dialog.qd_use_classifier_checkbox.isChecked() is False
        for widget in dependent_controls:
            assert widget.isEnabled() is False

        dialog.qd_use_classifier_checkbox.setChecked(True)
        _process_events(qapp)
        gpu_enabled_with_classifier = dialog.default_gpu_button.isEnabled()
        for widget in dependent_controls:
            assert widget.isEnabled() is True

        dialog.qd_use_classifier_checkbox.setChecked(False)
        _process_events(qapp)
        for widget in dependent_controls:
            assert widget.isEnabled() is False
        assert dialog.default_gpu_button.isEnabled() is False

        dialog.qd_use_classifier_checkbox.setChecked(True)
        _process_events(qapp)
        assert dialog.default_gpu_button.isEnabled() is gpu_enabled_with_classifier

        dialog.detectionTypeComboBox.setCurrentText("MNI")
        _process_events(qapp)
        assert dialog.stackedWidget.currentIndex() == 0
        dialog.detectionTypeComboBox.setCurrentText("STE")
        _process_events(qapp)
        assert dialog.stackedWidget.currentIndex() == 1
        dialog.detectionTypeComboBox.setCurrentText("HIL")
        _process_events(qapp)
        assert dialog.stackedWidget.currentIndex() == 2

        dialog.update_edf_info(["tiny_raw.fif", "200", "3", "400"])
        _process_events(qapp)
        assert dialog.run_button.isEnabled() is True
    finally:
        dialog.close()
        _process_events(qapp)


def test_quick_detection_clamps_filter_defaults_below_nyquist(qapp):
    dialog = HFOQuickDetector(backend=_DummyQuickDetectionBackend())
    try:
        dialog.show()
        _process_events(qapp)

        dialog.update_edf_info(["tiny_raw.fif", "200", "3", "400"])
        _process_events(qapp)

        nyquist = 100.0
        assert float(dialog.qd_fs_input.text()) < nyquist
        assert float(dialog.qd_fp_input.text()) < float(dialog.qd_fs_input.text())
        validator = dialog.qd_fs_input.validator()
        assert validator is not None
        assert validator.top() < nyquist
    finally:
        dialog.close()
        _process_events(qapp)


def test_dialog_primary_buttons_follow_default_action_convention(qapp):
    quick_dialog = HFOQuickDetector(backend=_DummyQuickDetectionBackend())
    channel_dialog = ChannelSelectionWindow(backend=_DummyChannelBackend(), main_window_model=_DummyChannelManager())
    bipolar_dialog = BipolarChannelSelectionWindow(
        main_window_model=_DummyChannelManager(),
        backend=_DummyChannelBackend(),
        waveform_plot=_DummyWaveformPlot(),
    )
    shortcut_dialog = WaveformShortcutSettingsDialog(
        [{"id": "window_2s", "label": "Set window to 2 s", "default": "1"}],
    )
    try:
        quick_dialog.show()
        channel_dialog.show()
        bipolar_dialog.show()
        shortcut_dialog.show()
        _process_events(qapp)

        quick_cancel = quick_dialog.cancel_button.button(QtWidgets.QDialogButtonBox.Cancel)
        assert quick_dialog.run_button.isDefault() is True
        assert quick_cancel is not None and quick_cancel.autoDefault() is False

        assert channel_dialog.ok_button.isDefault() is True
        assert channel_dialog.cancel_button.autoDefault() is False

        assert bipolar_dialog.ok_button.isDefault() is True
        assert bipolar_dialog.cancel_button.autoDefault() is False

        save_button = next(button for button in shortcut_dialog.findChildren(QtWidgets.QPushButton) if button.text() == "Save")
        cancel_button = next(button for button in shortcut_dialog.findChildren(QtWidgets.QPushButton) if button.text() == "Cancel")
        assert save_button.isDefault() is True
        assert cancel_button.autoDefault() is False
    finally:
        quick_dialog.close()
        channel_dialog.close()
        bipolar_dialog.close()
        shortcut_dialog.close()
        _process_events(qapp)


def test_waveform_shortcut_dialog_resets_trackpad_sensitivity_to_default(qapp):
    dialog = WaveformShortcutSettingsDialog(
        [{"id": "window_2s", "label": "Set window to 2 s", "default": "1"}],
        trackpad_sensitivity="gentle",
    )
    try:
        dialog.show()
        _process_events(qapp)

        assert dialog.current_trackpad_sensitivity() == "gentle"
        dialog.reset_to_defaults()
        _process_events(qapp)
        assert dialog.current_trackpad_sensitivity() == "default"
    finally:
        dialog.close()
        _process_events(qapp)


def test_quick_detection_fields_submit_on_return(monkeypatch, tmp_path, qapp):
    dialog = HFOQuickDetector(backend=_DummyQuickDetectionBackend())
    workers = []
    try:
        dialog.show()
        _process_events(qapp)
        _prepare_quick_detection_dialog(dialog, tmp_path, qapp)
        monkeypatch.setattr(dialog.threadpool, "start", lambda worker: workers.append(worker))

        dialog.qd_hil_min_window_input.setFocus(QtCore.Qt.OtherFocusReason)
        QtTest.QTest.keyClick(dialog.qd_hil_min_window_input, QtCore.Qt.Key_Return)
        _process_events(qapp)

        assert len(workers) == 1
        assert dialog.running is True
        dialog._run_cleanup()
        _process_events(qapp)
    finally:
        dialog.close()
        _process_events(qapp)


def test_annotation_review_actions_only_enable_when_selection_is_valid(qapp):
    window = Annotation(backend=_DummyReviewBackend())
    try:
        window.show()
        _process_events(qapp)

        assert window.Accept.isEnabled() is False
        assert window.PredictionScopeBox.isEnabled() is False
        assert window.UnannotatedOnlyCheckBox.isEnabled() is False
        assert window.PrevMatchButton.isEnabled() is False
        assert window.NextMatchButton.isEnabled() is False
        assert window.match_summary_label.text() == "Run classification to jump by prediction bucket."

        window.select_annotation_option("Artifact")
        _process_events(qapp)
        assert window.EventDropdown_Box.currentText() == "Artifact"
        assert window.Accept.isEnabled() is True

        window.update_button_clicked()
        _process_events(qapp)
        assert window.backend.event_features.index == 1
        assert window.EventDropdown_Box.currentText() == window.dropdown_placeholder
        assert window.Accept.isEnabled() is False
    finally:
        window.close()
        _process_events(qapp)


def test_channel_selection_bulk_toggles_track_current_selection(qapp):
    manager = _DummyChannelManager()
    dialog = ChannelSelectionWindow(backend=_DummyChannelBackend(), main_window_model=manager)
    try:
        dialog.show()
        _process_events(qapp)

        assert dialog.check_box_all.checkState() == 2
        assert dialog.check_box_none.checkState() == 0

        dialog.select_channels(False)
        _process_events(qapp)
        assert dialog.check_box_all.checkState() == 0
        assert dialog.check_box_none.checkState() == 2

        dialog.channel_checkboxes["A2Ref"].setChecked(True)
        _process_events(qapp)
        assert dialog.check_box_all.checkState() == 1
        assert dialog.check_box_none.checkState() == 1

        dialog.get_channels_to_show()
        assert manager.channel_updates == [(["A2Ref"], True)]
        assert manager.update_count == 1
    finally:
        dialog.close()
        _process_events(qapp)


def test_quick_detection_uses_detector_specific_unique_export_names_and_visible_status(qapp, tmp_path):
    dialog = HFOQuickDetector(backend=_DummyQuickDetectionBackend())
    try:
        dialog.show()
        _process_events(qapp)

        recording_path = tmp_path / "sample_raw.fif"
        recording_path.write_text("placeholder")
        dialog.fname = str(recording_path)
        dialog.filename = recording_path.name
        dialog.update_edf_info([recording_path.name, "2000", "4", "12000"])
        dialog.detectionTypeComboBox.setCurrentText("MNI")
        _process_events(qapp)

        first_excel = Path(dialog.build_output_path(".xlsx"))
        first_npz = Path(dialog.build_output_path(".npz"))
        assert first_excel.name == "sample_raw_mni.xlsx"
        assert first_npz.name == "sample_raw_mni.npz"

        first_excel.write_text("taken")
        assert Path(dialog.build_output_path(".xlsx")).name == "sample_raw_mni_2.xlsx"

        dialog.backend.event_features = _QuickDetectionEventFeatures(12)
        dialog._run_finished(
            {
                "excel_output": str(first_excel),
                "npz_output": str(first_npz),
            }
        )
        dialog._run_cleanup()
        _process_events(qapp)

        assert dialog.run_status_value_label.text() == "MNI complete"
        assert "12 events ready" in dialog.run_status_detail_label.text()
        assert "sample_raw_mni.xlsx" in dialog.run_status_detail_label.text()
        assert "sample_raw_mni.npz" in dialog.run_status_detail_label.text()
    finally:
        dialog.close()
        _process_events(qapp)


def test_quick_detection_rejects_noisy_numeric_and_device_inputs(qapp, tmp_path):
    dialog = HFOQuickDetector(backend=_DummyQuickDetectionBackend())
    try:
        dialog.show()
        _process_events(qapp)

        recording_path = tmp_path / "sample_raw.fif"
        recording_path.write_text("placeholder")
        dialog.fname = str(recording_path)
        dialog.filename = recording_path.name
        dialog.update_edf_info([recording_path.name, "2000", "4", "12000"])
        dialog.detectionTypeComboBox.setCurrentText("MNI")
        dialog.qd_mni_epoch_time_input.setText("nan")

        with pytest.raises(ValueError, match="finite number"):
            dialog.collect_run_configuration()

        dialog.init_default_mni_input_params()
        dialog.qd_use_classifier_checkbox.setChecked(True)
        dialog.qd_classifier_device_input.setText("gpu??")

        with pytest.raises(ValueError, match="Device must be either cpu or cuda:0"):
            dialog.get_classifier_param()

        dialog.qd_classifier_device_input.setText("cpu")
        dialog.qd_classifier_batch_size_input.setText("0")

        with pytest.raises(ValueError, match="Batch size must be greater than 0"):
            dialog.get_classifier_param()
    finally:
        dialog.close()
        _process_events(qapp)


def test_quick_detection_run_requires_selected_outputs_and_stays_idle(monkeypatch, qapp, tmp_path):
    dialog = HFOQuickDetector(backend=_DummyQuickDetectionBackend())
    captured = {}

    class _CapturedMessageBox:
        def exec_(self):
            captured["executed"] = True
            return 0

    def _capture_message_box(parent=None, **kwargs):
        captured.update(kwargs)
        return _CapturedMessageBox()

    monkeypatch.setattr(quick_detection_module, "build_themed_message_box", _capture_message_box)

    try:
        dialog.show()
        _prepare_quick_detection_dialog(dialog, tmp_path, qapp)
        dialog.qd_excel_checkbox.setChecked(False)
        dialog.qd_npz_checkbox.setChecked(False)
        _process_events(qapp)

        dialog.run()
        _process_events(qapp, cycles=6)

        assert captured["title"] == "Quick Detection"
        assert captured["text"] == "Quick detection could not start."
        assert "Select at least one output format." in captured["informative_text"]
        assert captured["executed"] is True
        assert dialog.running is False
        assert dialog.run_button.isEnabled() is True
        assert dialog.run_status_value_label.text().startswith("Ready for")
    finally:
        dialog.close()
        _process_events(qapp)


def test_quick_detection_run_requires_artifact_source_when_classifier_is_enabled(monkeypatch, qapp, tmp_path):
    dialog = HFOQuickDetector(backend=_DummyQuickDetectionBackend())
    captured = {}

    class _CapturedMessageBox:
        def exec_(self):
            captured["executed"] = True
            return 0

    def _capture_message_box(parent=None, **kwargs):
        captured.update(kwargs)
        return _CapturedMessageBox()

    monkeypatch.setattr(quick_detection_module, "build_themed_message_box", _capture_message_box)

    try:
        dialog.show()
        _prepare_quick_detection_dialog(dialog, tmp_path, qapp)
        dialog.qd_use_classifier_checkbox.setChecked(True)
        dialog.qd_use_spikes_checkbox.setChecked(False)
        dialog.qd_use_ehfo_checkbox.setChecked(False)
        dialog._set_model_path_display(dialog.qd_classifier_artifact_filename_display, "")
        _process_events(qapp)

        dialog.run()
        _process_events(qapp, cycles=6)

        assert captured["title"] == "Quick Detection"
        assert captured["text"] == "Quick detection could not start."
        assert "Artifact model file is required when classifier is enabled." in captured["informative_text"]
        assert captured["executed"] is True
        assert dialog.running is False
        assert dialog.run_button.isEnabled() is True
    finally:
        dialog.close()
        _process_events(qapp)


def test_quick_detection_run_sets_busy_state_before_worker_completion(monkeypatch, qapp, tmp_path):
    dialog = HFOQuickDetector(backend=_DummyQuickDetectionBackend())
    started_workers = []

    try:
        dialog.show()
        recording_path = _prepare_quick_detection_dialog(dialog, tmp_path, qapp)
        monkeypatch.setattr(dialog.threadpool, "start", lambda worker: started_workers.append(worker))

        dialog.run()
        _process_events(qapp, cycles=6)

        assert len(started_workers) == 1
        assert dialog.running is True
        assert dialog.run_button.isEnabled() is False
        assert dialog.cancel_button.isEnabled() is False
        assert dialog.scrollArea.isEnabled() is False
        assert dialog.windowTitle() == "Quick Detection - Running..."
        assert dialog.run_status_value_label.text() == f"Running {dialog._selected_detector_name()}"

        dialog.backend.event_features = _QuickDetectionEventFeatures(5)
        dialog._run_finished({"npz_output": str(recording_path.with_suffix(".npz"))})
        dialog._run_cleanup()
        _process_events(qapp, cycles=6)

        assert dialog.running is False
        assert dialog.run_button.isEnabled() is True
        assert dialog.cancel_button.isEnabled() is True
        assert dialog.scrollArea.isEnabled() is True
        assert dialog.windowTitle() == "Quick Detection - Complete"
        assert dialog.run_status_value_label.text() == f"{dialog._selected_detector_name()} complete"
        assert "5 events ready" in dialog.run_status_detail_label.text()
    finally:
        if dialog.running:
            dialog.running = False
            dialog.set_ui_enabled(True)
        dialog.close()
        _process_events(qapp)


def test_bipolar_selection_defaults_to_a_valid_pair_and_guards_primary_action(qapp):
    manager = _DummyChannelManager()
    backend = _DummyChannelBackend()
    plot = _DummyWaveformPlot()
    dialog = BipolarChannelSelectionWindow(
        main_window_model=manager,
        backend=backend,
        waveform_plot=plot,
    )
    try:
        dialog.show()
        _process_events(qapp)

        assert dialog.ch_1_dropdown.currentIndex() == 0
        assert dialog.ch_2_dropdown.currentIndex() == 1
        assert dialog.ok_button.isEnabled() is True
        assert dialog.selection_status_label.text() == "Ready to create the derived bipolar channel from the selected pair."

        dialog.ch_2_dropdown.setCurrentIndex(0)
        _process_events(qapp)
        assert dialog.ok_button.isEnabled() is False
        assert dialog.selection_status_label.text() == "Choose two different source channels to enable Create Channel."

        dialog.ch_2_dropdown.setCurrentIndex(2)
        _process_events(qapp)
        assert dialog.ok_button.isEnabled() is True
        assert dialog.selection_status_label.text() == "Ready to create the derived bipolar channel from the selected pair."

        dialog.check_channels()
        assert backend.added_pairs == [("B10Ref", "A1Ref")]
        assert plot.last_channel_names == ["B10Ref", "A2Ref", "A1Ref", "B10Ref#-#A1Ref"]
        assert manager.channel_updates[-1] == (
            ["B10Ref", "A2Ref", "A1Ref", "B10Ref#-#A1Ref"],
            False,
        )
    finally:
        dialog.close()
        _process_events(qapp)
