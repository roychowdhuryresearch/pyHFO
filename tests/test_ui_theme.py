import numpy as np
from PyQt5 import QtGui, QtWidgets

from src.hfo_feature import HFO_Feature
from src.ui.annotation import Annotation
from src.ui.bipolar_channel_selection import BipolarChannelSelectionWindow
from src.models.main_window_model import MainWindowModel
from src.ui.main_window import MainWindow
from src.ui.quick_detection import HFOQuickDetector
from src.ui.ui_tokens import resolve_ui_density
from src.param.param_classifier import ParamClassifier
from src.utils.utils_gui import apply_subwindow_theme


def _process_events(qapp, cycles=6):
    for _ in range(cycles):
        qapp.processEvents()


class _DummyAnnotationBackend:
    def __init__(self):
        self.biomarker_type = "HFO"
        self.sample_freq = 2000
        self.param_filter = None
        self.filter_data = None
        self.channel_names = np.array(["A1"])
        self.eeg_data = np.zeros((1, 8000))
        self.event_features = HFO_Feature(np.array(["A1"]), np.array([[1000, 1100]]), sample_freq=2000)

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


class _DummyBipolarBackend:
    def __init__(self):
        self.channel_names = np.array(["B10Ref", "A2Ref", "A1Ref"])

    def get_eeg_data(self):
        return np.zeros((3, 20)), self.channel_names

    def add_bipolar_channel(self, *_args, **_kwargs):
        pass


class _DummyChannelManager:
    def set_channels_to_plot(self, *_args, **_kwargs):
        pass


class _DummyWaveformPlot:
    def update_channel_names(self, *_args, **_kwargs):
        pass


def test_subwindow_theme_unifies_popup_chrome_and_input_heights(qapp):
    dialog = QtWidgets.QDialog()
    layout = QtWidgets.QVBoxLayout(dialog)

    combo = QtWidgets.QComboBox(dialog)
    combo.addItems(["HFO", "Spindle"])
    combo.setMinimumHeight(36)
    line_edit = QtWidgets.QLineEdit(dialog)
    line_edit.setMinimumHeight(34)
    spin_box = QtWidgets.QSpinBox(dialog)
    spin_box.setMinimumHeight(32)

    layout.addWidget(combo)
    layout.addWidget(line_edit)
    layout.addWidget(spin_box)

    apply_subwindow_theme(dialog)

    expected_height = resolve_ui_density(dialog.screen()).compact_input_height
    stylesheet = dialog.styleSheet()

    assert "QComboBox QAbstractItemView" in stylesheet
    assert "QComboBox::down-arrow" in stylesheet
    assert "QMenu::item" in stylesheet
    assert "QToolButton::menu-indicator" in stylesheet
    assert "QCheckBox::indicator:checked" in stylesheet
    assert "QCheckBox::indicator:indeterminate" in stylesheet
    assert "checkbox_check.svg" in stylesheet
    assert combo.minimumHeight() == expected_height
    assert combo.maximumHeight() == expected_height
    assert line_edit.minimumHeight() == expected_height
    assert line_edit.maximumHeight() == expected_height
    assert spin_box.minimumHeight() == expected_height
    assert spin_box.maximumHeight() == expected_height


def test_main_window_uses_shared_popup_theme_without_local_combo_override(monkeypatch, qapp):
    monkeypatch.setattr(MainWindowModel, "init_error_terminal_display", lambda self: None)

    window = MainWindow()
    try:
        window.show()
        _process_events(qapp)

        stylesheet = window.styleSheet()
        expected_height = window.view.ui_density.compact_input_height

        assert "QComboBox QAbstractItemView" in stylesheet
        assert "QComboBox::down-arrow" in stylesheet
        assert "QMenu::item" in stylesheet
        assert "QToolButton::menu-indicator" in stylesheet
        assert "QCheckBox::indicator:checked" in stylesheet
        assert "QCheckBox::indicator:indeterminate" in stylesheet
        assert "checkbox_check.svg" in stylesheet
        assert window.combo_box_biomarker.styleSheet() == ""

        for widget in (
            window.combo_box_biomarker,
            window.active_run_selector,
            window.detector_mode_combo,
            window.classifier_mode_combo,
        ):
            assert widget.minimumHeight() == expected_height
            assert widget.maximumHeight() == expected_height
    finally:
        window.close()
        _process_events(qapp)


def test_main_window_uses_shared_semantic_roles_for_cards_and_tables(monkeypatch, qapp):
    monkeypatch.setattr(MainWindowModel, "init_error_terminal_display", lambda self: None)

    window = MainWindow()
    try:
        window.show()
        _process_events(qapp)

        stylesheet = window.styleSheet()

        assert 'QLabel[pageTitle="true"]' in stylesheet
        assert 'QLabel[heroTitle="true"]' in stylesheet
        assert 'QFrame[softCard="true"]' in stylesheet
        assert 'QTableWidget[summaryTable="true"]' in stylesheet
        assert 'QLabel[chip="true"]' in stylesheet
        assert 'QTextEdit[consolePanel="true"]' in stylesheet
        assert "#InspectorSinglePage" in stylesheet
        assert 'QTabWidget[inspectorTabs="true"]' in stylesheet
        assert 'QPushButton:pressed' in stylesheet
        assert 'QPushButton[clickFeedback="true"]' in stylesheet
        assert 'QPushButton[inspectorPrimary="true"][clickFeedback="true"]' in stylesheet
        assert 'QToolButton[runLauncher="true"]' in stylesheet

        assert window.STDTextEdit.styleSheet() == ""

        assert window.waveform_header_title.property("waveformHeaderTitle") is True
        assert not hasattr(window, "signal_workspace_header")
        assert window.event_position_label.property("waveformBadge") is True
        assert window.event_channels_button.property("waveformTool") is True
        assert window.waveform_toolbar_frame.property("waveformToolbarShell") is True
        assert window.decision_runs_value.property("metricValue") is True
        assert window.activity_level_dot.property("statusDot") is True
        assert window.activity_level_dot.property("activityLevel") == "info"
        assert window.tabWidget.property("inspectorTabs") is True
        assert window.new_run_button.property("runLauncher") is True
        assert window.new_run_button.property("inspectorSecondary") is None
        assert window.new_run_button.height() <= window.view.ui_density.compact_input_height + 2
        assert window.active_run_param_table.property("summaryTable") is True
        assert window.active_classifier_table.property("summaryTable") is True
    finally:
        window.close()
        _process_events(qapp)


def test_annotation_window_uses_shared_dropdown_theme_and_compact_combo_heights(qapp):
    window = Annotation(backend=_DummyAnnotationBackend())
    try:
        window.show()
        _process_events(qapp)

        stylesheet = window.styleSheet()
        expected_height = window.ui_density.compact_input_height

        assert "QComboBox QAbstractItemView" in stylesheet
        assert "QComboBox::down-arrow" in stylesheet
        assert "QMenu::item" in stylesheet
        assert 'QFrame[surfaceCard="true"]' in stylesheet
        assert 'QLabel[sectionTitle="true"]' in stylesheet
        assert 'QLabel[helperText="true"]' in stylesheet

        assert window.progress_card.styleSheet() == ""
        assert window.prediction_scope_card.styleSheet() == ""
        assert window.progress_card.property("surfaceCard") is True
        assert window.prediction_scope_card.property("surfaceCard") is True
        assert window.progress_caption_label.property("sectionTitle") is True
        assert window.match_summary_label.property("helperText") is True

        for combo in window.findChildren(QtWidgets.QComboBox):
            assert combo.height() >= expected_height
            assert combo.height() >= combo.sizeHint().height()
            assert combo.height() >= combo.minimumSizeHint().height()
    finally:
        window.close()
        _process_events(qapp)


def test_quick_detection_dialog_uses_shared_semantic_fields_and_readonly_styles(qapp):
    dialog = HFOQuickDetector(backend=_DummyQuickDetectionBackend())
    try:
        dialog.show()
        _process_events(qapp)

        stylesheet = dialog.styleSheet()
        expected_height = dialog.ui_density.compact_input_height

        assert 'QLabel[fieldLabel="true"]' in stylesheet
        assert 'QLabel[fieldUnit="true"]' in stylesheet
        assert 'QLineEdit[readOnlyField="true"]' in stylesheet
        assert dialog.run_setup_card.property("surfaceCard") is True

        assert dialog.label_8.property("fieldLabel") is True
        assert dialog.label_105.property("fieldLabel") is True
        assert dialog.label_40.property("fieldUnit") is True
        assert dialog.run_button.parent() is dialog.run_setup_card
        assert dialog.run_status_value_label.text() == "Waiting for recording"
        combo_palette = dialog.detectionTypeComboBox.palette()
        assert combo_palette.color(QtGui.QPalette.Text) != combo_palette.color(QtGui.QPalette.Base)

        for display in (
            dialog.qd_classifier_artifact_filename_display,
            dialog.qd_classifier_spike_filename_display,
            dialog.qd_classifier_ehfo_filename_display,
        ):
            assert display.property("readOnlyField") is True
            assert display.minimumHeight() == expected_height
    finally:
        dialog.close()
        _process_events(qapp)


def test_bipolar_dialog_uses_rebuilt_compact_shell(qapp):
    dialog = BipolarChannelSelectionWindow(
        main_window_model=_DummyChannelManager(),
        backend=_DummyBipolarBackend(),
        waveform_plot=_DummyWaveformPlot(),
    )
    try:
        dialog.show()
        _process_events(qapp)

        stylesheet = dialog.styleSheet()
        assert 'QFrame[surfaceCard="true"]' in stylesheet
        assert 'QPushButton[accentButton="true"]:disabled' in stylesheet
        assert dialog.label_3.property("helperText") is True
        assert dialog.label.property("fieldLabel") is True
        assert dialog.label_2.property("fieldLabel") is True
        assert dialog.ok_button.text() == "Create Channel"
    finally:
        dialog.close()
        _process_events(qapp)
