import numpy as np
from PyQt5 import QtCore, QtWidgets

from src.hfo_feature import HFO_Feature
from src.models.main_window_model import MainWindowModel
from src.param.param_classifier import ParamClassifier
from src.spindle_feature import SpindleFeature
from src.ui.annotation import Annotation
from src.ui.main_window import MainWindow
from src.ui.quick_detection import HFOQuickDetector
from src.utils.analysis_session import DetectionRun


_LAYOUT_WIDGET_TYPES = (
    QtWidgets.QAbstractButton,
    QtWidgets.QComboBox,
    QtWidgets.QLineEdit,
    QtWidgets.QAbstractSpinBox,
    QtWidgets.QLabel,
)


def _process_events(qapp, cycles=6):
    for _ in range(cycles):
        qapp.processEvents()


def _widget_rect_in_root(root, widget):
    top_left = widget.mapTo(root, QtCore.QPoint(0, 0))
    rect = QtCore.QRect(top_left, widget.size())
    if rect.width() > 2 and rect.height() > 2:
        rect.adjust(1, 1, -1, -1)
    return rect


def _has_layout_widget_ancestor(widget, container):
    parent = widget.parentWidget()
    while parent is not None and parent is not container:
        if isinstance(parent, _LAYOUT_WIDGET_TYPES):
            return True
        parent = parent.parentWidget()
    return False


def _collect_visible_layout_widgets(container):
    widgets = []
    for widget in container.findChildren(QtWidgets.QWidget):
        if not isinstance(widget, _LAYOUT_WIDGET_TYPES):
            continue
        if _has_layout_widget_ancestor(widget, container):
            continue
        if not widget.isVisibleTo(container):
            continue
        if widget.width() <= 0 or widget.height() <= 0:
            continue
        widgets.append(widget)
    return widgets


def _collect_visible_buttons(container):
    buttons = []
    for widget in _collect_visible_layout_widgets(container):
        if isinstance(widget, QtWidgets.QAbstractButton) and widget.text():
            buttons.append(widget)
    return buttons


def _assert_widgets_do_not_overlap(root, widgets):
    visible_widgets = [widget for widget in widgets if widget is not None and widget.isVisibleTo(root)]
    for index, widget in enumerate(visible_widgets):
        rect = _widget_rect_in_root(root, widget)
        for other in visible_widgets[index + 1 :]:
            other_rect = _widget_rect_in_root(root, other)
            assert not rect.intersects(other_rect), (
                f"{widget.objectName() or widget.text()} overlaps "
                f"{other.objectName() or other.text()}"
            )


def _assert_buttons_fit_text(buttons):
    for button in buttons:
        assert button.width() >= button.sizeHint().width(), (
            f"{button.objectName() or button.text()} width {button.width()} < {button.sizeHint().width()}"
        )


def _assert_widgets_fit_height_hint(widgets):
    for widget in widgets:
        assert widget.height() >= widget.sizeHint().height(), (
            f"{widget.objectName() or widget.text()} height {widget.height()} < {widget.sizeHint().height()}"
        )


def _assert_plot_artists_fit_canvas(canvas):
    if hasattr(canvas, "get_label_bounding_rects"):
        canvas_rect = QtCore.QRectF(0.0, 0.0, float(canvas.width()), float(canvas.height()))
        tolerance = 6.0
        for index, rect in enumerate(canvas.get_label_bounding_rects()):
            assert rect.left() >= canvas_rect.left() - tolerance, f"{type(canvas).__name__}[{index}] label clipped on the left"
            assert rect.top() >= canvas_rect.top() - tolerance, f"{type(canvas).__name__}[{index}] label clipped on the top"
            assert rect.right() <= canvas_rect.right() + tolerance, f"{type(canvas).__name__}[{index}] label clipped on the right"
            assert rect.bottom() <= canvas_rect.bottom() + tolerance, f"{type(canvas).__name__}[{index}] label clipped on the bottom"
        return

    renderer = canvas.figure.canvas.get_renderer()
    figure_bbox = canvas.figure.bbox
    for ax_index, ax in enumerate(np.ravel(canvas.axs)):
        title_artist = getattr(ax, "_left_title", ax.title)
        for artist_name, artist in (
            ("title", title_artist),
            ("xlabel", ax.xaxis.label),
            ("ylabel", ax.yaxis.label),
        ):
            bbox = artist.get_window_extent(renderer=renderer)
            assert bbox.x0 >= 0, f"{type(canvas).__name__}[{ax_index}] {artist_name} clipped on the left"
            assert bbox.y0 >= 0, f"{type(canvas).__name__}[{ax_index}] {artist_name} clipped on the bottom"
            assert bbox.x1 <= figure_bbox.x1, f"{type(canvas).__name__}[{ax_index}] {artist_name} clipped on the right"
            assert bbox.y1 <= figure_bbox.y1, f"{type(canvas).__name__}[{ax_index}] {artist_name} clipped on the top"


def _assert_container_layout(root, container):
    widgets = _collect_visible_layout_widgets(container)
    assert widgets, f"No visible layout widgets found in {container.objectName() or type(container).__name__}"
    if container is getattr(root, "waveform_toolbar_frame", None):
        return
    _assert_widgets_do_not_overlap(root, widgets)
    _assert_buttons_fit_text(_collect_visible_buttons(container))


def _assert_vertical_order(root, widgets):
    visible_widgets = [widget for widget in widgets if widget is not None and widget.isVisibleTo(root)]
    previous_bottom = None
    for widget in visible_widgets:
        rect = _widget_rect_in_root(root, widget)
        if previous_bottom is not None:
            assert rect.top() >= previous_bottom, (
                f"{widget.objectName() or widget.text()} moved above the previous section"
            )
        previous_bottom = rect.bottom()


def _assert_horizontal_gap(root, left_widget, right_widget, minimum_gap):
    left_rect = _widget_rect_in_root(root, left_widget)
    right_rect = _widget_rect_in_root(root, right_widget)
    gap = right_rect.left() - left_rect.right()
    assert gap >= minimum_gap, (
        f"{left_widget.objectName() or left_widget.text()} and "
        f"{right_widget.objectName() or right_widget.text()} gap {gap} < {minimum_gap}"
    )


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


class _DummyReviewBackend:
    def __init__(self, biomarker_type="HFO", event_features=None):
        self.biomarker_type = biomarker_type
        self.sample_freq = 2000
        self.param_filter = None
        self.filter_data = None
        if event_features is None:
            if biomarker_type == "Spindle":
                event_features = SpindleFeature(
                    np.array(["A1", "A2"]),
                    np.array([1000, 3000]),
                    np.array([1120, 3120]),
                    sample_freq=2000,
                )
            else:
                event_features = HFO_Feature(
                    np.array(["A1", "A2"]),
                    np.array([[1000, 1100], [3000, 3120]]),
                    sample_freq=2000,
                )
        self.event_features = event_features
        self.channel_names = np.unique(np.array(getattr(event_features, "channel_names", np.array(["A1", "A2"]))))
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


def _create_main_window(monkeypatch, qapp):
    monkeypatch.setattr(MainWindowModel, "init_error_terminal_display", lambda self: None)
    window = MainWindow()
    window.show()
    _process_events(qapp)
    return window


def _load_recording(window, tiny_fif_path, qapp):
    results = window.model.read_edf(str(tiny_fif_path), None)
    window.model.update_edf_info(results)
    _process_events(qapp, cycles=20)


def _seed_hfo_runs(window):
    first_feature = HFO_Feature(
        np.array([str(window.model.backend.channel_names[0]), str(window.model.backend.channel_names[1])]),
        np.array([[10, 20], [40, 56]]),
        sample_freq=int(window.model.backend.sample_freq),
    )
    second_feature = HFO_Feature(
        np.array([str(window.model.backend.channel_names[0]), str(window.model.backend.channel_names[0])]),
        np.array([[12, 22], [60, 74]]),
        sample_freq=int(window.model.backend.sample_freq),
    )
    for detector_name, feature in (("STE", first_feature), ("MNI", second_feature)):
        run = DetectionRun.create(
            biomarker_type="HFO",
            detector_name=detector_name,
            selected_channels=list(window.model.backend.channel_names),
            param_filter=getattr(window.model.backend, "param_filter", None),
            param_detector=getattr(window.model.backend, "param_detector", None),
            param_classifier=getattr(window.model.backend, "param_classifier", None),
            event_features=feature,
            classified=False,
        )
        window.model.backend.analysis_session.add_run(run)
    active_run = window.model.backend.analysis_session.get_active_run()
    if active_run is not None:
        window.model.backend.analysis_session.accept_run(active_run.run_id)
        window.model.backend.event_features = active_run.event_features
        window.model.backend.detected = True
    window.model.update_run_management_panel()


def test_main_window_layout_regression_covers_dynamic_sections(monkeypatch, qapp, tiny_fif_path):
    window = _create_main_window(monkeypatch, qapp)
    try:
        _assert_buttons_fit_text(
            [
                window.empty_open_button,
                window.empty_load_session_button,
                window.empty_quick_button,
            ]
        )
        _assert_widgets_do_not_overlap(
            window,
            [
                window.empty_open_button,
                window.empty_load_session_button,
                window.empty_quick_button,
            ],
        )

        _load_recording(window, tiny_fif_path, qapp)
        for container in (
            window.run_actions_card,
            window.results_card,
            window.prepare_tab_compact_container,
            window.statistics_box,
            window.waveform_toolbar_frame,
        ):
            _assert_container_layout(window, container)

        workflow_matrix = (
            (window.new_hfo_run_action, ["STE", "MNI", "HIL"], ["Hugging Face CPU", "Hugging Face GPU", "Custom"]),
            (window.new_spindle_run_action, ["YASA"], ["Hugging Face CPU", "Hugging Face GPU", "Custom"]),
            (window.new_spike_run_action, ["Review"], ["Review only"]),
        )

        for action, detector_modes, classifier_modes in workflow_matrix:
            action.trigger()
            _process_events(qapp, cycles=30)

            if window.use_spike_checkbox.isVisibleTo(window) and window.use_ehfo_checkbox.isVisibleTo(window):
                _assert_horizontal_gap(window, window.use_spike_checkbox, window.use_ehfo_checkbox, 14)

            for detector_mode in detector_modes:
                if window.detector_mode_combo.findText(detector_mode) >= 0:
                    window.detector_mode_combo.setCurrentText(detector_mode)
                    _process_events(qapp, cycles=8)
                _assert_container_layout(window, window.detector_tab)

            for classifier_mode in classifier_modes:
                if window.classifier_mode_combo.findText(classifier_mode) >= 0:
                    window.classifier_mode_combo.setCurrentText(classifier_mode)
                    _process_events(qapp, cycles=8)
                _assert_container_layout(window, window.classifier_tab)

            _assert_container_layout(window, window.run_actions_card)
            _assert_container_layout(window, window.results_card)
            _assert_container_layout(window, window.statistics_box)
            _assert_container_layout(window, window.waveform_toolbar_frame)
    finally:
        window.close()
        _process_events(qapp)


def test_main_window_layout_stays_stable_across_supported_window_sizes(monkeypatch, qapp, tiny_fif_path):
    window = _create_main_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        for size in (
            (1024, 680),
            (1200, 820),
            (1600, 980),
        ):
            window.resize(*size)
            _process_events(qapp, cycles=20)

            for container in (
                window.run_actions_card,
                window.results_card,
                window.prepare_tab_compact_container,
                window.statistics_box,
                window.waveform_toolbar_frame,
            ):
                _assert_container_layout(window, container)
    finally:
        window.close()
        _process_events(qapp)


def test_run_statistics_dialog_layout_stays_stable_across_supported_window_sizes(monkeypatch, qapp, tiny_fif_path):
    window = _create_main_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)
        _seed_hfo_runs(window)

        window.model.show_run_comparison()
        _process_events(qapp, cycles=12)
        dialog = window.run_stats_dialog
        runs_group = window.run_table.parentWidget()

        for size in (
            (860, 680),
            (1180, 760),
            (1480, 920),
        ):
            dialog.resize(*size)
            _process_events(qapp, cycles=12)

            _assert_container_layout(dialog, runs_group)
            _assert_buttons_fit_text(
                [
                    window.run_stats_activate_button,
                    window.run_stats_accept_button,
                    window.run_stats_export_button,
                    window.run_stats_report_button,
                ]
            )
            _assert_widgets_do_not_overlap(
                dialog,
                [
                    window.run_stats_activate_button,
                    window.run_stats_accept_button,
                    window.run_stats_export_button,
                    window.run_stats_report_button,
                ],
            )
    finally:
        window.close()
        _process_events(qapp)


def test_quick_detection_layout_regression_covers_detector_and_classifier_panels(qapp):
    dialog = HFOQuickDetector(backend=_DummyQuickDetectionBackend())
    try:
        dialog.show()
        _process_events(qapp)
        dialog.qd_use_classifier_checkbox.setChecked(True)
        _process_events(qapp, cycles=8)

        for container in (
            dialog.run_setup_card,
            dialog.qd_edfInfo,
            dialog.qd_filters,
            dialog.classifier_groupbox_4,
            dialog.qd_saveAs,
        ):
            _assert_container_layout(dialog, container)

        detector_panels = (
            ("MNI", dialog.qd_MNI_detector),
            ("STE", dialog.qd_STE_detector),
            ("HIL", dialog.qd_HIL_detector),
        )
        for detector_name, panel in detector_panels:
            dialog.detectionTypeComboBox.setCurrentText(detector_name)
            _process_events(qapp, cycles=8)
            _assert_container_layout(dialog, panel)
    finally:
        dialog.close()
        _process_events(qapp)


def test_quick_detection_layout_stays_stable_across_supported_window_sizes(qapp):
    dialog = HFOQuickDetector(backend=_DummyQuickDetectionBackend())
    try:
        dialog.show()
        _process_events(qapp)
        dialog.qd_use_classifier_checkbox.setChecked(True)
        _process_events(qapp, cycles=8)

        for size in (
            (960, 700),
            (1180, 820),
            (1380, 1180),
        ):
            dialog.resize(*size)
            _process_events(qapp, cycles=12)

            for container in (
                dialog.run_setup_card,
                dialog.qd_edfInfo,
                dialog.qd_filters,
                dialog.classifier_groupbox_4,
                dialog.qd_saveAs,
            ):
                _assert_container_layout(dialog, container)

            _assert_widgets_do_not_overlap(
                dialog,
                [
                    dialog.default_cpu_button,
                    dialog.default_gpu_button,
                    dialog.qd_use_spikes_checkbox,
                    dialog.qd_use_ehfo_checkbox,
                    dialog.qd_npz_checkbox,
                    dialog.qd_excel_checkbox,
                ],
            )
            _assert_horizontal_gap(dialog, dialog.qd_use_spikes_checkbox, dialog.qd_use_ehfo_checkbox, 8)
    finally:
        dialog.close()
        _process_events(qapp)


def test_annotation_layout_regression_covers_side_panel_and_view_controls(qapp):
    window = Annotation(backend=_DummyReviewBackend())
    try:
        window.show()
        _process_events(qapp)

        _assert_buttons_fit_text(
            [
                window.PreviousButton,
                window.NextButton,
                window.PrevPendingButton,
                window.NextPendingButton,
                window.ClearAnnotationButton,
                window.Accept,
                window.ResetViewButton,
                window.ZoomInButton,
                window.ZoomOutButton,
                window.PanLeftButton,
                window.PanRightButton,
                window.PanUpButton,
                window.PanDownButton,
            ]
        )
        _assert_widgets_fit_height_hint(
            [
                window.AnotationDropdownBox,
                window.EventDropdown_Box,
                window.IntervalDropdownBox,
                window.PredictionScopeBox,
                window.spinBox_minFreq,
                window.spinBox_maxFreq,
                window.BackViewButton,
                window.ForwardViewButton,
                window.ExportSnapshotButton,
            ]
        )
        _assert_widgets_do_not_overlap(
            window,
            [
                window.PreviousButton,
                window.NextButton,
                window.PrevPendingButton,
                window.NextPendingButton,
                window.ClearAnnotationButton,
            ],
        )

        for container in (
            window.progress_card,
            window.prediction_scope_card,
            window.groupBox_2,
            window.groupBox,
            window.viewControlsGroupBox,
        ):
            _assert_container_layout(window, container)

        _assert_plot_artists_fit_canvas(window.annotation_controller.model.waveform_plot)
        _assert_plot_artists_fit_canvas(window.annotation_controller.model.fft_plot)
        assert window.annotation_controller.model.fft_plot.height() >= 156

        _assert_vertical_order(
            window,
            [
                window.progress_card,
                window.AnotationDropdownBox,
                window.prediction_scope_card,
                window.groupBox,
                window.EventDropdown_Box,
                window.IntervalDropdownBox,
            ],
        )
        _assert_widgets_do_not_overlap(
            window,
            [
                window.progress_card,
                window.AnotationDropdownBox,
                window.prediction_scope_card,
                window.groupBox,
                window.EventDropdown_Box,
                window.IntervalDropdownBox,
            ],
        )
    finally:
        window.close()
        _process_events(qapp)


def test_spindle_annotation_layout_matches_hfo_layout_rules(qapp):
    window = Annotation(backend=_DummyReviewBackend(biomarker_type="Spindle"))
    try:
        window.show()
        _process_events(qapp)

        _assert_container_layout(window, window.viewControlsGroupBox)
        _assert_container_layout(window, window.prediction_scope_card)
        _assert_container_layout(window, window.groupBox_2)
        _assert_container_layout(window, window.groupBox)
        _assert_widgets_do_not_overlap(
            window,
            [
                window.PreviousButton,
                window.NextButton,
                window.PrevPendingButton,
                window.NextPendingButton,
                window.ClearAnnotationButton,
                window.SyncViewsCheckBox,
                window.BackViewButton,
                window.ForwardViewButton,
                window.ExportSnapshotButton,
            ],
        )
    finally:
        window.close()
        _process_events(qapp)


def test_annotation_view_controls_keep_stable_layout_when_window_resizes(qapp):
    window = Annotation(backend=_DummyReviewBackend())
    try:
        window.show()
        _process_events(qapp)

        controls = [
            window.ResetViewButton,
            window.ZoomInButton,
            window.ZoomOutButton,
            window.PanLeftButton,
            window.PanRightButton,
            window.PanUpButton,
            window.PanDownButton,
            window.SyncViewsCheckBox,
            window.BackViewButton,
            window.ForwardViewButton,
            window.ExportSnapshotButton,
        ]

        for size in (
            (window.minimumWidth(), window.minimumHeight()),
            (1400, 900),
        ):
            window.resize(*size)
            _process_events(qapp, cycles=10)

            _assert_container_layout(window, window.viewControlsGroupBox)
            _assert_widgets_do_not_overlap(window, controls)
            assert window.SyncViewsCheckBox.width() <= window.SyncViewsCheckBox.sizeHint().width() + 24
    finally:
        window.close()
        _process_events(qapp)
