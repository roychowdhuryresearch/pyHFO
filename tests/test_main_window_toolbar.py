import json

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets, QtTest

from src.hfo_feature import HFO_Feature
from src.models.main_window_model import MainWindowModel
from src.ui.main_window import MainWindow
from src.utils.analysis_session import DetectionRun


def _process_events(qapp, cycles=6):
    for _ in range(cycles):
        qapp.processEvents()


def _commit_spinbox_text(spinbox, text, qapp, cycles=6):
    line_edit = spinbox.lineEdit()
    assert line_edit is not None
    line_edit.setText(str(text))
    spinbox.interpretText()
    spinbox.editingFinished.emit()
    _process_events(qapp, cycles=cycles)


def _submit_line_edit_text(widget, text, qapp, cycles=6):
    widget.setFocus(QtCore.Qt.OtherFocusReason)
    widget.clear()
    if text is not None:
        widget.setText(str(text))
    if hasattr(widget, "returnPressed"):
        widget.returnPressed.emit()
    else:
        QtTest.QTest.keyClick(widget, QtCore.Qt.Key_Return)
    _process_events(qapp, cycles=cycles)


def _clear_waveform_shortcut_settings():
    settings = QtCore.QSettings("PyBrain", "PyBrain")
    settings.remove("waveform_shortcuts")
    settings.remove("waveform_interaction")
    settings.sync()


def _create_window(monkeypatch, qapp, clear_shortcut_settings=True):
    monkeypatch.setattr(MainWindowModel, "init_error_terminal_display", lambda self: None)
    if clear_shortcut_settings:
        _clear_waveform_shortcut_settings()
    window = MainWindow()
    window.show()
    _process_events(qapp)
    return window


def _load_recording(window, tiny_fif_path, qapp):
    results = window.model.read_edf(str(tiny_fif_path), None)
    window.model.update_edf_info(results)
    _process_events(qapp, cycles=20)


def _run_hfo_detector(window, detector_name, qapp):
    window.model._filter(None)
    window.model.filtering_complete()
    window.detector_mode_combo.setCurrentText(detector_name)
    _process_events(qapp, cycles=6)
    assert window.model.apply_selected_detector_parameters(show_feedback=False) is True
    window.model.backend.detect_biomarker()
    window.model._detect_finished()
    _process_events(qapp, cycles=12)


def _assert_widget_is_stably_mounted(widget):
    assert widget is not None
    assert widget.parent() is not None
    assert widget.size().width() > 0
    assert widget.size().height() > 0


def _widget_rect_in_window(window, widget):
    top_left = widget.mapTo(window, QtCore.QPoint(0, 0))
    return QtCore.QRect(top_left, widget.size())


def _widget_debug_name(widget):
    if widget is None:
        return "<none>"
    object_name = widget.objectName()
    if object_name:
        return object_name
    text_getter = getattr(widget, "text", None)
    if callable(text_getter):
        text_value = text_getter()
        if text_value:
            return text_value
    return type(widget).__name__


def _assert_widgets_do_not_overlap_in_window(window, widgets):
    visible_widgets = [widget for widget in widgets if widget is not None and widget.isVisibleTo(window)]
    for index, widget in enumerate(visible_widgets):
        rect = _widget_rect_in_window(window, widget)
        for other in visible_widgets[index + 1 :]:
            other_rect = _widget_rect_in_window(window, other)
            assert not rect.intersects(other_rect), (
                f"{_widget_debug_name(widget)} overlaps "
                f"{_widget_debug_name(other)}"
            )


def _assert_widgets_are_contained_in_window(window, container, widgets):
    container_rect = _widget_rect_in_window(window, container)
    for widget in widgets:
        if widget is None or not widget.isVisibleTo(window):
            continue
        rect = _widget_rect_in_window(window, widget)
        assert container_rect.contains(rect), (
            f"{_widget_debug_name(widget)} extends outside "
            f"{_widget_debug_name(container)}"
        )


def _assert_workflow_controls_remain_mounted(window):
    controls = (
        window.combo_box_biomarker,
        window.active_run_selector,
        window.detector_mode_combo,
        window.classifier_mode_combo,
        window.overview_filter_button,
        window.annotation_button,
        window.save_npz_button,
        window.save_csv_button,
        window.new_run_button,
        window.accept_run_button,
        window.compare_runs_button,
        window.prev_event_button,
        window.next_event_button,
        window.snapshot_button,
        window.review_channels_button,
        window.referential_tool_button,
        window.average_reference_button,
        window.auto_bipolar_button,
        window.overlap_review_button,
        window.event_channels_button,
        window.all_channels_button,
        window.neighbor_channels_button,
        window.measure_tool_button,
        window.go_to_time_button,
        window.waveform_reset_view_button,
    )
    for control in controls:
        _assert_widget_is_stably_mounted(control)


def _assert_compact_workflow_inputs(window):
    expected_height = window.view.ui_density.compact_input_height
    for widget in (
        window.combo_box_biomarker,
        window.active_run_selector,
        window.detector_mode_combo,
        window.classifier_mode_combo,
        window.ste_rms_window_input,
        window.ste_min_window_input,
        window.ste_peak_threshold_input,
    ):
        assert widget.minimumHeight() == expected_height
        assert widget.maximumHeight() == expected_height
        assert widget.height() == expected_height


def _assert_classifier_custom_sources_follow_mode(window):
    custom_mode = window.classifier_mode_combo.currentText() == "Custom"
    assert window.groupBox.isVisible() is custom_mode
    assert window.groupBox_2.isVisible() is custom_mode
    assert window.classifier_apply_button.isVisible() is custom_mode


def _disconnect_if_connected(signal, slot):
    try:
        signal.disconnect(slot)
    except TypeError:
        pass


class _FakeWheelEvent:
    def __init__(self, pos, delta_x=0, delta_y=0, modifiers=QtCore.Qt.NoModifier, use_pixel=False):
        self._pos = QtCore.QPoint(pos)
        self._angle_delta = QtCore.QPoint(0, 0) if use_pixel else QtCore.QPoint(int(delta_x), int(delta_y))
        self._pixel_delta = QtCore.QPoint(int(delta_x), int(delta_y)) if use_pixel else QtCore.QPoint(0, 0)
        self._modifiers = modifiers
        self.accepted = False

    def type(self):
        return QtCore.QEvent.Wheel

    def pos(self):
        return QtCore.QPoint(self._pos)

    def angleDelta(self):
        return QtCore.QPoint(self._angle_delta)

    def pixelDelta(self):
        return QtCore.QPoint(self._pixel_delta)

    def modifiers(self):
        return self._modifiers

    def accept(self):
        self.accepted = True


def _dispatch_waveform_wheel(window, delta_x=0, delta_y=0, modifiers=QtCore.Qt.NoModifier, use_pixel=True):
    view = window.waveform_plot.main_waveform_plot_controller.view
    graphics_view = getattr(view.plot_widget, "graphics_widget", None) or view.plot_widget
    x_range = view.plot_widget.getPlotItem().vb.viewRange()[0]
    target_x = float((x_range[0] + x_range[1]) / 2.0)
    target_y = float(view._visible_channel_locations[0]) if len(view._visible_channel_locations) else 0.0
    scene_pos = view.plot_widget.getPlotItem().vb.mapViewToScene(QtCore.QPointF(target_x, target_y))
    viewport_pos = graphics_view.mapFromScene(scene_pos)
    event = _FakeWheelEvent(viewport_pos, delta_x=delta_x, delta_y=delta_y, modifiers=modifiers, use_pixel=use_pixel)
    handled = view.eventFilter(view._wheel_event_source, event)
    return handled, event


def _dispatch_waveform_native_zoom(window, value):
    view = window.waveform_plot.main_waveform_plot_controller.view
    graphics_view = getattr(view.plot_widget, "graphics_widget", None) or view.plot_widget
    x_range = view.plot_widget.getPlotItem().vb.viewRange()[0]
    target_x = float((x_range[0] + x_range[1]) / 2.0)
    target_y = float(view._visible_channel_locations[0]) if len(view._visible_channel_locations) else 0.0
    scene_pos = view.plot_widget.getPlotItem().vb.mapViewToScene(QtCore.QPointF(target_x, target_y))
    viewport_pos = graphics_view.mapFromScene(scene_pos)
    local_pos = QtCore.QPointF(viewport_pos)
    event = QtGui.QNativeGestureEvent(
        QtCore.Qt.ZoomNativeGesture,
        local_pos,
        local_pos,
        local_pos,
        float(value),
        1,
        0,
    )
    handled = view.eventFilter(view._wheel_event_source, event)
    return handled, event


def _click_spinbox_stepper(widget, subcontrol):
    option = QtWidgets.QStyleOptionSpinBox()
    widget.initStyleOption(option)
    rect = widget.style().subControlRect(QtWidgets.QStyle.CC_SpinBox, option, subcontrol, widget)
    assert rect.isValid()
    QtTest.QTest.mouseClick(widget, QtCore.Qt.LeftButton, pos=rect.center())


def _assert_main_workspace_busy(window, busy):
    expected_enabled = not busy
    for widget in (
        getattr(window, "widget", None),
        getattr(window, "widget_2", None),
        getattr(window, "case_console_panel", None),
        getattr(window, "decision_desk_panel", None),
        getattr(window, "inspector_dock", None),
    ):
        if widget is not None:
            assert widget.isEnabled() is expected_enabled
    menu_bar = window.menuBar()
    if menu_bar is not None:
        assert menu_bar.isEnabled() is expected_enabled


def test_waveform_toolbar_uses_direct_tool_rows_and_icons(monkeypatch, qapp):
    window = _create_window(monkeypatch, qapp)
    try:
        assert window.event_position_label.text() == "No events"
        assert window.waveform_toolbar_frame.property("metricCard") is True
        assert window.waveform_toolbar_frame.property("waveformToolbarShell") is True
        assert window.waveform_mode_frame.isVisible() is False
        assert window.prev_event_button.parent() is not None
        assert window.open_review_button.parent() is not None
        assert window.overlap_review_button.parent() is not None
        assert window.review_channels_button.parent() is not None
        assert window.go_to_time_button.parent() is not None
        assert window.pending_event_button.text() == "Pending"
        assert window.referential_tool_button.text() == "Ref"
        assert window.average_reference_button.text() == "Avg Ref"
        assert window.auto_bipolar_button.text() == "Auto Bp"
        assert window.neighbor_channels_button.text() == "Neighbors"
        assert window.clean_view_button.text() == "Clean"
        assert window.measure_tool_button.text() == "Measure"
        assert window.montage_status_badge.isVisible() is False
        assert window.montage_break_badge.isVisible() is False
        assert window.measurement_status_badge.isVisible() is False
        assert window.highlight_channel_button.text() == "Highlight"
        assert window.highlight_channel_button.isCheckable() is True
        assert window.neighbor_channels_button.isCheckable() is True
        assert window.hotspot_tool_button.isCheckable() is True
        assert window.prev_event_button.text() == "Prev"
        assert window.next_event_button.text() == "Next"
        assert window.snapshot_button.text() == "Snap"
        assert window.go_to_time_button.text() == "Go"
        assert window.display_time_window_input.prefix() == "Win "
        assert window.display_time_window_input.suffix() == " s"
        assert window.display_time_window_input.buttonSymbols() == QtWidgets.QAbstractSpinBox.UpDownArrows
        assert window.display_time_window_input.property("waveformStepper") is True
        assert window.Time_Increment_Input.prefix() == "Step "
        assert window.Time_Increment_Input.suffix() == "%"
        assert window.n_channel_input.prefix() == "Vis "
        assert window.waveform_amplitude_input.prefix() == "Amp "
        assert window.waveform_amplitude_input.suffix() == "x"
        assert window.waveform_amplitude_input.buttonSymbols() == QtWidgets.QAbstractSpinBox.UpDownArrows
        assert window.waveform_amplitude_input.property("waveformStepper") is True
        assert window.go_to_time_input.prefix() == "Go "
        assert window.go_to_time_input.suffix() == " s"
        assert window.go_to_time_input.property("waveformStepper") is False
        preset_height = window.graph_window_preset_buttons[0].height()
        assert window.waveform_amplitude_input.height() == preset_height
        assert window.go_to_time_input.height() == preset_height
        assert window.event_position_label.height() == preset_height
        assert window.prev_event_button.height() == preset_height
        assert window.measurement_status_badge.height() == preset_height
        assert window.montage_status_badge.height() == preset_height
        assert window.event_position_label.parent().property("waveformSlot") is True
        assert window.montage_status_badge.parent().property("waveformSlot") is True
        assert window.measurement_status_badge.parent().property("waveformSlot") is True
        assert window.waveform_amplitude_input.parent().property("waveformSlot") is True
        assert window.go_to_time_input.parent().property("waveformSlot") is True
        assert window.montage_status_badge.parent().width() >= window.montage_status_badge.fontMetrics().horizontalAdvance("Adj. Bipolar") + 12
        assert window.montage_status_slot.isVisible() is False
        assert window.montage_break_slot.isVisible() is False
        assert window.measurement_status_slot.isVisible() is False
        assert window.label_10.isHidden() is True
        assert window.label_9.isHidden() is True
        assert window.label_14.isHidden() is True
        assert window.raw_tool_button.text() == "Raw"
        assert window.cursor_tool_button.isCheckable() is True
        assert window.hotspot_tool_button.text() == "Hotspot"
        assert [button.text() for button in window.graph_window_preset_buttons] == ["2 s", "5 s", "10 s", "20 s"]
        assert all(not button.isVisibleTo(window) for button in window.graph_window_preset_buttons)
        assert [button.text() for button in window.graph_channel_preset_buttons] == ["8 ch", "16 ch", "32 ch", "Max"]

        assert not window.prev_event_button.icon().isNull()
        assert not window.normalize_tool_button.icon().isNull()
        assert not window.filtered_tool_button.icon().isNull()
        assert not window.montage_tool_button.icon().isNull()
        assert not window.next_event_button.icon().isNull()
        assert not window.snapshot_button.icon().isNull()
        assert not window.go_to_time_button.icon().isNull()
        assert not window.review_channels_button.icon().isNull()
        assert window.prev_event_button.toolButtonStyle() == QtCore.Qt.ToolButtonIconOnly
        assert window.normalize_tool_button.toolButtonStyle() == QtCore.Qt.ToolButtonTextBesideIcon
        assert window.review_channels_button.toolButtonStyle() == QtCore.Qt.ToolButtonTextBesideIcon
        assert window.widget_2.maximumHeight() >= 120
        assert window.waveform_toolbar_frame.minimumHeight() >= 120
        assert window.waveform_toolbar_frame.minimumHeight() == window.waveform_toolbar_frame.maximumHeight()

        toolbar_rows = [
            frame
            for frame in window.waveform_toolbar_frame.findChildren(QtWidgets.QFrame)
            if frame.property("waveformRow") is True
        ]
        assert len(toolbar_rows) == 3
    finally:
        window.close()
        _process_events(qapp)


def test_waveform_toolbar_rows_do_not_overlap_when_dynamic_status_badges_are_visible(monkeypatch, qapp):
    window = _create_window(monkeypatch, qapp)
    try:
        window.montage_status_badge.setText("Adj. Bipolar")
        window.montage_status_badge.setVisible(True)
        window.montage_status_slot.setVisible(True)
        window.montage_break_badge.setText("2 gaps")
        window.montage_break_badge.setVisible(True)
        window.montage_break_slot.setVisible(True)
        window.measurement_status_badge.setText("dt 4.056 s | dA 2.9 uV")
        window.measurement_status_badge.setVisible(True)
        window.measurement_status_slot.setVisible(True)
        window.event_position_label.setText("23/120 O1-Ref#-#1")
        _process_events(qapp, cycles=20)

        toolbar_row_frames = [
            frame
            for frame in window.waveform_toolbar_frame.findChildren(QtWidgets.QFrame)
            if frame.property("waveformRow") is True
        ]
        assert len(toolbar_row_frames) == 3

        for index, frame in enumerate(toolbar_row_frames):
            rect = _widget_rect_in_window(window, frame)
            assert _widget_rect_in_window(window, window.waveform_toolbar_frame).contains(rect)
            for other in toolbar_row_frames[index + 1 :]:
                other_rect = _widget_rect_in_window(window, other)
                assert not rect.intersects(other_rect), (
                    f"{_widget_debug_name(frame)} overlaps {_widget_debug_name(other)}"
                )

        _assert_widgets_are_contained_in_window(
            window,
            window.waveform_toolbar_frame,
            (
                window.montage_status_slot,
                window.montage_break_slot,
                window.measurement_status_slot,
                window.event_position_slot,
            ),
        )
    finally:
        window.close()
        _process_events(qapp)


def test_waveform_toolbar_collapses_empty_status_slots_and_restores_them_when_needed(monkeypatch, qapp):
    window = _create_window(monkeypatch, qapp)
    try:
        collapsed_highlight_x = window.highlight_channel_button.x()
        collapsed_hotspot_x = window.hotspot_tool_button.x()
        jump_input_width = window.go_to_time_input.width()
        assert window.montage_status_slot.isVisible() is False
        assert window.measurement_status_slot.isVisible() is False

        window.montage_status_badge.setText("Adj. Bipolar")
        window.montage_status_badge.setVisible(True)
        window.montage_status_slot.setVisible(True)
        window.measurement_status_badge.setText("dt 300.0 ms | dA 124.9 uV")
        window.measurement_status_badge.setVisible(True)
        window.measurement_status_slot.setVisible(True)
        window.event_position_label.setText("999/999 SUPER-LONG-CHANNEL-NAME")
        window.go_to_time_input.setMaximum(9999.99)
        window.go_to_time_input.setValue(987.65)
        _process_events(qapp)

        assert window.montage_status_slot.isVisible() is True
        assert window.measurement_status_slot.isVisible() is True
        assert window.highlight_channel_button.x() > collapsed_highlight_x
        assert window.hotspot_tool_button.x() > collapsed_hotspot_x
        assert window.go_to_time_input.width() == jump_input_width

        window.montage_status_badge.setVisible(False)
        window.montage_status_slot.setVisible(False)
        window.measurement_status_badge.setVisible(False)
        window.measurement_status_slot.setVisible(False)
        _process_events(qapp)

        assert window.highlight_channel_button.x() == collapsed_highlight_x
        assert window.hotspot_tool_button.x() == collapsed_hotspot_x
    finally:
        window.close()
        _process_events(qapp)


def test_top_toolbar_exposes_shortcut_settings_action(monkeypatch, qapp):
    window = _create_window(monkeypatch, qapp)
    try:
        assert window.actionWaveform_Shortcuts_toolbar.text() == "Shortcuts"
        toolbar_action_texts = [action.text() for action in window.toolBar.actions() if action.text()]
        assert "Shortcuts" in toolbar_action_texts
        assert not any(action.isSeparator() for action in window.toolBar.actions())
    finally:
        window.close()
        _process_events(qapp)


def test_waveform_review_uses_the_lower_toolbar_and_removes_the_upper_panel(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        assert not hasattr(window, "signal_workspace_header")
        assert not hasattr(window, "graph_cursor_button")
        assert not hasattr(window, "graph_next_pending_button")
        assert not hasattr(window, "graph_label_buttons")
        assert not hasattr(window, "graph_run_stats_button")
        assert window.montage_tool_button.text() == "Montage"
        assert window.event_channels_button.text() == "Events"
        assert window.all_channels_button.text() == "All"
        assert window.snapshot_button.isEnabled() is False
        assert window.event_channels_button.isEnabled() is False
        assert window.pending_event_button.isEnabled() is False
        assert window.referential_tool_button.isEnabled() is False
        assert window.average_reference_button.isEnabled() is False
        assert window.auto_bipolar_button.isEnabled() is False
        assert window.montage_status_badge.isVisible() is False
        assert window.montage_break_badge.isVisible() is False
        assert window.highlight_channel_button.isEnabled() is False

        _load_recording(window, tiny_fif_path, qapp)
        backend_channels = [str(channel) for channel in window.model.backend.channel_names]

        assert window.waveform_mode_frame.isVisible() is True
        assert window.waveform_source_mode_badge.text() == "Source Ref"
        assert window.waveform_scope_mode_badge.text() == "Scope All"
        assert window.waveform_tool_mode_badge.text() == "Tool Browse"
        assert window.waveform_reset_view_button.isEnabled() is False
        assert window.snapshot_button.isEnabled() is True
        assert window.zoom_out_button.isEnabled() is True
        assert window.zoom_in_button.isEnabled() is True
        assert window.event_channels_button.isEnabled() is False
        assert window.referential_tool_button.isEnabled() is True
        assert window.referential_tool_button.isChecked() is True
        assert window.average_reference_button.isEnabled() is True
        assert window.average_reference_button.isChecked() is False
        assert window.auto_bipolar_button.isEnabled() is True
        assert window.montage_status_badge.isVisible() is True
        assert window.montage_status_badge.text() == "Adj. Bipolar"
        assert window.montage_break_badge.isVisible() is False
        assert window.highlight_channel_button.isEnabled() is False

        class _FakeReviewFeatures:
            def __init__(self):
                self.starts = np.array([20, 80, 140])
                self.ends = np.array([32, 94, 154])
                self.channel_names = backend_channels[:2] + [backend_channels[0]]
                self.annotated = np.zeros(3, dtype=int)
                self.index = 0

            def get_current(self):
                return self.channel_names[self.index], self.starts[self.index], self.ends[self.index]

            def get_current_visible_position(self):
                return self.index

            def get_review_progress(self):
                reviewed = int(np.sum(self.annotated > 0))
                total = len(self.starts)
                return {"reviewed": reviewed, "remaining": total - reviewed, "total": total}

            def get_next_unannotated(self):
                self.index = 1
                return self.channel_names[self.index], self.starts[self.index], self.ends[self.index]

        fake_features = _FakeReviewFeatures()
        monkeypatch.setattr(window.model, "_get_active_event_features", lambda: fake_features)
        monkeypatch.setattr(window.model.backend, "get_channel_ranking", lambda _run_id=None: [{"channel_name": "Fz", "total_events": 3}])
        window.model.backend.detected = True
        window.model.update_waveform_toolbar_state()

        window.graph_window_preset_buttons[1].click()
        _process_events(qapp, cycles=6)
        assert window.display_time_window_input.value() == min(window.display_time_window_input.maximum(), 5.0)

        window.graph_channel_preset_buttons[0].click()
        _process_events(qapp, cycles=6)
        assert window.n_channel_input.value() == min(window.n_channel_input.maximum(), 8)

        assert window.event_channels_button.isEnabled() is True
        assert window.pending_event_button.isEnabled() is True
        window.event_channels_button.click()
        _process_events(qapp, cycles=12)
        assert window.event_channels_button.isChecked() is True
        assert window.waveform_plot.get_channels_to_plot() == backend_channels[:2]
        assert window.waveform_scope_mode_badge.text() == "Scope Events (2)"
        assert window.waveform_reset_view_button.isEnabled() is True

        window.event_channels_button.click()
        _process_events(qapp, cycles=12)
        assert window.event_channels_button.isChecked() is False
        assert window.all_channels_button.isChecked() is True
    finally:
        window.close()
        _process_events(qapp)


def test_waveform_toolbar_can_highlight_the_selected_channel_without_collapsing_the_channel_list(
    monkeypatch, qapp, tiny_fif_path
):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)
        backend_channels = [str(channel) for channel in window.model.backend.channel_names]
        target_channel = backend_channels[-1]

        window.n_channel_input.setValue(1)
        window.model.waveform_plot_button_clicked()
        _process_events(qapp, cycles=4)

        window.waveform_plot.main_waveform_plot_controller.view.set_highlighted_channel(target_channel)
        window.model.update_waveform_toolbar_state()
        _process_events(qapp, cycles=4)

        assert window.highlight_channel_button.isEnabled() is True
        assert window.highlight_channel_button.isChecked() is False
        window.highlight_channel_button.click()
        _process_events(qapp, cycles=8)
        assert window.highlight_channel_button.isChecked() is True
        assert window.waveform_plot.get_channels_to_plot() == backend_channels
        assert window.waveform_plot.get_highlighted_channel() == target_channel
        assert window.waveform_plot.first_channel_to_plot == len(backend_channels) - 1

        window.highlight_channel_button.click()
        _process_events(qapp, cycles=8)
        assert window.highlight_channel_button.isChecked() is False
        assert window.waveform_plot.get_channels_to_plot() == backend_channels
        assert window.waveform_plot.get_highlighted_channel() is None
    finally:
        window.close()
        _process_events(qapp)


def test_channel_table_highlight_keeps_the_current_channel_list(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)
        backend_channels = [str(channel) for channel in window.model.backend.channel_names]

        window.n_channel_input.setValue(1)
        window.model.waveform_plot_button_clicked()
        _process_events(qapp, cycles=4)

        window.channel_table.setProperty("channel_names", backend_channels)
        window.model.highlight_channel_from_table(len(backend_channels) - 1, 0)
        _process_events(qapp, cycles=8)

        assert window.highlight_channel_button.isChecked() is True
        assert window.waveform_plot.get_channels_to_plot() == backend_channels
        assert window.waveform_plot.get_highlighted_channel() == backend_channels[-1]
        assert window.waveform_plot.first_channel_to_plot == len(backend_channels) - 1
    finally:
        window.close()
        _process_events(qapp)


def test_waveform_toolbar_auto_bipolar_toggles_between_source_and_derived_channels(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        assert window.auto_bipolar_button.isEnabled() is True
        assert window.waveform_plot.get_channels_to_plot() == ["A1Ref", "A2Ref", "B10Ref"]

        window.auto_bipolar_button.click()
        _process_events(qapp, cycles=10)
        assert window.auto_bipolar_button.isChecked() is True
        assert window.waveform_plot.get_channels_to_plot() == ["A1Ref#-#A2Ref"]

        window.auto_bipolar_button.click()
        _process_events(qapp, cycles=10)
        assert window.auto_bipolar_button.isChecked() is False
        assert window.waveform_plot.get_channels_to_plot() == ["A1Ref", "A2Ref", "B10Ref"]
    finally:
        window.close()
        _process_events(qapp)


def test_waveform_toolbar_average_reference_toggles_between_source_and_derived_channels(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        assert window.referential_tool_button.isChecked() is True
        assert window.average_reference_button.isEnabled() is True
        assert window.waveform_plot.get_channels_to_plot() == ["A1Ref", "A2Ref", "B10Ref"]

        window.average_reference_button.click()
        _process_events(qapp, cycles=10)
        assert window.average_reference_button.isChecked() is True
        assert window.referential_tool_button.isChecked() is False
        assert window.waveform_plot.get_channels_to_plot() == ["A1Ref#-#AVG", "A2Ref#-#AVG", "B10Ref#-#AVG"]

        label_items = window.waveform_plot.main_waveform_plot_controller.view._channel_label_items
        assert [item.toPlainText() for item in label_items] == ["A1", "A2", "B10"]
        assert [item.toolTip() for item in label_items] == [
            "A1\nSource: A1Ref - Average Ref",
            "A2\nSource: A2Ref - Average Ref",
            "B10\nSource: B10Ref - Average Ref",
        ]

        window.referential_tool_button.click()
        _process_events(qapp, cycles=10)
        assert window.referential_tool_button.isChecked() is True
        assert window.average_reference_button.isChecked() is False
        assert window.waveform_plot.get_channels_to_plot() == ["A1Ref", "A2Ref", "B10Ref"]
    finally:
        window.close()
        _process_events(qapp)


def test_waveform_auto_bipolar_surfaces_clinician_labels_and_chain_break_hints(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        monkeypatch.setattr(
            window.model.backend,
            "get_auto_bipolar_metadata",
            lambda: {
                "montage_kind": "conventional_eeg",
                "present_pairs": [
                    {
                        "derived_name": "A1Ref#-#A2Ref",
                        "display_name": "Fp1-F3",
                        "source_channel_1": "EEG A1Ref",
                        "source_channel_2": "A2Ref",
                        "chain_name": "left_parasagittal",
                    }
                ],
                "chain_breaks": [{"chain_name": "left_temporal"}],
            },
        )

        window.model.update_waveform_toolbar_state()
        _process_events(qapp, cycles=6)
        assert "Broken chains: Left Temporal" in window.auto_bipolar_button.toolTip()
        assert window.montage_status_badge.text() == "Double Banana"
        assert window.montage_status_badge.isVisible() is True
        assert window.montage_break_badge.text() == "1 Break"
        assert window.montage_break_badge.isVisible() is True

        window.auto_bipolar_button.click()
        _process_events(qapp, cycles=12)

        label_items = window.waveform_plot.main_waveform_plot_controller.view._channel_label_items
        assert [item.toPlainText() for item in label_items] != ["A1Ref#-#A2Ref"]
        assert [item.toolTip() for item in label_items] == [
            "Fp1-F3\nSource: EEG A1Ref - A2Ref\nChain: Left Parasagittal"
        ]
    finally:
        window.close()
        _process_events(qapp)


def test_montage_badges_open_a_detail_dialog_with_break_and_alias_summary(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        monkeypatch.setattr(
            window.model.backend,
            "get_auto_bipolar_metadata",
            lambda: {
                "montage_kind": "conventional_eeg",
                "present_pairs": [
                    {
                        "display_name": "Fp1-F3",
                    },
                    {
                        "display_name": "F3-C3",
                    },
                ],
                "missing_pairs": [
                    {
                        "display_name": "C3-P3",
                    },
                    {
                        "display_name": "P3-O1",
                    },
                ],
                "chain_summaries": [
                    {
                        "chain_name": "left_parasagittal",
                        "is_visible": True,
                    }
                ],
                "chain_breaks": [
                    {
                        "chain_name": "left_parasagittal",
                        "missing_pairs": ["C3-P3", "P3-O1"],
                    }
                ],
                "normalized_channels": [
                    {
                        "clean_name": "T7",
                        "canonical_name": "T3",
                        "uses_alias": True,
                    },
                    {
                        "clean_name": "P7",
                        "canonical_name": "T5",
                        "uses_alias": True,
                    },
                ],
                "warnings": [
                    {
                        "type": "unrecognized_channels",
                        "channels": ["POL X1"],
                    }
                ],
            },
        )

        captured = {}

        def _capture_exec(message_box):
            captured["title"] = message_box.windowTitle()
            captured["text"] = message_box.text()
            captured["informative_text"] = message_box.informativeText()
            captured["detailed_text"] = message_box.detailedText()
            return QtWidgets.QMessageBox.Ok

        monkeypatch.setattr(QtWidgets.QMessageBox, "exec_", _capture_exec)

        window.model.update_waveform_toolbar_state()
        _process_events(qapp, cycles=6)

        assert window.montage_status_badge.toolTip().endswith("Click for details")
        assert window.montage_break_badge.toolTip().endswith("Click for details")

        window.montage_break_badge.click()
        _process_events(qapp, cycles=3)

        assert captured["text"] == "Double Banana montage ready"
        assert "Valid pairs: 2 / 4" in captured["informative_text"]
        assert "Visible chains: Left Parasagittal" in captured["informative_text"]
        assert "Broken chains: Left Parasagittal" in captured["informative_text"]
        assert "Alias substitutions: T3 <- T7, T5 <- P7" in captured["informative_text"]
        assert "Ignored channels: POL X1" in captured["informative_text"]
        assert "Broken chains" in captured["detailed_text"]
        assert "- Left Parasagittal: C3-P3, P3-O1" in captured["detailed_text"]
        assert "Missing pairs" in captured["detailed_text"]
        assert "- C3-P3" in captured["detailed_text"]
        assert "- P3-O1" in captured["detailed_text"]
    finally:
        window.close()
        _process_events(qapp)


def test_top_toolbar_shortcuts_action_opens_shortcut_settings(monkeypatch, qapp):
    opened = {"count": 0}

    def _stub_open_shortcuts(self):
        opened["count"] += 1

    monkeypatch.setattr(MainWindowModel, "init_error_terminal_display", lambda self: None)
    monkeypatch.setattr(MainWindowModel, "open_waveform_shortcut_settings", _stub_open_shortcuts)
    _clear_waveform_shortcut_settings()
    window = MainWindow()
    window.show()
    _process_events(qapp)
    try:
        window.actionWaveform_Shortcuts_toolbar.trigger()
        _process_events(qapp, cycles=3)
        assert opened["count"] == 1
    finally:
        window.close()
        _process_events(qapp)


def test_waveform_shortcut_settings_menu_remaps_shortcuts(monkeypatch, qapp, tiny_fif_path):
    _clear_waveform_shortcut_settings()
    settings = QtCore.QSettings("PyBrain", "PyBrain")
    settings.setValue("waveform_shortcuts/enabled", True)
    settings.setValue("waveform_interaction/trackpad_sensitivity", "gentle")
    settings.setValue(
        "waveform_shortcuts/keymap",
        json.dumps(
            {
                "toggle_cursor": "Y",
                "window_5s": "K",
            }
        ),
    )
    settings.sync()

    window = _create_window(monkeypatch, qapp, clear_shortcut_settings=False)
    try:
        assert window.waveform_shortcut_settings_action.text() == "Waveform Shortcuts..."
        _load_recording(window, tiny_fif_path, qapp)
        assert window.waveform_plot.get_trackpad_sensitivity() == "gentle"

        focused = QtWidgets.QApplication.focusWidget()
        if focused is not None:
            focused.clearFocus()
        window.setFocus(QtCore.Qt.OtherFocusReason)
        _process_events(qapp, cycles=3)
        initial_window_value = window.display_time_window_input.value()

        QtTest.QTest.keyClick(window, QtCore.Qt.Key_C)
        _process_events(qapp, cycles=6)
        assert window.waveform_plot.is_cursor_enabled() is False

        QtTest.QTest.keyClick(window, QtCore.Qt.Key_Y)
        _process_events(qapp, cycles=6)
        assert window.cursor_tool_button.isChecked() is True
        assert window.waveform_plot.is_cursor_enabled() is True

        QtTest.QTest.keyClick(window, QtCore.Qt.Key_2)
        _process_events(qapp, cycles=6)
        assert window.display_time_window_input.value() == initial_window_value

        QtTest.QTest.keyClick(window, QtCore.Qt.Key_K)
        _process_events(qapp, cycles=6)
        assert window.display_time_window_input.value() == min(window.display_time_window_input.maximum(), 5.0)
    finally:
        window.close()
        _clear_waveform_shortcut_settings()
        _process_events(qapp)


def test_waveform_shortcut_settings_can_disable_shortcuts(monkeypatch, qapp, tiny_fif_path):
    _clear_waveform_shortcut_settings()
    settings = QtCore.QSettings("PyBrain", "PyBrain")
    settings.setValue("waveform_shortcuts/enabled", False)
    settings.setValue("waveform_shortcuts/keymap", json.dumps({"toggle_cursor": "Y"}))
    settings.sync()

    window = _create_window(monkeypatch, qapp, clear_shortcut_settings=False)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        focused = QtWidgets.QApplication.focusWidget()
        if focused is not None:
            focused.clearFocus()
        window.setFocus(QtCore.Qt.OtherFocusReason)
        _process_events(qapp, cycles=3)

        QtTest.QTest.keyClick(window, QtCore.Qt.Key_Y)
        _process_events(qapp, cycles=6)
        assert window.waveform_plot.is_cursor_enabled() is False
    finally:
        window.close()
        _clear_waveform_shortcut_settings()
        _process_events(qapp)


def test_waveform_shortcuts_open_montage_tool(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        focused = QtWidgets.QApplication.focusWidget()
        if focused is not None:
            focused.clearFocus()
        window.setFocus(QtCore.Qt.OtherFocusReason)
        _process_events(qapp, cycles=3)

        assert not hasattr(window, "bipolar_channel_selection_window")

        QtTest.QTest.keyClick(window, QtCore.Qt.Key_B)
        _process_events(qapp, cycles=6)

        dialog = getattr(window, "bipolar_channel_selection_window", None)
        assert dialog is not None
        assert dialog.isVisible() is True
        dialog.close()
        _process_events(qapp, cycles=3)
    finally:
        window.close()
        _process_events(qapp)


def test_waveform_shortcuts_drive_waveform_review_actions(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        backend_channels = [str(channel) for channel in window.model.backend.channel_names]

        class _FakeReviewFeatures:
            def __init__(self):
                self.starts = np.array([20, 80], dtype=int)
                self.ends = np.array([32, 94], dtype=int)
                self.channel_names = backend_channels[:2] if len(backend_channels) >= 2 else backend_channels
                self.annotated = np.zeros(len(self.starts), dtype=int)
                self.index = 0

            def get_current(self):
                return self.channel_names[self.index], self.starts[self.index], self.ends[self.index]

            def get_current_visible_position(self):
                return self.index

            def get_review_progress(self):
                reviewed = int(np.sum(self.annotated > 0))
                total = len(self.starts)
                return {"reviewed": reviewed, "remaining": total - reviewed, "total": total}

            def get_next_unannotated(self):
                self.index = 1
                return self.channel_names[self.index], self.starts[self.index], self.ends[self.index]

            def get_prev_unannotated(self):
                self.index = 0
                return self.channel_names[self.index], self.starts[self.index], self.ends[self.index]

        fake_features = _FakeReviewFeatures()
        monkeypatch.setattr(window.model, "_get_active_event_features", lambda: fake_features)
        monkeypatch.setattr(
            window.model.backend,
            "get_channel_ranking",
            lambda _run_id=None: [{"channel_name": backend_channels[0], "total_events": 3}],
        )
        window.model.backend.detected = True
        window.model.update_waveform_toolbar_state()
        _process_events(qapp, cycles=6)
        focused = QtWidgets.QApplication.focusWidget()
        if focused is not None:
            focused.clearFocus()
        window.setFocus(QtCore.Qt.OtherFocusReason)
        _process_events(qapp, cycles=3)

        QtTest.QTest.keyClick(window, QtCore.Qt.Key_2)
        _process_events(qapp, cycles=6)
        assert window.display_time_window_input.value() == min(window.display_time_window_input.maximum(), 5.0)

        QtTest.QTest.keyClick(window, QtCore.Qt.Key_1, QtCore.Qt.ShiftModifier)
        _process_events(qapp, cycles=6)
        assert window.n_channel_input.value() == min(window.n_channel_input.maximum(), 8)

        window.raw_tool_button.click()
        _process_events(qapp, cycles=6)
        assert window.raw_tool_button.isChecked() is True
        assert window.filtered_tool_button.isChecked() is False

        QtTest.QTest.keyClick(window, QtCore.Qt.Key_R)
        _process_events(qapp, cycles=6)
        assert window.measure_tool_button.isChecked() is True
        assert window.waveform_plot.is_measurement_enabled() is True

        QtTest.QTest.keyClick(window, QtCore.Qt.Key_Escape)
        _process_events(qapp, cycles=6)
        assert window.measure_tool_button.isChecked() is False
        assert window.waveform_plot.is_measurement_enabled() is False

        QtTest.QTest.keyClick(window, QtCore.Qt.Key_C)
        _process_events(qapp, cycles=6)
        assert window.cursor_tool_button.isChecked() is True
        assert window.waveform_plot.is_cursor_enabled() is True

        QtTest.QTest.keyClick(window, QtCore.Qt.Key_Escape)
        _process_events(qapp, cycles=6)
        assert window.cursor_tool_button.isChecked() is False
        assert window.waveform_plot.is_cursor_enabled() is False

        QtTest.QTest.keyClick(window, QtCore.Qt.Key_E)
        _process_events(qapp, cycles=12)
        assert window.event_channels_button.isChecked() is True
        assert window.waveform_plot.get_channels_to_plot() == fake_features.channel_names

        QtTest.QTest.keyClick(window, QtCore.Qt.Key_H)
        _process_events(qapp, cycles=12)
        assert window.waveform_plot.get_channels_to_plot() == [backend_channels[0]]
        assert window.hotspot_tool_button.isChecked() is True

        QtTest.QTest.keyClick(window, QtCore.Qt.Key_H)
        _process_events(qapp, cycles=12)
        assert window.waveform_plot.get_channels_to_plot() == backend_channels
        assert window.hotspot_tool_button.isChecked() is False

        QtTest.QTest.keyClick(window, QtCore.Qt.Key_A)
        _process_events(qapp, cycles=12)
        assert window.auto_bipolar_button.isChecked() is True
        assert window.waveform_plot.get_channels_to_plot() == ["A1Ref#-#A2Ref"]

        QtTest.QTest.keyClick(window, QtCore.Qt.Key_A)
        _process_events(qapp, cycles=12)
        assert window.auto_bipolar_button.isChecked() is False
        assert window.waveform_plot.get_channels_to_plot() == backend_channels

        QtTest.QTest.keyClick(window, QtCore.Qt.Key_M)
        _process_events(qapp, cycles=12)
        assert window.average_reference_button.isChecked() is True
        assert window.waveform_plot.get_channels_to_plot() == [
            "A1Ref#-#AVG",
            "A2Ref#-#AVG",
            "B10Ref#-#AVG",
        ]

        QtTest.QTest.keyClick(window, QtCore.Qt.Key_V)
        _process_events(qapp, cycles=12)
        assert window.referential_tool_button.isChecked() is True
        assert window.average_reference_button.isChecked() is False
        assert window.waveform_plot.get_channels_to_plot() == backend_channels

        window.n_channel_input.setValue(1)
        window.model.waveform_plot_button_clicked()
        _process_events(qapp, cycles=6)

        window.waveform_plot.main_waveform_plot_controller.view.set_highlighted_channel(backend_channels[-1])
        window.model.update_waveform_toolbar_state()
        _process_events(qapp, cycles=6)

        QtTest.QTest.keyClick(window, QtCore.Qt.Key_G)
        _process_events(qapp, cycles=8)
        assert window.highlight_channel_button.isChecked() is True
        assert window.waveform_plot.get_channels_to_plot() == backend_channels
        assert window.waveform_plot.get_highlighted_channel() == backend_channels[-1]
        assert window.waveform_plot.first_channel_to_plot == len(backend_channels) - 1

        window.model.show_all_channels()
        window.waveform_plot.main_waveform_plot_controller.view.set_highlighted_channel(backend_channels[1])
        window.model.update_waveform_toolbar_state()
        _process_events(qapp, cycles=6)

        QtTest.QTest.keyClick(window, QtCore.Qt.Key_T)
        _process_events(qapp, cycles=8)
        assert window.waveform_plot.get_channels_to_plot() == backend_channels[:2]

        window.model.backend.raw.info["bads"] = [backend_channels[-1]]
        window.model.update_waveform_toolbar_state()
        _process_events(qapp, cycles=6)

        window.model.show_all_channels()
        _process_events(qapp, cycles=6)
        QtTest.QTest.keyClick(window, QtCore.Qt.Key_D)
        _process_events(qapp, cycles=8)
        assert window.waveform_plot.get_channels_to_plot() == backend_channels[:-1]

        QtTest.QTest.keyClick(window, QtCore.Qt.Key_N)
        _process_events(qapp, cycles=8)
        assert window.event_position_label.text().startswith("2/2")
    finally:
        window.close()
        _process_events(qapp)


def test_waveform_measure_tool_reports_interval_and_amplitude(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        channel_name = str(window.model.backend.channel_names[0])
        sfreq = float(window.model.backend.get_edf_info()["sfreq"])
        second_time = 2.0 / sfreq

        raw_window, _ = window.model.backend.get_eeg_data(0, 3, False)
        channel_index = list(window.model.backend.channel_names).index(channel_name)
        expected_delta = abs(float(raw_window[channel_index, 2] - raw_window[channel_index, 0]))

        window.measure_tool_button.click()
        _process_events(qapp, cycles=6)
        assert window.waveform_plot.is_measurement_enabled() is True
        assert window.measurement_status_badge.text() == "Click A"

        window.model._handle_waveform_measurement_point(channel_name, 0.0)
        _process_events(qapp, cycles=6)
        assert window.measurement_status_badge.text() == "Click B"
        assert "Click a second point" in window.measurement_status_badge.toolTip()

        window.model._handle_waveform_measurement_point(channel_name, second_time)
        _process_events(qapp, cycles=6)

        expected_text = f"dt {second_time * 1000.0:.1f} ms | dA {expected_delta:.1f} uV"
        assert window.measurement_status_badge.text() == expected_text
        assert window.measurement_status_badge.isVisible() is True
        assert "A:" in window.measurement_status_badge.toolTip()
        assert "B:" in window.measurement_status_badge.toolTip()

        view = window.waveform_plot.main_waveform_plot_controller.view
        assert len(view.measurement_points) == 2
        assert view.measurement_label.isVisible() is True

        window.measure_tool_button.click()
        _process_events(qapp, cycles=6)
        assert window.waveform_plot.is_measurement_enabled() is False
        assert window.measurement_status_badge.isVisible() is False
    finally:
        window.close()
        _process_events(qapp)


def test_waveform_reset_view_button_clears_sticky_modes(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        window.auto_bipolar_button.click()
        _process_events(qapp, cycles=10)
        window.measure_tool_button.click()
        _process_events(qapp, cycles=6)

        assert window.waveform_source_mode_badge.text().startswith("Source ")
        assert "Bipolar" in window.waveform_source_mode_badge.text()
        assert window.waveform_tool_mode_badge.text() == "Tool Measure"
        assert window.waveform_reset_view_button.isEnabled() is True

        window.waveform_reset_view_button.click()
        _process_events(qapp, cycles=12)

        assert window.auto_bipolar_button.isChecked() is False
        assert window.measure_tool_button.isChecked() is False
        assert window.waveform_plot.is_measurement_enabled() is False
        assert window.all_channels_button.isChecked() is True
        assert window.waveform_source_mode_badge.text() == "Source Ref"
        assert window.waveform_scope_mode_badge.text() == "Scope All"
        assert window.waveform_tool_mode_badge.text() == "Tool Browse"
    finally:
        window.close()
        _process_events(qapp)


def test_neighbor_tool_focuses_adjacent_channels_for_highlighted_target(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)
        backend_channels = [str(channel) for channel in window.model.backend.channel_names]

        window.waveform_plot.main_waveform_plot_controller.view.set_highlighted_channel(backend_channels[1])
        window.model.update_waveform_toolbar_state()
        _process_events(qapp, cycles=6)

        assert window.neighbor_channels_button.isEnabled() is True
        window.neighbor_channels_button.click()
        _process_events(qapp, cycles=8)

        assert window.neighbor_channels_button.isChecked() is True
        assert window.waveform_plot.get_channels_to_plot() == backend_channels[:2]

        window.neighbor_channels_button.click()
        _process_events(qapp, cycles=8)

        assert window.neighbor_channels_button.isChecked() is False
        assert window.all_channels_button.isChecked() is True
        assert window.waveform_plot.get_channels_to_plot() == backend_channels
    finally:
        window.close()
        _process_events(qapp)


def test_clean_view_hides_bad_source_channels(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)
        backend_channels = [str(channel) for channel in window.model.backend.channel_names]
        window.model.backend.raw.info["bads"] = [backend_channels[-1]]
        window.model.update_waveform_toolbar_state()
        _process_events(qapp, cycles=6)

        assert window.clean_view_button.isEnabled() is True
        window.clean_view_button.click()
        _process_events(qapp, cycles=8)

        assert window.waveform_plot.get_channels_to_plot() == backend_channels[:-1]
        assert window.clean_view_button.isChecked() is True
    finally:
        window.close()
        _process_events(qapp)


def test_waveform_shortcuts_do_not_fire_while_editing_inputs(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        window.fp_input.setFocus()
        _process_events(qapp, cycles=3)

        QtTest.QTest.keyClick(window.fp_input, QtCore.Qt.Key_C)
        _process_events(qapp, cycles=6)

        assert window.waveform_plot.is_cursor_enabled() is False
    finally:
        window.close()
        _process_events(qapp)


def test_waveform_toolbar_defaults_to_all_channels_scope_when_recording_is_loaded(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        assert window.waveform_placeholder.isVisible() is False
        assert window.workflow_status_label.isVisible() is False
        assert window.all_channels_button.isEnabled() is True
        assert window.all_channels_button.isChecked() is True
        assert window.event_channels_button.isEnabled() is False
        assert window.event_channels_button.isChecked() is False
    finally:
        window.close()
        _process_events(qapp)


def test_open_file_workflow_keeps_filter_controls_alive_after_loading(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        monkeypatch.setattr(window.model, "_run_open_dialog", lambda *args, **kwargs: str(tiny_fif_path))
        window.model.open_file()
        _process_events(qapp, cycles=120)

        assert window.waveform_placeholder.isVisible() is False
        assert window.workflow_status_label.isVisible() is False
        assert window.overview_filter_button.isEnabled() is True
        for widget in (
            window.overview_filter_button,
            window.fp_input,
            window.fs_input,
        ):
            assert widget.parent() is not None
            assert widget.isVisible() is True
            assert widget.isHidden() is False
            assert widget.width() > 0
            assert widget.height() > 0

        assert window.fp_input.isReadOnly() is False
        assert window.fs_input.isReadOnly() is False
    finally:
        window.close()
        _process_events(qapp)


def test_loaded_recording_clamps_filter_defaults_below_nyquist(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        nyquist = float(window.main_sampfreq.text()) / 2.0
        assert float(window.fs_input.text()) < nyquist
        assert float(window.fp_input.text()) < float(window.fs_input.text())
        validator = window.fs_input.validator()
        assert validator is not None
        assert validator.top() < nyquist
    finally:
        window.close()
        _process_events(qapp)


def test_apply_style_buttons_use_explicit_action_labels(monkeypatch, qapp):
    window = _create_window(monkeypatch, qapp)
    try:
        assert window.overview_filter_button.text() == "Filter"
        assert window.detector_apply_button.text() == "Apply"
        assert window.classifier_apply_button.text() == "Apply"
        assert window.classifier_save_button.text() == "Apply"
        assert window.STE_save_button.text() == "Apply"
        assert window.MNI_save_button.text() == "Apply"
        assert window.HIL_save_button.text() == "Apply"
        if hasattr(window, "YASA_save_button"):
            assert window.YASA_save_button.text() == "Apply"
    finally:
        window.close()
        _process_events(qapp)


def test_filter_button_shows_busy_feedback_and_syncs_filtered_view(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    workers = []
    try:
        _load_recording(window, tiny_fif_path, qapp)
        window.fp_input.setText("20")
        window.fs_input.setText("80")
        monkeypatch.setattr(window.threadpool, "start", lambda worker: workers.append(worker))

        window.model.filter_data()
        _process_events(qapp, cycles=6)

        assert len(workers) == 1
        assert window.overview_filter_button.text() == "Filtering"
        assert window.overview_filter_button.isEnabled() is False
        assert window.overview_filter_button.property("workflowBusy") is True
        assert window.fp_input.isEnabled() is False
        assert window.fs_input.isEnabled() is False
        _assert_main_workspace_busy(window, True)
        if hasattr(window, "status_filter_chip"):
            assert window.status_filter_chip.text() == "Filtering..."

        window.model._filter(None)
        workers[0].signals.result.emit([])
        _process_events(qapp, cycles=12)

        assert window.overview_filter_button.text() == "Filter"
        assert window.overview_filter_button.isEnabled() is True
        assert window.overview_filter_button.property("workflowBusy") is False
        assert window.fp_input.isEnabled() is True
        assert window.fs_input.isEnabled() is True
        _assert_main_workspace_busy(window, False)
        if hasattr(window, "status_filter_chip"):
            assert window.status_filter_chip.text() == "Filter ready"
        assert window.toggle_filtered_checkbox.isChecked() is True
        assert window.filtered_tool_button.isChecked() is True
        assert window.raw_tool_button.isChecked() is False
        assert window.show_filtered is True
        assert window.is_data_filtered is True
    finally:
        window.close()
        _process_events(qapp)


def test_filter_fields_submit_on_return(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    workers = []
    try:
        _load_recording(window, tiny_fif_path, qapp)
        monkeypatch.setattr(window.threadpool, "start", lambda worker: workers.append(worker))

        _submit_line_edit_text(window.fp_input, "20", qapp)

        assert len(workers) == 1
        assert window.overview_filter_button.text() == "Filtering"

        window.model._filter(None)
        workers[0].signals.result.emit([])
        _process_events(qapp, cycles=12)
    finally:
        window.close()
        _process_events(qapp)


def test_detector_fields_submit_on_return(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)
        window.detector_mode_combo.setCurrentText("MNI")
        _process_events(qapp, cycles=6)

        _submit_line_edit_text(window.mni_epoch_time_input, "10", qapp)

        assert window.model.backend.param_detector is not None
        assert str(window.model.backend.param_detector.detector_type).upper() == "MNI"
        assert window.detector_run_button.isEnabled() is True
    finally:
        window.close()
        _process_events(qapp)


def test_classifier_fields_submit_on_return(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)
        window.classifier_mode_combo.setCurrentText("Custom")
        _process_events(qapp, cycles=10)

        window.classifier_artifact_card_name.setText("org/artifact")
        window.classifier_spike_card_name.setText("org/spike")
        window.classifier_ehfo_card_name.setText("org/ehfo")
        window.classifier_device_input.setText("cpu")

        _submit_line_edit_text(window.classifier_batch_size_input, "8", qapp)

        classifier_param = window.model.backend.get_classifier_param()
        assert classifier_param is not None
        assert classifier_param.artifact_card == "org/artifact"
        assert classifier_param.batch_size == 8
    finally:
        window.close()
        _process_events(qapp)


def test_filter_error_restores_idle_ui_without_marking_signal_filtered(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    workers = []
    try:
        _load_recording(window, tiny_fif_path, qapp)
        window.fp_input.setText("20")
        window.fs_input.setText("80")
        monkeypatch.setattr(window.threadpool, "start", lambda worker: workers.append(worker))
        monkeypatch.setattr(QtWidgets.QMessageBox, "exec_", lambda self: QtWidgets.QMessageBox.Ok)

        window.model.filter_data()
        _process_events(qapp, cycles=6)

        assert len(workers) == 1
        _assert_main_workspace_busy(window, True)
        workers[0].signals.error.emit((RuntimeError, RuntimeError("boom"), "traceback"))
        workers[0].signals.finished.emit()
        _process_events(qapp, cycles=12)

        assert window.overview_filter_button.text() == "Filter"
        assert window.overview_filter_button.isEnabled() is True
        assert window.overview_filter_button.property("workflowBusy") is False
        assert window.fp_input.isEnabled() is True
        assert window.fs_input.isEnabled() is True
        _assert_main_workspace_busy(window, False)
        if hasattr(window, "status_filter_chip"):
            assert window.status_filter_chip.text() == "Raw signal"
        assert window.toggle_filtered_checkbox.isChecked() is False
        assert window.filtered_tool_button.isChecked() is False
        assert window.raw_tool_button.isChecked() is True
        assert window.is_data_filtered is False
    finally:
        window.close()
        _process_events(qapp)


def test_detection_workflow_freezes_workspace_until_worker_finishes(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    workers = []
    try:
        _load_recording(window, tiny_fif_path, qapp)
        monkeypatch.setattr(window.threadpool, "start", lambda worker: workers.append(worker))

        class _FakeFeatures:
            annotated = np.array([])

            def get_num_biomarker(self):
                return 2

            def get_review_progress(self):
                return {"reviewed": 0, "remaining": 2, "total": 2}

        window.model.run_selected_detector_workflow()
        _process_events(qapp, cycles=6)

        assert len(workers) == 1
        assert window.detector_run_button.text() == "Detecting"
        assert window.detector_run_button.property("workflowBusy") is True
        assert window.ste_detect_button.text() == "Detecting"
        assert window.ste_detect_button.property("workflowBusy") is True
        _assert_main_workspace_busy(window, True)

        window.model.backend.detected = True
        window.model.backend.event_features = _FakeFeatures()
        workers[0].signals.result.emit([])
        _process_events(qapp, cycles=12)

        assert window.detector_run_button.text() == "Run STE"
        assert window.detector_run_button.property("workflowBusy") is False
        assert window.ste_detect_button.text() == "Run"
        assert window.ste_detect_button.property("workflowBusy") is False
        _assert_main_workspace_busy(window, False)
    finally:
        window.close()
        _process_events(qapp)


def test_classification_workflow_freezes_workspace_until_worker_finishes(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    workers = []
    try:
        _load_recording(window, tiny_fif_path, qapp)
        monkeypatch.setattr(window.threadpool, "start", lambda worker: workers.append(worker))

        class _FakeFeatures:
            annotated = np.array([])

            def get_num_biomarker(self):
                return 1

            def get_review_progress(self):
                return {"reviewed": 0, "remaining": 1, "total": 1}

        window.model.backend.detected = True
        window.model.backend.event_features = _FakeFeatures()
        window.model.run_classifier_workflow()
        _process_events(qapp, cycles=6)

        assert len(workers) == 1
        assert window.classifier_run_button.text() == "Classifying"
        assert window.classifier_run_button.property("workflowBusy") is True
        assert window.detect_all_button.text() == "Classifying"
        assert window.detect_all_button.property("workflowBusy") is True
        _assert_main_workspace_busy(window, True)

        window.model.backend.classified = True
        workers[0].signals.result.emit([])
        _process_events(qapp, cycles=12)

        assert window.classifier_run_button.text() == "Classify"
        assert window.classifier_run_button.property("workflowBusy") is False
        assert window.detect_all_button.text() == "Run Classification"
        assert window.detect_all_button.property("workflowBusy") is False
        _assert_main_workspace_busy(window, False)
    finally:
        window.close()
        _process_events(qapp)


def test_run_actions_uses_uniform_compact_input_heights(monkeypatch, qapp):
    window = _create_window(monkeypatch, qapp)
    try:
        _assert_compact_workflow_inputs(window)
    finally:
        window.close()
        _process_events(qapp)


def test_run_actions_and_results_use_clear_labels(monkeypatch, qapp):
    window = _create_window(monkeypatch, qapp)
    try:
        assert window.compare_runs_button.text() == "Run Stats"
        assert window.annotation_button.text() == "Open Review"
        assert window.save_npz_button.text() == "Save Session"
        assert window.save_csv_button.text() == "Export Workbook"
        assert window.open_review_button.text() == "Review"

        menu_buttons = {
            button.text()
            for button in window.findChildren(QtWidgets.QToolButton)
            if button.property("inspectorMenu") is True
        }
        assert "Run" in menu_buttons
        assert "More" in menu_buttons
        assert sum(
            1
            for button in window.findChildren(QtWidgets.QToolButton)
            if button.property("inspectorMenu") is True and button.text() == "More"
        ) == 3
    finally:
        window.close()
        _process_events(qapp)


def test_workflow_controls_transition_cleanly_between_empty_and_loaded_states(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _assert_workflow_controls_remain_mounted(window)
        _assert_compact_workflow_inputs(window)

        assert window.combo_box_biomarker.currentText() == "HFO"
        assert window.active_run_selector.isEnabled() is False
        assert window.annotation_button.isEnabled() is False
        assert window.save_npz_button.isEnabled() is False
        assert window.save_csv_button.isEnabled() is False
        assert window.compare_runs_button.isEnabled() is False
        assert window.overview_filter_button.isEnabled() is False
        assert window.review_channels_button.isEnabled() is False
        assert window.all_channels_button.isEnabled() is False
        assert window.event_channels_button.isEnabled() is False
        assert window.go_to_time_button.isEnabled() is False
        assert window.snapshot_button.isEnabled() is False
        assert window.prev_event_button.isEnabled() is False
        assert window.next_event_button.isEnabled() is False

        _load_recording(window, tiny_fif_path, qapp)

        _assert_workflow_controls_remain_mounted(window)
        _assert_compact_workflow_inputs(window)

        assert window.active_run_selector.isEnabled() is False
        assert window.annotation_button.isEnabled() is False
        assert window.save_npz_button.isEnabled() is False
        assert window.save_csv_button.isEnabled() is False
        assert window.compare_runs_button.isEnabled() is False
        assert window.overview_filter_button.isEnabled() is True
        assert window.review_channels_button.isEnabled() is True
        assert window.all_channels_button.isEnabled() is True
        assert window.event_channels_button.isEnabled() is False
        assert window.go_to_time_button.isEnabled() is True
        assert window.snapshot_button.isEnabled() is True
        assert window.prev_event_button.isEnabled() is False
        assert window.next_event_button.isEnabled() is False
    finally:
        window.close()
        _process_events(qapp)


def test_new_run_actions_keep_workflow_selections_stable_across_biomarkers(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        assert window.new_run_button.menu() is not None
        assert [action.text() for action in window.new_run_button.menu().actions()] == [
            "New HFO Run",
            "New Spindle Run",
            "New Spike Run",
        ]

        workflow_matrix = (
            ("Spindle", window.new_spindle_run_action, ["YASA"], False, ["Hugging Face CPU", "Hugging Face GPU", "Custom"], True),
            ("Spike", window.new_spike_run_action, ["RMS/LL"], False, ["Review only"], False),
            ("HFO", window.new_hfo_run_action, ["STE", "MNI", "HIL"], True, ["Hugging Face CPU", "Hugging Face GPU", "Custom"], True),
        )

        for biomarker, action, detector_items, detector_enabled, classifier_items, classifier_enabled in workflow_matrix:
            action.trigger()
            _process_events(qapp, cycles=30)

            assert window.combo_box_biomarker.currentText() == biomarker
            assert window.model.biomarker_type == biomarker
            assert getattr(window.model.backend, "eeg_data", None) is not None
            assert [window.detector_mode_combo.itemText(i) for i in range(window.detector_mode_combo.count())] == detector_items
            assert window.detector_mode_combo.isEnabled() is detector_enabled
            assert [window.classifier_mode_combo.itemText(i) for i in range(window.classifier_mode_combo.count())] == classifier_items
            assert window.classifier_mode_combo.isEnabled() is classifier_enabled

            _assert_workflow_controls_remain_mounted(window)
            _assert_compact_workflow_inputs(window)
    finally:
        window.close()
        _process_events(qapp)


def test_detector_and_classifier_selections_stay_in_sync_with_active_workflow(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        window.detector_mode_combo.setCurrentText("MNI")
        _process_events(qapp)
        assert window.detector_subtabs.currentIndex() == 1
        assert window.detector_subtabs.tabText(window.detector_subtabs.currentIndex()) == "MNI"

        window.detector_subtabs.setCurrentIndex(2)
        _process_events(qapp)
        assert window.detector_mode_combo.currentText() == "HIL"

        window.classifier_mode_combo.setCurrentText("Custom")
        _process_events(qapp)
        _assert_classifier_custom_sources_follow_mode(window)

        window.classifier_mode_combo.setCurrentText("Hugging Face CPU")
        _process_events(qapp)
        _assert_classifier_custom_sources_follow_mode(window)

        window.new_spike_run_action.trigger()
        _process_events(qapp, cycles=30)

        assert window.detector_mode_combo.currentText() == "RMS/LL"
        assert window.classifier_mode_combo.currentText() == "Review only"
        assert window.classifier_mode_combo.isEnabled() is False
        assert window.detector_run_button.text() == "Run RMS/LL"
        assert window.detector_run_button.isEnabled() is True
        assert window.classifier_run_button.text() == "Unavailable"
        assert window.classifier_run_button.isEnabled() is False
        _assert_classifier_custom_sources_follow_mode(window)

        window.new_hfo_run_action.trigger()
        _process_events(qapp, cycles=30)

        assert [window.detector_subtabs.tabText(i) for i in range(window.detector_subtabs.count())] == ["STE", "MNI", "HIL"]
        assert [window.classifier_mode_combo.itemText(i) for i in range(window.classifier_mode_combo.count())] == [
            "Hugging Face CPU",
            "Hugging Face GPU",
            "Custom",
        ]
        _assert_compact_workflow_inputs(window)
        _assert_classifier_custom_sources_follow_mode(window)
    finally:
        window.close()
        _process_events(qapp)


def test_report_and_run_stats_actions_follow_the_first_detected_run(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        assert window.save_report_button.isEnabled() is False
        assert window.compare_runs_button.isEnabled() is False
        assert window.accept_run_button.isEnabled() is False
        assert window.active_run_selector.isEnabled() is False
        assert window.run_stats_activate_button.isEnabled() is False
        assert window.run_stats_accept_button.isEnabled() is False
        assert window.run_stats_report_button.isEnabled() is False

        def _fake_detect_biomarker():
            feature = HFO_Feature(
                np.array([str(window.model.backend.channel_names[0])]),
                np.array([[10, 20]]),
                sample_freq=int(window.model.backend.sample_freq),
            )
            window.model.backend.event_features = feature
            window.model.backend.detected = True
            run = DetectionRun.create(
                biomarker_type="HFO",
                detector_name="STE",
                selected_channels=list(window.model.backend.channel_names),
                param_filter=getattr(window.model.backend, "param_filter", None),
                param_detector=getattr(window.model.backend, "param_detector", None),
                param_classifier=getattr(window.model.backend, "param_classifier", None),
                event_features=feature,
                classified=False,
            )
            window.model.backend.analysis_session.add_run(run)

        monkeypatch.setattr(window.model.backend, "detect_biomarker", _fake_detect_biomarker)
        _run_hfo_detector(window, "STE", qapp)

        assert window.save_report_button.isEnabled() is True
        assert window.compare_runs_button.isEnabled() is True
        assert window.accept_run_button.isEnabled() is True
        assert window.active_run_selector.isEnabled() is True
        assert window.run_stats_activate_button.isEnabled() is True
        assert window.run_stats_accept_button.isEnabled() is True
        assert window.run_stats_report_button.isEnabled() is True
        assert window.run_stats_report_button.isEnabled() is window.save_report_button.isEnabled()
        assert window.open_review_button.isEnabled() is window.annotation_button.isEnabled()
        assert window.active_run_selector.currentText().startswith("HFO • STE •")

        window.compare_runs_button.click()
        _process_events(qapp, cycles=8)
        assert window.run_stats_dialog.isVisible() is True

        window.accept_run_button.click()
        _process_events(qapp, cycles=8)
        current_label = window.active_run_selector.itemText(window.active_run_selector.currentIndex())
        assert "accepted" in current_label
    finally:
        window.close()
        _process_events(qapp)


def test_open_review_guard_disables_stale_review_button_when_active_run_has_no_events(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        class _EmptyEventFeatures:
            def get_num_biomarker(self):
                return 0

        captured = {}

        def _capture_information(parent, title, text, *args, **kwargs):
            captured["title"] = title
            captured["text"] = text
            return QtWidgets.QMessageBox.Ok

        monkeypatch.setattr(QtWidgets.QMessageBox, "information", _capture_information)

        window.model.backend.event_features = _EmptyEventFeatures()
        window.annotation_button.setEnabled(True)
        window.model.open_annotation()
        _process_events(qapp, cycles=6)

        assert captured["title"] == "No Events To Review"
        assert "There are no events in the active run to review yet." in captured["text"]
        assert window.annotation_button.isEnabled() is False
    finally:
        window.close()
        _process_events(qapp)


def test_go_to_time_button_clamps_to_recording_bounds(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        total_time = float(window.waveform_plot.get_total_time())

        window.display_time_window_input.setValue(0.5)
        window.model.waveform_plot_button_clicked()
        _process_events(qapp, cycles=6)

        window.go_to_time_input.setValue(999.0)
        window.go_to_time_button.click()
        _process_events(qapp, cycles=6)

        expected_tail = max(0.0, total_time - 0.5)
        assert abs(float(window.waveform_plot.t_start) - expected_tail) < 1e-6
        assert abs(float(window.go_to_time_input.value()) - round(expected_tail, 2)) < 1e-6

        window.display_time_window_input.setValue(total_time)
        window.model.waveform_plot_button_clicked()
        _process_events(qapp, cycles=6)

        window.go_to_time_input.setValue(999.0)
        window.go_to_time_button.click()
        _process_events(qapp, cycles=6)

        assert abs(float(window.waveform_plot.t_start)) < 1e-6
        assert abs(float(window.go_to_time_input.value())) < 1e-6
    finally:
        window.close()
        _process_events(qapp)


def test_waveform_amplitude_input_scales_the_main_window_signal(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        plot_model = window.waveform_plot.main_waveform_plot_controller.model
        base_data, _, base_scale_length, base_offset_value = plot_model.get_all_current_eeg_data_to_display()

        window.waveform_amplitude_input.setValue(2.0)
        window.model.waveform_plot_button_clicked()
        _process_events(qapp, cycles=6)

        scaled_data, _, scaled_scale_length, scaled_offset_value = plot_model.get_all_current_eeg_data_to_display()

        assert window.waveform_plot.get_vertical_amplitude_scale() == 2.0
        assert scaled_offset_value == base_offset_value
        assert abs(scaled_scale_length - (base_scale_length * 2.0)) < 1e-6
        assert np.allclose(scaled_data, base_data * 2.0)
    finally:
        window.close()
        _process_events(qapp)


def test_waveform_amplitude_stepper_updates_signal_immediately(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        initial_scale = float(window.waveform_plot.get_vertical_amplitude_scale())
        _click_spinbox_stepper(window.waveform_amplitude_input, QtWidgets.QStyle.SC_SpinBoxUp)
        _process_events(qapp, cycles=6)

        assert float(window.waveform_amplitude_input.value()) > initial_scale
        assert float(window.waveform_plot.get_vertical_amplitude_scale()) == float(window.waveform_amplitude_input.value())
    finally:
        window.close()
        _process_events(qapp)


def test_waveform_window_stepper_updates_plot_immediately(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        initial_window = float(window.waveform_plot.get_time_window())
        _click_spinbox_stepper(window.display_time_window_input, QtWidgets.QStyle.SC_SpinBoxDown)
        _process_events(qapp, cycles=6)

        assert float(window.display_time_window_input.value()) < initial_window
        assert float(window.waveform_plot.get_time_window()) == float(window.display_time_window_input.value())
    finally:
        window.close()
        _process_events(qapp)


def test_waveform_wheel_vertical_scroll_moves_through_channels(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        window.n_channel_input.setValue(1)
        window.model.waveform_plot_button_clicked()
        _process_events(qapp, cycles=6)

        initial_first = int(window.waveform_plot.first_channel_to_plot)
        initial_channel = window.waveform_plot.get_channels_to_plot()[initial_first]
        handled, event = _dispatch_waveform_wheel(window, delta_y=-96)
        _process_events(qapp, cycles=6)

        assert handled is True
        assert event.accepted is True
        assert int(window.waveform_plot.first_channel_to_plot) > initial_first
        assert window.waveform_plot.get_channels_to_plot()[window.waveform_plot.first_channel_to_plot] != initial_channel
    finally:
        window.close()
        _process_events(qapp)


def test_waveform_wheel_horizontal_scroll_moves_through_time(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        window.display_time_window_input.setValue(0.5)
        window.Time_Increment_Input.setValue(50)
        window.waveform_plot.t_start = 0.5
        window.model.waveform_plot_button_clicked()
        _process_events(qapp, cycles=6)

        initial_start = float(window.waveform_plot.t_start)
        handled, event = _dispatch_waveform_wheel(window, delta_x=-40)
        _process_events(qapp, cycles=6)

        assert handled is True
        assert event.accepted is True
        assert float(window.waveform_plot.t_start) > initial_start
        assert float(window.go_to_time_input.value()) > 0.0
    finally:
        window.close()
        _process_events(qapp)


def test_waveform_trackpad_pinch_zoom_changes_time_window(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        window.display_time_window_input.setValue(1.0)
        window.model.waveform_plot_button_clicked()
        _process_events(qapp, cycles=6)

        initial_window = float(window.display_time_window_input.value())
        handled, event = _dispatch_waveform_native_zoom(window, 0.25)
        _process_events(qapp, cycles=6)

        assert handled is True
        assert event.isAccepted() is True
        assert float(window.display_time_window_input.value()) < initial_window
        assert float(window.display_time_window_input.value()) > initial_window * 0.75
    finally:
        window.close()
        _process_events(qapp)


def test_waveform_toolbar_numeric_fields_apply_on_edit_commit(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        _commit_spinbox_text(window.display_time_window_input, "0.5", qapp)
        assert abs(float(window.waveform_plot.get_time_window()) - 0.5) < 1e-6

        _commit_spinbox_text(window.Time_Increment_Input, "50", qapp)
        assert abs(float(window.waveform_plot.get_time_increment()) - 50.0) < 1e-6

        _commit_spinbox_text(window.n_channel_input, "1", qapp)
        assert int(window.waveform_plot.get_n_channels_to_plot()) == 1
    finally:
        window.close()
        _process_events(qapp)


def test_workers_spinbox_applies_on_edit_commit(monkeypatch, qapp):
    window = _create_window(monkeypatch, qapp)
    try:
        assert window.model.backend is not None
        target_jobs = min(2, window.n_jobs_spinbox.maximum())

        _commit_spinbox_text(window.n_jobs_spinbox, str(target_jobs), qapp)

        assert int(window.model.backend.n_jobs) == target_jobs
        assert int(window.n_jobs_spinbox.value()) == target_jobs
    finally:
        window.close()
        _process_events(qapp)


def test_custom_classifier_layout_keeps_source_groups_below_header_controls(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        window.classifier_mode_combo.setCurrentText("Custom")
        _process_events(qapp, cycles=20)

        combo_rect = _widget_rect_in_window(window, window.classifier_mode_combo)
        run_rect = _widget_rect_in_window(window, window.classifier_run_button)
        header_rect = combo_rect.united(run_rect)
        local_rect = _widget_rect_in_window(window, window.groupBox)
        hub_rect = _widget_rect_in_window(window, window.groupBox_2)
        apply_rect = _widget_rect_in_window(window, window.classifier_apply_button)

        assert window.classifier_custom_sources_frame.isVisible() is True
        assert window.classifier_custom_apply_frame.isVisible() is True
        assert window.groupBox.isVisible() is True
        assert window.groupBox_2.isVisible() is True
        assert not local_rect.intersects(header_rect)
        assert not hub_rect.intersects(header_rect)
        assert local_rect.top() > header_rect.bottom()
        assert hub_rect.top() > local_rect.bottom()
        assert apply_rect.top() > hub_rect.bottom()
        assert window.classifier_run_button.width() >= window.classifier_run_button.sizeHint().width()
        assert window.classifier_apply_button.width() >= window.classifier_apply_button.sizeHint().width()
    finally:
        window.close()
        _process_events(qapp)


def test_detector_and_classifier_action_paths_apply_all_main_workflow_modes(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        for detector_name in ["STE", "MNI", "HIL"]:
            window.detector_mode_combo.setCurrentText(detector_name)
            _process_events(qapp, cycles=6)
            window.detector_apply_button.click()
            _process_events(qapp, cycles=6)
            assert window.model.backend.param_detector.detector_type.upper() == detector_name
            assert window.detector_run_button.text() == f"Run {detector_name}"

        for classifier_mode in ["Hugging Face CPU", "Hugging Face GPU", "Custom"]:
            window.classifier_mode_combo.setCurrentText(classifier_mode)
            _process_events(qapp, cycles=6)
            if classifier_mode == "Custom":
                window.classifier_device_input.setText("cpu")
                window.classifier_apply_button.click()
                _process_events(qapp, cycles=6)
                assert window.model.backend.param_classifier.device == "cpu"
            else:
                assert window.model.apply_classifier_setup() is True
                _process_events(qapp, cycles=6)
            assert window.model.backend.param_classifier is not None

        window.new_spindle_run_action.trigger()
        _process_events(qapp, cycles=30)
        assert window.combo_box_biomarker.currentText() == "Spindle"
        window.detector_apply_button.click()
        _process_events(qapp, cycles=6)
        assert window.model.backend.param_detector.detector_type.upper() == "YASA"

        window.classifier_mode_combo.setCurrentText("Custom")
        _process_events(qapp, cycles=6)
        window.classifier_device_input.setText("cpu")
        window.classifier_apply_button.click()
        _process_events(qapp, cycles=6)
        assert window.model.backend.param_classifier is not None
        assert window.model.backend.param_classifier.device == "cpu"

        window.new_spike_run_action.trigger()
        _process_events(qapp, cycles=30)
        assert window.combo_box_biomarker.currentText() == "Spike"
        assert window.detector_run_button.isEnabled() is True
        assert window.classifier_run_button.isEnabled() is False
        assert window.detector_apply_button.isEnabled() is True
        window.detector_apply_button.click()
        _process_events(qapp, cycles=6)
        assert window.model.backend.param_detector.detector_type.upper() == "RMS/LL"
    finally:
        window.close()
        _process_events(qapp)


def test_inspector_action_buttons_flash_click_feedback_after_activation(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        assert window.detector_run_button.property("clickFeedbackInstalled") is True
        assert window.classifier_apply_button.property("clickFeedbackInstalled") is True
        assert window.ste_detect_button.property("clickFeedbackInstalled") is True
        assert window.mni_detect_button.property("clickFeedbackInstalled") is True
        assert window.hil_detect_button.property("clickFeedbackInstalled") is True

        _disconnect_if_connected(window.detector_run_button.clicked, window.model.run_selected_detector_workflow)
        QtTest.QTest.mouseClick(window.detector_run_button, QtCore.Qt.LeftButton)
        _process_events(qapp)
        assert window.detector_run_button.property("clickFeedback") is True

        QtTest.QTest.qWait(window.view._button_feedback_duration_ms + 80)
        _process_events(qapp)
        assert window.detector_run_button.property("clickFeedback") is False

        window.classifier_mode_combo.setCurrentText("Custom")
        _process_events(qapp, cycles=12)
        _disconnect_if_connected(window.classifier_apply_button.clicked, window.model.apply_classifier_setup)
        QtTest.QTest.mouseClick(window.classifier_apply_button, QtCore.Qt.LeftButton)
        _process_events(qapp)
        assert window.classifier_apply_button.property("clickFeedback") is True

        QtTest.QTest.qWait(window.view._button_feedback_duration_ms + 80)
        _process_events(qapp)
        assert window.classifier_apply_button.property("clickFeedback") is False
    finally:
        window.close()
        _process_events(qapp)


def test_detector_apply_rejects_invalid_inputs_instead_of_reusing_stale_params(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)
        window.detector_mode_combo.setCurrentText("MNI")
        _process_events(qapp, cycles=6)

        assert window.model.apply_selected_detector_parameters(show_feedback=False) is True
        assert window.model.backend.param_detector.detector_type.upper() == "MNI"
        assert window.mni_detect_button.isEnabled() is True

        monkeypatch.setattr(QtWidgets.QMessageBox, "exec_", lambda self: 0)
        window.mni_epoch_time_input.setText("nan")
        _process_events(qapp, cycles=6)

        assert window.model.apply_selected_detector_parameters(show_feedback=False) is False
        assert window.model.backend.param_detector.detector_type.upper() == "MNI"
        assert window.mni_detect_button.isEnabled() is False
    finally:
        window.close()
        _process_events(qapp)


def test_custom_classifier_setup_rejects_invalid_device_without_fake_success(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)
        baseline = window.model.backend.get_classifier_param()
        baseline_device = baseline.device
        baseline_model_type = baseline.model_type

        window.classifier_mode_combo.setCurrentText("Custom")
        _process_events(qapp, cycles=6)
        window.classifier_device_input.setText("gpu??")
        monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args, **kwargs: QtWidgets.QMessageBox.Ok)

        assert window.model.apply_classifier_setup() is False
        classifier_param = window.model.backend.get_classifier_param()
        assert classifier_param.device == baseline_device
        assert classifier_param.model_type == baseline_model_type
    finally:
        window.close()
        _process_events(qapp)


def test_custom_classifier_setup_requires_artifact_source(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        window.classifier_mode_combo.setCurrentText("Custom")
        _process_events(qapp, cycles=6)
        window.classifier_device_input.setText("cpu")
        window.classifier_artifact_filename.setText("")
        window.classifier_artifact_card_name.setText("")
        monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args, **kwargs: QtWidgets.QMessageBox.Ok)

        assert window.model.apply_classifier_setup() is False
    finally:
        window.close()
        _process_events(qapp)


def test_active_run_selector_and_run_stats_follow_current_workflow_only(monkeypatch, qapp, tiny_fif_path):
    window = _create_window(monkeypatch, qapp)
    try:
        _load_recording(window, tiny_fif_path, qapp)

        class _FakeRun:
            def __init__(self):
                class _DetectorParam:
                    def to_dict(self):
                        return {}

                class _DetectorWrapper:
                    detector_param = _DetectorParam()

                self.biomarker_type = "HFO"
                self.detector_name = "STE"
                self.run_id = "run-ste"
                self.summary = {"num_events": 3, "num_channels": 2}
                self.param_filter = None
                self.param_detector = _DetectorWrapper()
                self.param_classifier = None
                self.classified = False
                self.created_at = "2026-04-03T15:47:56Z"

        class _FakeSession:
            def __init__(self, active_run):
                self._active_run = active_run

            def get_active_run(self):
                return self._active_run

            def get_accepted_run(self):
                return None

            def is_run_visible(self, run_id):
                return run_id == self._active_run.run_id

        active_run = _FakeRun()
        hfo_backend = window.model.case_backends["HFO"]
        hfo_backend.analysis_session = _FakeSession(active_run)
        hfo_backend.get_run_summaries = lambda: [
            {
                "run_id": active_run.run_id,
                "detector_name": active_run.detector_name,
                "num_events": active_run.summary["num_events"],
                "num_channels": active_run.summary["num_channels"],
                "visible": True,
                "accepted": False,
                "created_at": active_run.created_at,
                "display_name": active_run.detector_name,
            }
        ]
        hfo_backend.get_channel_ranking = lambda *_args, **_kwargs: []
        hfo_backend.compare_runs = lambda: {"pairwise_overlap": []}
        window.model.update_run_management_panel()

        assert window.active_run_selector.isEnabled() is True
        assert window.active_run_selector.count() == 1
        assert window.compare_runs_button.isEnabled() is True
        assert window.active_run_selector.currentText().startswith("HFO • STE")

        window.new_spindle_run_action.trigger()
        _process_events(qapp, cycles=30)

        assert window.combo_box_biomarker.currentText() == "Spindle"
        assert window.active_run_selector.count() == 1
        assert window.active_run_selector.currentText() == "No runs yet"
        assert window.compare_runs_button.isEnabled() is True
        assert window.run_table.rowCount() == 1

        window.new_spike_run_action.trigger()
        _process_events(qapp, cycles=30)

        assert window.combo_box_biomarker.currentText() == "Spike"
        assert window.active_run_selector.currentText() == "No runs yet"
        assert window.compare_runs_button.isEnabled() is True
        assert window.run_table.rowCount() == 1
    finally:
        window.close()
        _process_events(qapp)
