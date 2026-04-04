import os
import sys
import warnings
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets, uic

from src.controllers import AnnotationController
from src.ui.ui_tokens import resolve_ui_density
from src.utils.utils_gui import (
    apply_subwindow_theme,
    fit_window_to_screen,
    safe_connect_signal_slot,
    set_accent_button,
    style_value_badge,
)

ROOT_DIR = Path(__file__).parent


class Annotation(QtWidgets.QMainWindow):
    def __init__(self, backend=None, main_window=None, close_signal=None, biomarker_type="HFO"):
        super().__init__(main_window)
        self.backend = backend
        self.biomarker_type = getattr(backend, "biomarker_type", biomarker_type)
        self.ui_density = resolve_ui_density(self.screen())
        self.ui = uic.loadUi(os.path.join(ROOT_DIR, "annotation.ui"), self)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.annotation_controller = AnnotationController(self, backend)
        self.threadpool = QtCore.QThreadPool()
        self.close_signal = close_signal
        self.shortcut_objects = []

        self.dropdown_placeholder = "--- Event Type ---"
        self.annotation_options = {
            "HFO": ["Pathological", "Physiological", "Artifact"],
            "Spindle": ["Real", "Spike", "Artifact"],
        }
        self.default_prediction_scope = "All"

        self.setWindowTitle(f"PyBrain {self.biomarker_type} Review Workspace")
        self.setWindowIcon(QtGui.QIcon(os.path.join(ROOT_DIR, "images/icon.png")))
        self._init_review_dropdowns()
        self._build_dynamic_controls()
        self._configure_status_bar()
        self._wire_signals()
        self._apply_review_styles()
        self._add_shortcuts()

        self.init_waveform_plot()
        self.init_fft_plot()
        self.init_annotation_dropdown()
        self.refresh_prediction_scope_controls()
        self.update_infos()
        self.on_history_state_changed(
            self.annotation_controller.can_go_back(),
            self.annotation_controller.can_go_forward(),
        )
        self.setInitialSize()
        self._update_plot_size_targets()
        self.setWindowModality(QtCore.Qt.ApplicationModal)

    def _init_review_dropdowns(self):
        self.EventDropdown_Box.clear()
        self.EventDropdown_Box.addItem(self.dropdown_placeholder)
        for label in self.annotation_options.get(self.biomarker_type, []):
            self.EventDropdown_Box.addItem(label)
        self.EventDropdown_Box.setCurrentIndex(0)

        self.IntervalDropdownBox.clear()
        if self.biomarker_type == "HFO":
            self.IntervalDropdownBox.addItems(["1s", "0.5s", "0.25s"])
        elif self.biomarker_type == "Spindle":
            self.IntervalDropdownBox.addItems(["4s", "3.5s"])
        self.IntervalDropdownBox.setCurrentIndex(0)

    def _build_dynamic_controls(self):
        self._build_navigation_controls()
        self._build_view_controls()
        self._build_info_labels()
        self._build_progress_summary()
        self._build_prediction_scope_controls()
        self._configure_frequency_controls()
        self._reflow_right_panel()
        if hasattr(self, "Accept"):
            self.Accept.setText("Save and Next")
        if hasattr(self, "actionStart_Over"):
            self.actionStart_Over.setText("Clear Current Annotation")

    def _build_navigation_controls(self):
        navigation_layout = getattr(self, "horizontalLayout", None)
        if navigation_layout is None:
            return
        navigation_layout.setSpacing(4)

        self.PrevPendingButton = QtWidgets.QPushButton("Prev Pending", self)
        self.NextPendingButton = QtWidgets.QPushButton("Next Pending", self)
        self.ClearAnnotationButton = QtWidgets.QPushButton("Clear Label", self)
        for button, tooltip in (
            (self.PrevPendingButton, "Jump to the previous unannotated event"),
            (self.NextPendingButton, "Jump to the next unannotated event"),
            (self.ClearAnnotationButton, "Clear the current review label"),
        ):
            button.setToolTip(tooltip)
            button.setMinimumWidth(self.ui_density.annotation_jump_button_width)
            button.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
            navigation_layout.addWidget(button)

        for button in (self.PreviousButton, self.NextButton):
            button.setMinimumWidth(self.ui_density.annotation_nav_button_width)
            button.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)

    def _build_prediction_scope_controls(self):
        parent = getattr(self, "centralwidget", self)
        if parent is None:
            return

        self.prediction_scope_card = QtWidgets.QFrame(parent)
        self.prediction_scope_card.setObjectName("predictionScopeCard")
        self.prediction_scope_card.setProperty("surfaceCard", True)
        self.prediction_scope_card.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        card_layout = QtWidgets.QVBoxLayout(self.prediction_scope_card)
        card_layout.setContentsMargins(10, 8, 10, 8)
        card_layout.setSpacing(6)

        header_layout = QtWidgets.QHBoxLayout()
        self.prediction_scope_label = QtWidgets.QLabel("Prediction Scope", self.prediction_scope_card)
        self.prediction_scope_label.setProperty("fieldLabel", True)
        self.PredictionScopeBox = QtWidgets.QComboBox(self.prediction_scope_card)
        self.PredictionScopeBox.setMinimumWidth(132)
        header_layout.addWidget(self.prediction_scope_label)
        header_layout.addWidget(self.PredictionScopeBox, 1)

        button_layout = QtWidgets.QHBoxLayout()
        self.PrevMatchButton = QtWidgets.QPushButton("Prev Match", self.prediction_scope_card)
        self.NextMatchButton = QtWidgets.QPushButton("Next Match", self.prediction_scope_card)
        for button in (self.PrevMatchButton, self.NextMatchButton):
            button.setMinimumWidth(self.ui_density.annotation_jump_button_width)
            button.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
            button_layout.addWidget(button)

        self.UnannotatedOnlyCheckBox = QtWidgets.QCheckBox("Unannotated only", self.prediction_scope_card)
        self.UnannotatedOnlyCheckBox.setToolTip("Limit match navigation to events that have not been reviewed yet")

        self.match_summary_label = QtWidgets.QLabel(self.prediction_scope_card)
        self.match_summary_label.setWordWrap(True)
        self.match_summary_label.setProperty("helperText", True)

        card_layout.addLayout(header_layout)
        card_layout.addLayout(button_layout)
        card_layout.addWidget(self.UnannotatedOnlyCheckBox)
        card_layout.addWidget(self.match_summary_label)

    def _build_view_controls(self):
        controls_layout = getattr(self, "viewControlsHLayout", None)
        groupbox = getattr(self, "viewControlsGroupBox", None)
        if controls_layout is None:
            return
        controls_layout.setSpacing(4)
        controls_layout.setContentsMargins(4, 4, 4, 4)

        if groupbox is not None:
            groupbox.setMinimumHeight(max(44, self.ui_density.compact_button_height + 24))
            groupbox.setMaximumHeight(QtWidgets.QWIDGETSIZE_MAX)
            groupbox.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self.SyncViewsCheckBox = QtWidgets.QCheckBox("Sync X", self)
        self.SyncViewsCheckBox.setToolTip("Synchronize the time axis across all review plots")
        controls_layout.addWidget(self.SyncViewsCheckBox)

        self.BackViewButton = QtWidgets.QToolButton(self)
        self.BackViewButton.setText("Back")
        self.BackViewButton.setToolTip("Go back to the previous view")
        controls_layout.addWidget(self.BackViewButton)

        self.ForwardViewButton = QtWidgets.QToolButton(self)
        self.ForwardViewButton.setText("Forward")
        self.ForwardViewButton.setToolTip("Go forward to the next view")
        controls_layout.addWidget(self.ForwardViewButton)

        self.ExportSnapshotButton = QtWidgets.QToolButton(self)
        self.ExportSnapshotButton.setText("Save")
        self.ExportSnapshotButton.setToolTip("Export a PNG snapshot of the current review workspace")
        controls_layout.addWidget(self.ExportSnapshotButton)

        for button in (self.BackViewButton, self.ForwardViewButton, self.ExportSnapshotButton):
            button.setMinimumWidth(self.ui_density.annotation_view_button_width)
            button.setAutoRaise(False)

    def _build_info_labels(self):
        info_layout = getattr(self, "gridLayout_3", None)
        if info_layout is None:
            return

        review_layout = QtWidgets.QHBoxLayout()
        review_label = QtWidgets.QLabel("Reviewed:", self.groupBox)
        review_label.setProperty("fieldLabel", True)
        self.reviewed_textbox = QtWidgets.QLabel(self.groupBox)
        review_layout.addWidget(review_label)
        review_layout.addWidget(self.reviewed_textbox)
        info_layout.addLayout(review_layout, 4, 0, 1, 1)

    def _build_progress_summary(self):
        parent = getattr(self, "centralwidget", self)
        if parent is None:
            return

        self.progress_card = QtWidgets.QFrame(parent)
        self.progress_card.setObjectName("progressCard")
        self.progress_card.setProperty("surfaceCard", True)
        self.progress_card.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        progress_layout = QtWidgets.QVBoxLayout(self.progress_card)
        progress_layout.setContentsMargins(10, 8, 10, 8)
        progress_layout.setSpacing(4)

        self.progress_caption_label = QtWidgets.QLabel("Annotation Progress", self.progress_card)
        self.progress_caption_label.setObjectName("progressCaption")
        self.progress_caption_label.setProperty("sectionTitle", True)

        self.progress_textbox = QtWidgets.QLabel(self.progress_card)
        self.progress_textbox.setObjectName("progressTextbox")
        self.progress_textbox.setWordWrap(False)

        progress_layout.addWidget(self.progress_caption_label)
        progress_layout.addWidget(self.progress_textbox)

    def _configure_frequency_controls(self):
        groupbox = getattr(self, "groupBox_2", None)
        layout = getattr(self, "gridLayout", None)
        if groupbox is None or layout is None:
            return

        groupbox.setTitle("Frequency Range")
        groupbox.setProperty("tightGroup", True)
        groupbox.setMinimumHeight(72)
        groupbox.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        layout.setContentsMargins(10, 8, 10, 8)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(4)

        self.label.setText("Min Hz")
        self.label_2.setText("Max Hz")
        self.label.setProperty("fieldLabel", True)
        self.label_2.setProperty("fieldLabel", True)
        self.SetFreqLimit.setText("Apply")
        self.SetFreqLimit.setToolTip("Apply the current frequency range to the TF and FFT views")
        self.SetFreqLimit.setMinimumWidth(96)
        self.SetFreqLimit.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.spinBox_minFreq.setFixedWidth(88)
        self.spinBox_maxFreq.setFixedWidth(88)

        for widget in (self.label, self.label_2, self.spinBox_minFreq, self.spinBox_maxFreq, self.SetFreqLimit):
            layout.removeWidget(widget)

        layout.addWidget(self.label, 0, 0, 1, 1)
        layout.addWidget(self.spinBox_minFreq, 0, 1, 1, 1)
        layout.addWidget(self.label_2, 0, 2, 1, 1)
        layout.addWidget(self.spinBox_maxFreq, 0, 3, 1, 1)
        layout.addWidget(self.SetFreqLimit, 0, 4, 1, 1)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 0)
        layout.setColumnStretch(3, 1)
        layout.setColumnStretch(4, 0)

    def _find_layout_item(self, grid_layout, target_layout):
        if grid_layout is None or target_layout is None:
            return None
        for index in range(grid_layout.count()):
            item = grid_layout.itemAt(index)
            if item is not None and item.layout() is target_layout:
                return item
        return None

    def _reflow_right_panel(self):
        left_panel = getattr(self, "left_panel", None)
        if left_panel is None:
            return

        for widget in (
            getattr(self, "progress_card", None),
            getattr(self, "AnotationDropdownBox", None),
            getattr(self, "groupBox", None),
            getattr(self, "EventDropdown_Box", None),
            getattr(self, "IntervalDropdownBox", None),
        ):
            if widget is not None:
                left_panel.removeWidget(widget)

        for layout in (
            getattr(self, "horizontalLayout", None),
            getattr(self, "FFT_layout", None),
            getattr(self, "freq_control_layout", None),
            getattr(self, "horizontalLayout_2", None),
        ):
            item = self._find_layout_item(left_panel, layout)
            if item is not None:
                left_panel.removeItem(item)

        left_panel.addWidget(self.progress_card, 0, 0, 1, 1)
        left_panel.addLayout(self.horizontalLayout, 1, 0, 1, 1)
        left_panel.addWidget(self.AnotationDropdownBox, 2, 0, 1, 1)
        left_panel.addLayout(self.FFT_layout, 3, 0, 1, 1)
        left_panel.addWidget(self.prediction_scope_card, 4, 0, 1, 1)
        left_panel.addLayout(self.freq_control_layout, 5, 0, 1, 1)
        left_panel.addWidget(self.groupBox, 6, 0, 1, 1)
        left_panel.addWidget(self.EventDropdown_Box, 7, 0, 1, 1)
        left_panel.addWidget(self.IntervalDropdownBox, 8, 0, 1, 1)
        left_panel.addLayout(self.horizontalLayout_2, 9, 0, 1, 1)

        for row in range(10):
            left_panel.setRowStretch(row, 0)
        left_panel.setRowStretch(3, 1)
        left_panel.setRowStretch(10, 0)

    def _wire_signals(self):
        if self.close_signal is not None:
            safe_connect_signal_slot(self.close_signal, self.close)

        safe_connect_signal_slot(self.PreviousButton.clicked, self.plot_prev)
        safe_connect_signal_slot(self.NextButton.clicked, self.plot_next)
        safe_connect_signal_slot(self.Accept.clicked, self.update_button_clicked)
        safe_connect_signal_slot(self.SetFreqLimit.clicked, self.update_frequency)
        safe_connect_signal_slot(self.spinBox_minFreq.editingFinished, self._submit_frequency_range)
        safe_connect_signal_slot(self.spinBox_maxFreq.editingFinished, self._submit_frequency_range)
        safe_connect_signal_slot(self.ResetViewButton.clicked, self.reset_view)
        safe_connect_signal_slot(self.ZoomInButton.clicked, self.zoom_in)
        safe_connect_signal_slot(self.ZoomOutButton.clicked, self.zoom_out)
        safe_connect_signal_slot(self.PanLeftButton.clicked, self.pan_left)
        safe_connect_signal_slot(self.PanRightButton.clicked, self.pan_right)
        safe_connect_signal_slot(self.PanUpButton.clicked, self.pan_up)
        safe_connect_signal_slot(self.PanDownButton.clicked, self.pan_down)
        safe_connect_signal_slot(self.IntervalDropdownBox.currentIndexChanged, self.update_interval)
        safe_connect_signal_slot(self.EventDropdown_Box.currentIndexChanged, self._update_accept_button_state)

        safe_connect_signal_slot(self.PrevPendingButton.clicked, self.plot_prev_unannotated)
        safe_connect_signal_slot(self.NextPendingButton.clicked, self.plot_next_unannotated)
        safe_connect_signal_slot(self.ClearAnnotationButton.clicked, self.clear_current_annotation)
        safe_connect_signal_slot(self.PredictionScopeBox.currentIndexChanged, self.on_prediction_scope_changed)
        safe_connect_signal_slot(self.UnannotatedOnlyCheckBox.toggled, self.on_prediction_scope_changed)
        safe_connect_signal_slot(self.PrevMatchButton.clicked, self.plot_prev_match)
        safe_connect_signal_slot(self.NextMatchButton.clicked, self.plot_next_match)
        safe_connect_signal_slot(self.SyncViewsCheckBox.toggled, self.annotation_controller.set_sync_views)
        safe_connect_signal_slot(self.BackViewButton.clicked, self.go_back_view)
        safe_connect_signal_slot(self.ForwardViewButton.clicked, self.go_forward_view)
        safe_connect_signal_slot(self.ExportSnapshotButton.clicked, self.export_snapshot)
        if hasattr(self, "actionStart_Over"):
            safe_connect_signal_slot(self.actionStart_Over.triggered, self.clear_current_annotation)

    def _configure_status_bar(self):
        self.hover_status_label = QtWidgets.QLabel(self)
        self.hover_status_label.setText("")
        self.statusbar.addWidget(self.hover_status_label, 1)

        self.hint_status_label = QtWidgets.QLabel(self)
        self.hint_status_label.setText(
            "Shift-drag: box zoom | Alt-drag: FFT ROI | Wheel: zoom | Drag: pan | Esc: clear FFT ROI"
        )
        self.statusbar.addPermanentWidget(self.hint_status_label, 1)

    def _apply_review_styles(self):
        if hasattr(self, "gridLayout_4"):
            self.gridLayout_4.setColumnStretch(0, 10)
            self.gridLayout_4.setColumnStretch(1, 6)
            self.gridLayout_4.setHorizontalSpacing(14)
        if hasattr(self, "VisulaizationVerticalLayout"):
            self.VisulaizationVerticalLayout.setContentsMargins(0, 0, 0, 0)
            self.VisulaizationVerticalLayout.setSpacing(0)
        if hasattr(self, "FFT_layout"):
            self.FFT_layout.setContentsMargins(0, 0, 0, 0)
        if self.menuBar() is not None:
            self.menuBar().hide()
        for label, width in (
            (self.model_textbox, 176),
            (self.channel_name_textbox, 168),
            (self.start_textbox, 96),
            (self.end_textbox, 96),
            (self.length_textbox, 96),
            (self.reviewed_textbox, 168),
            (self.progress_textbox, 248),
        ):
            style_value_badge(label, min_width=width)
        for label in (self.model_textbox, self.reviewed_textbox, self.progress_textbox):
            label.setWordWrap(True)
        self.hint_status_label.setProperty("helperText", True)
        self.hover_status_label.setProperty("helperText", True)
        set_accent_button(self.Accept)
        apply_subwindow_theme(
            self,
            extra_stylesheet="""
                QMainWindow { background: #f5f7fb; }
                QStatusBar { padding-left: 8px; }
                QLabel[dataBadge="true"] { padding: 3px 6px; border-radius: 6px; }
            """,
        )
        self._ensure_accessible_control_heights()

    def _ensure_accessible_control_heights(self):
        for widget in (
            getattr(self, "AnotationDropdownBox", None),
            getattr(self, "EventDropdown_Box", None),
            getattr(self, "IntervalDropdownBox", None),
            getattr(self, "PredictionScopeBox", None),
            getattr(self, "spinBox_minFreq", None),
            getattr(self, "spinBox_maxFreq", None),
        ):
            if widget is None:
                continue
            widget.setFixedHeight(max(widget.height(), widget.sizeHint().height(), widget.minimumSizeHint().height()))

        for button in (
            getattr(self, "BackViewButton", None),
            getattr(self, "ForwardViewButton", None),
            getattr(self, "ExportSnapshotButton", None),
        ):
            if button is None:
                continue
            button.setMinimumHeight(max(button.minimumHeight(), button.sizeHint().height(), button.minimumSizeHint().height()))

        if hasattr(self, "viewControlsGroupBox"):
            required_height = max(
                self.viewControlsGroupBox.minimumHeight(),
                self.viewControlsGroupBox.sizeHint().height(),
                self.viewControlsGroupBox.minimumSizeHint().height(),
            )
            self.viewControlsGroupBox.setMinimumHeight(required_height)

    def _add_shortcuts(self):
        shortcut_specs = [
            (QtGui.QKeySequence(QtCore.Qt.Key_Right), self.plot_next),
            (QtGui.QKeySequence(QtCore.Qt.Key_Left), self.plot_prev),
            (QtGui.QKeySequence("A"), self.plot_prev),
            (QtGui.QKeySequence("D"), self.plot_next),
            (QtGui.QKeySequence(QtCore.Qt.Key_Return), self.update_button_clicked),
            (QtGui.QKeySequence(QtCore.Qt.Key_Enter), self.update_button_clicked),
            (QtGui.QKeySequence(QtCore.Qt.Key_Backspace), self.clear_current_annotation),
            (QtGui.QKeySequence(QtCore.Qt.Key_Escape), self.clear_fft_roi),
        ]

        if self.biomarker_type == "HFO":
            shortcut_specs.extend(
                [
                    (QtGui.QKeySequence("1"), lambda: self.select_annotation_option("Pathological")),
                    (QtGui.QKeySequence("2"), lambda: self.select_annotation_option("Physiological")),
                    (QtGui.QKeySequence("3"), lambda: self.select_annotation_option("Artifact")),
                ]
            )
        elif self.biomarker_type == "Spindle":
            shortcut_specs.extend(
                [
                    (QtGui.QKeySequence("1"), lambda: self.select_annotation_option("Real")),
                    (QtGui.QKeySequence("2"), lambda: self.select_annotation_option("Spike")),
                    (QtGui.QKeySequence("3"), lambda: self.select_annotation_option("Artifact")),
                ]
            )

        for sequence, handler in shortcut_specs:
            self.shortcut_objects.append(QtWidgets.QShortcut(sequence, self, activated=handler))

    def init_waveform_plot(self):
        self.annotation_controller.create_waveform_plot()
        waveform_plot = self.annotation_controller.model.waveform_plot
        safe_connect_signal_slot(waveform_plot.hover_text_changed, self.on_hover_text_changed)
        safe_connect_signal_slot(waveform_plot.fft_window_selected, self.on_fft_window_selected)
        safe_connect_signal_slot(waveform_plot.history_state_changed, self.on_history_state_changed)

    def init_fft_plot(self):
        self.annotation_controller.create_fft_plot()

    def setInitialSize(self):
        fit_window_to_screen(
            self,
            default_width=self.ui_density.annotation_window_default_width,
            default_height=self.ui_density.annotation_window_default_height,
            min_width=self.ui_density.annotation_window_min_width,
            min_height=self.ui_density.annotation_window_min_height,
            width_ratio=0.9,
            height_ratio=0.9,
        )

    def _update_plot_size_targets(self):
        fft_plot = getattr(getattr(self.annotation_controller, "model", None), "fft_plot", None)
        if fft_plot is None:
            return
        target_height = max(156, min(240, int(self.height() * 0.24)))
        fft_plot.setMinimumHeight(target_height)
        fft_plot.updateGeometry()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_plot_size_targets()

    def on_hover_text_changed(self, text):
        self.hover_status_label.setText(text)

    def on_history_state_changed(self, can_go_back, can_go_forward):
        self.BackViewButton.setEnabled(can_go_back)
        self.ForwardViewButton.setEnabled(can_go_forward)

    def _update_accept_button_state(self, *_args):
        valid = self.EventDropdown_Box.currentText() in self.annotation_options.get(self.biomarker_type, [])
        self.Accept.setEnabled(valid)

    def get_current_interval(self):
        interval_text = self.IntervalDropdownBox.currentText()
        try:
            return float(interval_text.rstrip("s"))
        except (ValueError, AttributeError):
            return 1.0

    def get_current_freq_limit(self):
        min_freq = self.spinBox_minFreq.value()
        max_freq = self.spinBox_maxFreq.value()
        if min_freq >= max_freq:
            warnings.warn(
                "Invalid frequency range. Returning default values (10, 500).",
                UserWarning,
            )
            return 10, 500
        return min_freq, max_freq

    def _show_message(self, message, timeout=3000):
        self.statusbar.showMessage(message, timeout)

    def _navigate_to_event(self, event_tuple, reset_intervals=True):
        if not event_tuple:
            self._show_message("No matching event found.")
            return False
        channel, start, end = event_tuple
        if reset_intervals:
            default_interval = self.get_current_interval()
            self.annotation_controller.update_plots(
                start,
                end,
                channel,
                reset_intervals=True,
                default_interval=default_interval,
            )
        else:
            self.annotation_controller.update_plots(start, end, channel)
        self.update_infos()
        return True

    def _clear_fft_roi_internal(self, announce=False):
        self.annotation_controller.clear_fft_window()
        channel, start, end = self.annotation_controller.get_current_event()
        self.annotation_controller.model.fft_plot.plot(start, end, channel)
        if announce:
            self._show_message("Cleared FFT ROI")

    def plot_prev(self):
        self._clear_fft_roi_internal(announce=False)
        self._navigate_to_event(self.annotation_controller.get_previous_event())

    def plot_next(self):
        self._clear_fft_roi_internal(announce=False)
        self._navigate_to_event(self.annotation_controller.get_next_event())

    def plot_prev_unannotated(self):
        self._clear_fft_roi_internal(announce=False)
        event_tuple = self.annotation_controller.get_prev_unannotated_event()
        if event_tuple is None:
            self._show_message("All events are already annotated.")
            return
        self._navigate_to_event(event_tuple)

    def plot_next_unannotated(self):
        self._clear_fft_roi_internal(announce=False)
        event_tuple = self.annotation_controller.get_next_unannotated_event()
        if event_tuple is None:
            self._show_message("All events are already annotated.")
            return
        self._navigate_to_event(event_tuple)

    def plot_jump(self):
        selected_index = self.AnotationDropdownBox.currentIndex()
        if selected_index < 0:
            return
        self._clear_fft_roi_internal(announce=False)
        self._navigate_to_event(self.annotation_controller.get_jumped_event(selected_index))

    def _navigate_to_matching_event(self, direction):
        scope = self.get_current_prediction_scope()
        if scope == self.default_prediction_scope:
            self._show_message("Choose a prediction scope before using match navigation.")
            return
        event_features = getattr(self.backend, "event_features", None)
        if event_features is None:
            self._show_message("No events are available.")
            return
        self._clear_fft_roi_internal(announce=False)
        if direction > 0:
            event_tuple = event_features.get_next_matching(scope, unannotated_only=self.get_unannotated_only())
        else:
            event_tuple = event_features.get_prev_matching(scope, unannotated_only=self.get_unannotated_only())
        if event_tuple is None:
            qualifier = " unannotated" if self.get_unannotated_only() else ""
            self._show_message(f"No matching{qualifier} {scope.lower()} events found.")
            return
        self._navigate_to_event(event_tuple)

    def _set_annotation_selector(self, annotation):
        blocker = QtCore.QSignalBlocker(self.EventDropdown_Box)
        if annotation in self.annotation_options.get(self.biomarker_type, []):
            self.EventDropdown_Box.setCurrentText(annotation)
        else:
            self.EventDropdown_Box.setCurrentIndex(0)
        del blocker
        self._update_accept_button_state()

    def _update_progress_info(self):
        progress = self.annotation_controller.get_review_progress()
        counts = self.annotation_controller.get_annotation_counts()
        self.progress_textbox.setText(
            f"{progress['reviewed']} reviewed / {progress['total']} total | {progress['remaining']} remaining"
        )
        count_text = ", ".join(f"{label}: {count}" for label, count in counts.items())
        if hasattr(self.backend.event_features, "get_current_visible_position"):
            current_position = self.backend.event_features.get_current_visible_position()
        else:
            current_position = self.backend.event_features.index
        summary = f"Event {current_position + 1} of {progress['total']} | Reviewed {progress['reviewed']}/{progress['total']}"
        self.setWindowTitle(f"PyBrain {self.biomarker_type} Review Workspace - {summary}")
        self.progress_textbox.setToolTip(count_text)

    def _submit_frequency_range(self):
        if self.SetFreqLimit.isEnabled():
            self.SetFreqLimit.click()

    def get_current_prediction_scope(self):
        return self.PredictionScopeBox.currentText() or self.default_prediction_scope

    def get_unannotated_only(self):
        return self.UnannotatedOnlyCheckBox.isChecked()

    def refresh_prediction_scope_controls(self):
        event_features = getattr(self.backend, "event_features", None)
        options = (
            event_features.get_prediction_scope_options()
            if event_features is not None and hasattr(event_features, "get_prediction_scope_options")
            else [self.default_prediction_scope]
        )
        current_scope = self.get_current_prediction_scope()
        if current_scope not in options:
            current_scope = self.default_prediction_scope

        blocker = QtCore.QSignalBlocker(self.PredictionScopeBox)
        self.PredictionScopeBox.clear()
        self.PredictionScopeBox.addItems(options)
        self.PredictionScopeBox.setCurrentText(current_scope)
        del blocker

        self.PredictionScopeBox.setEnabled(len(options) > 1)
        self._update_prediction_scope_status()

    def _update_prediction_scope_status(self):
        event_features = getattr(self.backend, "event_features", None)
        if event_features is None or not hasattr(event_features, "get_matching_indexes"):
            self.match_summary_label.setText("Prediction filtering is unavailable.")
            self.UnannotatedOnlyCheckBox.setEnabled(False)
            self.PrevMatchButton.setEnabled(False)
            self.NextMatchButton.setEnabled(False)
            return

        options = event_features.get_prediction_scope_options()
        scope = self.get_current_prediction_scope()
        unannotated_only = self.get_unannotated_only()
        matches = event_features.get_matching_indexes(scope, unannotated_only=unannotated_only)
        self.UnannotatedOnlyCheckBox.setEnabled(len(options) > 1)

        if len(options) <= 1:
            self.match_summary_label.setText("Run classification to jump by prediction bucket.")
            self.PrevMatchButton.setEnabled(False)
            self.NextMatchButton.setEnabled(False)
            return

        if scope == self.default_prediction_scope:
            self.match_summary_label.setText("Select a prediction scope to jump among matching events.")
            self.PrevMatchButton.setEnabled(False)
            self.NextMatchButton.setEnabled(False)
            return

        scope_label = scope + (" + Unannotated" if unannotated_only else "")
        if len(matches) == 0:
            self.match_summary_label.setText(f"Scope: {scope_label} | No matching events")
            self.PrevMatchButton.setEnabled(False)
            self.NextMatchButton.setEnabled(False)
            return

        current_index = getattr(event_features, "index", 0)
        in_scope_positions = np.where(matches == current_index)[0]
        if len(in_scope_positions) > 0:
            current_pos = int(in_scope_positions[0]) + 1
            self.match_summary_label.setText(f"Scope: {scope_label} | {current_pos}/{len(matches)} matches")
        else:
            self.match_summary_label.setText(f"Scope: {scope_label} | {len(matches)} matches | current event outside scope")
        self.PrevMatchButton.setEnabled(True)
        self.NextMatchButton.setEnabled(True)

    def update_infos(self):
        info = self.backend.event_features.get_current_info()
        fs = self.backend.sample_freq
        self.channel_name_textbox.setText(info["channel_name"])
        self.channel_name_textbox.setToolTip(info["channel_name"])
        self.start_textbox.setText(f"{round(info['start_index'] / fs, 3)} s")
        self.end_textbox.setText(f"{round(info['end_index'] / fs, 3)} s")
        self.length_textbox.setText(f"{round((info['end_index'] - info['start_index']) / fs, 3)} s")

        current_display_index = info.get("display_index", 0)
        dropdown_blocker = QtCore.QSignalBlocker(self.AnotationDropdownBox)
        self.AnotationDropdownBox.setCurrentIndex(current_display_index)
        del dropdown_blocker

        prediction = info.get("prediction")
        annotation = info.get("annotation")
        overlap_tag = info.get("overlap_tag")
        model_text = prediction or "No model suggestion"
        if overlap_tag:
            model_text = f"{model_text} | {overlap_tag}"
        self.model_textbox.setText(model_text)
        self.model_textbox.setToolTip(model_text)
        reviewed_text = annotation or "Unannotated"
        if overlap_tag:
            reviewed_text = f"{reviewed_text} | {overlap_tag}"
        self.reviewed_textbox.setText(reviewed_text)
        self.reviewed_textbox.setToolTip(reviewed_text)
        self._set_annotation_selector(annotation)
        self._update_dropdown_item(current_display_index)
        self._update_progress_info()
        self._update_prediction_scope_status()

    def _update_dropdown_item(self, index):
        if 0 <= index < self.AnotationDropdownBox.count():
            self.AnotationDropdownBox.setItemText(index, self.backend.event_features.get_annotation_text(index))

    def update_button_clicked(self):
        selected_text = self.EventDropdown_Box.currentText()
        if selected_text not in self.annotation_options.get(self.biomarker_type, []):
            self._show_message("Choose a review label before saving.")
            return
        selected_index, item_text = self.annotation_controller.set_doctor_annotation(selected_text)
        self.AnotationDropdownBox.setItemText(selected_index, item_text)
        self.update_infos()
        self.plot_next()

    def clear_current_annotation(self):
        selected_index, item_text = self.annotation_controller.clear_current_annotation()
        self.AnotationDropdownBox.setItemText(selected_index, item_text)
        self.update_infos()
        self._show_message("Cleared the current review label.")

    def init_annotation_dropdown(self):
        self.AnotationDropdownBox.clear()
        event_count = (
            self.backend.event_features.get_num_biomarker()
            if hasattr(self.backend.event_features, "get_num_biomarker")
            else len(self.backend.event_features.annotated)
        )
        for i in range(event_count):
            self.AnotationDropdownBox.addItem(self.backend.event_features.get_annotation_text(i))
        safe_connect_signal_slot(self.AnotationDropdownBox.activated, self.plot_jump)

    def select_annotation_option(self, annotation):
        if annotation in self.annotation_options.get(self.biomarker_type, []):
            self.EventDropdown_Box.setCurrentText(annotation)
            self._show_message(f"Selected {annotation}")

    def on_prediction_scope_changed(self):
        self._update_prediction_scope_status()

    def plot_prev_match(self):
        self._navigate_to_matching_event(direction=-1)

    def plot_next_match(self):
        self._navigate_to_matching_event(direction=1)

    def update_plots(self):
        channel, start, end = self.annotation_controller.get_current_event()
        self.annotation_controller.update_plots(start, end, channel)

    def update_interval(self):
        interval = self.get_current_interval()
        self.annotation_controller.set_current_interval(interval)
        self.update_plots()
        self.on_history_state_changed(
            self.annotation_controller.can_go_back(),
            self.annotation_controller.can_go_forward(),
        )

    def update_frequency(self):
        min_freq, max_freq = self.get_current_freq_limit()
        self.annotation_controller.set_current_freq_limit(min_freq, max_freq)
        channel, start, end = self.annotation_controller.get_current_event()
        self.annotation_controller.update_plots(start, end, channel)
        self._show_message(f"Updated TF/FFT frequency range to {min_freq}-{max_freq} Hz")

    def reset_view(self):
        channel, start, end = self.annotation_controller.get_current_event()
        default_interval = self.get_current_interval()
        self.annotation_controller.model.waveform_plot.reset_intervals_to_default(default_interval)
        self.annotation_controller.reset_to_default_view()
        self.annotation_controller.model.fft_plot.plot(start, end, channel)

    def go_back_view(self):
        if self.annotation_controller.go_back_view():
            self._show_message("Moved back to the previous view.")

    def go_forward_view(self):
        if self.annotation_controller.go_forward_view():
            self._show_message("Moved forward to the next view.")

    def zoom_in(self):
        self.annotation_controller.model.waveform_plot.zoom_by_factor(1 / 1.5)

    def zoom_out(self):
        self.annotation_controller.model.waveform_plot.zoom_by_factor(1.5)

    def pan_left(self):
        self.annotation_controller.model.waveform_plot.pan_horizontal(-0.2)

    def pan_right(self):
        self.annotation_controller.model.waveform_plot.pan_horizontal(0.2)

    def pan_up(self):
        self.annotation_controller.model.waveform_plot.pan_vertical(0.2)

    def pan_down(self):
        self.annotation_controller.model.waveform_plot.pan_vertical(-0.2)

    def on_fft_window_selected(self, time_window):
        self.annotation_controller.set_fft_window(time_window)
        channel, start, end = self.annotation_controller.get_current_event()
        self.annotation_controller.model.fft_plot.plot(start, end, channel)
        if time_window is None:
            self._show_message("Using the default FFT window.")
        else:
            self._show_message(f"FFT ROI set to {time_window[0]:.4f}-{time_window[1]:.4f} s")

    def clear_fft_roi(self):
        self._clear_fft_roi_internal(announce=True)

    def export_snapshot(self):
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Review Snapshot",
            str(Path.cwd() / "review_snapshot.png"),
            "PNG Image (*.png)",
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".png"):
            file_path = f"{file_path}.png"
        self.grab().save(file_path)
        self._show_message(f"Saved snapshot to {file_path}")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = Annotation()
    mainWindow.show()
    sys.exit(app.exec_())
