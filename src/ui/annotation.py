import os
import sys
import warnings
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets, uic

from src.controllers import AnnotationController
from src.utils.utils_gui import apply_subwindow_theme, safe_connect_signal_slot, set_accent_button, style_value_badge

ROOT_DIR = Path(__file__).parent


class Annotation(QtWidgets.QMainWindow):
    def __init__(self, backend=None, main_window=None, close_signal=None, biomarker_type="HFO"):
        super().__init__(main_window)
        self.backend = backend
        self.biomarker_type = getattr(backend, "biomarker_type", biomarker_type)
        self.ui = uic.loadUi(os.path.join(ROOT_DIR, "annotation.ui"), self)
        self.annotation_controller = AnnotationController(self, backend)
        self.threadpool = QtCore.QThreadPool()
        self.close_signal = close_signal
        self.shortcut_objects = []

        self.dropdown_placeholder = "--- Event Type ---"
        self.annotation_options = {
            "HFO": ["Pathological", "Physiological", "Artifact"],
            "Spindle": ["Real", "Spike", "Artifact"],
        }

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
        self.update_infos()
        self.on_history_state_changed(
            self.annotation_controller.can_go_back(),
            self.annotation_controller.can_go_forward(),
        )
        self.setInitialSize()
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
        if hasattr(self, "Accept"):
            self.Accept.setText("Save and Next")
        if hasattr(self, "actionStart_Over"):
            self.actionStart_Over.setText("Clear Current Annotation")

    def _build_navigation_controls(self):
        navigation_layout = getattr(self, "horizontalLayout", None)
        if navigation_layout is None:
            return

        self.PrevPendingButton = QtWidgets.QPushButton("Prev Pending", self)
        self.NextPendingButton = QtWidgets.QPushButton("Next Pending", self)
        self.ClearAnnotationButton = QtWidgets.QPushButton("Clear Label", self)
        for button, tooltip in (
            (self.PrevPendingButton, "Jump to the previous unannotated event"),
            (self.NextPendingButton, "Jump to the next unannotated event"),
            (self.ClearAnnotationButton, "Clear the current review label"),
        ):
            button.setToolTip(tooltip)
            navigation_layout.addWidget(button)

    def _build_view_controls(self):
        controls_layout = getattr(self, "viewControlsHLayout", None)
        groupbox = getattr(self, "viewControlsGroupBox", None)
        if controls_layout is None:
            return

        if groupbox is not None:
            groupbox.setMaximumHeight(72)

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

    def _build_info_labels(self):
        info_layout = getattr(self, "gridLayout_3", None)
        if info_layout is None:
            return

        review_layout = QtWidgets.QHBoxLayout()
        review_label = QtWidgets.QLabel("Reviewed:", self.groupBox)
        self.reviewed_textbox = QtWidgets.QLabel(self.groupBox)
        self.reviewed_textbox.setStyleSheet("background-color: rgb(235, 235, 235);")
        review_layout.addWidget(review_label)
        review_layout.addWidget(self.reviewed_textbox)
        info_layout.addLayout(review_layout, 4, 0, 1, 1)

        progress_layout = QtWidgets.QHBoxLayout()
        progress_label = QtWidgets.QLabel("Progress:", self.groupBox)
        self.progress_textbox = QtWidgets.QLabel(self.groupBox)
        self.progress_textbox.setStyleSheet("background-color: rgb(235, 235, 235);")
        progress_layout.addWidget(progress_label)
        progress_layout.addWidget(self.progress_textbox)
        info_layout.addLayout(progress_layout, 5, 0, 1, 1)

    def _wire_signals(self):
        if self.close_signal is not None:
            safe_connect_signal_slot(self.close_signal, self.close)

        safe_connect_signal_slot(self.PreviousButton.clicked, self.plot_prev)
        safe_connect_signal_slot(self.NextButton.clicked, self.plot_next)
        safe_connect_signal_slot(self.Accept.clicked, self.update_button_clicked)
        safe_connect_signal_slot(self.SetFreqLimit.clicked, self.update_frequency)
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
        for label, width in (
            (self.model_textbox, 160),
            (self.channel_name_textbox, 160),
            (self.start_textbox, 96),
            (self.end_textbox, 96),
            (self.length_textbox, 96),
            (self.reviewed_textbox, 160),
            (self.progress_textbox, 280),
        ):
            style_value_badge(label, min_width=width)
        self.progress_textbox.setWordWrap(True)
        self.hint_status_label.setProperty("mutedText", True)
        set_accent_button(self.Accept)
        apply_subwindow_theme(
            self,
            extra_stylesheet="""
                QMainWindow { background: #f5f7fb; }
                QStatusBar { padding-left: 8px; }
            """,
        )

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
        screen = QtWidgets.QApplication.primaryScreen()
        rect = screen.availableGeometry()
        width = int(rect.width() * 0.78)
        height = int(rect.height() * 0.84)
        self.resize(width, height)
        self.setMinimumSize(QtCore.QSize(int(width * 0.7), int(height * 0.7)))

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
        summary = f"Event {self.backend.event_features.index + 1} of {progress['total']} | Reviewed {progress['reviewed']}/{progress['total']}"
        self.setWindowTitle(f"PyBrain {self.biomarker_type} Review Workspace - {summary}")
        self.progress_textbox.setToolTip(count_text)

    def update_infos(self):
        info = self.backend.event_features.get_current_info()
        fs = self.backend.sample_freq
        self.channel_name_textbox.setText(info["channel_name"])
        self.start_textbox.setText(f"{round(info['start_index'] / fs, 3)} s")
        self.end_textbox.setText(f"{round(info['end_index'] / fs, 3)} s")
        self.length_textbox.setText(f"{round((info['end_index'] - info['start_index']) / fs, 3)} s")

        dropdown_blocker = QtCore.QSignalBlocker(self.AnotationDropdownBox)
        self.AnotationDropdownBox.setCurrentIndex(self.backend.event_features.index)
        del dropdown_blocker

        prediction = info.get("prediction")
        annotation = info.get("annotation")
        self.model_textbox.setText(prediction or "No model suggestion")
        self.reviewed_textbox.setText(annotation or "Unannotated")
        self._set_annotation_selector(annotation)
        self._update_dropdown_item(self.backend.event_features.index)
        self._update_progress_info()

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
        for i in range(len(self.backend.event_features.annotated)):
            self.AnotationDropdownBox.addItem(self.backend.event_features.get_annotation_text(i))
        safe_connect_signal_slot(self.AnotationDropdownBox.activated, self.plot_jump)

    def select_annotation_option(self, annotation):
        if annotation in self.annotation_options.get(self.biomarker_type, []):
            self.EventDropdown_Box.setCurrentText(annotation)
            self._show_message(f"Selected {annotation}")

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
