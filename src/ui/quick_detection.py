from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5 import uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import sys
import os
import math
from pathlib import Path
from src.hfo_app import HFO_App
from src.param.param_classifier import ParamClassifier
from src.param.param_detector import ParamDetector, ParamSTE, ParamMNI, ParamHIL
from src.param.param_filter import ParamFilter
from src.ui.ui_tokens import resolve_ui_density
from src.utils.utils_gui import *

import multiprocessing as mp
try:
    import torch
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None

ROOT_DIR = Path(__file__).parent


class HFOQuickDetector(QtWidgets.QDialog):
    def __init__(self, backend=None, main_window=None, close_signal = None):
        super(HFOQuickDetector, self).__init__()
        self.ui_density = resolve_ui_density(self.screen())
        self.ui = uic.loadUi(os.path.join(ROOT_DIR, 'quick_detection.ui'), self)
        self.setWindowTitle("HFO Quick Detector")
        self.setWindowIcon(QtGui.QIcon(os.path.join(ROOT_DIR, 'images/icon1.png')))
        self.filename = None
        self.detector = ""
        self.threadpool = QThreadPool()
        self._run_result = None
        self._run_failed = False
        self.running = False
        safe_connect_signal_slot(self.detectionTypeComboBox.currentIndexChanged['int'],
                                 lambda: self.update_detector_tab(self.detectionTypeComboBox.currentText()))
        self.detectionTypeComboBox.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(self)
        safe_connect_signal_slot(self.qd_loadEDF_button.clicked, self.open_file)
        if backend is None:
            self.backend = HFO_App()
        else:
            self.backend = backend
        self.init_default_filter_input_params()
        self.init_default_mni_input_params()
        self.init_default_ste_input_params()
        self.init_default_hil_input_params()
        safe_connect_signal_slot(self.qd_choose_artifact_model_button.clicked,
                                 lambda: self.choose_model_file("artifact"))
        safe_connect_signal_slot(self.qd_choose_spike_model_button.clicked,
                                 lambda: self.choose_model_file("spike"))
        safe_connect_signal_slot(self.qd_choose_ehfo_model_button.clicked,
                                 lambda: self.choose_model_file("ehfo"))

        safe_connect_signal_slot(self.run_button.clicked, self.run)
        self.run_button.setEnabled(False)
        safe_connect_signal_slot(self.qd_use_classifier_checkbox.toggled, self._update_classifier_controls_enabled)

        #set n_jobs min and max
        self.n_jobs_spinbox.setMinimum(1)
        self.n_jobs_spinbox.setMaximum(mp.cpu_count())

        #set default n_jobs
        self.n_jobs_spinbox.setValue(self.backend.n_jobs)

        self.main_window = main_window

        #classifier default buttons
        safe_connect_signal_slot(self.default_cpu_button.clicked, self.set_classifier_param_cpu_default)
        safe_connect_signal_slot(self.default_gpu_button.clicked, self.set_classifier_param_gpu_default)
        if torch is None or not torch.cuda.is_available():
            self.default_gpu_button.setEnabled(False)
        if self.backend.get_classifier_param() is None:
            self.backend.set_default_cpu_classifier()
        self.classifier_source_preference = getattr(self.backend.get_classifier_param(), "source_preference", "auto")
        self.set_classifier_param_display()
        self.qd_npz_checkbox.setChecked(True)
        self._update_classifier_controls_enabled(self.qd_use_classifier_checkbox.isChecked())
        safe_connect_signal_slot(self.qd_excel_checkbox.toggled, self._refresh_run_setup_feedback)
        safe_connect_signal_slot(self.qd_npz_checkbox.toggled, self._refresh_run_setup_feedback)

        safe_connect_signal_slot(self.cancel_button.clicked, self.close)

        self.close_signal = close_signal
        if self.close_signal is not None:
            safe_connect_signal_slot(self.close_signal, self.close)

        # Disable the full configuration area while the worker is active.
        self.controls_to_disable = [self.scrollArea]
        self._apply_dialog_theme()
        self._restore_detector_combo_palette()
        self._configure_input_conventions()
        self._refresh_run_setup_feedback()

    def _apply_dialog_theme(self):
        self.ui_density = resolve_ui_density(self.screen())
        fit_window_to_screen(
            self,
            default_width=900,
            default_height=780,
            min_width=780,
            min_height=660,
            width_ratio=0.78,
            height_ratio=0.82,
        )
        self.qd_filename.setWordWrap(True)
        for label, width in (
            (self.qd_filename, 260),
            (self.qd_sampfreq, 110),
            (self.qd_numchannels, 110),
            (self.qd_length, 110),
        ):
            style_value_badge(label, min_width=width, selectable=True)
        self.qd_filename.setText("No recording loaded")
        self.qd_sampfreq.setText("--")
        self.qd_numchannels.setText("--")
        self.qd_length.setText("--")
        self._compact_dialog_layouts()
        set_accent_button(self.run_button)
        cancel_button = self.cancel_button.button(QtWidgets.QDialogButtonBox.Cancel)
        if cancel_button is not None:
            cancel_button.setText("Close")
        apply_subwindow_theme(self)
        self._apply_widget_level_density()

    def _compact_dialog_layouts(self):
        density = self.ui_density

        root_layout = self.layout()
        if root_layout is not None:
            root_layout.setContentsMargins(8, 8, 8, 8)
            root_layout.setSpacing(6)

        scroll_layout = self.scrollAreaWidgetContents.layout()
        if scroll_layout is not None:
            scroll_layout.setContentsMargins(0, 0, 0, 0)
            scroll_layout.setSpacing(6)

        main_layout = self.gridLayout_9
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setHorizontalSpacing(6)
        main_layout.setVerticalSpacing(6)

        top_layout = self.gridLayout_8
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setHorizontalSpacing(8)
        top_layout.setVerticalSpacing(4)
        top_layout.setColumnStretch(0, 0)
        top_layout.setColumnStretch(1, 1)

        control_page = self.findChild(QtWidgets.QWidget, "page_17")
        if control_page is not None and control_page.layout() is not None:
            control_page.setMaximumWidth(220)
            control_page.layout().setContentsMargins(0, 0, 0, 0)
            control_page.layout().setHorizontalSpacing(4)
            control_page.layout().setVerticalSpacing(4)

        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setSpacing(6)

        self._tighten_groupbox(self.qd_edfInfo, title="Recording")
        self._tighten_groupbox(self.qd_filters, title="Filter")
        self._tighten_groupbox(self.qd_MNI_detector, title="MNI")
        self._tighten_groupbox(self.qd_STE_detector, title="STE")
        self._tighten_groupbox(self.qd_HIL_detector, title="HIL")
        self._tighten_groupbox(self.classifier_groupbox_4, title="Classifier")
        self._tighten_groupbox(self.qd_saveAs, title="Export")

        edf_layout = self.qd_edfInfo.layout()
        if isinstance(edf_layout, QtWidgets.QFormLayout):
            edf_layout.setContentsMargins(8, 8, 8, 8)
            edf_layout.setHorizontalSpacing(8)
            edf_layout.setVerticalSpacing(6)
            edf_layout.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            edf_layout.setFormAlignment(QtCore.Qt.AlignTop)

        self._compact_header_controls()
        self._compact_filter_group()
        self._compact_detector_grids()
        self._compact_classifier_group()
        self._compact_export_group()

        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        for stacked in (self.stackedWidget_2, self.stackedWidget, self.stackedWidget_3, self.stackedWidget_4):
            stacked.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self._rebuild_run_setup_panel()

    def _tighten_groupbox(self, groupbox, *, title=None):
        if groupbox is None:
            return
        if title is not None:
            groupbox.setTitle(title)
        layout = groupbox.layout()
        if layout is not None:
            layout.setContentsMargins(8, 8, 8, 8)
            if hasattr(layout, "setHorizontalSpacing"):
                layout.setHorizontalSpacing(6)
            if hasattr(layout, "setVerticalSpacing"):
                layout.setVerticalSpacing(4)

    def _compact_header_controls(self):
        self.qd_loadEDF_button.setText("Load Recording")
        self.qd_loadEDF_button.setMinimumHeight(self.ui_density.compact_button_height)
        self.qd_loadEDF_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.label_8.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.label_8.setProperty("fieldLabel", True)
        self.n_jobs_spinbox.setMaximumWidth(64)
        self.n_jobs_spinbox.setMinimumHeight(self.ui_density.compact_input_height)
        self.detectionTypeComboBox.setMinimumHeight(self.ui_density.compact_input_height)
        self.detectionTypeComboBox.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.run_button.setText("Run Detection")
        self.run_button.setMinimumHeight(self.ui_density.compact_button_height)

    def _restore_detector_combo_palette(self):
        combo = getattr(self, "detectionTypeComboBox", None)
        if combo is None:
            return
        combo.setPalette(QtWidgets.QApplication.palette(combo))
        view = combo.view()
        if view is not None:
            view.setPalette(QtWidgets.QApplication.palette(view))

    def _rebuild_run_setup_panel(self):
        if getattr(self, "_run_setup_panel_built", False):
            return

        setup_host = getattr(self, "page_17", None)
        setup_layout = setup_host.layout() if setup_host is not None else None
        if setup_host is None or setup_layout is None:
            return

        while setup_layout.count():
            item = setup_layout.takeAt(0)
            child = item.widget()
            if child is not None:
                child.hide()
        setup_host.setMaximumWidth(240)

        setup_card = QtWidgets.QFrame(setup_host)
        setup_card.setProperty("surfaceCard", True)
        setup_card_layout = QtWidgets.QVBoxLayout(setup_card)
        setup_card_layout.setContentsMargins(10, 10, 10, 10)
        setup_card_layout.setSpacing(8)

        title = QtWidgets.QLabel("Run Setup", setup_card)
        title.setProperty("sectionTitle", True)
        subtitle = QtWidgets.QLabel("Load a recording, choose the detector, then launch a compact run from here.", setup_card)
        subtitle.setProperty("helperText", True)
        subtitle.setWordWrap(True)
        setup_card_layout.addWidget(title)
        setup_card_layout.addWidget(subtitle)

        self.qd_loadEDF_button.setParent(setup_card)
        self.qd_loadEDF_button.show()
        setup_card_layout.addWidget(self.qd_loadEDF_button)

        detector_row = QtWidgets.QFrame(setup_card)
        detector_layout = QtWidgets.QVBoxLayout(detector_row)
        detector_layout.setContentsMargins(0, 0, 0, 0)
        detector_layout.setSpacing(4)
        detector_label = QtWidgets.QLabel("Detector", detector_row)
        detector_label.setProperty("fieldLabel", True)
        self.detectionTypeComboBox.setParent(detector_row)
        self.detectionTypeComboBox.show()
        detector_layout.addWidget(detector_label)
        detector_layout.addWidget(self.detectionTypeComboBox)
        setup_card_layout.addWidget(detector_row)

        jobs_row = QtWidgets.QFrame(setup_card)
        jobs_layout = QtWidgets.QHBoxLayout(jobs_row)
        jobs_layout.setContentsMargins(0, 0, 0, 0)
        jobs_layout.setSpacing(6)
        self.label_8.setParent(jobs_row)
        self.n_jobs_spinbox.setParent(jobs_row)
        self.label_8.show()
        self.n_jobs_spinbox.show()
        self.n_jobs_spinbox.setMaximumWidth(72)
        jobs_layout.addWidget(self.label_8)
        jobs_layout.addWidget(self.n_jobs_spinbox, 0, QtCore.Qt.AlignLeft)
        jobs_layout.addStretch(1)
        setup_card_layout.addWidget(jobs_row)

        self.run_button.setParent(setup_card)
        self.run_button.show()
        self.run_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        setup_card_layout.addWidget(self.run_button)

        status_card = QtWidgets.QFrame(setup_card)
        status_card.setProperty("softCard", True)
        status_layout = QtWidgets.QVBoxLayout(status_card)
        status_layout.setContentsMargins(8, 8, 8, 8)
        status_layout.setSpacing(4)
        self.run_status_value_label = QtWidgets.QLabel("Waiting for recording", status_card)
        self.run_status_value_label.setProperty("sectionTitle", True)
        self.run_status_detail_label = QtWidgets.QLabel(
            "Load a recording to enable quick detection and export results next to the source file.",
            status_card,
        )
        self.run_status_detail_label.setProperty("helperText", True)
        self.run_status_detail_label.setWordWrap(True)
        status_layout.addWidget(self.run_status_value_label)
        status_layout.addWidget(self.run_status_detail_label)
        setup_card_layout.addWidget(status_card)
        setup_card_layout.addStretch(1)

        setup_layout.addWidget(setup_card, 0, 0)

        button_layout = getattr(self, "horizontalLayout_2", None)
        cancel_button = self.cancel_button.button(QtWidgets.QDialogButtonBox.Cancel)
        if button_layout is not None and cancel_button is not None:
            button_layout.removeWidget(self.run_button)
            cancel_button.setText("Close")

        self.run_setup_card = setup_card
        self._run_setup_panel_built = True

    def _selected_detector_name(self):
        for candidate in (
            getattr(self, "detector", ""),
            getattr(self.detectionTypeComboBox, "currentText", lambda: "")(),
        ):
            normalized = str(candidate or "").strip().upper()
            if normalized in {"MNI", "STE", "HIL"}:
                return normalized
        return ""

    def _set_run_feedback(self, headline, detail):
        if hasattr(self, "run_status_value_label"):
            self.run_status_value_label.setText(headline)
            self.run_status_value_label.setToolTip(detail or headline)
        if hasattr(self, "run_status_detail_label"):
            self.run_status_detail_label.setText(detail)
            self.run_status_detail_label.setToolTip(detail or headline)

    def _clear_run_feedback_state(self):
        self._run_result = None
        self._run_failed = False
        if not self.running:
            self.setWindowTitle("HFO Quick Detector")

    def _parse_float_input(self, raw_value, field_label, *, positive=False, non_negative=False):
        text = str(raw_value).strip()
        if not text:
            raise ValueError(f"{field_label} is required.")
        try:
            value = float(text)
        except (TypeError, ValueError):
            raise ValueError(f"{field_label} must be a number.")
        if not math.isfinite(value):
            raise ValueError(f"{field_label} must be a finite number.")
        if positive and value <= 0:
            raise ValueError(f"{field_label} must be greater than 0.")
        if non_negative and value < 0:
            raise ValueError(f"{field_label} must be non-negative.")
        return value

    def _parse_int_input(self, raw_value, field_label, *, positive=False, non_negative=False):
        text = str(raw_value).strip()
        if not text:
            raise ValueError(f"{field_label} is required.")
        try:
            value = int(text)
        except (TypeError, ValueError):
            raise ValueError(f"{field_label} must be an integer.")
        if positive and value <= 0:
            raise ValueError(f"{field_label} must be greater than 0.")
        if non_negative and value < 0:
            raise ValueError(f"{field_label} must be non-negative.")
        return value

    def _refresh_run_setup_feedback(self, *_args):
        detector_name = self._selected_detector_name() or "Detection"
        if self.running:
            self._set_run_feedback(
                f"Running {detector_name}",
                "Filtering, detecting, and exporting the current recording. This panel will update when the run finishes.",
            )
            return

        if self._run_failed:
            self._set_run_feedback(
                f"{detector_name} failed",
                "Quick detection stopped before export. Review the error details, adjust the settings, and try again.",
            )
            return

        if self._run_result:
            event_features = getattr(self.backend, "event_features", None)
            event_count = event_features.get_num_biomarker() if event_features is not None else 0
            exports = [Path(path).name for path in self._run_result.values() if path]
            export_text = ", ".join(exports) if exports else "No files exported"
            self._set_run_feedback(
                f"{detector_name} complete",
                f"{event_count} events ready. Saved: {export_text}.",
            )
            return

        if getattr(self, "fname", None):
            recording_name = self.filename or Path(self.fname).name
            selected_outputs = []
            if getattr(self, "qd_excel_checkbox", None) is not None and self.qd_excel_checkbox.isChecked():
                selected_outputs.append("Excel workbook")
            if getattr(self, "qd_npz_checkbox", None) is not None and self.qd_npz_checkbox.isChecked():
                selected_outputs.append("session file")
            output_text = " and ".join(selected_outputs) if selected_outputs else "selected outputs"
            self._set_run_feedback(
                f"Ready for {detector_name}",
                f"Run {detector_name} on {recording_name}. Detector-specific filenames will be used for each {output_text}.",
            )
            return

        self._set_run_feedback(
            "Waiting for recording",
            "Load a recording to enable quick detection and export results next to the source file.",
        )

    def _apply_label_density(self, labels):
        for label in labels:
            label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            label.setProperty("fieldLabel", True)

    def _apply_unit_density(self, labels):
        for label in labels:
            label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            label.setProperty("fieldUnit", True)

    def _apply_line_edit_density(self, widgets, *, max_width=150):
        for widget in widgets:
            widget.setMaximumWidth(max_width)
            widget.setMinimumHeight(self.ui_density.compact_input_height)
            widget.setAlignment(QtCore.Qt.AlignCenter)

    def _compact_filter_group(self):
        layout = self.qd_filters.layout()
        if isinstance(layout, QtWidgets.QGridLayout):
            layout.setColumnMinimumWidth(3, 10)
            layout.setColumnStretch(2, 1)
            layout.setColumnStretch(6, 1)
        self._apply_label_density([self.label_105, self.label_106, self.label_109, self.label_110])
        self._apply_line_edit_density(
            [self.qd_fp_input, self.qd_fs_input, self.qd_rp_input, self.qd_rs_input],
            max_width=134,
        )

    def _compact_detector_grids(self):
        mni_layout = self.qd_MNI_detector.layout()
        if isinstance(mni_layout, QtWidgets.QGridLayout):
            mni_layout.setColumnStretch(1, 1)
            mni_layout.setColumnStretch(4, 1)
        self._apply_label_density([
            self.label_29, self.label_30, self.label_33, self.label_34, self.label_32,
            self.label_37, self.label_71, self.label_38, self.label_31, self.label_36,
        ])
        self._apply_unit_density([self.label_40, self.label_48, self.label_39, self.label_35, self.label_42, self.label_47, self.label_41])
        self._apply_line_edit_density([
            self.qd_mni_epoch_time_input, self.qd_mni_epoch_chf_input, self.qd_mni_chf_percentage_input,
            self.qd_mni_min_window_input, self.qd_mni_min_gap_time_input, self.qd_mni_baseline_window_input,
            self.qd_mni_baseline_shift_input, self.qd_mni_baseline_threshold_input,
            self.qd_mni_threshold_percentage_input, self.qd_mni_baseline_time_input,
        ], max_width=118)

        ste_layout = self.qd_STE_detector.layout()
        if isinstance(ste_layout, QtWidgets.QGridLayout):
            ste_layout.setColumnStretch(1, 1)
            ste_layout.setColumnStretch(4, 1)
        self._apply_label_density([
            self.label_103, self.label_104, self.label_97, self.label_100, self.label_102, self.label_107, self.label_101,
        ])
        self._apply_unit_density([self.label_108, self.label_96, self.label_95, self.label_111])
        self._apply_line_edit_density([
            self.qd_ste_min_oscillation_input, self.qd_ste_rms_threshold_input, self.qd_ste_rms_window_input,
            self.qd_ste_epoch_length_input, self.qd_ste_min_window_input, self.qd_ste_peak_threshold_input,
            self.qd_ste_min_gap_input,
        ], max_width=118)

        hil_layout = self.qd_HIL_detector.layout()
        if isinstance(hil_layout, QtWidgets.QGridLayout):
            hil_layout.setColumnStretch(1, 1)
            hil_layout.setColumnStretch(4, 1)
        self._apply_label_density([
            self.label_1031, self.label_1041, self.label_1071, self.label_971, self.label_1011, self.label_1001,
        ])
        self._apply_unit_density([self.label_1081, self.label_9, self.label_961, self.label_951, self.label_1111])
        self._apply_line_edit_density([
            self.qd_hil_sample_freq_input, self.qd_hil_pass_band_input, self.qd_hil_stop_band_input,
            self.qd_hil_min_window_input, self.qd_hil_epoch_time_input, self.qd_hil_sd_threshold_input,
        ], max_width=118)

    def _compact_classifier_group(self):
        self.qd_use_classifier_checkbox.setText("Run classifier")
        self.default_cpu_button.setText("Hugging Face CPU")
        self.default_gpu_button.setText("Hugging Face GPU")
        self.qd_use_spikes_checkbox.setText("spkHFO")
        self.qd_use_ehfo_checkbox.setText("eHFO")

        self.gridLayout_11.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_11.setHorizontalSpacing(6)
        self.gridLayout_11.setVerticalSpacing(4)
        self.gridLayout_11.setColumnStretch(1, 1)
        self.gridLayout_11.setColumnStretch(2, 0)

        self._apply_label_density([self.label_69, self.label_46, self.label_10, self.label_5, self.label_6, self.label_7, self.label_70, self.label_27])
        self._apply_line_edit_density([
            self.qd_ignore_sec_before_input, self.qd_ignore_sec_after_input,
            self.qd_classifier_device_input, self.qd_classifier_batch_size_input,
        ], max_width=82)

        for layout in (self.horizontalLayout, self.horizontalLayout_3, self.horizontalLayout_4, self.horizontalLayout_6):
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)

        for button in (
            self.qd_choose_artifact_model_button,
            self.qd_choose_spike_model_button,
            self.qd_choose_ehfo_model_button,
            self.default_cpu_button,
            self.default_gpu_button,
        ):
            button.setMinimumWidth(86)
            button.setMinimumHeight(self.ui_density.compact_button_height)

        for display in (
            self.qd_classifier_artifact_filename_display,
            self.qd_classifier_spike_filename_display,
            self.qd_classifier_ehfo_filename_display,
        ):
            display.setMinimumHeight(self.ui_density.compact_input_height)
            display.setReadOnly(True)
            display.setProperty("readOnlyField", True)
            display.setPlaceholderText("No model selected")
            display.installEventFilter(self)
            self._set_model_path_display(display, display.property("fullPath") or "")

    def _update_classifier_controls_enabled(self, enabled):
        classifier_controls = (
            self.default_cpu_button,
            self.default_gpu_button,
            self.qd_use_spikes_checkbox,
            self.qd_use_ehfo_checkbox,
            self.qd_choose_artifact_model_button,
            self.qd_choose_spike_model_button,
            self.qd_choose_ehfo_model_button,
            self.qd_classifier_artifact_filename_display,
            self.qd_classifier_spike_filename_display,
            self.qd_classifier_ehfo_filename_display,
            self.qd_ignore_sec_before_input,
            self.qd_ignore_sec_after_input,
            self.qd_classifier_device_input,
            self.qd_classifier_batch_size_input,
        )
        for widget in classifier_controls:
            widget.setEnabled(bool(enabled))

    def _compact_export_group(self):
        layout = self.qd_saveAs.layout()
        if layout is not None:
            layout.setContentsMargins(8, 6, 8, 6)
            if hasattr(layout, "setVerticalSpacing"):
                layout.setVerticalSpacing(4)
        self.qd_npz_checkbox.setText("Session (.npz)")
        self.qd_excel_checkbox.setText("Workbook (.xlsx)")

    def _apply_widget_level_density(self):
        density = self.ui_density
        self.run_button.setMinimumHeight(density.compact_button_height)
        cancel_button = self.cancel_button.button(QtWidgets.QDialogButtonBox.Cancel)
        if cancel_button is not None:
            cancel_button.setMinimumHeight(density.compact_button_height)

    def _configure_numeric_line_edit(
        self,
        widget,
        *,
        integer=False,
        minimum=None,
        maximum=None,
        decimals=6,
        placeholder=None,
        tooltip=None,
    ):
        if widget is None or not hasattr(widget, "setValidator"):
            return
        if integer:
            lower = int(0 if minimum is None else minimum)
            upper = int(1_000_000 if maximum is None else maximum)
            validator = QtGui.QIntValidator(lower, upper, widget)
        else:
            lower = float(0.0 if minimum is None else minimum)
            upper = float(1_000_000_000.0 if maximum is None else maximum)
            validator = QtGui.QDoubleValidator(lower, upper, int(decimals), widget)
            validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        widget.setValidator(validator)
        if placeholder is not None and hasattr(widget, "setPlaceholderText"):
            widget.setPlaceholderText(str(placeholder))
        if tooltip:
            widget.setToolTip(str(tooltip))
            widget.setStatusTip(str(tooltip))

    def _recording_sample_frequency(self):
        try:
            sample_freq = float(getattr(self.backend, "sample_freq", 0.0) or 0.0)
        except (TypeError, ValueError):
            sample_freq = 0.0
        return sample_freq if math.isfinite(sample_freq) and sample_freq > 0 else 0.0

    def _recording_nyquist_frequency(self):
        sample_freq = self._recording_sample_frequency()
        return sample_freq / 2.0 if sample_freq > 0 else 0.0

    def _filter_stop_band_limit(self):
        nyquist = self._recording_nyquist_frequency()
        if nyquist <= 0:
            return 0.0
        filter_space = 0.5
        return max(0.0, min((nyquist * 0.99) - 0.1, nyquist - filter_space - 0.1))

    def _format_numeric_text(self, value):
        if value is None:
            return ""
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not math.isfinite(numeric_value):
            return ""
        if numeric_value.is_integer():
            return str(int(numeric_value))
        return f"{numeric_value:g}"

    def _recommended_filter_param(self):
        default_param = ParamFilter()
        sample_freq = self._recording_sample_frequency()
        if sample_freq <= 0:
            return default_param

        max_stop_band = max(1.1, self._filter_stop_band_limit())
        stop_band = min(float(default_param.fs), max_stop_band)
        preferred_gap = max(5.0, stop_band * 0.15)
        pass_band = min(float(default_param.fp), stop_band - preferred_gap)
        if pass_band <= 0:
            pass_band = max(1.0, stop_band * 0.8)
        pass_band = min(pass_band, stop_band - 0.1)
        pass_band = max(1.0, pass_band)
        if pass_band >= stop_band:
            pass_band = max(1.0, stop_band - 0.1)

        default_param.fp = round(pass_band, 3)
        default_param.fs = round(stop_band, 3)
        default_param.sample_freq = sample_freq
        return default_param

    def _sync_filter_input_constraints(self):
        nyquist = self._recording_nyquist_frequency()
        maximum_frequency = self._filter_stop_band_limit() or None
        tooltip = "Enter a positive frequency."
        if nyquist > 0 and maximum_frequency is not None:
            tooltip = (
                f"Enter a positive frequency below {maximum_frequency:.2f} Hz "
                f"(Nyquist {nyquist:.2f} Hz)."
            )
        for widget in (self.qd_fp_input, self.qd_fs_input):
            self._configure_numeric_line_edit(
                widget,
                minimum=0.0,
                maximum=maximum_frequency,
                tooltip=tooltip,
            )
        self._configure_numeric_line_edit(self.qd_rp_input, minimum=0.0)
        self._configure_numeric_line_edit(self.qd_rs_input, minimum=0.0)

    def _sync_filter_inputs_from_param(self, filter_param=None):
        if filter_param is None:
            filter_param = self._recommended_filter_param()
        self.qd_fp_input.setText(self._format_numeric_text(getattr(filter_param, "fp", "")))
        self.qd_fs_input.setText(self._format_numeric_text(getattr(filter_param, "fs", "")))
        self.qd_rp_input.setText(self._format_numeric_text(getattr(filter_param, "rp", "")))
        self.qd_rs_input.setText(self._format_numeric_text(getattr(filter_param, "rs", "")))
        self._sync_filter_input_constraints()

    def _build_filter_param_from_inputs(self):
        fp = self._parse_float_input(self.qd_fp_input.text(), "Filter pass band", positive=True)
        fs = self._parse_float_input(self.qd_fs_input.text(), "Filter stop band", positive=True)
        rp = self._parse_float_input(self.qd_rp_input.text(), "Filter ripple", positive=True)
        rs = self._parse_float_input(self.qd_rs_input.text(), "Filter attenuation", positive=True)
        stop_band_limit = self._filter_stop_band_limit()
        if stop_band_limit > 0 and fs > stop_band_limit:
            raise ValueError(f"Filter stop band must stay below {stop_band_limit:.2f} Hz for this recording.")
        if fp >= fs:
            raise ValueError("Filter pass band must be lower than the stop band.")
        return ParamFilter().from_dict(
            {
                "fp": fp,
                "fs": fs,
                "rp": rp,
                "rs": rs,
                "sample_freq": self._recording_sample_frequency() or None,
            }
        )

    def _submit_run_from_fields(self):
        if self.run_button.isEnabled():
            self.run_button.click()

    def _configure_default_buttons(self):
        self.run_button.setDefault(True)
        self.run_button.setAutoDefault(True)
        cancel_button = self.cancel_button.button(QtWidgets.QDialogButtonBox.Cancel)
        if cancel_button is not None:
            cancel_button.setDefault(False)
            cancel_button.setAutoDefault(False)

    def _configure_input_conventions(self):
        numeric_fields = (
            self.qd_mni_epoch_time_input,
            self.qd_mni_epoch_chf_input,
            self.qd_mni_chf_percentage_input,
            self.qd_mni_min_window_input,
            self.qd_mni_min_gap_time_input,
            self.qd_mni_threshold_percentage_input,
            self.qd_mni_baseline_window_input,
            self.qd_mni_baseline_shift_input,
            self.qd_mni_baseline_threshold_input,
            self.qd_mni_baseline_time_input,
            self.qd_ste_rms_window_input,
            self.qd_ste_min_window_input,
            self.qd_ste_min_gap_input,
            self.qd_ste_epoch_length_input,
            self.qd_ste_min_oscillation_input,
            self.qd_ste_rms_threshold_input,
            self.qd_ste_peak_threshold_input,
            self.qd_hil_sample_freq_input,
            self.qd_hil_pass_band_input,
            self.qd_hil_stop_band_input,
            self.qd_hil_epoch_time_input,
            self.qd_hil_sd_threshold_input,
            self.qd_hil_min_window_input,
            self.qd_ignore_sec_before_input,
            self.qd_ignore_sec_after_input,
        )
        for widget in (self.qd_fp_input, self.qd_fs_input, self.qd_rp_input, self.qd_rs_input):
            safe_connect_signal_slot(widget.returnPressed, self._submit_run_from_fields)
        self._sync_filter_input_constraints()
        for widget in numeric_fields:
            self._configure_numeric_line_edit(widget, minimum=0.0)
            safe_connect_signal_slot(widget.returnPressed, self._submit_run_from_fields)

        self._configure_numeric_line_edit(
            self.qd_classifier_batch_size_input,
            integer=True,
            minimum=1,
        )
        safe_connect_signal_slot(self.qd_classifier_batch_size_input.returnPressed, self._submit_run_from_fields)

        self.qd_classifier_device_input.setPlaceholderText("cpu or cuda:0")
        self.qd_classifier_device_input.setToolTip("Classifier device. Use cpu or cuda:0.")
        self.qd_classifier_device_input.setStatusTip("Classifier device. Use cpu or cuda:0.")
        safe_connect_signal_slot(self.qd_classifier_device_input.returnPressed, self._submit_run_from_fields)

        self._configure_default_buttons()

    def _set_model_path_display(self, widget, path):
        if widget is None:
            return
        full_path = str(path or "")
        widget.setProperty("fullPath", full_path)
        widget.setToolTip(full_path)
        if not full_path:
            widget.blockSignals(True)
            widget.clear()
            widget.blockSignals(False)
            return

        metrics = widget.fontMetrics()
        available_width = max(80, widget.width() - 14)
        compact_path = self._compact_model_path(full_path)
        elided = metrics.elidedText(compact_path, QtCore.Qt.ElideMiddle, available_width)
        widget.blockSignals(True)
        widget.setText(elided)
        widget.setCursorPosition(0)
        widget.blockSignals(False)

    def _compact_model_path(self, full_path):
        path = Path(full_path)
        if not full_path:
            return ""
        if not path.name:
            return full_path
        parent_name = path.parent.name
        if parent_name:
            return f".../{parent_name}/{path.name}"
        return path.name

    def _get_model_path(self, widget):
        if widget is None:
            return ""
        stored_path = widget.property("fullPath")
        if stored_path:
            return str(stored_path)
        return widget.text().strip()

    def eventFilter(self, obj, event):
        tracked_displays = {
            getattr(self, "qd_classifier_artifact_filename_display", None),
            getattr(self, "qd_classifier_spike_filename_display", None),
            getattr(self, "qd_classifier_ehfo_filename_display", None),
        }
        if obj in tracked_displays and event.type() in {QtCore.QEvent.Resize, QtCore.QEvent.Show}:
            QtCore.QTimer.singleShot(0, lambda widget=obj: self._set_model_path_display(widget, self._get_model_path(widget)))
        return super().eventFilter(obj, event)

    def _create_file_dialog(self, title, name_filter):
        dialog = QFileDialog(self, title)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter(name_filter)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        apply_subwindow_theme(dialog)
        dialog.raise_()
        dialog.activateWindow()
        return dialog

    def open_file(self):
        dialog = self._create_file_dialog(
            "Open File",
            "Recordings Files (*.edf *.eeg *.vhdr *.vmrk *.fif *.fif.gz)",
        )
        if not dialog.exec_():
            return
        selected_files = dialog.selectedFiles()
        fname = selected_files[0] if selected_files else ""
        if fname:
            worker = Worker(self.read_edf, fname)
            safe_connect_signal_slot(worker.signals.result, self.update_edf_info)
            safe_connect_signal_slot(worker.signals.error, lambda error: self.handle_worker_error("Open recording", error))
            self.threadpool.start(worker)

    def read_edf(self, fname, progress_callback):
        self.fname = fname
        self.backend.load_edf(fname)
        eeg_data,channel_names=self.backend.get_eeg_data()
        edf_info=self.backend.get_edf_info()
        filename = os.path.basename(fname)
        self.filename = filename
        sample_freq = str(self.backend.sample_freq)
        num_channels = str(len(self.backend.channel_names))
        length = str(self.backend.eeg_data.shape[1])
        return [filename, sample_freq, num_channels, length]
    
    @pyqtSlot(object)
    def update_edf_info(self, results):
        self.filename = results[0]
        self._clear_run_feedback_state()
        try:
            self.backend.sample_freq = float(results[1])
        except (TypeError, ValueError):
            self.backend.sample_freq = getattr(self.backend, "sample_freq", 0.0)
        self.qd_filename.setText(results[0])
        self.qd_sampfreq.setText(results[1])
        self.qd_numchannels.setText(results[2])
        self.qd_length.setText(results[3])
        self._sync_filter_inputs_from_param(self._recommended_filter_param())
        self.run_button.setEnabled(True)
        self._refresh_run_setup_feedback()


    def update_detector_tab(self, index):
        if index == "MNI":
            self.stackedWidget.setCurrentIndex(0)
            self.detector = "MNI"
        elif index == "STE":
            self.stackedWidget.setCurrentIndex(1)
            self.detector = "STE"
        elif index == "HIL":
            self.stackedWidget.setCurrentIndex(2)
            self.detector = "HIL"
        else:
            self.detector = ""
        if not self.running:
            self._clear_run_feedback_state()
            self._refresh_run_setup_feedback()

        # filter stuff
    def init_default_filter_input_params(self):
        self._sync_filter_inputs_from_param(self._recommended_filter_param())

    def get_filter_param(self):
        return self._build_filter_param_from_inputs()

    def init_default_mni_input_params(self):
        """this is how I got the params, I reversed it here
        epoch_time = self.mni_epoch_time_input.text()
        epo_CHF = self.mni_epoch_CHF_input.text() 
        per_CHF = self.mni_chf_percentage_input.text()
        min_win = self.mni_min_window_input.text()
        min_gap = self.mni_min_gap_time_input.text()
        thrd_perc = self.mni_threshold_percentage_input.text()
        base_seg = self.mni_baseline_window_input.text()
        base_shift = self.mni_baseline_shift_input.text()
        base_thrd = self.mni_baseline_threshold_input.text()
        base_min = self.mni_baseline_min_time_input.text()
        """
        default_params=ParamMNI(200)
        self.qd_mni_epoch_time_input.setText(str(default_params.epoch_time))
        self.qd_mni_epoch_chf_input.setText(str(default_params.epo_CHF))
        self.qd_mni_chf_percentage_input.setText(str(default_params.per_CHF))
        self.qd_mni_min_window_input.setText(str(default_params.min_win))
        self.qd_mni_min_gap_time_input.setText(str(default_params.min_gap))
        self.qd_mni_threshold_percentage_input.setText(str(default_params.thrd_perc*100))
        self.qd_mni_baseline_window_input.setText(str(default_params.base_seg))
        self.qd_mni_baseline_shift_input.setText(str(default_params.base_shift))
        self.qd_mni_baseline_threshold_input.setText(str(default_params.base_thrd))
        self.qd_mni_baseline_time_input.setText(str(default_params.base_min))

    def get_mni_params(self):
        epoch_time = self.qd_mni_epoch_time_input.text()
        epo_CHF = self.qd_mni_epoch_chf_input.text() 
        per_CHF = self.qd_mni_chf_percentage_input.text()
        min_win = self.qd_mni_min_window_input.text()
        min_gap = self.qd_mni_min_gap_time_input.text()
        thrd_perc = self.qd_mni_threshold_percentage_input.text()
        base_seg = self.qd_mni_baseline_window_input.text()
        base_shift = self.qd_mni_baseline_shift_input.text()
        base_thrd = self.qd_mni_baseline_threshold_input.text()
        base_min = self.qd_mni_baseline_time_input.text()
        
        param_dict = {"sample_freq":2000,"pass_band":1, "stop_band":80, #these are placeholder params, will be updated later
                    "epoch_time":self._parse_float_input(epoch_time, "MNI epoch time", positive=True),
                    "epo_CHF":self._parse_float_input(epo_CHF, "MNI epoch CHF", positive=True),
                    "per_CHF":self._parse_float_input(per_CHF, "MNI CHF percentage", positive=True),
                    "min_win":self._parse_float_input(min_win, "MNI minimum window", positive=True),
                    "min_gap":self._parse_float_input(min_gap, "MNI minimum gap", non_negative=True),
                    "base_seg":self._parse_float_input(base_seg, "MNI baseline window", positive=True),
                    "thrd_perc":self._parse_float_input(thrd_perc, "MNI threshold percentage", positive=True)/100,
                    "base_shift":self._parse_float_input(base_shift, "MNI baseline shift", non_negative=True),
                    "base_thrd":self._parse_float_input(base_thrd, "MNI baseline threshold", positive=True),
                    "base_min":self._parse_float_input(base_min, "MNI baseline minimum time", non_negative=True),
                    "n_jobs":self.backend.n_jobs}
        detector_params = {"detector_type":"MNI", "detector_param":param_dict}
        return ParamDetector.from_dict(detector_params)

    def init_default_ste_input_params(self):
        default_params=ParamSTE(2000)
        self.qd_ste_rms_window_input.setText(str(default_params.rms_window))
        self.qd_ste_rms_threshold_input.setText(str(default_params.rms_thres))
        self.qd_ste_min_window_input.setText(str(default_params.min_window))
        self.qd_ste_epoch_length_input.setText(str(default_params.epoch_len))
        self.qd_ste_min_gap_input.setText(str(default_params.min_gap))
        self.qd_ste_min_oscillation_input.setText(str(default_params.min_osc))
        self.qd_ste_peak_threshold_input.setText(str(default_params.peak_thres))

    def get_ste_params(self):
        rms_window_raw = self.qd_ste_rms_window_input.text()
        min_window_raw = self.qd_ste_min_window_input.text()
        min_gap_raw = self.qd_ste_min_gap_input.text()
        epoch_len_raw = self.qd_ste_epoch_length_input.text()
        min_osc_raw = self.qd_ste_min_oscillation_input.text()
        rms_thres_raw = self.qd_ste_rms_threshold_input.text()
        peak_thres_raw = self.qd_ste_peak_threshold_input.text()
        param_dict={"sample_freq":2000,"pass_band":1, "stop_band":80, #these are placeholder params, will be updated later
                    "rms_window":self._parse_float_input(rms_window_raw, "STE RMS window", positive=True),
                    "min_window":self._parse_float_input(min_window_raw, "STE minimum window", positive=True),
                    "min_gap":self._parse_float_input(min_gap_raw, "STE minimum gap", non_negative=True),
                    "epoch_len":self._parse_float_input(epoch_len_raw, "STE epoch length", positive=True),
                    "min_osc":self._parse_float_input(min_osc_raw, "STE minimum oscillations", positive=True),
                    "rms_thres":self._parse_float_input(rms_thres_raw, "STE RMS threshold", positive=True),
                    "peak_thres":self._parse_float_input(peak_thres_raw, "STE peak threshold", positive=True),
                    "n_jobs":self.backend.n_jobs}
        detector_params={"detector_type":"STE", "detector_param":param_dict}
        return ParamDetector.from_dict(detector_params)
    
    def init_default_hil_input_params(self):
        default_params = ParamHIL(2000)
        self.qd_hil_sample_freq_input.setText(str(default_params.sample_freq))
        self.qd_hil_pass_band_input.setText(str(default_params.pass_band))
        self.qd_hil_stop_band_input.setText(str(default_params.stop_band))
        self.qd_hil_epoch_time_input.setText(str(default_params.epoch_time))
        self.qd_hil_sd_threshold_input.setText(str(default_params.sd_threshold))
        self.qd_hil_min_window_input.setText(str(default_params.min_window))

    def get_hil_params(self):
        sample_freq_raw = self.qd_hil_sample_freq_input.text()
        pass_band_raw = self.qd_hil_pass_band_input.text()
        stop_band_raw = self.qd_hil_stop_band_input.text()
        epoch_time_raw = self.qd_hil_epoch_time_input.text()
        sd_threshold_raw = self.qd_hil_sd_threshold_input.text()
        min_window_raw = self.qd_hil_min_window_input.text()
        
        param_dict = {
            "sample_freq": self._parse_float_input(sample_freq_raw, "HIL sample frequency", positive=True),
            "pass_band": self._parse_float_input(pass_band_raw, "HIL pass band", positive=True),
            "stop_band": self._parse_float_input(stop_band_raw, "HIL stop band", positive=True),
            "epoch_time": self._parse_float_input(epoch_time_raw, "HIL epoch time", positive=True),
            "sd_threshold": self._parse_float_input(sd_threshold_raw, "HIL SD threshold", positive=True),
            "min_window": self._parse_float_input(min_window_raw, "HIL minimum window", positive=True),
            "n_jobs": self.backend.n_jobs
        }
        detector_params = {"detector_type": "HIL", "detector_param": param_dict}
        return ParamDetector.from_dict(detector_params)
        
    def get_classifier_param(self):
        artifact_path = self._get_model_path(self.qd_classifier_artifact_filename_display)
        spike_path = self._get_model_path(self.qd_classifier_spike_filename_display)
        ehfo_path = self._get_model_path(self.qd_classifier_ehfo_filename_display)
        base_param = self.backend.get_classifier_param()
        artifact_card = getattr(base_param, "artifact_card", None) if base_param is not None else None
        spike_card = getattr(base_param, "spike_card", None) if base_param is not None else None
        ehfo_card = getattr(base_param, "ehfo_card", None) if base_param is not None else None
        use_spike = self.qd_use_spikes_checkbox.isChecked()
        use_ehfo = self.qd_use_ehfo_checkbox.isChecked()
        device, model_type = self.normalize_classifier_device(self.qd_classifier_device_input.text())
        batch_size = self._parse_int_input(self.qd_classifier_batch_size_input.text(), "Batch size", positive=True)

        if not artifact_path:
            raise ValueError("Artifact model file is required when classifier is enabled.")
        if use_spike and not spike_path:
            raise ValueError("Spike model file is required when spkHFO is enabled.")
        if use_ehfo and not ehfo_path:
            raise ValueError("eHFO model file is required when eHFO is enabled.")

        classifier_param = ParamClassifier(artifact_path=artifact_path, spike_path=spike_path, ehfo_path=ehfo_path,
                                           artifact_card=artifact_card, spike_card=spike_card, ehfo_card=ehfo_card,
                                           use_spike=use_spike, use_ehfo=use_ehfo,
                                           device=device, batch_size=batch_size, model_type=model_type,
                                           source_preference=getattr(self, "classifier_source_preference", "auto"))

        seconds_before = self._parse_float_input(
            self.qd_ignore_sec_before_input.text(),
            "Ignore window before events",
            non_negative=True,
        )
        seconds_after = self._parse_float_input(
            self.qd_ignore_sec_after_input.text(),
            "Ignore window after events",
            non_negative=True,
        )
        return {"classifier_param": classifier_param, "use_spike": use_spike, "use_ehfo": use_ehfo,
                "seconds_before": seconds_before, "seconds_after": seconds_after}
    
    def set_classifier_param_display(self):
        classifier_param = self.backend.get_classifier_param()
        self.classifier_source_preference = getattr(classifier_param, "source_preference", "auto")

        #set also the input fields
        self._set_model_path_display(self.qd_classifier_artifact_filename_display, classifier_param.artifact_path or "")
        self._set_model_path_display(self.qd_classifier_spike_filename_display, classifier_param.spike_path or "")
        self._set_model_path_display(self.qd_classifier_ehfo_filename_display, classifier_param.ehfo_path or "")
        self.qd_use_spikes_checkbox.setChecked(classifier_param.use_spike)
        self.qd_use_ehfo_checkbox.setChecked(classifier_param.use_ehfo)
        self.qd_classifier_device_input.setText(str(classifier_param.device))
        self.qd_classifier_batch_size_input.setText(str(classifier_param.batch_size))

    def set_classifier_param_gpu_default(self):
        self.backend.set_default_gpu_classifier()
        self.set_classifier_param_display()
    
    def set_classifier_param_cpu_default(self):
        self.backend.set_default_cpu_classifier()
        self.set_classifier_param_display()

    def choose_model_file(self, model_type):
        dialog = self._create_file_dialog("Open file", ".tar files (*.tar)")
        if not dialog.exec_():
            return
        selected_files = dialog.selectedFiles()
        fname = selected_files[0] if selected_files else ""
        self.classifier_source_preference = "local"
        if model_type == "artifact":
            self._set_model_path_display(self.qd_classifier_artifact_filename_display, fname)
        elif model_type == "spike":
            self._set_model_path_display(self.qd_classifier_spike_filename_display, fname)
        elif model_type == "ehfo":
            self._set_model_path_display(self.qd_classifier_ehfo_filename_display, fname)

    def normalize_classifier_device(self, device_text):
        normalized = device_text.strip().lower()
        if normalized == "cpu":
            return "cpu", "default_cpu"
        if normalized in {"cuda", "cuda:0", "gpu"}:
            if torch is None or not torch.cuda.is_available():
                raise ValueError("GPU classifier is unavailable on this machine. Use cpu instead.")
            return "cuda:0", "default_gpu"
        raise ValueError("Device must be either cpu or cuda:0.")

    def get_output_stem(self):
        if not getattr(self, "fname", None):
            raise ValueError("Load a recording before running quick detection.")

        recording_path = Path(self.fname)
        recording_name = recording_path.name
        if recording_name.endswith(".fif.gz"):
            return recording_name[:-7]
        return recording_path.stem

    def build_output_path(self, extension):
        recording_path = Path(self.fname)
        detector_name = (self._selected_detector_name() or "quick").lower()
        output_stem = f"{self.get_output_stem()}_{detector_name}"
        candidate = recording_path.with_name(f"{output_stem}{extension}")
        suffix_index = 2
        while candidate.exists():
            candidate = recording_path.with_name(f"{output_stem}_{suffix_index}{extension}")
            suffix_index += 1
        return str(candidate)

    def collect_run_configuration(self):
        n_jobs = int(self.n_jobs_spinbox.value())
        filter_param = self.get_filter_param()

        if self.detector == "MNI":
            detector_param = self.get_mni_params()
        elif self.detector == "STE":
            detector_param = self.get_ste_params()
        elif self.detector == "HIL":
            detector_param = self.get_hil_params()
        else:
            raise ValueError("Select a detector before running quick detection.")

        detector_param.detector_param.n_jobs = n_jobs

        save_as_excel = self.qd_excel_checkbox.isChecked()
        save_as_npz = self.qd_npz_checkbox.isChecked()
        if not save_as_excel and not save_as_npz:
            raise ValueError("Select at least one output format.")

        classifier_config = None
        if self.qd_use_classifier_checkbox.isChecked():
            classifier_config = self.get_classifier_param()

        return {
            "n_jobs": n_jobs,
            "filter_param": filter_param,
            "detector_param": detector_param,
            "classifier": classifier_config,
            "save_as_excel": save_as_excel,
            "save_as_npz": save_as_npz,
            "excel_output": self.build_output_path(".xlsx"),
            "npz_output": self.build_output_path(".npz"),
        }

    def handle_worker_error(self, task_name, error_tuple):
        _, value, traceback_text = error_tuple
        msg = build_themed_message_box(
            self,
            icon=QMessageBox.Critical,
            title=f"{task_name} Failed",
            text=f"{task_name} failed",
            informative_text=str(value),
            detailed_text=traceback_text,
        )
        msg.exec_()
    
    def _detect(self, progress_callback):
        #call detect HFO function on backend
        self.backend.detect_biomarker()
        return []
    
    def detect_biomarkers(self):
        # print("Detecting HFOs...")
        worker=Worker(self._detect)
        safe_connect_signal_slot(worker.signals.result, self._detect_finished)
        self.threadpool.start(worker)

    # def _detect_finished(self):
    #     #right now do nothing beyond message handler saying that 
    #     # it has detected HFOs
    #     # self.message_handler("HFOs detected")

    def filter_data(self):
        # print("Filtering data...")
        worker=Worker(self._filter)
        safe_connect_signal_slot(worker.signals.finished, self.filtering_complete)
        self.threadpool.start(worker)

    def _filter(self, progress_callback):
        self.backend.filter_eeg_data(self.filter_params)

    # def filtering_complete(self):
    #     self.message_handler('Filtering COMPLETE!')

    def _classify(self,classify_spikes,use_ehfo=False,seconds_to_ignore_before=0,seconds_to_ignore_after=0):
        self.backend.classify_artifacts([seconds_to_ignore_before,seconds_to_ignore_after])
        if classify_spikes:
            self.backend.classify_spikes()
        if use_ehfo and hasattr(self.backend, "classify_ehfos"):
            self.backend.classify_ehfos()
        return []

    # def _classify_finished(self):
    #     self.message_handler("Classification finished!..")

    def classify(self,params):
        #set the parameters
        self.backend.set_classifier(params["classifier_param"])
        seconds_to_ignore_before = params["seconds_before"]
        seconds_to_ignore_after = params["seconds_after"]
        self._classify(
            params["use_spike"],
            params["use_ehfo"],
            seconds_to_ignore_before,
            seconds_to_ignore_after,
        )


    def set_ui_enabled(self, enabled):
        """Enable or disable UI controls during detection"""
        for control in self.controls_to_disable:
            try:
                control.setEnabled(enabled)
            except Exception:
                pass  # Skip if control doesn't exist
        
        # Change cursor to indicate busy/ready state
        if not enabled:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        else:
            try:
                QtWidgets.QApplication.restoreOverrideCursor()
            except Exception:
                pass

    def _run(self, run_config, progress_callback):
        self.backend.set_n_jobs(run_config["n_jobs"])
        self.backend.filter_eeg_data(run_config["filter_param"])
        self.backend.set_detector(run_config["detector_param"])
        self.backend.detect_biomarker()

        classifier_config = run_config["classifier"]
        if classifier_config is not None:
            self.classify(classifier_config)

        outputs = {}
        if run_config["save_as_excel"]:
            self.backend.export_excel(run_config["excel_output"])
            outputs["excel_output"] = run_config["excel_output"]
        if run_config["save_as_npz"]:
            self.backend.export_app(run_config["npz_output"])
            outputs["npz_output"] = run_config["npz_output"]
        return outputs
    
    def run(self):
        try:
            run_config = self.collect_run_configuration()
        except Exception as exc:
            msg = build_themed_message_box(
                self,
                icon=QMessageBox.Warning,
                title="Quick Detection",
                text="Quick detection could not start.",
                informative_text=str(exc),
            )
            msg.exec_()
            return

        worker=Worker(self._run, run_config)
        self.running = True
        self._run_result = None
        self._run_failed = False
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.set_ui_enabled(False)
        self.setWindowTitle("Quick Detection - Running...")
        self._refresh_run_setup_feedback()
        safe_connect_signal_slot(worker.signals.result, self._run_finished)
        safe_connect_signal_slot(worker.signals.error, self._run_error)
        safe_connect_signal_slot(worker.signals.finished, self._run_cleanup)
        self.threadpool.start(worker)

    def _run_finished(self, result=None):
        self._run_result = result or {}
        self._run_failed = False
        self.setWindowTitle("Quick Detection - Complete")
        self._refresh_run_setup_feedback()

    def _run_error(self, error_tuple):
        self._run_result = None
        self._run_failed = True
        self.setWindowTitle("Quick Detection - Failed")
        self._refresh_run_setup_feedback()
        self.handle_worker_error("Quick detection", error_tuple)

    def _run_cleanup(self):
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        self.set_ui_enabled(True)
        self.running = False
        try:
            QtWidgets.QApplication.restoreOverrideCursor()
        except Exception:
            pass
        if not self._run_failed:
            self.setWindowTitle("Quick Detection - Complete")
        self._refresh_run_setup_feedback()
        
    def reject(self):
        if self.running:
            reply = build_themed_message_box(
                self,
                icon=QMessageBox.Warning,
                title="Detection Running",
                text="Detection is currently running.",
                informative_text="Are you sure you want to close this window?",
                buttons=QMessageBox.Yes | QMessageBox.No,
                default_button=QMessageBox.No,
            )
            if reply.exec_() == QMessageBox.No:
                return
        
        # Restore cursor if it was changed
        try:
            QtWidgets.QApplication.restoreOverrideCursor()
        except Exception:
            pass

        if self.main_window is not None:
            self.main_window.set_quick_detect_open(False)
        super().reject()
        
        


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = HFOQuickDetector()
    mainWindow.show()
    sys.exit(app.exec_())
