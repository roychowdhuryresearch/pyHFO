from pathlib import Path
from queue import Queue
import os
import tempfile
from datetime import datetime
import math
import json

from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import pyqtSignal
import multiprocessing as mp
try:
    import torch
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
from src.hfo_app import HFO_App
from src.hfo_feature import HFO_Feature
from src.spindle_app import SpindleApp
from src.ui.channels_selection import ChannelSelectionWindow
from src.param.param_classifier import ParamClassifier
from src.param.param_detector import ParamDetector, ParamSTE, ParamMNI, ParamHIL, ParamYASA
from src.param.param_filter import ParamFilter, ParamFilterSpindle
from src.ui.bipolar_channel_selection import BipolarChannelSelectionWindow
from src.ui.annotation import Annotation
from src.ui.waveform_shortcut_settings import WaveformShortcutSettingsDialog
from src.utils.utils_gui import *
from src.ui.plot_waveform import *
from src.utils.app_state import build_base_checkpoint, checkpoint_array, checkpoint_get, checkpoint_version
from src.utils.session_store import load_session_checkpoint
from src.utils.utils_detector import has_hil, has_yasa
from src.utils.utils_montage import (
    BIPOLAR_TOKEN,
    get_adjacent_contact_neighbor_channels,
    get_conventional_eeg_neighbor_channels,
)
from src.utils.reporting import export_analysis_report, export_clinical_summary_workbook
from src.utils.analysis_session import build_run_comparison
from src.views.shared_plot_handle import SharedPlotHandle


class MainWindowModel(QObject):
    def __init__(self, main_window):
        super(MainWindowModel, self).__init__()
        self.window = main_window
        self.backend = None
        self.biomarker_type = None
        self.case_backends = {}
        self.current_recording_path = None
        self._suspend_default_configuration = False
        self._busy_task_state = None
        self.waveform_shortcuts_enabled = True
        self.waveform_shortcut_bindings = {}
        self.waveform_trackpad_sensitivity = "default"
        self.waveform_measurement_state = {"points": [], "summary": None}
        self._highlight_channel_mode = False
        self._waveform_channel_wheel_remainder = 0.0
        self._load_waveform_shortcut_preferences()
        self._load_waveform_interaction_preferences()

    def _ui_object_is_alive(self, obj):
        if obj is None:
            return False
        try:
            obj.objectName()
        except RuntimeError:
            return False
        return True

    def _set_toolbar_slot_text(self, widget, text, *, tooltip=""):
        if widget is None:
            return
        full_text = str(text or "")
        widget.setText(full_text)
        resolved_tooltip = str(tooltip or full_text)
        widget.setToolTip(resolved_tooltip)
        widget.setStatusTip(resolved_tooltip)

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
            validator = QIntValidator(lower, upper, widget)
        else:
            lower = float(0.0 if minimum is None else minimum)
            upper = float(1_000_000_000.0 if maximum is None else maximum)
            validator = QDoubleValidator(lower, upper, int(decimals), widget)
            validator.setNotation(QDoubleValidator.StandardNotation)
        widget.setValidator(validator)
        if placeholder is not None and hasattr(widget, "setPlaceholderText"):
            widget.setPlaceholderText(str(placeholder))
        if tooltip:
            widget.setToolTip(str(tooltip))
            widget.setStatusTip(str(tooltip))

    def _recording_sample_frequency(self):
        sample_freq = 0.0
        if self.backend is not None:
            try:
                sample_freq = float(getattr(self.backend, "sample_freq", 0.0) or 0.0)
            except (TypeError, ValueError):
                sample_freq = 0.0
        if sample_freq <= 0:
            try:
                sample_freq = float(getattr(self.window, "sample_freq", 0.0) or 0.0)
            except (TypeError, ValueError):
                sample_freq = 0.0
        return sample_freq if math.isfinite(sample_freq) and sample_freq > 0 else 0.0

    def _recording_nyquist_frequency(self):
        sample_freq = self._recording_sample_frequency()
        return sample_freq / 2.0 if sample_freq > 0 else 0.0

    def _filter_space(self):
        default_param = self._filter_param_class()()
        try:
            return float(getattr(default_param, "space", 0.5) or 0.5)
        except (TypeError, ValueError):
            return 0.5

    def _filter_stop_band_limit(self):
        nyquist = self._recording_nyquist_frequency()
        if nyquist <= 0:
            return 0.0
        filter_space = self._filter_space()
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
        if self.biomarker_type == "Spindle":
            default_param = ParamFilterSpindle()
            min_pass_band = 0.1
            gap_floor = 0.5
        else:
            default_param = ParamFilter()
            min_pass_band = 1.0
            gap_floor = 5.0

        sample_freq = self._recording_sample_frequency()
        if sample_freq <= 0:
            return default_param

        max_stop_band = max(min_pass_band + 0.1, self._filter_stop_band_limit())
        stop_band = min(float(default_param.fs), max_stop_band)
        preferred_gap = max(gap_floor, stop_band * 0.15)
        pass_band = min(float(default_param.fp), stop_band - preferred_gap)
        if pass_band <= 0:
            pass_band = max(min_pass_band, stop_band * (0.45 if self.biomarker_type == "Spindle" else 0.8))
        pass_band = min(pass_band, stop_band - 0.1)
        pass_band = max(min_pass_band, pass_band)
        if pass_band >= stop_band:
            pass_band = max(min_pass_band, stop_band - 0.1)

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
        for attr in ("fp_input", "fs_input"):
            self._configure_numeric_line_edit(
                getattr(self.window, attr, None),
                minimum=0.0,
                maximum=maximum_frequency,
                tooltip=tooltip,
            )
        self._configure_numeric_line_edit(getattr(self.window, "rp_input", None), minimum=0.0)
        self._configure_numeric_line_edit(getattr(self.window, "rs_input", None), minimum=0.0)

    def _sync_filter_inputs_from_param(self, filter_param=None):
        if filter_param is None:
            filter_param = getattr(self.backend, "param_filter", None)
        if filter_param is None:
            filter_param = self._recommended_filter_param()
        if hasattr(self.window, "fp_input"):
            self.window.fp_input.setText(self._format_numeric_text(getattr(filter_param, "fp", "")))
        if hasattr(self.window, "fs_input"):
            self.window.fs_input.setText(self._format_numeric_text(getattr(filter_param, "fs", "")))
        if hasattr(self.window, "rp_input"):
            self.window.rp_input.setText(self._format_numeric_text(getattr(filter_param, "rp", "")))
        if hasattr(self.window, "rs_input"):
            self.window.rs_input.setText(self._format_numeric_text(getattr(filter_param, "rs", "")))
        self._sync_filter_input_constraints()

    def _build_filter_param_from_inputs(self):
        fp = self._parse_float_input(self.window.fp_input.text(), "Filter pass band", positive=True)
        fs = self._parse_float_input(self.window.fs_input.text(), "Filter stop band", positive=True)
        rp = self._parse_float_input(self.window.rp_input.text(), "Filter ripple", positive=True)
        rs = self._parse_float_input(self.window.rs_input.text(), "Filter attenuation", positive=True)
        stop_band_limit = self._filter_stop_band_limit()
        if stop_band_limit > 0 and fs > stop_band_limit:
            raise ValueError(f"Filter stop band must stay below {stop_band_limit:.2f} Hz for this recording.")
        if fp >= fs:
            raise ValueError("Filter pass band must be lower than the stop band.")
        return self._filter_param_class().from_dict(
            {
                "fp": fp,
                "fs": fs,
                "rp": rp,
                "rs": rs,
                "sample_freq": self._recording_sample_frequency() or None,
            }
        )

    def _click_button_if_enabled(self, button):
        if button is None or not self._ui_object_is_alive(button) or not button.isEnabled():
            return
        button.click()

    def _submit_filter_from_fields(self):
        self._click_button_if_enabled(getattr(self.window, "overview_filter_button", None))

    def _submit_detector_from_fields(self):
        self._click_button_if_enabled(getattr(self.window, "detector_apply_button", None))

    def _submit_classifier_from_fields(self):
        apply_button = getattr(self.window, "classifier_apply_button", None)
        if apply_button is not None and apply_button.isVisible() and apply_button.isEnabled():
            apply_button.click()
            return
        if (
            self.backend is not None
            and getattr(self.backend, "eeg_data", None) is not None
            and self.biomarker_supports_classification()
        ):
            self.apply_classifier_setup()

    def _configure_main_window_input_conventions(self):
        filter_fields = ("fp_input", "fs_input", "rp_input", "rs_input")
        for attr in filter_fields:
            widget = getattr(self.window, attr, None)
            if widget is not None and hasattr(widget, "returnPressed"):
                safe_connect_signal_slot(widget.returnPressed, self._submit_filter_from_fields)
        self._sync_filter_input_constraints()

        detector_field_specs = {
            "ste_rms_window_input": {"minimum": 0.0},
            "ste_min_window_input": {"minimum": 0.0},
            "ste_min_gap_input": {"minimum": 0.0},
            "ste_epoch_length_input": {"minimum": 0.0},
            "ste_min_oscillation_input": {"minimum": 0.0},
            "ste_rms_threshold_input": {"minimum": 0.0},
            "ste_peak_threshold_input": {"minimum": 0.0},
            "mni_epoch_time_input": {"minimum": 0.0},
            "mni_epoch_chf_input": {"minimum": 0.0},
            "mni_chf_percentage_input": {"minimum": 0.0},
            "mni_min_window_input": {"minimum": 0.0},
            "mni_min_gap_time_input": {"minimum": 0.0},
            "mni_threshold_percentage_input": {"minimum": 0.0},
            "mni_baseline_window_input": {"minimum": 0.0},
            "mni_baseline_shift_input": {"minimum": 0.0},
            "mni_baseline_threshold_input": {"minimum": 0.0},
            "mni_baseline_min_time_input": {"minimum": 0.0},
            "hil_sample_freq_input": {"minimum": 0.0},
            "hil_pass_band_input": {"minimum": 0.0},
            "hil_stop_band_input": {"minimum": 0.0},
            "hil_epoch_time_input": {"minimum": 0.0},
            "hil_sd_threshold_input": {"minimum": 0.0},
            "hil_min_window_input": {"minimum": 0.0},
            "yasa_freq_sp_low_input": {"minimum": 0.0},
            "yasa_freq_sp_high_input": {"minimum": 0.0},
            "yasa_freq_broad_low_input": {"minimum": 0.0},
            "yasa_freq_broad_high_input": {"minimum": 0.0},
            "yasa_duration_low_input": {"minimum": 0.0},
            "yasa_duration_high_input": {"minimum": 0.0},
            "yasa_min_distance_input": {"minimum": 0.0},
            "yasa_thresh_rel_pow_input": {"minimum": 0.0},
            "yasa_thresh_corr_input": {"minimum": 0.0},
            "yasa_thresh_rms_input": {"minimum": 0.0},
        }
        for attr, kwargs in detector_field_specs.items():
            widget = getattr(self.window, attr, None)
            self._configure_numeric_line_edit(widget, **kwargs)
            if widget is not None and hasattr(widget, "returnPressed"):
                safe_connect_signal_slot(widget.returnPressed, self._submit_detector_from_fields)

        classifier_submit_fields = (
            "overview_ignore_before_input",
            "overview_ignore_after_input",
            "classifier_artifact_filename",
            "classifier_spike_filename",
            "classifier_ehfo_filename",
            "classifier_artifact_card_name",
            "classifier_spike_card_name",
            "classifier_ehfo_card_name",
            "classifier_device_input",
            "classifier_batch_size_input",
        )
        self._configure_numeric_line_edit(
            getattr(self.window, "overview_ignore_before_input", None),
            minimum=0.0,
        )
        self._configure_numeric_line_edit(
            getattr(self.window, "overview_ignore_after_input", None),
            minimum=0.0,
        )
        self._configure_numeric_line_edit(
            getattr(self.window, "classifier_batch_size_input", None),
            integer=True,
            minimum=1,
        )
        device_input = getattr(self.window, "classifier_device_input", None)
        if device_input is not None and hasattr(device_input, "setPlaceholderText"):
            device_input.setPlaceholderText("cpu or cuda:0")
            device_input.setToolTip("Classifier device. Use cpu or cuda:0.")
            device_input.setStatusTip("Classifier device. Use cpu or cuda:0.")
        for attr in classifier_submit_fields:
            widget = getattr(self.window, attr, None)
            if widget is not None and hasattr(widget, "returnPressed"):
                safe_connect_signal_slot(widget.returnPressed, self._submit_classifier_from_fields)

    def _normalize_classifier_device_input(self, raw_value):
        normalized = str(raw_value).strip().lower()
        if normalized == "cpu":
            return "cpu", "default_cpu"
        if normalized in {"cuda", "cuda:0", "gpu"}:
            if not self.gpu:
                raise ValueError("GPU classifier is unavailable on this machine. Use cpu instead.")
            return "cuda:0", "default_gpu"
        raise ValueError("Device must be either cpu or cuda:0.")

    def _create_spike_review_backend(self):
        backend = HFO_App()
        backend.biomarker_type = "Spike"
        if hasattr(backend, "analysis_session"):
            backend.analysis_session.biomarker_type = "Spike"
        return backend

    def set_biomarker_type_and_init_backend(self, bio_type):
        self.biomarker_type = bio_type
        if bio_type in self.case_backends:
            self.backend = self.case_backends[bio_type]
            return
        if bio_type == 'HFO':
            self.backend = HFO_App()
        elif bio_type == 'Spindle':
            self.backend = SpindleApp()
        elif bio_type == 'Spike':
            self.backend = self._create_spike_review_backend()
        else:
            raise ValueError(f"Unsupported biomarker type: {bio_type}")
        self.case_backends[bio_type] = self.backend

    def get_biomarker_display_name(self):
        return self.biomarker_type or "event"

    def biomarker_supports_detection(self):
        return self.biomarker_type in {'HFO', 'Spindle'}

    def biomarker_supports_classification(self):
        return self.biomarker_type in {'HFO', 'Spindle'}

    def _filter_param_class(self):
        return ParamFilter if self.biomarker_type in {'HFO', 'Spike'} else ParamFilterSpindle

    def _instantiate_backend_for_biomarker(self, biomarker_type):
        if biomarker_type == 'HFO':
            return HFO_App()
        if biomarker_type == 'Spindle':
            return SpindleApp()
        if biomarker_type == 'Spike':
            return self._create_spike_review_backend()
        raise ValueError(f"Unsupported biomarker type: {biomarker_type}")

    def init_error_terminal_display(self):
        self.window._original_stdout = sys.stdout
        self.window._original_stderr = sys.stderr
        self.window.stdout = Queue()
        self.window.stderr = Queue()
        sys.stdout = WriteStream(self.window.stdout)
        sys.stderr = WriteStream(self.window.stderr)
        self.window.thread_stdout = STDOutReceiver(self.window.stdout)
        safe_connect_signal_slot(self.window.thread_stdout.std_received_signal, self.message_handler)
        self.window.thread_stdout.start()

        self.window.thread_stderr = STDErrReceiver(self.window.stderr)
        safe_connect_signal_slot(self.window.thread_stderr.std_received_signal, self.message_handler)
        self.window.thread_stderr.start()

    def shutdown(self):
        if hasattr(self.window, "thread_stdout"):
            self.window.thread_stdout.stop()
            self.window.thread_stdout.wait(500)
        if hasattr(self.window, "thread_stderr"):
            self.window.thread_stderr.stop()
            self.window.thread_stderr.wait(500)
        if hasattr(self.window, "_original_stdout"):
            sys.stdout = self.window._original_stdout
        if hasattr(self.window, "_original_stderr"):
            sys.stderr = self.window._original_stderr

    def _set_workflow_message(self, text):
        if hasattr(self.window, "workflow_status_label"):
            self.window.workflow_status_label.setText(text)

    def _sync_workspace_state(self):
        has_recording = bool(self.backend is not None and getattr(self.backend, "eeg_data", None) is not None)
        if hasattr(self.window, "view"):
            self.window.view.set_workspace_state(has_recording, self.get_biomarker_display_name())

    def _set_status_chip(self, attr_name, active, text):
        chip = getattr(self.window, attr_name, None)
        if chip is None:
            return
        chip.setText(text)
        chip.setProperty("active", active)
        chip.style().unpolish(chip)
        chip.style().polish(chip)

    def _unique_alive_widgets(self, widgets):
        unique = []
        seen = set()
        for widget in widgets or []:
            if not self._ui_object_is_alive(widget):
                continue
            widget_id = id(widget)
            if widget_id in seen:
                continue
            seen.add(widget_id)
            unique.append(widget)
        return unique

    def _busy_root_widgets(self):
        window = getattr(self, "window", None)
        if window is None:
            return []
        roots = [
            getattr(window, "widget", None),
            getattr(window, "widget_2", None),
            getattr(window, "case_console_panel", None),
            getattr(window, "decision_desk_panel", None),
            getattr(window, "inspector_dock", None),
            window.menuBar() if hasattr(window, "menuBar") else None,
        ]
        try:
            roots.extend(window.findChildren(QToolBar))
        except Exception:
            pass
        return self._unique_alive_widgets(roots)

    def _begin_busy_task(self, task_key, busy_text, buttons=None):
        if self._busy_task_state is not None:
            return False

        active_buttons = []
        for button in self._unique_alive_widgets(buttons):
            active_buttons.append(
                {
                    "widget": button,
                    "text": button.text(),
                    "tool_tip": button.toolTip(),
                    "status_tip": button.statusTip(),
                    "enabled": button.isEnabled(),
                    "busy_property": bool(button.property("workflowBusy")),
                }
            )
            button.setText(str(busy_text or "Working"))
            button.setToolTip(str(busy_text or "Working"))
            button.setStatusTip(str(busy_text or "Working"))
            set_dynamic_property(button, "workflowBusy", True)
            button.setEnabled(False)

        busy_roots = []
        for root in self._busy_root_widgets():
            busy_roots.append({"widget": root, "enabled": root.isEnabled()})
            root.setEnabled(False)

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._busy_task_state = {
            "task_key": str(task_key or ""),
            "buttons": active_buttons,
            "roots": busy_roots,
        }
        return True

    def _end_busy_task(self):
        state = self._busy_task_state
        if state is None:
            return
        self._busy_task_state = None

        if QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()

        for root_state in state.get("roots", []):
            widget = root_state.get("widget")
            if not self._ui_object_is_alive(widget):
                continue
            widget.setEnabled(bool(root_state.get("enabled", True)))

        for button_state in state.get("buttons", []):
            widget = button_state.get("widget")
            if not self._ui_object_is_alive(widget):
                continue
            widget.setText(str(button_state.get("text") or ""))
            widget.setToolTip(str(button_state.get("tool_tip") or ""))
            widget.setStatusTip(str(button_state.get("status_tip") or ""))
            set_dynamic_property(widget, "workflowBusy", bool(button_state.get("busy_property", False)))
            widget.setEnabled(bool(button_state.get("enabled", True)))

    def _filter_busy_buttons(self):
        return [getattr(self.window, "overview_filter_button", None)]

    def _detection_busy_buttons(self):
        sender_button = self.sender() if isinstance(self.sender(), (QPushButton, QToolButton)) else None
        detector_name = self._selected_detector_name()
        legacy_mapping = {
            "STE": getattr(self.window, "ste_detect_button", None),
            "MNI": getattr(self.window, "mni_detect_button", None),
            "HIL": getattr(self.window, "hil_detect_button", None),
            "YASA": getattr(self.window, "yasa_detect_button", None),
        }
        return self._unique_alive_widgets(
            [
                sender_button,
                getattr(self.window, "detector_run_button", None),
                legacy_mapping.get(detector_name),
            ]
        )

    def _classification_busy_buttons(self):
        sender_button = self.sender() if isinstance(self.sender(), (QPushButton, QToolButton)) else None
        return self._unique_alive_widgets(
            [
                sender_button,
                getattr(self.window, "classifier_run_button", None),
                getattr(self.window, "detect_all_button", None),
            ]
        )

    def _set_filter_controls_busy(self, busy):
        has_recording = bool(self.backend and getattr(self.backend, "eeg_data", None) is not None)
        filter_available = has_recording and self.biomarker_supports_detection()
        for attr_name in ("fp_input", "fs_input", "rp_input", "rs_input"):
            widget = getattr(self.window, attr_name, None)
            if widget is not None:
                widget.setEnabled(False if busy else filter_available)
        if busy:
            self._set_status_chip("status_filter_chip", True, "Filtering...")

    def _set_filtered_toggle_state(self, checked):
        checkbox = getattr(self.window, "toggle_filtered_checkbox", None)
        if checkbox is None:
            return
        blocker = QSignalBlocker(checkbox)
        checkbox.setChecked(bool(checked))
        del blocker

    def _handle_filter_worker_error(self, _error_tuple):
        self._set_filter_controls_busy(False)
        self.update_status_indicators()

    def update_status_indicators(self):
        has_file = bool(self.backend and self.backend.eeg_data is not None)
        selected_channels = 0
        if hasattr(self.window, "waveform_plot") and has_file:
            selected_channels = len(self.window.waveform_plot.get_channels_to_plot())
        has_channels = selected_channels > 0
        has_filter = bool(self.backend and hasattr(self.backend, "has_filtered_data") and self.backend.has_filtered_data())
        has_detect = bool(self.backend and getattr(self.backend, "detected", False))
        has_classify = bool(self.backend and getattr(self.backend, "classified", False))
        features = self._get_active_event_features()
        has_annotations = False
        if has_detect and features is not None:
            if hasattr(features, "get_review_progress"):
                try:
                    has_annotations = features.get_review_progress().get("reviewed", 0) > 0
                except Exception:
                    has_annotations = False
            if not has_annotations:
                has_annotations = bool(np.any(np.array(getattr(features, "annotated", np.array([]))) > 0))
        self._set_status_chip("status_loaded_chip", has_file, "Loaded" if has_file else "Load data")
        self._set_status_chip("status_channels_chip", has_channels, f"{selected_channels} channels" if has_channels else "Choose channels")
        self._set_status_chip("status_filter_chip", has_filter, "Filter ready" if has_filter else "Raw signal")
        self._set_status_chip("status_detect_chip", has_detect, "Detection complete" if has_detect else "Detect events")
        self._set_status_chip("status_classify_chip", has_classify, "Classified" if has_classify else "Classification optional")
        self._set_status_chip("status_annotate_chip", has_annotations, "Annotated" if has_annotations else "Review & annotate")
        self.update_overlap_review_button_state()
        self.update_waveform_toolbar_state()
        self.update_setup_action_state()

    def _get_active_event_features(self):
        if self.backend is None:
            return None
        session = getattr(self.backend, "analysis_session", None)
        if session is not None and hasattr(session, "get_active_run"):
            active_run = session.get_active_run()
            if active_run is not None and getattr(active_run, "event_features", None) is not None:
                return active_run.event_features
        return getattr(self.backend, "event_features", None)

    def _get_feature_event_count(self, features=None, *, raw=False):
        features = features if features is not None else self._get_active_event_features()
        if features is None:
            return 0
        if raw and hasattr(features, "get_raw_event_count"):
            try:
                return int(features.get_raw_event_count())
            except Exception:
                pass
        if not raw and hasattr(features, "get_num_biomarker"):
            try:
                return int(features.get_num_biomarker())
            except Exception:
                pass
        return int(len(getattr(features, "starts", [])))

    def _get_feature_channels(self, features=None):
        features = features if features is not None else self._get_active_event_features()
        if features is None:
            return []
        if hasattr(features, "get_visible_array"):
            try:
                return [str(channel) for channel in np.array(features.get_visible_array("channel_names")).tolist()]
            except Exception:
                pass
        return [str(channel) for channel in getattr(features, "channel_names", [])]

    def _get_feature_current_position(self, features=None):
        features = features if features is not None else self._get_active_event_features()
        if features is None:
            return 0
        if hasattr(features, "get_current_visible_position"):
            try:
                return int(features.get_current_visible_position())
            except Exception:
                pass
        return int(getattr(features, "index", 0))

    def update_overlap_review_button_state(self):
        button = getattr(self.window, "overlap_review_button", None)
        if button is None:
            return
        features = self._get_active_event_features() if self.biomarker_type == "HFO" else None
        raw_events = self._get_feature_event_count(features, raw=True)
        button.setVisible(self.biomarker_type == "HFO")
        button.setEnabled(bool(self.backend is not None and self.biomarker_type == "HFO" and raw_events > 0))
        if self.biomarker_type != "HFO":
            button.setToolTip("Cross-channel overlap review is only available for HFO runs.")
        elif raw_events == 0:
            button.setToolTip("Run HFO detection first, then review cross-channel overlaps for the active run.")
        else:
            button.setToolTip("Keep the first event in each overlap group, then tag or hide the later cross-channel overlaps.")

    def _set_button_checked(self, button, checked):
        if button is None:
            return
        blocker = QSignalBlocker(button)
        button.setChecked(bool(checked))
        del blocker

    def _set_report_export_enabled(self, enabled):
        if hasattr(self.window, "save_report_button"):
            self.window.save_report_button.setEnabled(bool(enabled))
        if hasattr(self.window, "run_stats_report_button"):
            self.window.run_stats_report_button.setEnabled(bool(enabled))

    def _get_auto_bipolar_metadata(self):
        if self.backend is None or not hasattr(self.backend, "get_auto_bipolar_metadata"):
            return {}
        try:
            return self.backend.get_auto_bipolar_metadata() or {}
        except Exception:
            return {}

    def _format_chain_name(self, chain_name):
        return str(chain_name or "").replace("_", " ").strip().title()

    def _format_auto_bipolar_button_tooltip(self, metadata):
        default_tooltip = "Automatically build conventional EEG bipolar chains or adjacent iEEG bipolar channels"
        if not metadata:
            return default_tooltip
        montage_kind = str(metadata.get("montage_kind", "") or "")
        pair_count = len(metadata.get("present_pairs", []) or [])
        if montage_kind == "conventional_eeg":
            tooltip = f"Apply conventional EEG bipolar montage ({pair_count} valid pairs)"
            chain_breaks = metadata.get("chain_breaks", []) or []
            if chain_breaks:
                break_names = ", ".join(self._format_chain_name(entry.get("chain_name")) for entry in chain_breaks[:3])
                tooltip += f". Broken chains: {break_names}"
            return tooltip
        if montage_kind == "adjacent_contacts":
            return f"Apply adjacent-contact bipolar montage ({pair_count} valid pairs)"
        return default_tooltip

    def _join_preview_list(self, items, limit=4):
        values = [str(item) for item in (items or []) if str(item)]
        if not values:
            return ""
        if len(values) <= limit:
            return ", ".join(values)
        hidden_count = len(values) - limit
        return ", ".join(values[:limit]) + f", +{hidden_count} more"

    def _auto_bipolar_badge_text(self, metadata):
        montage_kind = str(metadata.get("montage_kind", "") or "")
        if montage_kind == "conventional_eeg":
            return "Double Banana"
        if montage_kind == "adjacent_contacts":
            return "Adj. Bipolar"
        return ""

    def _format_measurement_duration(self, duration_seconds):
        duration_seconds = abs(float(duration_seconds or 0.0))
        if duration_seconds < 1.0:
            return f"{duration_seconds * 1000.0:.1f} ms"
        return f"{duration_seconds:.3f} s"

    def _format_measurement_amplitude(self, amplitude_uv):
        return f"{abs(float(amplitude_uv or 0.0)):.1f} uV"

    def _summarize_waveform_measurement(self, points):
        if len(points or []) < 2:
            return None
        point_a = dict(points[0])
        point_b = dict(points[1])
        duration_seconds = abs(float(point_b["time_seconds"]) - float(point_a["time_seconds"]))
        amplitude_delta_uv = abs(float(point_b["raw_value_uv"]) - float(point_a["raw_value_uv"]))
        same_channel = str(point_a.get("channel_name") or "") == str(point_b.get("channel_name") or "")
        return {
            "point_a": point_a,
            "point_b": point_b,
            "duration_seconds": duration_seconds,
            "amplitude_delta_uv": amplitude_delta_uv,
            "same_channel": same_channel,
            "badge_text": f"dt {self._format_measurement_duration(duration_seconds)} | dA {self._format_measurement_amplitude(amplitude_delta_uv)}",
            "overlay_text": f"dt {self._format_measurement_duration(duration_seconds)} | dA {self._format_measurement_amplitude(amplitude_delta_uv)}",
        }

    def _measurement_badge_payload(self):
        points = list(self.waveform_measurement_state.get("points") or [])
        summary = self.waveform_measurement_state.get("summary") or {}
        if len(points) >= 2 and summary:
            point_a = summary.get("point_a", {})
            point_b = summary.get("point_b", {})
            tooltip = (
                f"A: {point_a.get('channel_name', '')} @ {float(point_a.get('time_seconds', 0.0)):.4f} s"
                f" ({self._format_measurement_amplitude(point_a.get('raw_value_uv', 0.0))})\n"
                f"B: {point_b.get('channel_name', '')} @ {float(point_b.get('time_seconds', 0.0)):.4f} s"
                f" ({self._format_measurement_amplitude(point_b.get('raw_value_uv', 0.0))})"
            )
            return {
                "text": str(summary.get("badge_text") or ""),
                "tooltip": tooltip,
                "overlay_text": str(summary.get("overlay_text") or ""),
            }
        if len(points) == 1:
            point = points[0]
            return {
                "text": "Click B",
                "tooltip": (
                    f"A: {point.get('channel_name', '')} @ {float(point.get('time_seconds', 0.0)):.4f} s"
                    f" ({self._format_measurement_amplitude(point.get('raw_value_uv', 0.0))})\n"
                    "Click a second point to measure interval and amplitude difference."
                ),
                "overlay_text": "Click second point",
            }
        return {
            "text": "Click A",
            "tooltip": "Click the first waveform point to start a measurement.",
            "overlay_text": "",
        }

    def _sync_waveform_measurement_overlay(self):
        plot = getattr(self.window, "waveform_plot", None)
        if plot is None or not hasattr(plot, "set_measurement"):
            return
        payload = self._measurement_badge_payload()
        points = list(self.waveform_measurement_state.get("points") or [])
        if points:
            plot.set_measurement(points, summary_text=payload.get("overlay_text", ""))
        elif hasattr(plot, "clear_measurement"):
            plot.clear_measurement()

    def _reset_waveform_measurement_state(self, *, sync_overlay=True):
        self.waveform_measurement_state = {"points": [], "summary": None}
        if sync_overlay:
            self._sync_waveform_measurement_overlay()

    def _format_auto_bipolar_detail_dialog(self, metadata):
        if not metadata:
            return {
                "title": "Montage Details",
                "text": "No auto bipolar montage is available yet.",
                "informative_text": "Load a recording to inspect the available montage pairs.",
                "detailed_text": "",
            }

        montage_kind = str(metadata.get("montage_kind", "") or "")
        present_pairs = metadata.get("present_pairs", []) or []
        missing_pairs = metadata.get("missing_pairs", []) or []
        chain_summaries = metadata.get("chain_summaries", []) or []
        chain_breaks = metadata.get("chain_breaks", []) or []
        normalized_channels = metadata.get("normalized_channels", []) or []
        warnings = metadata.get("warnings", []) or []
        pair_count = len(present_pairs)
        total_pairs = pair_count + len(missing_pairs)

        if montage_kind == "conventional_eeg":
            title = "Montage Details"
            text = "Double Banana montage ready" if pair_count else "Double Banana montage unavailable"
        elif montage_kind == "adjacent_contacts":
            title = "Montage Details"
            text = "Adjacent-contact bipolar montage ready" if pair_count else "Adjacent-contact bipolar montage unavailable"
        else:
            title = "Montage Details"
            text = "Auto bipolar montage summary"

        informative_lines = []
        if total_pairs:
            informative_lines.append(f"Valid pairs: {pair_count} / {total_pairs}")
        else:
            informative_lines.append(f"Valid pairs: {pair_count}")

        visible_chains = [
            self._format_chain_name(entry.get("chain_name"))
            for entry in chain_summaries
            if entry.get("is_visible")
        ]
        if visible_chains:
            informative_lines.append(f"Visible chains: {self._join_preview_list(visible_chains, limit=5)}")

        broken_chain_names = [
            self._format_chain_name(entry.get("chain_name"))
            for entry in chain_breaks
            if entry.get("chain_name")
        ]
        if broken_chain_names:
            informative_lines.append(f"Broken chains: {self._join_preview_list(broken_chain_names, limit=4)}")

        alias_substitutions = []
        for channel_info in normalized_channels:
            canonical_name = str(channel_info.get("canonical_name") or "")
            clean_name = str(channel_info.get("clean_name") or "")
            if channel_info.get("uses_alias") and canonical_name and clean_name:
                alias_substitutions.append(f"{canonical_name} <- {clean_name}")
        if alias_substitutions:
            informative_lines.append(
                f"Alias substitutions: {self._join_preview_list(sorted(set(alias_substitutions)), limit=4)}"
            )

        unrecognized_channels = []
        for warning in warnings:
            if warning.get("type") == "unrecognized_channels":
                unrecognized_channels.extend(str(channel) for channel in (warning.get("channels") or []) if str(channel))
        if unrecognized_channels:
            informative_lines.append(
                f"Ignored channels: {self._join_preview_list(unrecognized_channels, limit=4)}"
            )

        detail_sections = []
        if present_pairs:
            present_labels = [str(entry.get("display_name") or "") for entry in present_pairs if str(entry.get("display_name") or "")]
            detail_sections.append("Present pairs")
            detail_sections.extend(f"- {label}" for label in present_labels)

        if chain_breaks:
            detail_sections.append("")
            detail_sections.append("Broken chains")
            for entry in chain_breaks:
                chain_name = self._format_chain_name(entry.get("chain_name"))
                missing = [str(label) for label in (entry.get("missing_pairs") or []) if str(label)]
                if missing:
                    detail_sections.append(f"- {chain_name}: {', '.join(missing)}")
                else:
                    detail_sections.append(f"- {chain_name}")

        if missing_pairs:
            detail_sections.append("")
            detail_sections.append("Missing pairs")
            detail_sections.extend(
                f"- {entry.get('display_name')}"
                for entry in missing_pairs
                if str(entry.get("display_name") or "")
            )

        if alias_substitutions:
            detail_sections.append("")
            detail_sections.append("Alias substitutions")
            detail_sections.extend(f"- {label}" for label in sorted(set(alias_substitutions)))

        if unrecognized_channels:
            detail_sections.append("")
            detail_sections.append("Ignored input channels")
            detail_sections.extend(f"- {channel}" for channel in unrecognized_channels)

        detailed_text = "\n".join(detail_sections).strip()
        return {
            "title": title,
            "text": text,
            "informative_text": "\n".join(informative_lines).strip(),
            "detailed_text": detailed_text,
        }

    def show_auto_bipolar_details(self):
        dialog_copy = self._format_auto_bipolar_detail_dialog(self._get_auto_bipolar_metadata())
        msg = QMessageBox(self.window)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle(dialog_copy["title"])
        msg.setText(dialog_copy["text"])
        msg.setInformativeText(dialog_copy["informative_text"])
        if dialog_copy["detailed_text"]:
            msg.setDetailedText(dialog_copy["detailed_text"])
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def _build_waveform_channel_presentation(self, metadata):
        presentation = {}
        for entry in metadata.get("present_pairs", []) or []:
            derived_name = str(entry.get("derived_name") or "")
            display_name = str(entry.get("display_name") or derived_name)
            if not derived_name:
                continue
            tooltip_lines = [display_name]
            source_channel_1 = str(entry.get("source_channel_1") or "")
            source_channel_2 = str(entry.get("source_channel_2") or "")
            if source_channel_1 and source_channel_2:
                tooltip_lines.append(f"Source: {source_channel_1} - {source_channel_2}")
            elif source_channel_1:
                tooltip_lines.append(f"Source: {source_channel_1}")
            chain_name = self._format_chain_name(entry.get("chain_name"))
            if chain_name and chain_name not in {"Adjacent Contacts", "Average Reference"}:
                tooltip_lines.append(f"Chain: {chain_name}")
            presentation[derived_name] = {
                "display_label": display_name,
                "tooltip": "\n".join(tooltip_lines),
            }
        return presentation

    def _sync_waveform_channel_presentation(self, metadata_override=None):
        plot = getattr(self.window, "waveform_plot", None)
        if plot is None or not hasattr(plot, "set_channel_presentation"):
            return
        metadata = metadata_override
        if metadata is None:
            current_channels = self._current_waveform_channel_names()
            auto_channels = self._auto_bipolar_waveform_channel_names()
            average_reference_channels = self._average_reference_waveform_channel_names()
            if auto_channels and current_channels == auto_channels:
                metadata = self._get_auto_bipolar_metadata()
            elif average_reference_channels and current_channels == average_reference_channels:
                metadata = self._get_average_reference_metadata()
            else:
                metadata = {}
        presentation = self._build_waveform_channel_presentation(metadata)
        if presentation:
            plot.set_channel_presentation(presentation)
        elif hasattr(plot, "clear_channel_presentation"):
            plot.clear_channel_presentation()

    def _channel_subset_matches(self, candidate_channels, *, current_channels=None, minimum_size=1):
        current_channels = (
            [str(channel) for channel in (current_channels or [])]
            if current_channels is not None
            else self._current_waveform_channel_names()
        )
        normalized_candidates = [str(channel) for channel in (candidate_channels or []) if str(channel)]
        return bool(len(normalized_candidates) >= minimum_size and current_channels == normalized_candidates)

    def _channel_subset_within(self, candidate_channels, *, current_channels=None, minimum_size=1):
        current_channels = (
            [str(channel) for channel in (current_channels or [])]
            if current_channels is not None
            else self._current_waveform_channel_names()
        )
        normalized_candidates = {str(channel) for channel in (candidate_channels or []) if str(channel)}
        return bool(
            len(current_channels) >= minimum_size
            and normalized_candidates
            and all(str(channel) in normalized_candidates for channel in current_channels)
        )

    def _is_recording_scope_active(self, current_channels=None):
        return self._channel_subset_matches(
            self._recording_waveform_channel_names(),
            current_channels=current_channels,
        )

    def _is_event_scope_active(self, event_channels, current_channels=None):
        return self._channel_subset_matches(
            event_channels,
            current_channels=current_channels,
        )

    def _hotspot_focus_channel_names(self, limit=8):
        if self.backend is None or not hasattr(self.backend, "get_channel_ranking"):
            return []
        active_run = getattr(getattr(self.backend, "analysis_session", None), "get_active_run", lambda: None)()
        accepted_run = getattr(getattr(self.backend, "analysis_session", None), "get_accepted_run", lambda: None)()
        ranking_run_id = getattr(accepted_run, "run_id", None) or getattr(active_run, "run_id", None)
        ranking = self.backend.get_channel_ranking(ranking_run_id) or []
        prioritized = [row for row in ranking if row.get("accepted_predicted", row.get("total_events", 0)) > 0]
        selected_rows = prioritized[: int(limit)] or ranking[: max(1, min(int(limit), len(ranking)))]
        return [str(row["channel_name"]) for row in selected_rows if row.get("channel_name")]

    def _is_hotspot_scope_active(self, current_channels=None, limit=8):
        return self._channel_subset_matches(
            self._hotspot_focus_channel_names(limit=limit),
            current_channels=current_channels,
        )

    def _abbreviate_channel_name(self, channel_name, *, max_length=18):
        channel_name = str(channel_name or "").strip()
        if not channel_name:
            return ""
        if len(channel_name) <= max_length:
            return channel_name
        return f"{channel_name[: max_length - 1]}..."

    def _waveform_source_badge_payload(
        self,
        *,
        has_recording,
        current_channels,
        average_reference_channels,
        auto_bipolar_channels,
        auto_bipolar_metadata,
    ):
        if not has_recording:
            return {
                "text": "Source --",
                "tooltip": "Load a recording to inspect waveform source modes.",
            }

        if self._channel_subset_within(auto_bipolar_channels, current_channels=current_channels):
            badge_label = self._auto_bipolar_badge_text(auto_bipolar_metadata) or "Auto bipolar"
            tooltip = (
                "Reviewing derived bipolar channels. Click Auto Bp again, Ref, or Reset View to return."
            )
            return {
                "text": f"Source {badge_label}",
                "tooltip": tooltip,
            }

        if self._channel_subset_within(average_reference_channels, current_channels=current_channels):
            return {
                "text": "Source Avg ref",
                "tooltip": "Reviewing average-reference derived channels. Click Avg Ref again, Ref, or Reset View to return.",
            }

        return {
            "text": "Source Ref",
            "tooltip": "Reviewing referential source channels.",
        }

    def _waveform_scope_badge_payload(
        self,
        *,
        has_recording,
        current_channels,
        event_channels,
        highlighted_channel,
    ):
        if not has_recording:
            return {
                "text": "Scope --",
                "tooltip": "Load a recording to inspect waveform channel scopes.",
            }

        highlighted_channel = str(highlighted_channel or "").strip()
        if self._is_neighbor_focus_active():
            label = self._abbreviate_channel_name(highlighted_channel) or "selected"
            return {
                "text": f"Scope Neighbors {label}",
                "tooltip": (
                    f"Showing the highlighted channel and adjacent neighbors around {highlighted_channel or 'the current selection'}. "
                    "Click Neighbors again, All, or Reset View to return."
                ),
            }

        if self._is_hotspot_scope_active(current_channels=current_channels):
            hotspot_channels = self._hotspot_focus_channel_names()
            top_label = self._abbreviate_channel_name(hotspot_channels[0]) if hotspot_channels else ""
            count_suffix = f" ({len(hotspot_channels)})" if len(hotspot_channels) > 1 else ""
            tooltip = "Showing the highest-priority review channels from the active or accepted run. Click Hotspot again, All, or Reset View to return."
            if top_label:
                tooltip = f"Showing hotspot review channels led by {hotspot_channels[0]}. Click Hotspot again, All, or Reset View to return."
            return {
                "text": f"Scope Hotspot{count_suffix}" if not top_label else f"Scope Hotspot {top_label}",
                "tooltip": tooltip,
            }

        if self._is_clean_view_active():
            excluded_count = len(self._get_clean_recording_metadata().get("excluded_channels") or [])
            suffix = f" ({excluded_count} hidden)" if excluded_count > 0 else ""
            return {
                "text": f"Scope Clean{suffix}",
                "tooltip": "Showing only clean source channels. Click Clean again, All, or Reset View to return.",
            }

        if self._is_event_scope_active(event_channels, current_channels=current_channels):
            return {
                "text": f"Scope Events ({len(event_channels)})",
                "tooltip": "Showing only channels with detected events in the active run. Click Events again, All, or Reset View to return.",
            }

        if self._highlight_channel_mode and highlighted_channel:
            return {
                "text": f"Scope Highlight {self._abbreviate_channel_name(highlighted_channel)}",
                "tooltip": (
                    f"Highlighting {highlighted_channel} while keeping the current channel list intact. "
                    "Click Highlight again or Reset View to clear it."
                ),
            }

        if self._is_recording_scope_active(current_channels=current_channels):
            return {
                "text": "Scope All",
                "tooltip": "Showing the full referential source channel list.",
            }

        return {
            "text": f"Scope Custom ({len(current_channels)})",
            "tooltip": "Showing a custom channel subset. Use All or Reset View to return to the default source list.",
        }

    def _waveform_tool_badge_payload(self, *, has_recording):
        if not has_recording:
            return {
                "text": "Tool --",
                "tooltip": "Load a recording to inspect waveform tools.",
            }
        plot = getattr(self.window, "waveform_plot", None)
        if plot is not None and hasattr(plot, "is_measurement_enabled") and plot.is_measurement_enabled():
            return {
                "text": "Tool Measure",
                "tooltip": "Measure mode is active. Click Measure again, press Esc, or use Reset View to exit.",
            }
        if plot is not None and hasattr(plot, "is_cursor_enabled") and plot.is_cursor_enabled():
            return {
                "text": "Tool Cursor",
                "tooltip": "Cursor mode is active. Click Cursor again, press Esc, or use Reset View to exit.",
            }
        return {
            "text": "Tool Browse",
            "tooltip": "No sticky inspect tool is active.",
        }

    def _update_waveform_context_badges(
        self,
        *,
        has_recording,
        current_channels,
        event_channels,
        highlighted_channel,
        average_reference_channels,
        auto_bipolar_channels,
        auto_bipolar_metadata,
    ):
        source_payload = self._waveform_source_badge_payload(
            has_recording=has_recording,
            current_channels=current_channels,
            average_reference_channels=average_reference_channels,
            auto_bipolar_channels=auto_bipolar_channels,
            auto_bipolar_metadata=auto_bipolar_metadata,
        )
        scope_payload = self._waveform_scope_badge_payload(
            has_recording=has_recording,
            current_channels=current_channels,
            event_channels=event_channels,
            highlighted_channel=highlighted_channel,
        )
        tool_payload = self._waveform_tool_badge_payload(has_recording=has_recording)

        if hasattr(self.window, "waveform_source_mode_badge"):
            self._set_toolbar_slot_text(
                self.window.waveform_source_mode_badge,
                source_payload["text"],
                tooltip=source_payload["tooltip"],
            )
        if hasattr(self.window, "waveform_scope_mode_badge"):
            self._set_toolbar_slot_text(
                self.window.waveform_scope_mode_badge,
                scope_payload["text"],
                tooltip=scope_payload["tooltip"],
            )
        if hasattr(self.window, "waveform_tool_mode_badge"):
            self._set_toolbar_slot_text(
                self.window.waveform_tool_mode_badge,
                tool_payload["text"],
                tooltip=tool_payload["tooltip"],
            )

        if hasattr(self.window, "waveform_reset_view_button"):
            reset_available = bool(
                has_recording
                and (
                    not self._is_recording_scope_active(current_channels=current_channels)
                    or self._channel_subset_within(auto_bipolar_channels, current_channels=current_channels)
                    or self._channel_subset_within(average_reference_channels, current_channels=current_channels)
                    or self._highlight_channel_mode
                    or tool_payload["text"] != "Tool Browse"
                )
            )
            reset_tooltip = "Return to the referential source channels and clear sticky waveform tools."
            if not reset_available:
                reset_tooltip = "Already in the default waveform view."
            self.window.waveform_reset_view_button.setEnabled(reset_available)
            self.window.waveform_reset_view_button.setToolTip(reset_tooltip)
            self.window.waveform_reset_view_button.setStatusTip(reset_tooltip)

    def _update_channel_source_toolbar_state(
        self,
        *,
        has_recording,
        current_channels,
        recording_channels,
        average_reference_channels,
        auto_bipolar_channels,
        auto_bipolar_metadata,
    ):
        if hasattr(self.window, "review_channels_button"):
            self.window.review_channels_button.setEnabled(self.window.Choose_Channels_Button.isEnabled())
        if hasattr(self.window, "montage_tool_button"):
            self.window.montage_tool_button.setEnabled(self.window.bipolar_button.isEnabled())
        if hasattr(self.window, "referential_tool_button"):
            self.window.referential_tool_button.setEnabled(has_recording)
            referential_active = self._channel_subset_within(recording_channels, current_channels=current_channels)
            self._set_button_checked(
                self.window.referential_tool_button,
                referential_active,
            )
            referential_tooltip = "Return to the referential source channel view."
            if referential_active:
                referential_tooltip = "Already reviewing referential source channels."
            self.window.referential_tool_button.setToolTip(referential_tooltip)
            self.window.referential_tool_button.setStatusTip(referential_tooltip)
        if hasattr(self.window, "average_reference_button"):
            average_reference_enabled = bool(has_recording and len(average_reference_channels) >= 2)
            self.window.average_reference_button.setEnabled(average_reference_enabled)
            average_reference_active = self._channel_subset_within(
                average_reference_channels,
                current_channels=current_channels,
            )
            self._set_button_checked(
                self.window.average_reference_button,
                average_reference_active,
            )
            average_reference_tooltip = "Show average-reference derived channels"
            if average_reference_active:
                average_reference_tooltip = "Showing average-reference derived channels. Click again, Ref, or Reset View to return."
            elif average_reference_enabled:
                average_reference_tooltip = (
                    f"Show average-reference derived channels ({len(average_reference_channels)} channels)"
                )
            self.window.average_reference_button.setToolTip(average_reference_tooltip)
            self.window.average_reference_button.setStatusTip(average_reference_tooltip)
        if hasattr(self.window, "auto_bipolar_button"):
            self.window.auto_bipolar_button.setEnabled(bool(has_recording and auto_bipolar_channels))
            auto_bipolar_active = self._channel_subset_within(
                auto_bipolar_channels,
                current_channels=current_channels,
            )
            self._set_button_checked(
                self.window.auto_bipolar_button,
                auto_bipolar_active,
            )
            auto_bipolar_tooltip = self._format_auto_bipolar_button_tooltip(auto_bipolar_metadata)
            if auto_bipolar_active:
                auto_bipolar_tooltip += ". Click again, Ref, or Reset View to return."
            self.window.auto_bipolar_button.setToolTip(auto_bipolar_tooltip)
            self.window.auto_bipolar_button.setStatusTip(auto_bipolar_tooltip)
        if hasattr(self.window, "montage_status_badge"):
            badge_text = self._auto_bipolar_badge_text(auto_bipolar_metadata) if has_recording else ""
            status_tooltip = self._format_auto_bipolar_button_tooltip(auto_bipolar_metadata) if badge_text else ""
            if status_tooltip:
                status_tooltip += "\nClick for details"
            self._set_toolbar_slot_text(self.window.montage_status_badge, badge_text, tooltip=status_tooltip)
            self.window.montage_status_badge.setVisible(bool(badge_text))
            self.window.montage_status_badge.setEnabled(bool(badge_text))
        if hasattr(self.window, "montage_status_slot"):
            self.window.montage_status_slot.setVisible(bool(badge_text))
        if hasattr(self.window, "montage_break_badge"):
            chain_breaks = auto_bipolar_metadata.get("chain_breaks", []) if isinstance(auto_bipolar_metadata, dict) else []
            break_count = len(chain_breaks or [])
            break_text = "" if break_count == 0 else f"{break_count} Break" + ("s" if break_count != 1 else "")
            if break_text:
                break_names = ", ".join(self._format_chain_name(entry.get("chain_name")) for entry in (chain_breaks or [])[:4])
                tooltip = f"Broken chains: {break_names}\nClick for details"
            else:
                tooltip = ""
            self._set_toolbar_slot_text(self.window.montage_break_badge, break_text, tooltip=tooltip)
            self.window.montage_break_badge.setVisible(bool(break_text))
            self.window.montage_break_badge.setEnabled(bool(break_text))
        if hasattr(self.window, "montage_break_slot"):
            self.window.montage_break_slot.setVisible(bool(break_text))

    def _update_channel_focus_toolbar_state(
        self,
        *,
        has_recording,
        current_channels,
        highlighted_channel,
        total_events,
        event_channels,
    ):
        clean_metadata = self._get_clean_recording_metadata() if has_recording else {}
        clean_channels = [str(channel) for channel in (clean_metadata.get("clean_channels") or [])]
        excluded_channels = [str(channel) for channel in (clean_metadata.get("excluded_channels") or [])]
        clean_enabled = bool(has_recording and clean_channels and excluded_channels)

        if hasattr(self.window, "highlight_channel_button"):
            self.window.highlight_channel_button.setEnabled(bool(has_recording and highlighted_channel))
            highlight_active = self._is_highlight_focus_active()
            self._set_button_checked(self.window.highlight_channel_button, highlight_active)
            highlight_tooltip = "Highlight the selected channel without hiding the other visible channels"
            if highlighted_channel:
                highlight_tooltip = f"Highlight {highlighted_channel} without hiding the other visible channels."
            if highlight_active:
                highlight_tooltip += " Click again or Reset View to clear it."
            self.window.highlight_channel_button.setToolTip(highlight_tooltip)
            self.window.highlight_channel_button.setStatusTip(highlight_tooltip)
        if hasattr(self.window, "neighbor_channels_button"):
            self.window.neighbor_channels_button.setEnabled(bool(has_recording and highlighted_channel))
            neighbor_active = self._is_neighbor_focus_active()
            self._set_button_checked(self.window.neighbor_channels_button, neighbor_active)
            neighbor_tooltip = "Focus the highlighted channel together with adjacent channels"
            if highlighted_channel:
                neighbor_tooltip = f"Focus {highlighted_channel} together with adjacent channels."
            if neighbor_active:
                neighbor_tooltip += " Click again, All, or Reset View to return."
            self.window.neighbor_channels_button.setToolTip(neighbor_tooltip)
            self.window.neighbor_channels_button.setStatusTip(neighbor_tooltip)
        if hasattr(self.window, "clean_view_button"):
            clean_active = self._channel_subset_matches(clean_channels, current_channels=current_channels)
            self.window.clean_view_button.setEnabled(clean_enabled)
            self._set_button_checked(
                self.window.clean_view_button,
                clean_active,
            )
            tooltip = "Hide explicitly bad or flat source channels"
            if clean_active:
                tooltip = "Showing only clean source channels. Click again, All, or Reset View to return."
            elif excluded_channels:
                tooltip = f"Hide {len(excluded_channels)} bad/flat channels from the source view"
            self.window.clean_view_button.setToolTip(tooltip)
            self.window.clean_view_button.setStatusTip(tooltip)
        if hasattr(self.window, "event_channels_button"):
            self.window.event_channels_button.setEnabled(total_events > 0)
            self._set_button_checked(
                self.window.event_channels_button,
                self._is_event_scope_active(event_channels, current_channels=current_channels),
            )
            event_scope_active = self._is_event_scope_active(event_channels, current_channels=current_channels)
            event_tooltip = "Only show channels with detected events in the active run"
            if event_scope_active:
                event_tooltip = "Showing only channels with detected events. Click again, All, or Reset View to return."
            elif total_events > 0:
                event_tooltip = f"Only show the {len(event_channels)} channels with detected events in the active run."
            self.window.event_channels_button.setToolTip(event_tooltip)
            self.window.event_channels_button.setStatusTip(event_tooltip)
        if hasattr(self.window, "all_channels_button"):
            self.window.all_channels_button.setEnabled(has_recording)
            self._set_button_checked(
                self.window.all_channels_button,
                self._is_recording_scope_active(current_channels=current_channels),
            )
            all_tooltip = "Return to the full referential source channel list."
            if self._is_recording_scope_active(current_channels=current_channels):
                all_tooltip = "Already showing the full referential source channel list."
            self.window.all_channels_button.setToolTip(all_tooltip)
            self.window.all_channels_button.setStatusTip(all_tooltip)
        if hasattr(self.window, "hotspot_tool_button"):
            hotspot_enabled = bool(has_recording and self._hotspot_focus_channel_names())
            self.window.hotspot_tool_button.setEnabled(hotspot_enabled)
            self._set_button_checked(
                self.window.hotspot_tool_button,
                self._is_hotspot_scope_active(current_channels=current_channels),
            )
            hotspot_tooltip = "Focus the most active channel set in the active or accepted run"
            if self._is_hotspot_scope_active(current_channels=current_channels):
                hotspot_tooltip = "Showing hotspot channels. Click again, All, or Reset View to return."
            elif hotspot_enabled:
                hotspot_channels = self._hotspot_focus_channel_names()
                hotspot_tooltip = (
                    f"Focus the top {len(hotspot_channels)} review channels in the active or accepted run."
                )
            self.window.hotspot_tool_button.setToolTip(hotspot_tooltip)
            self.window.hotspot_tool_button.setStatusTip(hotspot_tooltip)

    def _update_visible_channel_preset_state(self, *, has_recording):
        current_window = float(self.window.display_time_window_input.value()) if hasattr(self.window, "display_time_window_input") else 0.0
        visible_count = int(self.window.n_channel_input.value()) if hasattr(self.window, "n_channel_input") else 0
        max_visible_count = int(self.window.n_channel_input.maximum()) if hasattr(self.window, "n_channel_input") else 0

        for button in getattr(self.window, "graph_window_preset_buttons", []):
            preset = float(button.property("windowPreset") or 0.0)
            button.setEnabled(has_recording)
            self._set_button_checked(button, has_recording and abs(current_window - preset) < 1e-6)
        for button in getattr(self.window, "graph_channel_preset_buttons", []):
            preset = button.property("channelPreset")
            is_all = str(preset) == "all"
            enabled = bool(has_recording and max_visible_count > 0 and (is_all or max_visible_count >= int(preset)))
            button.setEnabled(enabled)
            checked = bool(
                enabled and (
                    (is_all and visible_count == max_visible_count)
                    or (not is_all and visible_count == int(preset))
                )
            )
            self._set_button_checked(button, checked)
        return current_window, visible_count, max_visible_count

    def update_waveform_toolbar_state(self):
        has_recording = bool(self.backend and getattr(self.backend, "eeg_data", None) is not None)
        features = self._get_active_event_features()
        total_events = self._get_feature_event_count(features)
        current_channels = self._current_waveform_channel_names() if has_recording else []
        recording_channels = self._recording_waveform_channel_names() if has_recording else []
        auto_bipolar_channels = self._auto_bipolar_waveform_channel_names() if has_recording else []
        average_reference_channels = self._average_reference_waveform_channel_names() if has_recording else []
        event_channels = []
        if features is not None:
            event_channels = list(dict.fromkeys(self._get_feature_channels(features)))
        review_progress = (
            features.get_review_progress()
            if features is not None and hasattr(features, "get_review_progress")
            else {"reviewed": 0, "remaining": 0, "total": total_events}
        )
        auto_bipolar_metadata = self._get_auto_bipolar_metadata() if has_recording else {}
        highlighted_channel = self._current_highlighted_waveform_channel()
        active_run = getattr(getattr(self.backend, "analysis_session", None), "get_active_run", lambda: None)() if self.backend else None
        accepted_run = getattr(getattr(self.backend, "analysis_session", None), "get_accepted_run", lambda: None)() if self.backend else None
        ranking_run_id = getattr(accepted_run, "run_id", None) or getattr(active_run, "run_id", None)
        ranking = self.backend.get_channel_ranking(ranking_run_id) if self.backend and hasattr(self.backend, "get_channel_ranking") else []
        top_channel = ranking[0] if ranking else None
        available_scopes = (
            features.get_prediction_scope_options()
            if features is not None and hasattr(features, "get_prediction_scope_options")
            else []
        )
        preferred_scope_labels = {
            "HFO": ["Artifact", "spkHFO", "eHFO"],
            "Spindle": ["Artifact", "Spike", "Non-artifact"],
            "Spike": [],
        }.get(self.biomarker_type or "", [])

        if hasattr(self.window, "normalize_tool_button"):
            self.window.normalize_tool_button.setEnabled(has_recording)
            self._set_button_checked(self.window.normalize_tool_button, self.window.normalize_vertical_input.isChecked())
        if hasattr(self.window, "raw_tool_button"):
            raw_enabled = has_recording
            raw_checked = bool(has_recording and not self.window.toggle_filtered_checkbox.isChecked())
            self.window.raw_tool_button.setEnabled(raw_enabled)
            self._set_button_checked(self.window.raw_tool_button, raw_checked)
        if hasattr(self.window, "filtered_tool_button"):
            self.window.filtered_tool_button.setEnabled(self.window.toggle_filtered_checkbox.isEnabled())
            self._set_button_checked(self.window.filtered_tool_button, self.window.toggle_filtered_checkbox.isChecked())
        if hasattr(self.window, "filter60_tool_button"):
            self.window.filter60_tool_button.setEnabled(self.window.Filter60Button.isEnabled())
            self._set_button_checked(self.window.filter60_tool_button, self.window.Filter60Button.isChecked())
        self._update_channel_source_toolbar_state(
            has_recording=has_recording,
            current_channels=current_channels,
            recording_channels=recording_channels,
            average_reference_channels=average_reference_channels,
            auto_bipolar_channels=auto_bipolar_channels,
            auto_bipolar_metadata=auto_bipolar_metadata,
        )
        self._update_channel_focus_toolbar_state(
            has_recording=has_recording,
            current_channels=current_channels,
            highlighted_channel=highlighted_channel,
            total_events=total_events,
            event_channels=event_channels,
        )
        if hasattr(self.window, "go_to_time_input"):
            self.window.go_to_time_input.setEnabled(has_recording)
        if hasattr(self.window, "go_to_time_button"):
            self.window.go_to_time_button.setEnabled(has_recording)
        if hasattr(self.window, "zoom_in_button"):
            self.window.zoom_in_button.setEnabled(has_recording)
        if hasattr(self.window, "zoom_out_button"):
            self.window.zoom_out_button.setEnabled(has_recording)
        if hasattr(self.window, "snapshot_button"):
            self.window.snapshot_button.setEnabled(has_recording)
        if hasattr(self.window, "open_review_button"):
            self.window.open_review_button.setEnabled(self.window.annotation_button.isEnabled())
        if hasattr(self.window, "pending_event_button"):
            pending_supported = bool(features is not None and hasattr(features, "get_next_unannotated"))
            self.window.pending_event_button.setEnabled(
                bool(total_events > 0 and review_progress["remaining"] > 0 and pending_supported)
            )
        if hasattr(self.window, "cursor_tool_button"):
            cursor_enabled = bool(has_recording and hasattr(self.window, "waveform_plot") and self.window.waveform_plot.is_cursor_enabled())
            self.window.cursor_tool_button.setEnabled(has_recording)
            self._set_button_checked(self.window.cursor_tool_button, cursor_enabled)
            cursor_tooltip = "Show a live crosshair cursor over the waveform"
            if cursor_enabled:
                cursor_tooltip = "Cursor mode is active. Click Cursor again, press Esc, or use Reset View to exit."
            self.window.cursor_tool_button.setToolTip(cursor_tooltip)
            self.window.cursor_tool_button.setStatusTip(cursor_tooltip)
        if hasattr(self.window, "measure_tool_button"):
            measure_enabled = bool(has_recording and hasattr(self.window, "waveform_plot") and self.window.waveform_plot.is_measurement_enabled())
            self.window.measure_tool_button.setEnabled(has_recording)
            self._set_button_checked(self.window.measure_tool_button, measure_enabled)
            measure_tooltip = "Click two waveform points to measure interval and amplitude difference"
            if measure_enabled:
                measure_tooltip = "Measure mode is active. Click Measure again, press Esc, or use Reset View to exit."
            self.window.measure_tool_button.setToolTip(measure_tooltip)
            self.window.measure_tool_button.setStatusTip(measure_tooltip)
        if hasattr(self.window, "measurement_status_badge"):
            measure_enabled = bool(has_recording and hasattr(self.window, "waveform_plot") and self.window.waveform_plot.is_measurement_enabled())
            badge_payload = self._measurement_badge_payload() if measure_enabled else {"text": "", "tooltip": ""}
            badge_text = str(badge_payload.get("text") or "")
            badge_tooltip = str(badge_payload.get("tooltip") or "")
            self._set_toolbar_slot_text(self.window.measurement_status_badge, badge_text, tooltip=badge_tooltip)
            self.window.measurement_status_badge.setVisible(bool(measure_enabled and badge_text))
        if hasattr(self.window, "measurement_status_slot"):
            self.window.measurement_status_slot.setVisible(bool(measure_enabled and badge_text))
        current_window, visible_count, max_visible_count = self._update_visible_channel_preset_state(
            has_recording=has_recording,
        )
        advance_value = float(self.window.Time_Increment_Input.value()) if hasattr(self.window, "Time_Increment_Input") else 0.0
        if hasattr(self.window, "graph_event_channels_button"):
            self.window.graph_event_channels_button.setEnabled(total_events > 0)
            self._set_button_checked(
                self.window.graph_event_channels_button,
                self._is_event_scope_active(event_channels, current_channels=current_channels),
            )
        if hasattr(self.window, "graph_cursor_button"):
            cursor_enabled = bool(has_recording and hasattr(self.window, "waveform_plot") and self.window.waveform_plot.is_cursor_enabled())
            self.window.graph_cursor_button.setEnabled(has_recording)
            self._set_button_checked(self.window.graph_cursor_button, cursor_enabled)
        if hasattr(self.window, "graph_zoom_out_review_button"):
            self.window.graph_zoom_out_review_button.setEnabled(has_recording)
        if hasattr(self.window, "graph_zoom_in_review_button"):
            self.window.graph_zoom_in_review_button.setEnabled(has_recording)
        if hasattr(self.window, "graph_snapshot_button"):
            self.window.graph_snapshot_button.setEnabled(has_recording)
        if hasattr(self.window, "graph_run_stats_button"):
            self.window.graph_run_stats_button.setEnabled(self.window.compare_runs_button.isEnabled())
        if hasattr(self.window, "graph_hotspot_button"):
            self.window.graph_hotspot_button.setEnabled(bool(ranking))
        if hasattr(self.window, "graph_accept_run_button"):
            self.window.graph_accept_run_button.setEnabled(self.window.accept_run_button.isEnabled())
        if hasattr(self.window, "graph_export_button"):
            self.window.graph_export_button.setEnabled(self.window.save_csv_button.isEnabled())
        if hasattr(self.window, "graph_next_pending_button"):
            self.window.graph_next_pending_button.setEnabled(total_events > 0 and review_progress["remaining"] > 0)
        annotation_labels = {
            "HFO": ["Pathological", "Physiological", "Artifact"],
            "Spindle": ["Real", "Spike", "Artifact"],
            "Spike": [],
        }.get(self.biomarker_type or "", [])
        visible_quick_labels = False
        for index, button in enumerate(getattr(self.window, "graph_label_buttons", [])):
            label = annotation_labels[index] if index < len(annotation_labels) else ""
            button.setProperty("quickLabel", label)
            if label:
                button.setText(label)
                button.setToolTip(f"Assign the label '{label}' to the current review target")
                button.setVisible(True)
                button.setEnabled(total_events > 0)
                visible_quick_labels = True
            else:
                button.setVisible(False)
        if hasattr(self.window, "graph_label_section"):
            self.window.graph_label_section.setVisible(visible_quick_labels)
        visible_prediction_scopes = False
        for index, button in enumerate(getattr(self.window, "graph_scope_buttons", [])):
            scope = preferred_scope_labels[index] if index < len(preferred_scope_labels) else ""
            button.setProperty("predictionScope", scope)
            if scope:
                button.setText(scope)
                button.setToolTip(f"Jump to the next review target in the '{scope}' model bucket")
                button.setVisible(True)
                button.setEnabled(scope in available_scopes and total_events > 0)
                visible_prediction_scopes = True
            else:
                button.setVisible(False)
        if hasattr(self.window, "graph_triage_section"):
            self.window.graph_triage_section.setVisible(visible_prediction_scopes)
        if hasattr(self.window, "graph_reviewed_status_badge"):
            text = (
                f"Reviewed {review_progress['reviewed']}/{review_progress['total']}"
                if has_recording
                else "Reviewed --"
            )
            self.window.graph_reviewed_status_badge.setText(text)
            self.window.graph_reviewed_status_badge.setToolTip(
                f"Reviewed {review_progress['reviewed']} of {review_progress['total']} visible events" if has_recording else ""
            )
        if hasattr(self.window, "graph_pending_status_badge"):
            text = f"Pending {review_progress['remaining']}" if has_recording else "Pending --"
            self.window.graph_pending_status_badge.setText(text)
            self.window.graph_pending_status_badge.setToolTip(
                f"{review_progress['remaining']} visible events still need review" if has_recording else ""
            )
        if hasattr(self.window, "graph_accepted_status_badge"):
            accepted_text = getattr(accepted_run, "detector_name", None) or "--"
            text = f"Run {accepted_text}" if has_recording else "Run --"
            self.window.graph_accepted_status_badge.setText(text)
            self.window.graph_accepted_status_badge.setToolTip(
                f"Accepted run: {accepted_text}" if has_recording and accepted_text != "--" else "No accepted run yet"
            )
        if hasattr(self.window, "graph_top_channel_status_badge"):
            if has_recording and top_channel:
                channel_text = str(top_channel["channel_name"])
                compact_channel = channel_text if len(channel_text) <= 10 else f"{channel_text[:9]}..."
                self.window.graph_top_channel_status_badge.setText(f"Top {compact_channel}")
                self.window.graph_top_channel_status_badge.setToolTip(
                    f"Top channel: {channel_text} ({top_channel['total_events']} events)"
                )
            else:
                self.window.graph_top_channel_status_badge.setText("Top --")
                self.window.graph_top_channel_status_badge.setToolTip("")
        if hasattr(self.window, "graph_window_status_badge"):
            self.window.graph_window_status_badge.setText(f"Window {current_window:.2f} s" if has_recording else "Window --")
            self.window.graph_window_status_badge.setToolTip(
                f"Current waveform time window: {current_window:.2f} s" if has_recording else ""
            )
        if hasattr(self.window, "graph_visible_status_badge"):
            visible_text = f"Visible {visible_count}"
            if max_visible_count > 0:
                visible_text = f"Visible {visible_count}/{max_visible_count}"
            self.window.graph_visible_status_badge.setText(visible_text if has_recording else "Visible --")
            self.window.graph_visible_status_badge.setToolTip(
                f"{visible_count} channels visible out of {max_visible_count} selected" if has_recording and max_visible_count > 0 else ""
            )
        if hasattr(self.window, "graph_advance_status_badge"):
            self.window.graph_advance_status_badge.setText(f"Advance {advance_value:.0f}%" if has_recording else "Advance --")
            self.window.graph_advance_status_badge.setToolTip(
                f"Waveform step size: {advance_value:.0f}% of the current window" if has_recording else ""
            )

        event_buttons = (
            getattr(self.window, "prev_event_button", None),
            getattr(self.window, "center_event_button", None),
            getattr(self.window, "next_event_button", None),
        )
        for button in event_buttons:
            if button is not None:
                button.setEnabled(total_events > 0)

        if hasattr(self.window, "event_position_label"):
            if total_events > 0:
                current_index = min(self._get_feature_current_position(features) + 1, total_events)
                current_info = {}
                if hasattr(features, "get_current_info"):
                    try:
                        current_info = features.get_current_info() or {}
                    except Exception:
                        current_info = {}
                channel_name = current_info.get("channel_name")
                if channel_name:
                    event_text = f"{current_index}/{total_events}  {channel_name}"
                else:
                    event_text = f"{current_index}/{total_events}"
            else:
                event_text = "No events"
            self._set_toolbar_slot_text(
                self.window.event_position_label,
                event_text,
                tooltip="" if event_text == "No events" else event_text,
            )
        self._update_waveform_context_badges(
            has_recording=has_recording,
            current_channels=current_channels,
            event_channels=event_channels,
            highlighted_channel=highlighted_channel,
            average_reference_channels=average_reference_channels,
            auto_bipolar_channels=auto_bipolar_channels,
            auto_bipolar_metadata=auto_bipolar_metadata,
        )
        self._sync_waveform_channel_presentation()

    def _connect_worker(self, worker, task_name, result_handler=None, finished_handler=None):
        safe_connect_signal_slot(worker.signals.error, lambda err: self.handle_worker_error(task_name, err))
        if result_handler is not None:
            safe_connect_signal_slot(worker.signals.result, result_handler)
        if finished_handler is not None:
            safe_connect_signal_slot(worker.signals.finished, finished_handler)
        self.window.threadpool.start(worker)

    def handle_worker_error(self, task_name, error_tuple):
        self._end_busy_task()
        _, value, traceback_text = error_tuple
        self.message_handler(f"{task_name} failed: {value}")
        self._set_workflow_message(f"{task_name} failed")
        msg = QMessageBox(self.window)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(f"{task_name} Failed")
        msg.setText(f"{task_name} failed")
        msg.setInformativeText(str(value))
        msg.setDetailedText(traceback_text)
        msg.exec_()

    def init_menu_bar(self):
        safe_connect_signal_slot(self.window.action_Open_EDF.triggered, self.open_file)
        safe_connect_signal_slot(self.window.actionQuick_Detection.triggered, self.open_quick_detection)
        safe_connect_signal_slot(self.window.action_Load_Detection.triggered, self.load_from_npz)

        ## top toolbar buttoms
        safe_connect_signal_slot(self.window.actionOpen_EDF_toolbar.triggered, self.open_file)
        safe_connect_signal_slot(self.window.actionQuick_Detection_toolbar.triggered, self.open_quick_detection)
        safe_connect_signal_slot(self.window.actionLoad_Detection_toolbar.triggered, self.load_from_npz)
        if hasattr(self.window, "actionWaveform_Shortcuts_toolbar"):
            safe_connect_signal_slot(self.window.actionWaveform_Shortcuts_toolbar.triggered, self.open_waveform_shortcut_settings)
        if hasattr(self.window, "empty_open_button"):
            safe_connect_signal_slot(self.window.empty_open_button.clicked, self.open_file)
        if hasattr(self.window, "empty_load_session_button"):
            safe_connect_signal_slot(self.window.empty_load_session_button.clicked, self.load_from_npz)
        if hasattr(self.window, "empty_quick_button"):
            safe_connect_signal_slot(self.window.empty_quick_button.clicked, self.open_quick_detection)
        if hasattr(self.window, "new_hfo_run_action"):
            safe_connect_signal_slot(self.window.new_hfo_run_action.triggered, lambda: self.start_run_mode("HFO"))
        if hasattr(self.window, "new_spindle_run_action"):
            safe_connect_signal_slot(self.window.new_spindle_run_action.triggered, lambda: self.start_run_mode("Spindle"))
        if hasattr(self.window, "new_spike_run_action"):
            safe_connect_signal_slot(self.window.new_spike_run_action.triggered, lambda: self.start_run_mode("Spike"))
        menu_bar = self.window.menuBar()
        settings_menu = getattr(self.window, "settings_menu", None)
        if settings_menu is None:
            settings_menu = menu_bar.addMenu("Settings")
            self.window.settings_menu = settings_menu
        if not hasattr(self.window, "waveform_shortcut_settings_action"):
            self.window.waveform_shortcut_settings_action = QAction("Waveform Shortcuts...", self.window)
            settings_menu.addAction(self.window.waveform_shortcut_settings_action)
        safe_connect_signal_slot(self.window.waveform_shortcut_settings_action.triggered, self.open_waveform_shortcut_settings)

    def init_waveform_display(self):
        # waveform display widget
        if hasattr(self.window, "waveform_graphics_widget") and self.window.waveform_graphics_widget is not None:
            self.window.waveform_graphics_widget.setParent(None)
        waveform_layout = self.window.widget.layout()
        self.window.waveform_graphics_widget = pg.GraphicsLayoutWidget()
        self.window.waveform_graphics_widget.setBackground("w")
        self.window.waveform_graphics_widget.setObjectName("waveformGraphicsWidget")
        self.window.waveform_graphics_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        try:
            self.window.waveform_graphics_widget.ci.layout.setContentsMargins(0, 0, 0, 0)
            self.window.waveform_graphics_widget.ci.layout.setVerticalSpacing(2)
            self.window.waveform_graphics_widget.ci.layout.setHorizontalSpacing(0)
            self.window.waveform_graphics_widget.ci.layout.setRowStretchFactor(0, 10)
            self.window.waveform_graphics_widget.ci.layout.setRowStretchFactor(1, 2)
            self.window.waveform_graphics_widget.ci.layout.setColumnStretchFactor(0, 0)
            self.window.waveform_graphics_widget.ci.layout.setColumnStretchFactor(1, 1)
        except Exception:
            pass

        gutter_plot_item = self.window.waveform_graphics_widget.addPlot(row=0, col=0)
        main_plot_item = self.window.waveform_graphics_widget.addPlot(row=0, col=1)
        mini_plot_item = self.window.waveform_graphics_widget.addPlot(row=1, col=1)
        try:
            gutter_plot_item.setMinimumWidth(96)
            gutter_plot_item.setMaximumWidth(96)
            mini_plot_item.setMaximumHeight(76)
            mini_plot_item.setMinimumHeight(56)
        except Exception:
            pass

        self.window.waveform_channel_gutter_widget = SharedPlotHandle(self.window.waveform_graphics_widget, gutter_plot_item)
        self.window.waveform_plot_widget = SharedPlotHandle(self.window.waveform_graphics_widget, main_plot_item)
        self.window.waveform_mini_widget = SharedPlotHandle(self.window.waveform_graphics_widget, mini_plot_item)
        waveform_layout.addWidget(self.window.waveform_graphics_widget, 0, 1, 2, 1)
        waveform_layout.setRowStretch(0, 1)
        waveform_layout.setRowStretch(1, 0)

    def set_backend(self, backend):
        self.backend = backend

    def filter_data(self):
        if self.backend is None:
            self.handle_unsupported_biomarker_mode("Filtering is not available for the current biomarker mode yet.")
            return
        self.message_handler("Filtering data...")
        try:
            filter_param = self._build_filter_param_from_inputs()
            self.backend.set_filter_parameter(filter_param)
        except (TypeError, ValueError) as exc:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText(f'Filter could not be constructed with the given parameters. {exc}')
            msg.setWindowTitle("Filter Construction Error")
            msg.exec_()
            return
        self._set_workflow_message("Filtering signal...")
        self._begin_busy_task("filter", "Filtering", self._filter_busy_buttons())
        self._set_filter_controls_busy(True)
        worker = Worker(self._filter)
        safe_connect_signal_slot(worker.signals.error, self._handle_filter_worker_error)
        self._connect_worker(worker, "Filtering", result_handler=lambda _result: self.filtering_complete())

    def create_center_waveform_and_mini_plot(self):
        self.window.channels_to_plot = []
        self.window.waveform_plot = CenterWaveformAndMiniPlotController(
            self.window.waveform_plot_widget,
            self.window.waveform_mini_widget,
            self.backend,
            channel_gutter_widget=getattr(self.window, "waveform_channel_gutter_widget", None),
        )
        self.window.waveform_plot.set_overlay_run_provider(self._get_case_visible_runs_for_waveform)

        # part of “clear everything if exit”, optimize in the future
        safe_connect_signal_slot(self.window.waveform_time_scroll_bar.valueChanged, self.scroll_time_waveform_plot)
        safe_connect_signal_slot(self.window.channel_scroll_bar.valueChanged, self.scroll_channel_waveform_plot)
        self.window.waveform_time_scroll_bar.valueChanged.disconnect(self.scroll_time_waveform_plot)
        self.window.channel_scroll_bar.valueChanged.disconnect(self.scroll_channel_waveform_plot)
        self.window.waveform_plot.connect_channel_selection(self._handle_waveform_channel_selection)
        self.window.waveform_plot.connect_measurement_selection(self._handle_waveform_measurement_point)
        self.window.waveform_plot.connect_wheel_time_scroll(self._handle_waveform_wheel_time_scroll)
        self.window.waveform_plot.connect_wheel_channel_scroll(self._handle_waveform_wheel_channel_scroll)
        self.window.waveform_plot.connect_time_zoom(self._handle_waveform_time_zoom)
        self._apply_waveform_interaction_preferences()

    def _current_highlighted_waveform_channel(self):
        plot = getattr(self.window, "waveform_plot", None)
        if plot is None or not hasattr(plot, "get_highlighted_channel"):
            return ""
        try:
            channel_name = plot.get_highlighted_channel()
        except Exception:
            return ""
        return str(channel_name) if channel_name else ""

    def _set_highlighted_waveform_channel(self, channel_name):
        plot = getattr(self.window, "waveform_plot", None)
        if plot is None:
            return
        try:
            plot.main_waveform_plot_controller.view.set_highlighted_channel(channel_name)
        except Exception:
            return

    def _handle_waveform_channel_selection(self, channel_name):
        if not channel_name:
            return
        self.update_waveform_toolbar_state()

    def _handle_waveform_measurement_point(self, channel_name, time_value):
        if not channel_name or self.backend is None or not hasattr(self.window, "waveform_plot"):
            return
        plot = self.window.waveform_plot
        if not hasattr(plot, "get_measurement_point"):
            return
        point = plot.get_measurement_point(channel_name, time_value)
        if not point:
            return

        existing_points = list(self.waveform_measurement_state.get("points") or [])
        if len(existing_points) >= 2:
            existing_points = []
        existing_points.append(point)
        summary = self._summarize_waveform_measurement(existing_points)
        self.waveform_measurement_state = {
            "points": existing_points,
            "summary": summary,
        }
        self._sync_waveform_measurement_overlay()
        self.update_waveform_toolbar_state()
        if summary:
            self._set_workflow_message(f"Measure: {summary['badge_text']}")
        else:
            self._set_workflow_message("Measure: first point selected")

    def _handle_waveform_wheel_time_scroll(self, step_delta):
        if self.backend is None or not hasattr(self.window, "waveform_plot"):
            return
        plot = self.window.waveform_plot
        total_time = max(0.0, float(plot.get_total_time()))
        time_window = max(0.1, float(plot.get_time_window()))
        time_increment = max(0.0, float(plot.get_time_increment()))
        step_size = time_window * time_increment / 100.0
        if step_size <= 0:
            step_size = max(0.05, time_window * 0.1)

        current_start = float(getattr(plot, "t_start", 0.0) or 0.0)
        max_start = max(0.0, total_time - time_window)
        new_start = max(0.0, min(current_start - float(step_delta) * step_size, max_start))
        if abs(new_start - current_start) < 1e-6:
            return

        plot.t_start = new_start
        self.waveform_plot_button_clicked()

    def _handle_waveform_wheel_channel_scroll(self, step_delta):
        if self.backend is None or not hasattr(self.window, "waveform_plot"):
            return
        plot = self.window.waveform_plot
        visible_channels = max(1, int(plot.get_n_channels_to_plot()))
        available_channels = len(plot.get_channels_to_plot() or [])
        max_first = max(0, available_channels - visible_channels)
        if max_first <= 0:
            self._waveform_channel_wheel_remainder = 0.0
            return

        self._waveform_channel_wheel_remainder -= float(step_delta)
        whole_steps = int(math.trunc(self._waveform_channel_wheel_remainder))
        if whole_steps == 0:
            return

        self._waveform_channel_wheel_remainder -= whole_steps
        current_first = int(getattr(plot, "first_channel_to_plot", 0) or 0)
        new_first = max(0, min(current_first + whole_steps, max_first))
        if new_first == current_first:
            self._waveform_channel_wheel_remainder = 0.0
            return

        plot.first_channel_to_plot = new_first
        self.waveform_plot_button_clicked()

    def _handle_waveform_time_zoom(self, zoom_delta, anchor_time):
        if self.backend is None or not hasattr(self.window, "waveform_plot"):
            return
        plot = self.window.waveform_plot
        total_time = max(0.1, float(plot.get_total_time()))
        current_window = max(0.1, float(plot.get_time_window()))
        current_start = max(0.0, float(getattr(plot, "t_start", 0.0) or 0.0))
        current_end = min(total_time, current_start + current_window)

        new_window = max(0.1, min(total_time, current_window * math.exp(-float(zoom_delta))))
        if abs(new_window - current_window) < 1e-6:
            return

        anchor_time = float(anchor_time)
        anchor_time = max(current_start, min(anchor_time, current_end))
        anchor_ratio = (anchor_time - current_start) / current_window if current_window > 0 else 0.5
        if not math.isfinite(anchor_ratio):
            anchor_ratio = 0.5
        anchor_ratio = max(0.0, min(anchor_ratio, 1.0))

        max_start = max(0.0, total_time - new_window)
        new_start = max(0.0, min(anchor_time - new_window * anchor_ratio, max_start))

        self.window.display_time_window_input.setValue(float(new_window))
        plot.t_start = float(new_start)
        self.waveform_plot_button_clicked()

    def init_classifier_param(self):
        self.window.classifier_param = ParamClassifier()
        # self.classifier_save_button.clicked.connect(self.hfo_app.set_classifier())

    def init_param(self, biomarker_type='HFO'):
        if biomarker_type == 'HFO':
            self.init_classifier_param()
            self.init_default_filter_input_params()
            self.init_default_ste_input_params()
            self.init_default_mni_input_params()
            self.init_default_hil_input_params()

            self.set_mni_input_len(8)
            self.set_ste_input_len(8)
            self.set_hil_input_len(8)
        elif biomarker_type == 'Spindle':
            self.init_classifier_param()
            self.init_default_filter_input_params()
            self.init_default_yasa_input_params()

            self.set_yasa_input_len(8)

    def init_default_filter_input_params(self):
        self._sync_filter_inputs_from_param(self._recommended_filter_param())

    def init_default_ste_input_params(self):
        default_params = ParamSTE(2000)
        self.window.ste_rms_window_input.setText(str(default_params.rms_window))
        self.window.ste_rms_threshold_input.setText(str(default_params.rms_thres))
        self.window.ste_min_window_input.setText(str(default_params.min_window))
        self.window.ste_epoch_length_input.setText(str(default_params.epoch_len))
        self.window.ste_min_gap_input.setText(str(default_params.min_gap))
        self.window.ste_min_oscillation_input.setText(str(default_params.min_osc))
        self.window.ste_peak_threshold_input.setText(str(default_params.peak_thres))

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
        default_params = ParamMNI(200)
        self.window.mni_epoch_time_input.setText(str(default_params.epoch_time))
        self.window.mni_epoch_chf_input.setText(str(default_params.epo_CHF))
        self.window.mni_chf_percentage_input.setText(str(default_params.per_CHF))
        self.window.mni_min_window_input.setText(str(default_params.min_win))
        self.window.mni_min_gap_time_input.setText(str(default_params.min_gap))
        self.window.mni_threshold_percentage_input.setText(str(default_params.thrd_perc * 100))
        self.window.mni_baseline_window_input.setText(str(default_params.base_seg))
        self.window.mni_baseline_shift_input.setText(str(default_params.base_shift))
        self.window.mni_baseline_threshold_input.setText(str(default_params.base_thrd))
        self.window.mni_baseline_min_time_input.setText(str(default_params.base_min))

    def init_default_hil_input_params(self):
        default_params = ParamHIL(2000)
        self.window.hil_sample_freq_input.setText(str(default_params.sample_freq))
        self.window.hil_pass_band_input.setText(str(default_params.pass_band))
        self.window.hil_stop_band_input.setText(str(default_params.stop_band))
        self.window.hil_epoch_time_input.setText(str(default_params.epoch_time))
        self.window.hil_sd_threshold_input.setText(str(default_params.sd_threshold))
        self.window.hil_min_window_input.setText(str(default_params.min_window))

    def init_default_yasa_input_params(self):
        default_params = ParamYASA(2000)
        self.window.yasa_freq_sp_low_input.setText(str(default_params.freq_sp[0]))
        self.window.yasa_freq_sp_high_input.setText(str(default_params.freq_sp[1]))
        self.window.yasa_freq_broad_low_input.setText(str(default_params.freq_broad[0]))
        self.window.yasa_freq_broad_high_input.setText(str(default_params.freq_broad[1]))
        self.window.yasa_duration_low_input.setText(str(default_params.duration[0]))
        self.window.yasa_duration_high_input.setText(str(default_params.duration[1]))
        self.window.yasa_min_distance_input.setText(str(default_params.min_distance))
        self.window.yasa_thresh_rel_pow_input.setText(str(default_params.rel_pow))
        self.window.yasa_thresh_corr_input.setText(str(default_params.corr))
        self.window.yasa_thresh_rms_input.setText(str(default_params.rms))

    def _settings_store(self):
        return QSettings("PyBrain", "PyBrain")

    def _parse_setting_bool(self, raw_value, default=True):
        if raw_value is None:
            return bool(default)
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, (int, float)):
            return bool(raw_value)
        normalized = str(raw_value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return bool(default)

    def _waveform_shortcut_specs(self):
        return [
            {"id": "window_2s", "label": "Set window to 2 s", "default": "1", "handler": lambda: self.apply_waveform_window_preset(2.0)},
            {"id": "window_5s", "label": "Set window to 5 s", "default": "2", "handler": lambda: self.apply_waveform_window_preset(5.0)},
            {"id": "window_10s", "label": "Set window to 10 s", "default": "3", "handler": lambda: self.apply_waveform_window_preset(10.0)},
            {"id": "window_20s", "label": "Set window to 20 s", "default": "4", "handler": lambda: self.apply_waveform_window_preset(20.0)},
            {"id": "visible_8", "label": "Show 8 channels", "default": "Shift+1", "handler": lambda: self.apply_visible_channel_preset(8)},
            {"id": "visible_16", "label": "Show 16 channels", "default": "Shift+2", "handler": lambda: self.apply_visible_channel_preset(16)},
            {"id": "visible_32", "label": "Show 32 channels", "default": "Shift+3", "handler": lambda: self.apply_visible_channel_preset(32)},
            {
                "id": "visible_all",
                "label": "Show full current channel subset",
                "default": "Shift+4",
                "handler": lambda: self.apply_visible_channel_preset("all"),
            },
            {"id": "zoom_out", "label": "Zoom out", "default": "[", "handler": lambda: self.zoom_waveform(2.0)},
            {"id": "zoom_in", "label": "Zoom in", "default": "]", "handler": lambda: self.zoom_waveform(0.5)},
            {
                "id": "toggle_filtered",
                "label": "Toggle raw / filtered",
                "default": "F",
                "handler": self._toggle_waveform_filtered_view,
                "required_widget": "toggle_filtered_checkbox",
            },
            {
                "id": "toggle_event_scope",
                "label": "Toggle event channels / all channels",
                "default": "E",
                "handler": self._toggle_waveform_event_scope,
                "required_widget": "event_channels_button",
            },
            {
                "id": "toggle_cursor",
                "label": "Toggle cursor",
                "default": "C",
                "handler": self._toggle_waveform_cursor_shortcut,
                "required_widget": "cursor_tool_button",
            },
            {
                "id": "toggle_measure",
                "label": "Toggle measure",
                "default": "R",
                "handler": self._toggle_waveform_measure_shortcut,
                "required_widget": "measure_tool_button",
            },
            {
                "id": "toggle_auto_bipolar",
                "label": "Toggle auto bipolar view",
                "default": "A",
                "handler": lambda: self.toggle_auto_bipolar_view(
                    self._current_waveform_channel_names() != self._auto_bipolar_waveform_channel_names()
                ),
                "required_widget": "auto_bipolar_button",
            },
            {
                "id": "show_referential",
                "label": "Show referential channels",
                "default": "V",
                "handler": self.show_referential_channels,
                "required_widget": "referential_tool_button",
            },
            {
                "id": "toggle_average_reference",
                "label": "Toggle average reference view",
                "default": "M",
                "handler": lambda: self.toggle_average_reference_view(not self._is_average_reference_active()),
                "required_widget": "average_reference_button",
            },
            {
                "id": "focus_highlighted_channel",
                "label": "Highlight selected channel",
                "default": "G",
                "handler": lambda: self.toggle_highlighted_waveform_channel_focus(not self._is_highlight_focus_active()),
                "required_widget": "highlight_channel_button",
            },
            {
                "id": "focus_neighbors",
                "label": "Focus highlighted channel and neighbors",
                "default": "T",
                "handler": lambda: self.toggle_neighbor_channel_focus(not self._is_neighbor_focus_active()),
                "required_widget": "neighbor_channels_button",
            },
            {
                "id": "toggle_clean_view",
                "label": "Toggle clean view",
                "default": "D",
                "handler": self.toggle_clean_view_shortcut,
                "required_widget": "clean_view_button",
            },
            {
                "id": "open_montage",
                "label": "Open montage / bipolar tool",
                "default": "B",
                "handler": self.open_bipolar_channel_selection,
                "required_widget": "montage_tool_button",
            },
            {
                "id": "hotspot",
                "label": "Highlight hotspot channels",
                "default": "H",
                "handler": lambda: self.toggle_hotspot_channels(not self._is_hotspot_scope_active()),
                "required_widget": "hotspot_tool_button",
            },
            {
                "id": "center_event",
                "label": "Center current event",
                "default": "Space",
                "handler": self.center_current_event,
                "required_widget": "center_event_button",
            },
            {
                "id": "next_pending",
                "label": "Jump to next unreviewed event",
                "default": "N",
                "handler": lambda: self.jump_to_pending_review(1),
                "required_widget": "pending_event_button",
            },
            {"id": "clear_inspect", "label": "Clear inspect mode", "default": "Esc", "handler": self._clear_waveform_inspect_modes},
        ]

    def _default_waveform_shortcut_bindings(self):
        return {spec["id"]: spec["default"] for spec in self._waveform_shortcut_specs()}

    def _normalize_trackpad_sensitivity(self, raw_value):
        normalized = str(raw_value or "default").strip().lower()
        if normalized not in {"gentle", "default", "fast"}:
            normalized = "default"
        return normalized

    def _normalize_shortcut_text(self, shortcut_value):
        if isinstance(shortcut_value, QKeySequence):
            return shortcut_value.toString(QKeySequence.PortableText).strip()
        text = str(shortcut_value or "").strip()
        if not text:
            return ""
        return QKeySequence(text).toString(QKeySequence.PortableText).strip()

    def _load_waveform_shortcut_preferences(self):
        defaults = self._default_waveform_shortcut_bindings()
        settings = self._settings_store()
        raw_enabled = settings.value("waveform_shortcuts/enabled", True)
        raw_bindings = settings.value("waveform_shortcuts/keymap", "{}")

        stored_bindings = {}
        if isinstance(raw_bindings, dict):
            stored_bindings = dict(raw_bindings)
        else:
            try:
                stored_bindings = json.loads(raw_bindings) if raw_bindings not in (None, "") else {}
            except (TypeError, ValueError):
                stored_bindings = {}

        bindings = defaults.copy()
        for shortcut_id in defaults:
            if shortcut_id in stored_bindings:
                bindings[shortcut_id] = self._normalize_shortcut_text(stored_bindings.get(shortcut_id, ""))

        if self._validate_waveform_shortcut_bindings(True, bindings):
            bindings = defaults

        self.waveform_shortcuts_enabled = self._parse_setting_bool(raw_enabled, True)
        self.waveform_shortcut_bindings = bindings

    def _load_waveform_interaction_preferences(self):
        settings = self._settings_store()
        raw_sensitivity = settings.value("waveform_interaction/trackpad_sensitivity", "default")
        self.waveform_trackpad_sensitivity = self._normalize_trackpad_sensitivity(raw_sensitivity)

    def _save_waveform_shortcut_preferences(self):
        settings = self._settings_store()
        settings.setValue("waveform_shortcuts/enabled", bool(self.waveform_shortcuts_enabled))
        settings.setValue("waveform_shortcuts/keymap", json.dumps(self.waveform_shortcut_bindings))

    def _save_waveform_interaction_preferences(self):
        settings = self._settings_store()
        settings.setValue("waveform_interaction/trackpad_sensitivity", self.waveform_trackpad_sensitivity)

    def _apply_waveform_interaction_preferences(self):
        plot = getattr(self.window, "waveform_plot", None)
        if plot is None or not hasattr(plot, "set_trackpad_sensitivity"):
            return
        plot.set_trackpad_sensitivity(self.waveform_trackpad_sensitivity)

    def _validate_waveform_shortcut_bindings(self, _enabled, bindings):
        seen = {}
        for spec in self._waveform_shortcut_specs():
            sequence_text = self._normalize_shortcut_text(bindings.get(spec["id"], ""))
            if not sequence_text:
                continue
            if "," in sequence_text:
                return f"{spec['label']} uses '{sequence_text}'. Use a single shortcut, not a multi-step sequence."
            parts = [part.strip() for part in sequence_text.split("+") if part.strip()]
            modifiers = parts[:-1]
            if any(modifier in {"Ctrl", "Meta", "Alt"} for modifier in modifiers):
                return (
                    f"{spec['label']} uses '{sequence_text}'. To avoid system conflicts, "
                    "use plain keys or Shift-only combinations."
                )
            previous = seen.get(sequence_text)
            if previous is not None:
                return f"{spec['label']} and {previous} both use '{sequence_text}'. Choose unique shortcuts."
            seen[sequence_text] = spec["label"]
        return ""

    def open_waveform_shortcut_settings(self):
        dialog_specs = [
            {"id": spec["id"], "label": spec["label"], "default": spec["default"]}
            for spec in self._waveform_shortcut_specs()
        ]
        dialog = WaveformShortcutSettingsDialog(
            dialog_specs,
            enabled=self.waveform_shortcuts_enabled,
            bindings=self.waveform_shortcut_bindings,
            validate_callback=self._validate_waveform_shortcut_bindings,
            trackpad_sensitivity=self.waveform_trackpad_sensitivity,
            parent=self.window,
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        self.waveform_shortcuts_enabled = dialog.shortcuts_enabled()
        self.waveform_shortcut_bindings = {
            shortcut_id: self._normalize_shortcut_text(shortcut_value)
            for shortcut_id, shortcut_value in dialog.current_bindings().items()
        }
        self.waveform_trackpad_sensitivity = self._normalize_trackpad_sensitivity(dialog.current_trackpad_sensitivity())
        self._save_waveform_shortcut_preferences()
        self._save_waveform_interaction_preferences()
        self._install_waveform_shortcuts()
        self._apply_waveform_interaction_preferences()
        self.message_handler("Updated waveform settings")
        self._set_workflow_message("Waveform settings updated")

    def _clear_waveform_shortcuts(self):
        for shortcut in getattr(self.window, "waveform_shortcut_objects", []):
            if shortcut is None:
                continue
            shortcut.setEnabled(False)
            shortcut.setParent(None)
            shortcut.deleteLater()
        self.window.waveform_shortcut_objects = []

    def _waveform_shortcuts_available(self):
        return bool(
            self.backend is not None
            and getattr(self.backend, "eeg_data", None) is not None
            and hasattr(self.window, "waveform_plot")
        )

    def _waveform_shortcut_focus_blocked(self):
        focus_widget = QApplication.focusWidget()
        if focus_widget is None:
            return False
        blocked_types = (
            QLineEdit,
            QTextEdit,
            QPlainTextEdit,
            QAbstractSpinBox,
            QComboBox,
            QAbstractItemView,
        )
        current = focus_widget
        while current is not None:
            if isinstance(current, blocked_types):
                return True
            current = current.parentWidget()
        return False

    def _run_waveform_shortcut(self, handler, required_widget=None):
        if not self._waveform_shortcuts_available() or self._waveform_shortcut_focus_blocked():
            return
        widget = required_widget
        if isinstance(required_widget, str):
            widget = getattr(self.window, required_widget, None)
        if widget is not None and not widget.isEnabled():
            return
        handler()

    def _toggle_waveform_filtered_view(self):
        checkbox = getattr(self.window, "toggle_filtered_checkbox", None)
        if checkbox is None or not checkbox.isEnabled():
            return
        checkbox.setChecked(not checkbox.isChecked())

    def _show_raw_waveform_view(self):
        checkbox = getattr(self.window, "toggle_filtered_checkbox", None)
        if checkbox is None:
            return
        if checkbox.isChecked():
            checkbox.setChecked(False)
        else:
            self.update_waveform_toolbar_state()

    def _current_waveform_channel_names(self):
        if not hasattr(self.window, "waveform_plot"):
            return []
        try:
            channels = self.window.waveform_plot.get_channels_to_plot() or []
        except Exception:
            return []
        return [str(channel) for channel in channels]

    def _recording_waveform_channel_names(self):
        if self.backend is None:
            return []
        if hasattr(self.backend, "get_recording_channel_names"):
            try:
                return [str(channel) for channel in np.array(self.backend.get_recording_channel_names()).tolist()]
            except Exception:
                return []
        return [str(channel) for channel in getattr(self.backend, "channel_names", []) if "#-#" not in str(channel)]

    def _auto_bipolar_waveform_channel_names(self):
        if self.backend is None:
            return []
        if hasattr(self.backend, "get_auto_bipolar_definitions"):
            try:
                definitions = self.backend.get_auto_bipolar_definitions() or []
            except Exception:
                return []
            return [str(derived_name) for derived_name, _channel_1, _channel_2 in definitions]
        if not hasattr(self.backend, "get_auto_bipolar_pairs"):
            return []
        try:
            pairs = self.backend.get_auto_bipolar_pairs() or []
        except Exception:
            return []
        return [f"{channel_1}#-#{channel_2}" for channel_1, channel_2 in pairs]

    def _get_average_reference_metadata(self):
        if self.backend is None or not hasattr(self.backend, "get_average_reference_metadata"):
            return {}
        try:
            return self.backend.get_average_reference_metadata() or {}
        except Exception:
            return {}

    def _average_reference_waveform_channel_names(self):
        if self.backend is None:
            return []
        if hasattr(self.backend, "get_average_reference_definitions"):
            try:
                definitions = self.backend.get_average_reference_definitions() or []
            except Exception:
                return []
            return [str(derived_name) for derived_name, _source_channel in definitions]
        return []

    def _get_clean_recording_metadata(self):
        if self.backend is None or not hasattr(self.backend, "get_clean_recording_channel_metadata"):
            return {
                "recording_channels": self._recording_waveform_channel_names(),
                "clean_channels": self._recording_waveform_channel_names(),
                "excluded_channels": [],
                "bad_channels": [],
                "flat_channels": [],
            }
        try:
            metadata = self.backend.get_clean_recording_channel_metadata() or {}
        except Exception:
            metadata = {}
        metadata.setdefault("recording_channels", self._recording_waveform_channel_names())
        metadata.setdefault("clean_channels", list(metadata.get("recording_channels") or []))
        metadata.setdefault("excluded_channels", [])
        metadata.setdefault("bad_channels", [])
        metadata.setdefault("flat_channels", [])
        return metadata

    def _ordered_channel_subset(self, ordered_source_channels, candidate_channels):
        candidate_set = {str(channel) for channel in (candidate_channels or []) if str(channel)}
        return [str(channel) for channel in (ordered_source_channels or []) if str(channel) in candidate_set]

    def _infer_auto_bipolar_neighbor_channels(self, highlighted_channel):
        highlighted_channel = str(highlighted_channel or "")
        if not highlighted_channel or BIPOLAR_TOKEN not in highlighted_channel:
            return []
        metadata = self._get_auto_bipolar_metadata()
        present_pairs = [entry for entry in (metadata.get("present_pairs", []) or []) if entry.get("derived_name")]
        if not present_pairs:
            return []
        target_entry = next(
            (entry for entry in present_pairs if str(entry.get("derived_name") or "") == highlighted_channel),
            None,
        )
        if target_entry is None:
            return []
        chain_name = str(target_entry.get("chain_name") or "")
        chain_entries = [
            entry
            for entry in present_pairs
            if str(entry.get("chain_name") or "") == chain_name
        ]
        chain_entries.sort(key=lambda entry: int(entry.get("display_order", 0)))
        target_index = next(
            (index for index, entry in enumerate(chain_entries) if str(entry.get("derived_name") or "") == highlighted_channel),
            None,
        )
        if target_index is None:
            return []
        neighbors = [highlighted_channel]
        if target_index > 0:
            neighbors.append(str(chain_entries[target_index - 1].get("derived_name") or ""))
        if target_index + 1 < len(chain_entries):
            neighbors.append(str(chain_entries[target_index + 1].get("derived_name") or ""))
        return [channel for channel in neighbors if channel]

    def _infer_neighbor_channels(self, highlighted_channel, radius=1):
        highlighted_channel = str(highlighted_channel or "")
        if not highlighted_channel:
            return []
        auto_neighbors = self._infer_auto_bipolar_neighbor_channels(highlighted_channel)
        if auto_neighbors:
            current_channels = self._current_waveform_channel_names()
            return self._ordered_channel_subset(current_channels, auto_neighbors)

        recording_channels = self._recording_waveform_channel_names()
        if not recording_channels:
            return []

        conventional_neighbors = get_conventional_eeg_neighbor_channels(recording_channels, highlighted_channel)
        if conventional_neighbors:
            return self._ordered_channel_subset(recording_channels, [highlighted_channel, *conventional_neighbors])

        adjacent_neighbors = get_adjacent_contact_neighbor_channels(recording_channels, highlighted_channel, radius=radius)
        if adjacent_neighbors:
            return self._ordered_channel_subset(recording_channels, [highlighted_channel, *adjacent_neighbors])

        if highlighted_channel in recording_channels:
            target_index = recording_channels.index(highlighted_channel)
            candidate_channels = [highlighted_channel]
            if target_index > 0:
                candidate_channels.append(recording_channels[target_index - 1])
            if target_index + 1 < len(recording_channels):
                candidate_channels.append(recording_channels[target_index + 1])
            return self._ordered_channel_subset(recording_channels, candidate_channels)
        return []

    def _highlight_focus_channel_names(self):
        channel_name = self._current_highlighted_waveform_channel()
        return [channel_name] if channel_name else []

    def _neighbor_focus_channel_names(self, radius=1):
        channel_name = self._current_highlighted_waveform_channel()
        if not channel_name:
            return []
        return self._infer_neighbor_channels(channel_name, radius=radius)

    def _is_highlight_focus_active(self):
        channel_name = self._current_highlighted_waveform_channel()
        current_channels = self._current_waveform_channel_names()
        return bool(self._highlight_channel_mode and channel_name and channel_name in current_channels)

    def _is_neighbor_focus_active(self, radius=1):
        current_channels = self._current_waveform_channel_names()
        neighbor_channels = self._neighbor_focus_channel_names(radius=radius)
        return bool(len(neighbor_channels) > 1 and current_channels == neighbor_channels)

    def _is_clean_view_active(self):
        current_channels = self._current_waveform_channel_names()
        metadata = self._get_clean_recording_metadata()
        clean_channels = [str(channel) for channel in (metadata.get("clean_channels") or [])]
        excluded_channels = [str(channel) for channel in (metadata.get("excluded_channels") or [])]
        return bool(clean_channels and excluded_channels and current_channels == clean_channels)

    def _is_average_reference_active(self):
        current_channels = self._current_waveform_channel_names()
        average_reference_channels = self._average_reference_waveform_channel_names()
        return bool(average_reference_channels and current_channels == average_reference_channels)

    def _toggle_waveform_event_scope(self):
        if self.backend is None:
            return
        features = self._get_active_event_features()
        event_channels = list(dict.fromkeys(self._get_feature_channels(features)))
        if not event_channels:
            return
        if self._current_waveform_channel_names() == event_channels:
            self.show_all_channels()
            return
        self.focus_event_channels()

    def focus_highlighted_waveform_channel(self):
        channel_name = self._current_highlighted_waveform_channel()
        if not channel_name:
            self.message_handler("Select a channel in the waveform gutter first")
            self._highlight_channel_mode = False
            self.update_waveform_toolbar_state()
            return False
        self._highlight_channel_mode = True
        if not self.highlight_waveform_channel(channel_name):
            self._highlight_channel_mode = False
            self.update_waveform_toolbar_state()
            return False
        return True

    def toggle_highlighted_waveform_channel_focus(self, enabled):
        if self.backend is None:
            return
        if enabled:
            self.focus_highlighted_waveform_channel()
        else:
            self._highlight_channel_mode = False
            self._set_highlighted_waveform_channel(None)
            self.update_waveform_toolbar_state()
            self.message_handler("Cleared channel highlight")
            self._set_workflow_message("Channel highlight cleared")

    def focus_neighbor_channels(self, radius=1):
        channel_name = self._current_highlighted_waveform_channel()
        if not channel_name:
            self.message_handler("Select a channel in the waveform gutter first")
            return
        channels = self._infer_neighbor_channels(channel_name, radius=radius)
        if not channels:
            self.message_handler(f"Could not infer adjacent neighbors for {channel_name}")
            return

        self.set_channels_to_plot(channels, display_all=True)
        self.window.channel_scroll_bar.setValue(0)
        if hasattr(self.window, "waveform_plot"):
            try:
                self.window.waveform_plot.main_waveform_plot_controller.view.set_highlighted_channel(channel_name)
            except Exception:
                pass
        if len(channels) == 1:
            self.message_handler(f"No adjacent neighbors inferred for {channel_name}; focused the highlighted channel")
            self._set_workflow_message(f"Focused {channel_name}")
        else:
            self.message_handler(f"Focused {channel_name} with {len(channels) - 1} adjacent channels")
            self._set_workflow_message(f"Target + neighbors: {channel_name}")
        self.update_waveform_toolbar_state()

    def toggle_neighbor_channel_focus(self, enabled, radius=1):
        if self.backend is None:
            return
        if enabled:
            self.focus_neighbor_channels(radius=radius)
        else:
            self.show_all_channels()

    def focus_clean_channels(self):
        metadata = self._get_clean_recording_metadata()
        clean_channels = [str(channel) for channel in (metadata.get("clean_channels") or [])]
        excluded_channels = [str(channel) for channel in (metadata.get("excluded_channels") or [])]
        if not clean_channels:
            self.message_handler("No clean source channels are available to display")
            return
        if not excluded_channels:
            self.show_all_channels()
            return
        self.set_channels_to_plot(clean_channels, display_all=False)
        self.window.channel_scroll_bar.setValue(0)
        hidden_count = len(excluded_channels)
        self.message_handler(f"Clean view ready ({hidden_count} excluded channels)")
        self._set_workflow_message("Reviewing clean source channels")
        self.update_waveform_toolbar_state()

    def toggle_clean_view(self, enabled):
        if self.backend is None:
            return
        if enabled:
            self.focus_clean_channels()
        else:
            self.show_all_channels()

    def toggle_clean_view_shortcut(self):
        self.toggle_clean_view(not self._is_clean_view_active())

    def show_referential_channels(self):
        self.show_all_channels()

    def toggle_average_reference_view(self, enabled):
        if self.backend is None:
            return
        if not enabled:
            self.show_all_channels()
            return
        if not hasattr(self.backend, "ensure_average_reference_channels"):
            self.message_handler("Average reference view is unavailable for this recording backend")
            self.update_waveform_toolbar_state()
            return
        average_reference_metadata = self._get_average_reference_metadata()
        derived_channels = [
            str(channel)
            for channel in np.array(self.backend.ensure_average_reference_channels()).tolist()
        ]
        if len(derived_channels) < 2:
            self.message_handler("Average reference needs at least two source channels")
            self.update_waveform_toolbar_state()
            return
        if hasattr(self.window, "waveform_plot"):
            self.window.waveform_plot.update_channel_names(self.backend.channel_names)
        self._sync_waveform_channel_presentation(average_reference_metadata)
        self._reset_waveform_measurement_state()
        self.set_channels_to_plot(derived_channels, display_all=True)
        self.window.channel_scroll_bar.setValue(0)
        self.message_handler(f"Average reference view ready ({len(derived_channels)} derived channels)")
        self._set_workflow_message("Reviewing average-reference channels")

    def toggle_auto_bipolar_view(self, enabled):
        if self.backend is None:
            return
        if not enabled:
            self.show_all_channels()
            return
        if not hasattr(self.backend, "ensure_auto_bipolar_channels"):
            self.message_handler("Auto bipolar view is unavailable for this recording backend")
            self.update_waveform_toolbar_state()
            return
        auto_bipolar_metadata = self._get_auto_bipolar_metadata()
        derived_channels = [str(channel) for channel in np.array(self.backend.ensure_auto_bipolar_channels()).tolist()]
        if not derived_channels:
            self.message_handler("Could not infer an EEG/iEEG bipolar montage from the current channel names")
            self.update_waveform_toolbar_state()
            return
        if hasattr(self.window, "waveform_plot"):
            self.window.waveform_plot.update_channel_names(self.backend.channel_names)
        self._sync_waveform_channel_presentation(auto_bipolar_metadata)
        self._reset_waveform_measurement_state()
        self.set_channels_to_plot(derived_channels, display_all=True)
        self.window.channel_scroll_bar.setValue(0)
        montage_kind = str(auto_bipolar_metadata.get("montage_kind", "") or "")
        montage_label = "auto bipolar"
        if montage_kind == "conventional_eeg":
            montage_label = "conventional EEG bipolar"
        elif montage_kind == "adjacent_contacts":
            montage_label = "adjacent-contact bipolar"
        chain_breaks = auto_bipolar_metadata.get("chain_breaks", []) or []
        break_suffix = ""
        if chain_breaks:
            break_names = ", ".join(self._format_chain_name(entry.get("chain_name")) for entry in chain_breaks[:3])
            break_suffix = f". Broken chains: {break_names}"
        self.message_handler(f"{montage_label.title()} view ready ({len(derived_channels)} derived channels){break_suffix}")
        self._set_workflow_message(f"Reviewing {montage_label} channels")

    def _toggle_waveform_cursor_shortcut(self):
        if self.backend is None or not hasattr(self.window, "waveform_plot"):
            return
        self.toggle_waveform_cursor(not self.window.waveform_plot.is_cursor_enabled())

    def _toggle_waveform_measure_shortcut(self):
        if self.backend is None or not hasattr(self.window, "waveform_plot"):
            return
        self.toggle_waveform_measure(not self.window.waveform_plot.is_measurement_enabled())

    def _clear_waveform_inspect_modes(self):
        if self.backend is None or not hasattr(self.window, "waveform_plot"):
            return
        if self.window.waveform_plot.is_cursor_enabled():
            self.toggle_waveform_cursor(False)
        if self.window.waveform_plot.is_measurement_enabled():
            self.toggle_waveform_measure(False)

    def _install_waveform_shortcuts(self):
        self._clear_waveform_shortcuts()
        if not self.waveform_shortcuts_enabled:
            return
        self.window.waveform_shortcut_objects = []
        for spec in self._waveform_shortcut_specs():
            sequence_text = self._normalize_shortcut_text(
                self.waveform_shortcut_bindings.get(spec["id"], spec["default"])
            )
            if not sequence_text:
                continue
            shortcut = QShortcut(QKeySequence(sequence_text), self.window)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(
                lambda current_spec=spec: self._run_waveform_shortcut(
                    current_spec["handler"],
                    current_spec.get("required_widget"),
                )
            )
            self.window.waveform_shortcut_objects.append(shortcut)

    def connect_signal_and_slot(self, biomarker_type='HFO'):
        # classifier default buttons
        safe_connect_signal_slot(self.window.default_cpu_button.clicked, self.set_classifier_param_cpu_default)
        safe_connect_signal_slot(self.window.default_gpu_button.clicked, self.set_classifier_param_gpu_default)

        # choose model files connection
        safe_connect_signal_slot(self.window.choose_artifact_model_button.clicked, lambda: self.choose_model_file("artifact"))
        safe_connect_signal_slot(self.window.choose_spike_model_button.clicked, lambda: self.choose_model_file("spike"))
        safe_connect_signal_slot(self.window.choose_ehfo_model_button.clicked, lambda: self.choose_model_file("ehfo"))

        # custom model param connection
        safe_connect_signal_slot(self.window.classifier_save_button.clicked, self.set_custom_classifier_param)

        # detect_all_button
        safe_connect_signal_slot(self.window.detect_all_button.clicked, self.classify)
        self.window.detect_all_button.setEnabled(False)
        # # self.detect_artifacts_button.clicked.connect(lambda : self.classify(False))

        safe_connect_signal_slot(self.window.save_csv_button.clicked, self.save_to_excel)
        self.window.save_csv_button.setEnabled(False)
        if hasattr(self.window, "save_report_button"):
            safe_connect_signal_slot(self.window.save_report_button.clicked, self.save_analysis_report)
            self.window.save_report_button.setEnabled(False)

        # set n_jobs min and max
        self.window.n_jobs_spinbox.setMinimum(1)
        self.window.n_jobs_spinbox.setMaximum(mp.cpu_count())

        # set default n_jobs
        self.window.n_jobs_spinbox.setValue(getattr(self.backend, "n_jobs", 1))
        safe_connect_signal_slot(self.window.n_jobs_ok_button.clicked, self.set_n_jobs)
        safe_connect_signal_slot(self.window.n_jobs_spinbox.editingFinished, self.set_n_jobs)

        safe_connect_signal_slot(self.window.save_npz_button.clicked, self.save_to_npz)
        self.window.save_npz_button.setEnabled(False)

        safe_connect_signal_slot(self.window.Filter60Button.toggled, self.switch_60)
        self.window.Filter60Button.setEnabled(False)
        safe_connect_signal_slot(self.window.toggle_filtered_checkbox.stateChanged, self.toggle_filtered)
        safe_connect_signal_slot(self.window.normalize_vertical_input.stateChanged, self.waveform_plot_button_clicked)

        safe_connect_signal_slot(self.window.bipolar_button.clicked, self.open_bipolar_channel_selection)
        self.window.bipolar_button.setEnabled(False)
        safe_connect_signal_slot(self.window.waveform_plot_button.clicked, self.waveform_plot_button_clicked)
        for widget in (
            self.window.display_time_window_input,
            self.window.Time_Increment_Input,
            self.window.n_channel_input,
        ):
            safe_connect_signal_slot(widget.editingFinished, self.waveform_plot_button_clicked)
        safe_connect_signal_slot(self.window.Choose_Channels_Button.clicked, self.open_channel_selection)
        if hasattr(self.window, "review_channels_button"):
            safe_connect_signal_slot(self.window.review_channels_button.clicked, self.open_channel_selection)
        if hasattr(self.window, "montage_tool_button"):
            safe_connect_signal_slot(self.window.montage_tool_button.clicked, self.open_bipolar_channel_selection)
        if hasattr(self.window, "referential_tool_button"):
            safe_connect_signal_slot(self.window.referential_tool_button.clicked, self.show_referential_channels)
        if hasattr(self.window, "average_reference_button"):
            safe_connect_signal_slot(self.window.average_reference_button.toggled, self.toggle_average_reference_view)
        if hasattr(self.window, "auto_bipolar_button"):
            safe_connect_signal_slot(self.window.auto_bipolar_button.toggled, self.toggle_auto_bipolar_view)
        if hasattr(self.window, "montage_status_badge"):
            safe_connect_signal_slot(self.window.montage_status_badge.clicked, self.show_auto_bipolar_details)
        if hasattr(self.window, "montage_break_badge"):
            safe_connect_signal_slot(self.window.montage_break_badge.clicked, self.show_auto_bipolar_details)
        if hasattr(self.window, "waveform_reset_view_button"):
            safe_connect_signal_slot(self.window.waveform_reset_view_button.clicked, self.reset_waveform_view_state)
        if hasattr(self.window, "highlight_channel_button"):
            safe_connect_signal_slot(
                self.window.highlight_channel_button.toggled,
                self.toggle_highlighted_waveform_channel_focus,
            )
        if hasattr(self.window, "neighbor_channels_button"):
            safe_connect_signal_slot(
                self.window.neighbor_channels_button.toggled,
                self.toggle_neighbor_channel_focus,
            )
        if hasattr(self.window, "clean_view_button"):
            safe_connect_signal_slot(self.window.clean_view_button.toggled, self.toggle_clean_view)
        if hasattr(self.window, "overlap_review_button"):
            safe_connect_signal_slot(self.window.overlap_review_button.clicked, self.open_overlap_review_dialog)
        if hasattr(self.window, "event_channels_button"):
            safe_connect_signal_slot(self.window.event_channels_button.toggled, self.toggle_event_channel_scope)
        if hasattr(self.window, "all_channels_button"):
            safe_connect_signal_slot(self.window.all_channels_button.clicked, self.show_all_channels)
        if hasattr(self.window, "cursor_tool_button"):
            safe_connect_signal_slot(self.window.cursor_tool_button.toggled, self.toggle_waveform_cursor)
        if hasattr(self.window, "measure_tool_button"):
            safe_connect_signal_slot(self.window.measure_tool_button.toggled, self.toggle_waveform_measure)
        if hasattr(self.window, "hotspot_tool_button"):
            safe_connect_signal_slot(self.window.hotspot_tool_button.toggled, self.toggle_hotspot_channels)
        if hasattr(self.window, "open_review_button"):
            safe_connect_signal_slot(self.window.open_review_button.clicked, self.open_annotation)
        if hasattr(self.window, "normalize_tool_button"):
            safe_connect_signal_slot(
                self.window.normalize_tool_button.toggled,
                lambda checked: self.window.normalize_vertical_input.setChecked(checked),
            )
        if hasattr(self.window, "raw_tool_button"):
            safe_connect_signal_slot(self.window.raw_tool_button.clicked, lambda _checked=False: self._show_raw_waveform_view())
        if hasattr(self.window, "filtered_tool_button"):
            safe_connect_signal_slot(
                self.window.filtered_tool_button.toggled,
                lambda checked: self.window.toggle_filtered_checkbox.setChecked(checked),
            )
        if hasattr(self.window, "filter60_tool_button"):
            safe_connect_signal_slot(
                self.window.filter60_tool_button.toggled,
                lambda checked: self.window.Filter60Button.setChecked(checked),
            )
        if hasattr(self.window, "prev_event_button"):
            safe_connect_signal_slot(self.window.prev_event_button.clicked, self.show_previous_event)
        if hasattr(self.window, "center_event_button"):
            safe_connect_signal_slot(self.window.center_event_button.clicked, self.center_current_event)
        if hasattr(self.window, "next_event_button"):
            safe_connect_signal_slot(self.window.next_event_button.clicked, self.show_next_event)
        if hasattr(self.window, "pending_event_button"):
            safe_connect_signal_slot(self.window.pending_event_button.clicked, lambda: self.jump_to_pending_review(1))
        if hasattr(self.window, "snapshot_button"):
            safe_connect_signal_slot(self.window.snapshot_button.clicked, self.export_waveform_snapshot)
        if hasattr(self.window, "go_to_time_button"):
            safe_connect_signal_slot(self.window.go_to_time_button.clicked, self.go_to_time_position)
        if hasattr(self.window, "go_to_time_input"):
            safe_connect_signal_slot(self.window.go_to_time_input.editingFinished, self.go_to_time_position)
        if hasattr(self.window, "zoom_in_button"):
            safe_connect_signal_slot(self.window.zoom_in_button.clicked, lambda: self.zoom_waveform(0.5))
        if hasattr(self.window, "zoom_out_button"):
            safe_connect_signal_slot(self.window.zoom_out_button.clicked, lambda: self.zoom_waveform(2.0))
        for button in getattr(self.window, "graph_window_preset_buttons", []):
            safe_connect_signal_slot(
                button.clicked,
                lambda _checked=False, current_button=button: self.apply_waveform_window_preset(
                    float(current_button.property("windowPreset") or 0.0)
                ),
            )
        for button in getattr(self.window, "graph_channel_preset_buttons", []):
            safe_connect_signal_slot(
                button.clicked,
                lambda _checked=False, current_button=button: self.apply_visible_channel_preset(
                    current_button.property("channelPreset")
                ),
            )
        if hasattr(self.window, "graph_event_channels_button"):
            safe_connect_signal_slot(self.window.graph_event_channels_button.clicked, self.focus_event_channels)
        if hasattr(self.window, "graph_hotspot_button"):
            safe_connect_signal_slot(self.window.graph_hotspot_button.clicked, self.focus_hotspot_channels)
        if hasattr(self.window, "graph_cursor_button"):
            safe_connect_signal_slot(self.window.graph_cursor_button.toggled, self.toggle_waveform_cursor)
        if hasattr(self.window, "graph_zoom_out_review_button"):
            safe_connect_signal_slot(self.window.graph_zoom_out_review_button.clicked, lambda: self.zoom_waveform(2.0))
        if hasattr(self.window, "graph_zoom_in_review_button"):
            safe_connect_signal_slot(self.window.graph_zoom_in_review_button.clicked, lambda: self.zoom_waveform(0.5))
        if hasattr(self.window, "graph_snapshot_button"):
            safe_connect_signal_slot(self.window.graph_snapshot_button.clicked, self.export_waveform_snapshot)
        self._install_waveform_shortcuts()

        # annotation button
        safe_connect_signal_slot(self.window.annotation_button.clicked, self.open_annotation)
        self.window.annotation_button.setEnabled(False)
        safe_connect_signal_slot(self.window.switch_run_button.clicked, self.choose_active_run)
        safe_connect_signal_slot(self.window.accept_run_button.clicked, self.accept_active_run)
        safe_connect_signal_slot(self.window.compare_runs_button.clicked, self.show_run_comparison)
        if hasattr(self.window, "run_table"):
            safe_connect_signal_slot(self.window.run_table.cellDoubleClicked, self.activate_run_from_table)
            safe_connect_signal_slot(self.window.run_table.cellClicked, self.handle_run_table_click)
        if hasattr(self.window, "channel_table"):
            safe_connect_signal_slot(self.window.channel_table.cellDoubleClicked, self.highlight_channel_from_table)
        if hasattr(self.window, "active_run_selector"):
            safe_connect_signal_slot(self.window.active_run_selector.currentIndexChanged, self.activate_run_from_selector)
        if hasattr(self.window, "prepare_tab_button"):
            safe_connect_signal_slot(self.window.prepare_tab_button.clicked, self.open_prepare_tab)
        if hasattr(self.window, "detector_mode_combo"):
            safe_connect_signal_slot(self.window.detector_mode_combo.currentIndexChanged, self.select_detector_mode)
        if hasattr(self.window, "detector_apply_button"):
            safe_connect_signal_slot(self.window.detector_apply_button.clicked, self.apply_selected_detector_parameters)
        if hasattr(self.window, "detector_run_button"):
            safe_connect_signal_slot(self.window.detector_run_button.clicked, self.run_selected_detector_workflow)
        if hasattr(self.window, "detector_subtabs"):
            safe_connect_signal_slot(self.window.detector_subtabs.currentChanged, self.sync_detector_mode_combo)
        if hasattr(self.window, "classifier_mode_combo"):
            safe_connect_signal_slot(self.window.classifier_mode_combo.currentIndexChanged, self.apply_classifier_mode)
        if hasattr(self.window, "classifier_apply_button"):
            safe_connect_signal_slot(self.window.classifier_apply_button.clicked, self.apply_classifier_setup)
        if hasattr(self.window, "classifier_run_button"):
            safe_connect_signal_slot(self.window.classifier_run_button.clicked, self.run_classifier_workflow)
        if hasattr(self.window, "run_stats_activate_button"):
            safe_connect_signal_slot(self.window.run_stats_activate_button.clicked, self.activate_selected_run_from_popup)
        if hasattr(self.window, "run_stats_accept_button"):
            safe_connect_signal_slot(self.window.run_stats_accept_button.clicked, self.accept_selected_run_from_popup)
        if hasattr(self.window, "run_stats_export_button"):
            safe_connect_signal_slot(self.window.run_stats_export_button.clicked, self.save_to_excel)
        if hasattr(self.window, "run_stats_report_button"):
            safe_connect_signal_slot(self.window.run_stats_report_button.clicked, self.save_analysis_report)
        self.window.switch_run_button.setEnabled(False)
        self.window.accept_run_button.setEnabled(False)
        self.window.compare_runs_button.setEnabled(False)
        self._set_report_export_enabled(False)

        self.window.Choose_Channels_Button.setEnabled(False)
        self.window.waveform_plot_button.setEnabled(False)

        # check if gpu is available
        self.gpu = bool(torch is not None and torch.cuda.is_available())
        # print(f"GPU available: {self.gpu}")
        if not self.gpu:
            # disable gpu buttons
            self.window.default_gpu_button.setEnabled(False)

        if biomarker_type == 'HFO':
            safe_connect_signal_slot(self.window.overview_filter_button.clicked, self.filter_data)
            # set filter button to be disabled by default
            self.window.overview_filter_button.setEnabled(False)
            # # self.show_original_button.clicked.connect(self.toggle_filtered)

            safe_connect_signal_slot(self.window.mni_detect_button.clicked, self.detect_HFOs)
            self.window.mni_detect_button.setEnabled(False)
            safe_connect_signal_slot(self.window.ste_detect_button.clicked, self.detect_HFOs)
            self.window.ste_detect_button.setEnabled(False)
            safe_connect_signal_slot(self.window.hil_detect_button.clicked, self.detect_HFOs)
            self.window.hil_detect_button.setEnabled(False)

            safe_connect_signal_slot(self.window.STE_save_button.clicked, self.save_ste_params)
            safe_connect_signal_slot(self.window.MNI_save_button.clicked, self.save_mni_params)
            safe_connect_signal_slot(self.window.HIL_save_button.clicked, self.save_hil_params)
            self.window.STE_save_button.setEnabled(False)
            self.window.MNI_save_button.setEnabled(False)
            self.window.HIL_save_button.setEnabled(False)
        elif biomarker_type == 'Spindle':
            safe_connect_signal_slot(self.window.overview_filter_button.clicked, self.filter_data)

            # set filter button to be disabled by default
            self.window.overview_filter_button.setEnabled(False)

            safe_connect_signal_slot(self.window.yasa_detect_button.clicked, self.detect_Spindles)
            self.window.yasa_detect_button.setEnabled(False)

            safe_connect_signal_slot(self.window.YASA_save_button.clicked, self.save_yasa_params)
            # self.window.YASA_save_button.setEnabled(False)
        elif biomarker_type == 'Spike':
            self.window.overview_filter_button.setEnabled(False)
            self.window.detect_all_button.setEnabled(False)
            self.window.save_csv_button.setEnabled(False)
            self.window.save_npz_button.setEnabled(False)
            self._set_report_export_enabled(False)
        self._configure_main_window_input_conventions()
        self.update_waveform_toolbar_state()
        self.update_setup_action_state()

    def set_classifier_param_display(self):
        if self.backend is None:
            return
        classifier_param = self.backend.get_classifier_param()
        if classifier_param is None:
            self.refresh_classifier_mode_ui()
            return

        self.window.overview_artifact_path_display.setText(
            self._get_classifier_source_display_value(classifier_param, "artifact_path", "artifact_card")
        )
        self.window.overview_spike_path_display.setText(
            self._get_classifier_source_display_value(classifier_param, "spike_path", "spike_card")
        )
        self.window.overview_ehfo_path_display.setText(
            self._get_classifier_source_display_value(classifier_param, "ehfo_path", "ehfo_card")
        )

        self.window.overview_use_spike_checkbox.setChecked(classifier_param.use_spike)
        self.window.overview_use_ehfo_checkbox.setChecked(classifier_param.use_ehfo)
        self.window.overview_device_display.setText(str(classifier_param.device))
        self.window.overview_batch_size_display.setText(str(classifier_param.batch_size))

        # set also the input fields
        self.window.classifier_artifact_filename.setText(classifier_param.artifact_path)
        self.window.classifier_spike_filename.setText(classifier_param.spike_path)
        self.window.classifier_ehfo_filename.setText(classifier_param.ehfo_path)

        self.window.classifier_artifact_card_name.setText(classifier_param.artifact_card)
        self.window.classifier_spike_card_name.setText(classifier_param.spike_card)
        self.window.classifier_ehfo_card_name.setText(classifier_param.ehfo_card)

        self.window.use_spike_checkbox.setChecked(classifier_param.use_spike)
        self.window.use_ehfo_checkbox.setChecked(classifier_param.use_ehfo)
        self.window.classifier_device_input.setText(str(classifier_param.device))
        self.window.classifier_batch_size_input.setText(str(classifier_param.batch_size))
        self.refresh_classifier_mode_ui()

    def set_classifier_param_gpu_default(self):
        if self.backend is None:
            self.handle_unsupported_biomarker_mode("GPU classifier setup is not available for the current biomarker mode yet.")
            return
        self.backend.set_default_gpu_classifier()
        self.set_classifier_param_display()

    def set_classifier_param_cpu_default(self):
        if self.backend is None:
            self.handle_unsupported_biomarker_mode("CPU classifier setup is not available for the current biomarker mode yet.")
            return
        self.backend.set_default_cpu_classifier()
        self.set_classifier_param_display()

    def set_custom_classifier_param(self):
        if self.backend is None:
            self.handle_unsupported_biomarker_mode("Classifier setup is not available for the current biomarker mode yet.")
            return False
        # local
        artifact_path = self.window.classifier_artifact_filename.text().strip()
        spike_path = self.window.classifier_spike_filename.text().strip()
        ehfo_path = self.window.classifier_ehfo_filename.text().strip()

        # hugging face
        artifact_card = self.window.classifier_artifact_card_name.text().strip()
        spike_card = self.window.classifier_spike_card_name.text().strip()
        ehfo_card = self.window.classifier_ehfo_card_name.text().strip()

        use_spike = self.window.use_spike_checkbox.isChecked()
        use_ehfo = self.window.use_ehfo_checkbox.isChecked()
        device, model_type = self._normalize_classifier_device_input(self.window.classifier_device_input.text())
        batch_size = self._parse_int_input(self.window.classifier_batch_size_input.text(), "Batch size", positive=True)

        if not (artifact_path or artifact_card):
            raise ValueError("Artifact classifier source is required in Custom mode.")
        if use_spike and not (spike_path or spike_card):
            raise ValueError("Spike classifier source is required when spike classification is enabled.")
        if use_ehfo and not (ehfo_path or ehfo_card):
            raise ValueError("eHFO classifier source is required when eHFO classification is enabled.")
        source_preference = self._infer_classifier_source_preference(
            artifact_path=artifact_path,
            spike_path=spike_path,
            ehfo_path=ehfo_path,
            artifact_card=artifact_card,
            spike_card=spike_card,
            ehfo_card=ehfo_card,
        )

        classifier_param = ParamClassifier(artifact_path=artifact_path, spike_path=spike_path, ehfo_path=ehfo_path,
                                           artifact_card=artifact_card, spike_card=spike_card, ehfo_card=ehfo_card,
                                           use_spike=use_spike, use_ehfo=use_ehfo,
                                           device=device, batch_size=batch_size, model_type=model_type,
                                           source_preference=source_preference)
        self.backend.set_classifier(classifier_param)
        self.set_classifier_param_display()
        return True

    def choose_model_file(self, model_type):
        fname, _ = QFileDialog.getOpenFileName(self.window, 'Open file', "", ".tar files (*.tar)")
        if model_type == "artifact":
            self.window.classifier_artifact_filename.setText(fname)
        elif model_type == "spike":
            self.window.classifier_spike_filename.setText(fname)
        elif model_type == "ehfo":
            self.window.classifier_ehfo_filename.setText(fname)

    def _classify(self):
        if self.backend is None or not self.biomarker_supports_classification():
            raise ValueError(f"Classification is not available for biomarker mode '{self.get_biomarker_display_name()}'.")
        threshold = 0.5
        seconds_to_ignore_before = self._parse_float_input(
            self.window.overview_ignore_before_input.text(),
            "Ignore window before events",
            non_negative=True,
        )
        seconds_to_ignore_after = self._parse_float_input(
            self.window.overview_ignore_after_input.text(),
            "Ignore window after events",
            non_negative=True,
        )
        self.backend.classify_artifacts([seconds_to_ignore_before, seconds_to_ignore_after], threshold)

        use_spike = self.window.overview_use_spike_checkbox.isChecked()
        use_ehfo = self.window.overview_use_ehfo_checkbox.isChecked()
        if use_spike:
            self.backend.classify_spikes()
        if use_ehfo and hasattr(self.backend, "classify_ehfos"):
            self.backend.classify_ehfos()
        return []

    def _classify_finished(self):
        self._end_busy_task()
        self.message_handler("Classification finished!..")
        self._set_workflow_message("Classification complete")
        self.update_statistics_label()
        self.window.waveform_plot.set_plot_biomarkers(self._has_case_visible_runs() or bool(self.backend and self.backend.detected))
        self.window.save_csv_button.setEnabled(True)
        self._set_report_export_enabled(True)
        self.update_status_indicators()

    def classify(self):
        self.message_handler(f"Classifying {self.get_biomarker_display_name()} events...")
        self._set_workflow_message("Running classifiers...")
        self._begin_busy_task("classification", "Classifying", self._classification_busy_buttons())
        worker = Worker(lambda progress_callback: self._classify())
        self._connect_worker(worker, "Classification", result_handler=self._classify_finished)

    def update_statistics_label(self):
        if not self._ui_object_is_alive(getattr(self, "window", None)):
            return
        if not self._ui_object_is_alive(getattr(self.window, "statistics_label", None)):
            return
        if self.backend is None or self.backend.event_features is None:
            self.window.statistics_label.setText("")
            self.update_run_management_panel()
            self.update_decision_overview()
            self._sync_workspace_state()
            return
        run_count = len(getattr(getattr(self.backend, "analysis_session", None), "runs", {}))
        active_run = getattr(self.backend, "analysis_session", None).get_active_run() if hasattr(self.backend, "analysis_session") else None
        accepted_run = getattr(self.backend, "analysis_session", None).get_accepted_run() if hasattr(self.backend, "analysis_session") else None
        top_channel = ""
        if hasattr(self.backend, "get_channel_ranking"):
            ranking = self.backend.get_channel_ranking()
            if ranking:
                top = ranking[0]
                top_channel = f"\n Top channel: {top['channel_name']} ({top['accepted_predicted']} accepted / {top['total_events']} total)"
        if self.biomarker_type == 'HFO':
            num_HFO = self.backend.event_features.get_num_biomarker()
            num_artifact = self.backend.event_features.get_num_artifact()
            num_spk_hfo = self.backend.event_features.get_num_spike()
            num_e_hfo = self.backend.event_features.get_num_ehfo()
            num_real = self.backend.event_features.get_num_real()

            self.window.statistics_label.setText(" Total HFOs: " + str(num_HFO) + \
                                          "\n Artifacts: " + str(num_artifact) + \
                                          "\n Real HFOs: " + str(num_real) + \
                                          "\n spkHFOs: " + str(num_spk_hfo) + \
                                                 "\n eHFOs: " + str(num_e_hfo) + \
                                                 f"\n Runs in session: {run_count}" + \
                                                 (f"\n Active run: {active_run.detector_name}" if active_run else "") + \
                                                 (f"\n Accepted run: {accepted_run.detector_name}" if accepted_run else "") + \
                                                 top_channel)
        elif self.biomarker_type == 'Spindle':
            num_spindle = self.backend.event_features.get_num_biomarker()
            num_artifact = self.backend.event_features.get_num_artifact()
            num_spike = self.backend.event_features.get_num_spike()
            num_real = self.backend.event_features.get_num_real()

            self.window.statistics_label.setText(" Total spindles: " + str(num_spindle) + \
                                                 "\n Artifacts: " + str(num_artifact) + \
                                                 "\n Spike-associated: " + str(num_spike) + \
                                                 "\n Accepted spindles: " + str(num_real) + \
                                                 f"\n Runs in session: {run_count}" + \
                                                 (f"\n Active run: {active_run.detector_name}" if active_run else "") + \
                                                 (f"\n Accepted run: {accepted_run.detector_name}" if accepted_run else "") + \
                                                 top_channel)
        elif self.biomarker_type == 'Spike':
            self.window.statistics_label.setText(
                " Spike review summary\n"
                " Use this mode to inspect signal, choose channels, and manage saved sessions.\n"
                " Detector-specific spike automation is not yet available in this workspace."
            )
        self.update_run_management_panel()
        self.update_decision_overview()
        self._sync_workspace_state()

    def update_run_management_panel(self):
        if not hasattr(self.window, "run_summary_label"):
            return
        workflow_name = self.get_biomarker_display_name()
        if self.backend is None:
            self.window.run_summary_label.setText(f"No {workflow_name.lower()} runs yet.")
            self.window.run_summary_label.setToolTip("Load a recording, then create a run to review.")
            if hasattr(self.window, "analysis_summary_label"):
                self.window.analysis_summary_label.setText("No saved runs yet")
                self.window.analysis_summary_label.setToolTip("Load a recording, then create a run to review.")
            self.window.switch_run_button.setEnabled(False)
            self.window.accept_run_button.setEnabled(False)
            self.window.compare_runs_button.setEnabled(False)
            if hasattr(self.window, "run_stats_activate_button"):
                self.window.run_stats_activate_button.setEnabled(False)
            if hasattr(self.window, "run_stats_accept_button"):
                self.window.run_stats_accept_button.setEnabled(False)
            if hasattr(self.window, "run_stats_export_button"):
                self.window.run_stats_export_button.setEnabled(False)
            self._set_report_export_enabled(False)
            self._sync_active_run_selector([], None)
            self.update_active_run_panel(None, None)
            self.populate_decision_tables([], [], [])
            return

        current_runs = self._collect_case_run_summaries(self.biomarker_type)
        case_runs = self._collect_case_run_summaries()
        visible_case_count = sum(1 for run in case_runs if run.get("visible"))
        session = getattr(self.backend, "analysis_session", None)
        active_run = session.get_active_run() if session is not None else None
        accepted_run = session.get_accepted_run() if session is not None else None
        if not current_runs:
            self.window.run_summary_label.setText(f"No {workflow_name.lower()} runs yet.")
            if case_runs:
                summary_text = f"No {workflow_name.lower()} runs yet • {len(case_runs)} case runs available"
                self.window.run_summary_label.setText(summary_text)
                self.window.run_summary_label.setToolTip(summary_text)
            else:
                self.window.run_summary_label.setToolTip(f"{workflow_name} detector settings are ready for the next run.")
            if hasattr(self.window, "analysis_summary_label"):
                if case_runs:
                    self.window.analysis_summary_label.setText(f"{workflow_name} detector settings are ready • case runs remain available")
                    self.window.analysis_summary_label.setToolTip(
                        f"No saved {workflow_name.lower()} runs yet. Existing case runs from other biomarker workflows remain available for comparison."
                    )
                else:
                    self.window.analysis_summary_label.setText(f"{workflow_name} detector settings are ready for the next run")
                    self.window.analysis_summary_label.setToolTip(
                        f"No saved {workflow_name.lower()} runs yet. Current detector settings will be used for the next detection."
                    )
            self.window.switch_run_button.setEnabled(False)
            self.window.accept_run_button.setEnabled(False)
            self.window.compare_runs_button.setEnabled(bool(case_runs))
            if hasattr(self.window, "run_stats_activate_button"):
                self.window.run_stats_activate_button.setEnabled(bool(case_runs))
            if hasattr(self.window, "run_stats_accept_button"):
                self.window.run_stats_accept_button.setEnabled(False)
            if hasattr(self.window, "run_stats_export_button"):
                self.window.run_stats_export_button.setEnabled(False)
            self._set_report_export_enabled(active_run is not None or accepted_run is not None)
            self._sync_active_run_selector([], active_run)
            self.update_active_run_panel(active_run, accepted_run)
            comparison_rows = self._compare_case_runs().get("pairwise_overlap", []) if case_runs else []
            self.populate_decision_tables(case_runs, [], comparison_rows)
            return

        summary_parts = [f"{len(current_runs)} {workflow_name.lower()} runs", f"{visible_case_count} case-visible"]
        if len(case_runs) != len(current_runs):
            summary_parts.append(f"{len(case_runs)} case runs")
        if active_run is not None:
            summary_parts.append(f"active {active_run.detector_name}")
        if accepted_run is not None:
            summary_parts.append(f"accepted {accepted_run.detector_name}")
        if hasattr(self.backend, "get_channel_ranking"):
            ranking = self.backend.get_channel_ranking(active_run.run_id if active_run else None)
        else:
            ranking = []
        summary_text = " • ".join(summary_parts)
        self.window.run_summary_label.setText(summary_text)
        self.window.run_summary_label.setToolTip(summary_text)
        if hasattr(self.window, "analysis_summary_label"):
            top_hint = f" • top {ranking[0]['channel_name']}" if ranking else ""
            self.window.analysis_summary_label.setText(summary_text + top_hint)
            self.window.analysis_summary_label.setToolTip(summary_text + top_hint)
        self.window.switch_run_button.setEnabled(len(current_runs) > 1)
        self.window.accept_run_button.setEnabled(active_run is not None)
        self.window.compare_runs_button.setEnabled(bool(case_runs))
        if hasattr(self.window, "run_stats_activate_button"):
            self.window.run_stats_activate_button.setEnabled(bool(case_runs))
        if hasattr(self.window, "run_stats_accept_button"):
            self.window.run_stats_accept_button.setEnabled(active_run is not None)
        if hasattr(self.window, "run_stats_export_button"):
            self.window.run_stats_export_button.setEnabled(hasattr(self.backend, "export_clinical_summary") or hasattr(self.backend, "export_excel"))
        self._set_report_export_enabled(active_run is not None or accepted_run is not None)
        self._sync_active_run_selector(current_runs, active_run)
        self.update_active_run_panel(active_run, accepted_run)
        comparison_rows = self._compare_case_runs().get("pairwise_overlap", [])
        self.populate_decision_tables(case_runs, ranking if 'ranking' in locals() else [], comparison_rows)

    def _default_report_path(self):
        default_path = os.path.expanduser("~")
        if self.backend and hasattr(self.backend, "edf_param") and self.backend.edf_param:
            edf_path = self.backend.edf_param.get("edf_fn", "")
            if edf_path:
                directory = os.path.dirname(edf_path) or default_path
                base_name = os.path.splitext(os.path.basename(edf_path))[0]
                default_path = os.path.join(directory, f"{base_name}_report.html")
        return default_path

    def _capture_waveform_snapshot_to_temp(self):
        if not hasattr(self.window, "waveformWiget"):
            return None
        pixmap = self.window.waveformWiget.grab()
        if pixmap.isNull():
            return None
        fd, temp_path = tempfile.mkstemp(prefix="pybrain_report_", suffix=".png")
        os.close(fd)
        if pixmap.save(temp_path):
            return temp_path
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        return None

    def save_analysis_report(self):
        if self.backend is None or not hasattr(self.backend, "analysis_session"):
            return

        default_path = self._default_report_path()
        fname, _ = QFileDialog.getSaveFileName(
            self.window,
            "Export Analysis Report",
            default_path,
            "HTML Report (*.html)",
            options=QFileDialog.DontUseNativeDialog,
        )
        if not fname:
            return

        snapshot_path = self._capture_waveform_snapshot_to_temp()
        try:
            report_path = export_analysis_report(
                fname,
                self.backend,
                biomarker_label=self.get_biomarker_display_name(),
                app_name=self.window.windowTitle() or "PyBrain",
                snapshot_source_path=snapshot_path,
                run_summaries=self._collect_case_run_summaries(),
                comparison_rows=self._compare_case_runs().get("pairwise_overlap", []),
                workbook_exporter=self._export_case_clinical_summary,
            )
        finally:
            if snapshot_path and os.path.exists(snapshot_path):
                try:
                    os.unlink(snapshot_path)
                except OSError:
                    pass

        self.message_handler(f"Exported analysis report: {os.path.basename(str(report_path))}")
        self._set_workflow_message("Analysis report exported")

    def _format_run_param_label(self, key):
        labels = {
            "fp": "Filter low",
            "fs": "Filter high",
            "rp": "Pass ripple",
            "rs": "Stop attenuation",
            "space": "Transition",
            "rms_window": "RMS window",
            "min_window": "Min window",
            "min_gap": "Min gap",
            "epoch_len": "Epoch length",
            "min_osc": "Min oscillations",
            "rms_thres": "RMS threshold",
            "peak_thres": "Peak threshold",
            "epoch_time": "Epoch time",
            "epo_CHF": "CHF epoch",
            "per_CHF": "CHF %",
            "min_win": "Min window",
            "thrd_perc": "Threshold %",
            "base_seg": "Baseline segment",
            "base_shift": "Baseline shift",
            "base_thrd": "Baseline threshold",
            "base_min": "Baseline minimum",
            "sd_threshold": "SD threshold",
            "freq_sp": "Spindle band",
            "freq_broad": "Broad band",
            "duration": "Duration",
            "min_distance": "Min distance",
            "corr": "Correlation",
            "rel_pow": "Relative power",
            "rms": "RMS",
            "model_type": "Model",
            "batch_size": "Batch size",
            "device": "Device",
        }
        return labels.get(key, key.replace("_", " ").title())

    def _format_run_param_value(self, value):
        if isinstance(value, (list, tuple)):
            return " - ".join(self._format_run_param_value(item) for item in value)
        if isinstance(value, float):
            if abs(value) >= 100:
                return f"{value:.0f}"
            if abs(value) >= 10:
                return f"{value:.1f}"
            if abs(value) >= 1:
                return f"{value:.2f}"
            return f"{value:.3f}".rstrip("0").rstrip(".")
        if isinstance(value, bool):
            return "Yes" if value else "No"
        return str(value)

    def _build_active_run_param_rows(self, run):
        if run is None:
            return []
        rows = []
        if run.param_filter is not None:
            filter_dict = getattr(run.param_filter, "to_dict", lambda: {})()
            fp = filter_dict.get("fp")
            fs = filter_dict.get("fs")
            if fp is not None and fs is not None:
                rows.append(("Filter band", f"{self._format_run_param_value(fp)} - {self._format_run_param_value(fs)} Hz"))
            for key in ("rp", "rs"):
                if key in filter_dict:
                    rows.append((self._format_run_param_label(key), self._format_run_param_value(filter_dict[key])))
        if run.param_detector is not None:
            rows.append(("Detector", str(run.detector_name)))
            detector_dict = getattr(run.param_detector.detector_param, "to_dict", lambda: {})()
            for key, value in detector_dict.items():
                if key in {"sample_freq", "n_jobs", "pass_band", "stop_band"}:
                    continue
                rows.append((self._format_run_param_label(key), self._format_run_param_value(value)))
        if run.classified and run.param_classifier is not None:
            classifier_dict = getattr(run.param_classifier, "to_dict", lambda: {})()
            for key in ("model_type", "device", "batch_size"):
                if key in classifier_dict:
                    rows.append((self._format_run_param_label(key), self._format_run_param_value(classifier_dict[key])))
        overlap_summary = dict(getattr(run, "summary", {}) or {})
        if overlap_summary.get("overlap_action") and overlap_summary.get("overlap_action") != HFO_Feature.OVERLAP_ACTION_DISABLED:
            rows.append(("Overlap review", overlap_summary.get("overlap_action_label", "Tag only")))
            rows.append(("Overlap tagged", str(overlap_summary.get("overlap_tagged", 0))))
            rows.append(("Min overlap", f"{self._format_run_param_value(overlap_summary.get('overlap_min_overlap_ms', 0.0))} ms"))
            rows.append(("Min channels", str(overlap_summary.get("overlap_min_channels", 2))))
            if overlap_summary.get("overlap_action") == HFO_Feature.OVERLAP_ACTION_HIDE:
                rows.append(("Visible after overlap", str(overlap_summary.get("overlap_visible", 0))))
        return rows[:12]

    def _build_current_config_rows(self):
        rows = []
        if self.backend is None:
            return rows
        if getattr(self.backend, "param_filter", None) is not None:
            filter_dict = self.backend.param_filter.to_dict()
            fp = filter_dict.get("fp")
            fs = filter_dict.get("fs")
            if fp is not None and fs is not None:
                rows.append(("Filter band", f"{self._format_run_param_value(fp)} - {self._format_run_param_value(fs)} Hz"))
        if getattr(self.backend, "param_detector", None) is not None:
            rows.append(("Detector", str(self.backend.param_detector.detector_type).upper()))
            detector_dict = self.backend.param_detector.detector_param.to_dict()
            for key, value in detector_dict.items():
                if key in {"sample_freq", "n_jobs", "pass_band", "stop_band"}:
                    continue
                rows.append((self._format_run_param_label(key), self._format_run_param_value(value)))
        return rows[:12]

    def _build_classifier_rows(self, classifier_param):
        if classifier_param is None:
            return []
        classifier_dict = getattr(classifier_param, "to_dict", lambda: {})()
        rows = []
        for key in ("model_type", "device", "batch_size"):
            value = classifier_dict.get(key)
            if value not in (None, "", False):
                rows.append((self._format_run_param_label(key), self._format_run_param_value(value)))
        if "use_spike" in classifier_dict:
            rows.append(("Spike model", self._format_run_param_value(classifier_dict.get("use_spike"))))
        if "use_ehfo" in classifier_dict:
            rows.append(("eHFO model", self._format_run_param_value(classifier_dict.get("use_ehfo"))))
        return rows[:6]

    def _populate_two_column_table(self, table, rows):
        if table is None:
            return
        table.setRowCount(len(rows))
        for row_index, (label, value) in enumerate(rows):
            label_item = QTableWidgetItem(label)
            value_item = QTableWidgetItem(value)
            label_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            value_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            table.setItem(row_index, 0, label_item)
            table.setItem(row_index, 1, value_item)
        table.resizeRowsToContents()
        header_height = table.horizontalHeader().height() if table.horizontalHeader() is not None else 24
        row_height = table.verticalHeader().defaultSectionSize() if table.verticalHeader() is not None else 18
        estimated_height = header_height + max(1, len(rows)) * row_height + 8
        target_height = min(table.maximumHeight(), max(table.minimumHeight(), estimated_height))
        table.setFixedHeight(target_height)

    def _sync_active_run_selector(self, runs, active_run):
        selector = getattr(self.window, "active_run_selector", None)
        if selector is None:
            return
        blocker = QSignalBlocker(selector)
        selector.clear()
        if not runs:
            selector.addItem("No runs yet", None)
            selector.setEnabled(False)
            del blocker
            return

        active_ref = None
        if active_run is not None:
            active_ref = (getattr(active_run, "biomarker_type", self.biomarker_type), getattr(active_run, "run_id", None))

        selected_index = 0
        for index, run in enumerate(runs):
            biomarker = run.get("biomarker_type", "")
            label = f"{biomarker} • {run.get('detector_name', '--')} • {run.get('num_events', 0)} ev"
            if run.get("accepted"):
                label += " • accepted"
            ref = (biomarker, run.get("run_id"))
            selector.addItem(label, ref)
            if active_ref is not None and ref == active_ref:
                selected_index = index
        selector.setCurrentIndex(selected_index)
        selector.setEnabled(True)
        del blocker

    def update_active_run_panel(self, active_run, accepted_run):
        if not hasattr(self.window, "active_run_name_label"):
            return
        if active_run is None:
            if self.backend is not None and getattr(self.backend, "param_detector", None) is not None:
                self.window.active_run_name_label.setText(f"{self.backend.param_detector.detector_type.upper()} • configured")
                self.window.active_run_meta_label.setText("Prepared for next run")
                self.window.active_run_meta_label.setToolTip("No saved run yet. These settings will be used for the next detection.")
                self.window.active_run_status_label.setText("Ready")
                self.window.active_run_status_label.setToolTip("Current detector settings are ready for the next detection run.")
                self.window.active_detector_label.setText(str(self.backend.param_detector.detector_type).upper())
                self.window.active_detector_label.setToolTip(f"Current detector: {self.backend.param_detector.detector_type.upper()}")
                classifier_param = getattr(self.backend, "param_classifier", None)
                if classifier_param is not None:
                    self.window.active_classifier_label.setText(str(classifier_param.model_type))
                    self.window.active_classifier_label.setToolTip(f"Classifier model: {classifier_param.model_type}")
                else:
                    self.window.active_classifier_label.setText("No classifier")
                    self.window.active_classifier_label.setToolTip("Classifier is not configured for the current detector settings.")
                rows = self._build_current_config_rows()
                if hasattr(self.window, "active_run_param_table"):
                    self._populate_two_column_table(self.window.active_run_param_table, rows)
                if hasattr(self.window, "active_classifier_table"):
                    self._populate_two_column_table(self.window.active_classifier_table, self._build_classifier_rows(classifier_param))
                return
            self.window.active_run_name_label.setText("No active run")
            self.window.active_run_meta_label.setText("Load a case to begin")
            self.window.active_run_meta_label.setToolTip("Load a recording, then create or activate a run.")
            self.window.active_run_status_label.setText("--")
            self.window.active_run_status_label.setToolTip("")
            self.window.active_detector_label.setText("Detector --")
            self.window.active_detector_label.setToolTip("")
            self.window.active_classifier_label.setText("Classifier --")
            self.window.active_classifier_label.setToolTip("")
            if hasattr(self.window, "active_run_param_table"):
                self.window.active_run_param_table.setRowCount(0)
            if hasattr(self.window, "active_classifier_table"):
                self.window.active_classifier_table.setRowCount(0)
            return

        biomarker = getattr(active_run, "biomarker_type", self.biomarker_type or "Event")
        self.window.active_run_name_label.setText(f"{active_run.detector_name} • {biomarker}")
        summary = active_run.summary or {}
        meta_parts = [
            f"{summary.get('num_events', 0)} events",
            f"{summary.get('num_channels', 0)} channels",
        ]
        if getattr(active_run, "created_at", None):
            meta_parts.append(active_run.created_at.replace("T", " ").replace("Z", " UTC"))
        meta_text = " • ".join(meta_parts)
        self.window.active_run_meta_label.setText(meta_text)
        self.window.active_run_meta_label.setToolTip(meta_text)

        status_parts = ["Active"]
        if accepted_run is not None and getattr(accepted_run, "run_id", None) == active_run.run_id:
            status_parts.append("Accepted")
        if hasattr(self.backend, "analysis_session") and self.backend.analysis_session.is_run_visible(active_run.run_id):
            status_parts.append("Visible")
        status_text = " • ".join(status_parts)
        self.window.active_run_status_label.setText(status_text)
        self.window.active_run_status_label.setToolTip(status_text)
        self.window.active_detector_label.setText(str(active_run.detector_name))
        self.window.active_detector_label.setToolTip(f"Active detector: {active_run.detector_name}")
        classifier_param = getattr(active_run, "param_classifier", None)
        if classifier_param is not None:
            self.window.active_classifier_label.setText(str(classifier_param.model_type))
            self.window.active_classifier_label.setToolTip(f"Classifier model: {classifier_param.model_type}")
        else:
            self.window.active_classifier_label.setText("Not run")
            self.window.active_classifier_label.setToolTip("Classifier has not been run for this analysis.")

        if hasattr(self.window, "active_run_param_table"):
            rows = self._build_active_run_param_rows(active_run)
            self._populate_two_column_table(self.window.active_run_param_table, rows)
        if hasattr(self.window, "active_classifier_table"):
            self._populate_two_column_table(self.window.active_classifier_table, self._build_classifier_rows(classifier_param))

    def _collect_case_run_summaries(self, biomarker_type=None):
        summaries = []
        target_backends = self.case_backends.items()
        if biomarker_type is not None:
            target_backends = [(biomarker_type, self.case_backends.get(biomarker_type))]
        for biomarker_type, backend in target_backends:
            if backend is None or not hasattr(backend, "get_run_summaries"):
                continue
            for run in backend.get_run_summaries():
                enriched = dict(run)
                enriched["biomarker_type"] = biomarker_type
                summaries.append(enriched)
        summaries.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return summaries

    def _collect_case_runs(self, biomarker_type=None):
        runs = []
        target_backends = self.case_backends.items()
        if biomarker_type is not None:
            target_backends = [(biomarker_type, self.case_backends.get(biomarker_type))]
        for _current_biomarker, backend in target_backends:
            session = getattr(backend, "analysis_session", None)
            if session is None:
                continue
            if hasattr(session, "list_runs"):
                session_runs = session.list_runs()
            else:
                session_runs = list(getattr(session, "runs", {}).values())
            for run in session_runs:
                if run is not None:
                    runs.append(run)
        runs.sort(key=lambda run: getattr(run, "created_at", ""), reverse=True)
        return runs

    def _get_case_visible_runs_for_waveform(self):
        active_run_id = getattr(getattr(self.backend, "analysis_session", None), "active_run_id", None)
        visible_runs = []
        for biomarker_type, backend in self.case_backends.items():
            session = getattr(backend, "analysis_session", None)
            if session is None or not hasattr(session, "get_visible_runs"):
                continue
            for run in session.get_visible_runs():
                if run is None or getattr(run, "event_features", None) is None:
                    continue
                visible_runs.append(
                    {
                        "run": run,
                        "is_active": biomarker_type == self.biomarker_type and run.run_id == active_run_id,
                        "is_current_biomarker": biomarker_type == self.biomarker_type,
                        "created_at": getattr(run, "created_at", ""),
                    }
                )
        visible_runs.sort(
            key=lambda item: (
                0 if item["is_active"] else 1,
                0 if item["is_current_biomarker"] else 1,
                item["created_at"],
            )
        )
        return [item["run"] for item in visible_runs]

    def _has_case_visible_runs(self):
        return bool(self._get_case_visible_runs_for_waveform())

    def _compare_case_runs(self, biomarker_type=None):
        return build_run_comparison(self._collect_case_runs(biomarker_type))

    def _get_selected_export_run(self):
        if self.backend is None or not hasattr(self.backend, "analysis_session"):
            return None
        session = self.backend.analysis_session
        return session.get_accepted_run() or session.get_active_run()

    def _export_case_clinical_summary(self, output_path):
        exported_run = self._get_selected_export_run()
        if exported_run is None:
            raise ValueError("No accepted or active run is available to export.")
        session = getattr(self.backend, "analysis_session", None)
        active_run = session.get_active_run() if session is not None else None
        accepted_run = session.get_accepted_run() if session is not None else None
        ranking_rows = self.backend.get_channel_ranking(exported_run.run_id) if hasattr(self.backend, "get_channel_ranking") else []
        return export_clinical_summary_workbook(
            output_path,
            exported_run=exported_run,
            run_summaries=self._collect_case_run_summaries(),
            ranking_rows=ranking_rows,
            comparison_rows=self._compare_case_runs().get("pairwise_overlap", []),
            active_run=active_run,
            accepted_run=accepted_run,
            decision_overrides={"comparison_scope": "case"},
        )

    def populate_decision_tables(self, runs, ranking, comparison_rows):
        if hasattr(self.window, "run_table"):
            self.window.run_table.setRowCount(len(runs))
            self.window.run_table.setProperty("run_refs", [(run.get("biomarker_type"), run.get("run_id")) for run in runs])
            active_row = -1
            for row, run in enumerate(runs[:]):
                status = []
                backend = self.case_backends.get(run.get("biomarker_type"))
                active_run_id = getattr(getattr(backend, "analysis_session", None), "active_run_id", None)
                is_active = run.get("run_id") == active_run_id and run.get("biomarker_type") == self.biomarker_type
                is_accepted = bool(run.get("accepted"))
                if is_active:
                    active_row = row
                if run.get("run_id") == active_run_id:
                    status.append("Active")
                if run.get("accepted"):
                    status.append("Accepted")
                values = [
                    "",
                    run.get("biomarker_type", self.biomarker_type or ""),
                    run.get("detector_name", ""),
                    str(run.get("num_events", "")),
                    ", ".join(status) or "Stored",
                ]
                for col, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    if col == 0:
                        item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                        item.setCheckState(Qt.Checked if run.get("visible") else Qt.Unchecked)
                    else:
                        item.setTextAlignment(Qt.AlignCenter)
                    item.setToolTip(run.get("display_name", run.get("detector_name", "")))
                    if is_accepted:
                        item.setBackground(QColor("#e5efe2"))
                    elif is_active:
                        item.setBackground(QColor("#e2ebf2"))
                    self.window.run_table.setItem(row, col, item)
            if active_row >= 0:
                self.window.run_table.selectRow(active_row)
        if hasattr(self.window, "channel_table"):
            display_rows = ranking[:8] if ranking else []
            self.window.channel_table.setRowCount(len(display_rows))
            self.window.channel_table.setProperty("channel_names", [channel.get("channel_name") for channel in display_rows])
            for row, channel in enumerate(display_rows):
                accepted_count = int(channel.get("accepted_predicted", 0))
                total_count = int(channel.get("total_events", 0))
                if accepted_count >= 10:
                    priority = "High"
                elif accepted_count >= 3:
                    priority = "Review"
                elif total_count > 0:
                    priority = "Monitor"
                else:
                    priority = "-"
                values = [
                    channel.get("channel_name", ""),
                    str(accepted_count),
                    str(total_count),
                    priority,
                ]
                for col, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setTextAlignment(Qt.AlignCenter)
                    if row == 0:
                        item.setBackground(QColor("#edf5e8"))
                    if col == 3 and priority == "High":
                        item.setBackground(QColor("#f6e3df"))
                    self.window.channel_table.setItem(row, col, item)
        if hasattr(self.window, "comparison_table"):
            display_rows = comparison_rows[:6] if comparison_rows else []
            self.window.comparison_table.setRowCount(len(display_rows))
            for row, item_row in enumerate(display_rows):
                only_text = f"{item_row.get('left_only', 0)}/{item_row.get('right_only', 0)}"
                values = [
                    item_row.get("left_label", item_row.get("left_detector", "")),
                    item_row.get("right_label", item_row.get("right_detector", "")),
                    str(item_row.get("overlap_events", 0)),
                    only_text,
                    f"{item_row.get('jaccard', 0):.2f}",
                ]
                for col, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setTextAlignment(Qt.AlignCenter)
                    if item_row.get("jaccard", 0) >= 0.5:
                        item.setBackground(QColor("#e8f1e6"))
                    self.window.comparison_table.setItem(row, col, item)

    def activate_run_from_table(self, row, _column):
        run_refs = self.window.run_table.property("run_refs") or []
        if row < 0 or row >= len(run_refs):
            return
        biomarker_type, run_id = run_refs[row]
        backend = self.case_backends.get(biomarker_type)
        if backend is None or not hasattr(backend, "activate_run"):
            return
        if biomarker_type != self.biomarker_type:
            self.start_run_mode(biomarker_type)
            QTimer.singleShot(0, lambda: self._activate_case_run(biomarker_type, run_id))
            return
        self._activate_case_run(biomarker_type, run_id)

    def _activate_case_run(self, biomarker_type, run_id):
        backend = self.case_backends.get(biomarker_type)
        if backend is None or not hasattr(backend, "activate_run"):
            return
        self.backend = backend
        backend.activate_run(run_id)
        self.message_handler(f"Activated {biomarker_type} run from registry")
        self._set_workflow_message("Switched active run from registry")
        self.refresh_run_dependent_views()

    def activate_run_from_selector(self, index):
        selector = getattr(self.window, "active_run_selector", None)
        if selector is None or index < 0:
            return
        run_ref = selector.itemData(index)
        if not run_ref:
            return
        biomarker_type, run_id = run_ref
        if biomarker_type != self.biomarker_type:
            self.start_run_mode(biomarker_type)
            QTimer.singleShot(0, lambda bt=biomarker_type, rid=run_id: self._activate_case_run(bt, rid))
            return
        self._activate_case_run(biomarker_type, run_id)

    def open_prepare_tab(self):
        if hasattr(self.window, "tabWidget"):
            self.window.tabWidget.setCurrentIndex(0)
        self._set_workflow_message("Filter and signal preparation")

    def _selected_detector_name(self):
        combo = getattr(self.window, "detector_mode_combo", None)
        if combo is not None and combo.count() > 0:
            return combo.currentText().strip().upper()
        if self.backend is not None and getattr(self.backend, "param_detector", None) is not None:
            return str(self.backend.param_detector.detector_type).strip().upper()
        return ""

    def update_setup_action_state(self):
        has_recording = bool(self.backend and getattr(self.backend, "eeg_data", None) is not None)
        detector_name = self._selected_detector_name() or self.get_biomarker_display_name().upper()
        supports_detection = self.biomarker_supports_detection()
        supports_classification = self.biomarker_supports_classification()
        has_detection_result = bool(self.backend and getattr(self.backend, "detected", False))
        yasa_ready = has_yasa()

        detector_apply = getattr(self.window, "detector_apply_button", None)
        detector_run = getattr(self.window, "detector_run_button", None)
        if detector_apply is not None:
            detector_apply.setEnabled(has_recording and supports_detection)
            detector_apply.setText("Apply")
            detector_apply.setToolTip(f"Apply the current {detector_name} detector parameters")
        if detector_run is not None:
            if self.biomarker_type == "Spike":
                detector_run.setText("Review Only")
                detector_run.setEnabled(False)
                detector_run.setToolTip("Spike mode currently supports review and import, not automated detection.")
            else:
                detector_run.setText(f"Run {detector_name}")
                enabled = has_recording and supports_detection and (self.biomarker_type != "Spindle" or yasa_ready)
                detector_run.setEnabled(enabled)
                if self.biomarker_type == "Spindle" and not yasa_ready:
                    detector_run.setToolTip("Install the optional 'yasa' package to run spindle detection.")
                else:
                    detector_run.setToolTip(f"Apply the visible settings and run a new {self.get_biomarker_display_name()} analysis")

        classifier_apply = getattr(self.window, "classifier_apply_button", None)
        classifier_run = getattr(self.window, "classifier_run_button", None)
        classifier_mode = getattr(self.window, "classifier_mode_combo", None)
        mode_text = classifier_mode.currentText() if classifier_mode is not None and classifier_mode.count() > 0 else ""
        custom_mode = mode_text == "Custom"
        if classifier_apply is not None:
            classifier_apply.setVisible(custom_mode and supports_classification)
            classifier_apply.setEnabled(has_recording and custom_mode and supports_classification)
        if classifier_run is not None:
            if self.biomarker_type == "Spike":
                classifier_run.setText("Review Only")
                classifier_run.setEnabled(False)
                classifier_run.setToolTip("Spike mode currently focuses on review rather than automated classification.")
            else:
                classifier_run.setText("Classify")
                classifier_run.setEnabled(has_detection_result and supports_classification)
                classifier_run.setToolTip("Classify the events in the active run after reviewing detection output.")

    def apply_selected_detector_parameters(self, _checked=False, show_feedback=True):
        if not self.biomarker_supports_detection():
            self.handle_unsupported_biomarker_mode(
                f"{self.get_biomarker_display_name()} mode currently focuses on review rather than automated detection."
            )
            return False
        if self.backend is None or getattr(self.backend, "eeg_data", None) is None:
            QMessageBox.information(self.window, "No Recording", "Load a recording before preparing detector settings.")
            return False

        detector_name = self._selected_detector_name()
        handlers = {
            ("HFO", "STE"): self.save_ste_params,
            ("HFO", "MNI"): self.save_mni_params,
            ("HFO", "HIL"): self.save_hil_params,
            ("Spindle", "YASA"): self.save_yasa_params,
        }
        handler = handlers.get((self.biomarker_type, detector_name))
        if handler is None:
            self.handle_unsupported_biomarker_mode(
                f"No configurable detector is available for {self.get_biomarker_display_name()} yet."
            )
            return False

        before_type = ""
        if getattr(self.backend, "param_detector", None) is not None:
            before_type = str(self.backend.param_detector.detector_type).upper()
        handler_succeeded = handler()
        after_type = ""
        if getattr(self.backend, "param_detector", None) is not None:
            after_type = str(self.backend.param_detector.detector_type).upper()
        success = bool(handler_succeeded) and after_type == detector_name and bool(after_type)
        if success and show_feedback:
            self.message_handler(f"{detector_name} parameters applied")
            self._set_workflow_message(f"{detector_name} settings ready")
        elif not success and show_feedback:
            if before_type == after_type and not after_type:
                self._set_workflow_message("Detector settings need attention")
            else:
                self._set_workflow_message(f"{detector_name} settings need attention")
        self.update_setup_action_state()
        return success

    def run_selected_detector_workflow(self):
        if not self.apply_selected_detector_parameters(show_feedback=False):
            return
        detector_name = self._selected_detector_name()
        self.message_handler(f"Starting {detector_name} detection run")
        if self.biomarker_type == "HFO":
            self.detect_HFOs()
        elif self.biomarker_type == "Spindle":
            self.detect_Spindles()
        else:
            self.handle_unsupported_biomarker_mode(
                f"{self.get_biomarker_display_name()} mode currently focuses on review rather than automated detection."
            )

    def _sync_classifier_execution_inputs(self):
        if hasattr(self.window, "overview_use_spike_checkbox") and hasattr(self.window, "use_spike_checkbox"):
            self.window.overview_use_spike_checkbox.setChecked(self.window.use_spike_checkbox.isChecked())
        if hasattr(self.window, "overview_use_ehfo_checkbox") and hasattr(self.window, "use_ehfo_checkbox"):
            self.window.overview_use_ehfo_checkbox.setChecked(self.window.use_ehfo_checkbox.isChecked())

    def apply_classifier_setup(self):
        if not self.biomarker_supports_classification():
            self.handle_unsupported_biomarker_mode(
                f"{self.get_biomarker_display_name()} mode currently focuses on review rather than automated classification."
            )
            return False
        if self.backend is None or getattr(self.backend, "eeg_data", None) is None:
            QMessageBox.information(self.window, "No Recording", "Load a recording before preparing classifier settings.")
            return False
        combo = getattr(self.window, "classifier_mode_combo", None)
        mode = combo.currentText() if combo is not None and combo.count() > 0 else "Hugging Face CPU"
        try:
            if mode == "Custom":
                self.set_custom_classifier_param()
            else:
                self._sync_classifier_execution_inputs()
                classifier_param = self.backend.get_classifier_param() if hasattr(self.backend, "get_classifier_param") else None
                if classifier_param is None:
                    if mode == "Hugging Face GPU":
                        self.set_classifier_param_gpu_default()
                    else:
                        self.set_classifier_param_cpu_default()
                    classifier_param = self.backend.get_classifier_param() if hasattr(self.backend, "get_classifier_param") else None
                if classifier_param is not None:
                    classifier_param.use_spike = self.window.use_spike_checkbox.isChecked()
                    classifier_param.use_ehfo = self.window.use_ehfo_checkbox.isChecked()
                    self.set_classifier_param_display()
            self.message_handler(f"{mode} classifier settings applied")
            self._set_workflow_message(f"{mode} classifier ready")
            self.update_setup_action_state()
            return True
        except Exception as exc:
            QMessageBox.critical(self.window, "Classifier Setup Failed", str(exc))
            return False

    def run_classifier_workflow(self):
        if not self.biomarker_supports_classification():
            self.handle_unsupported_biomarker_mode(
                f"{self.get_biomarker_display_name()} mode currently focuses on review rather than automated classification."
            )
            return
        if self.backend is None or not getattr(self.backend, "detected", False):
            QMessageBox.information(self.window, "Run Detection First", "Create a detection run first, then classify the active run.")
            return
        if not self.apply_classifier_setup():
            return
        self.classify()

    def select_detector_mode(self, index):
        combo = getattr(self.window, "detector_mode_combo", None)
        tabs = getattr(self.window, "detector_subtabs", None)
        if combo is None or tabs is None or index < 0:
            return
        if self.biomarker_type == "Spike":
            self._set_workflow_message("Spike review mode is active")
            self.update_setup_action_state()
            return
        blocker = QSignalBlocker(tabs)
        tabs.setCurrentIndex(min(index, max(0, tabs.count() - 1)))
        del blocker
        self._set_workflow_message(f"Detector set to {combo.currentText()}")
        self.update_setup_action_state()

    def sync_detector_mode_combo(self, index):
        combo = getattr(self.window, "detector_mode_combo", None)
        tabs = getattr(self.window, "detector_subtabs", None)
        if combo is None or tabs is None or index < 0:
            return
        blocker = QSignalBlocker(combo)
        combo.setCurrentIndex(min(index, max(0, combo.count() - 1)))
        del blocker

    def _set_classifier_custom_sources_visible(self, visible):
        for attr in ("groupBox", "groupBox_2"):
            widget = getattr(self.window, attr, None)
            if widget is not None:
                widget.setVisible(bool(visible))
        custom_button = getattr(self.window, "classifier_save_button", None)
        if custom_button is not None:
            custom_button.setVisible(False)
        custom_sources_frame = getattr(self.window, "classifier_custom_sources_frame", None)
        if custom_sources_frame is not None:
            custom_sources_frame.setVisible(bool(visible))
        custom_apply_frame = getattr(self.window, "classifier_custom_apply_frame", None)
        if custom_apply_frame is not None:
            custom_apply_frame.setVisible(bool(visible))
        header_button = getattr(self.window, "classifier_apply_button", None)
        if header_button is not None:
            header_button.setVisible(bool(visible))
        for attr in ("classifier_device_input", "classifier_batch_size_input"):
            widget = getattr(self.window, attr, None)
            if widget is not None:
                widget.setReadOnly(not visible)

    def _set_classifier_option_visibility(self):
        spike_toggle = getattr(self.window, "use_spike_checkbox", None)
        ehfo_toggle = getattr(self.window, "use_ehfo_checkbox", None)
        overview_spike_toggle = getattr(self.window, "overview_use_spike_checkbox", None)
        overview_ehfo_toggle = getattr(self.window, "overview_use_ehfo_checkbox", None)
        ignore_frame = getattr(self.window, "classifier_ignore_frame", None)
        device_label = getattr(self.window, "label_70", None)
        device_input = getattr(self.window, "classifier_device_input", None)
        batch_label = getattr(self.window, "label_27", None)
        batch_input = getattr(self.window, "classifier_batch_size_input", None)

        show_classifier_controls = self.biomarker_type in {"HFO", "Spindle"}
        show_ehfo = self.biomarker_type == "HFO"

        spike_text = "Use spk-HFO" if self.biomarker_type == "HFO" else "Use spike-associated"
        for toggle in (spike_toggle, overview_spike_toggle):
            if toggle is not None:
                toggle.setVisible(show_classifier_controls)
                toggle.setText(spike_text)
        for toggle in (ehfo_toggle, overview_ehfo_toggle):
            if toggle is not None:
                toggle.setVisible(show_ehfo)
                if not show_ehfo:
                    toggle.setChecked(False)
        if ignore_frame is not None:
            ignore_frame.setVisible(show_classifier_controls)
        for widget in (device_label, device_input, batch_label, batch_input):
            if widget is not None:
                widget.setVisible(show_classifier_controls)

    def _get_classifier_source_display_value(self, classifier_param, path_attr, card_attr):
        if classifier_param is None:
            return ""
        source_kind, source_value = classifier_param.preferred_model_source(
            getattr(classifier_param, path_attr, ""),
            getattr(classifier_param, card_attr, ""),
        )
        if source_value:
            return source_value
        fallback_card = getattr(classifier_param, card_attr, "")
        fallback_path = getattr(classifier_param, path_attr, "")
        return fallback_card or fallback_path or ""

    def _infer_classifier_source_preference(
        self,
        *,
        artifact_path="",
        spike_path="",
        ehfo_path="",
        artifact_card="",
        spike_card="",
        ehfo_card="",
    ):
        if any(str(value).strip() for value in (artifact_card, spike_card, ehfo_card)):
            return "huggingface"
        if any(str(value).strip() for value in (artifact_path, spike_path, ehfo_path)):
            return "local"
        return "auto"

    def apply_classifier_mode(self, index):
        combo = getattr(self.window, "classifier_mode_combo", None)
        if combo is None or index < 0:
            return
        mode = combo.currentText()
        if self.biomarker_type == "Spike":
            self._set_classifier_custom_sources_visible(False)
            self.window.detect_all_button.setEnabled(False)
            return
        if mode == "Hugging Face CPU":
            self.set_classifier_param_cpu_default()
            self._set_classifier_custom_sources_visible(False)
        elif mode == "Hugging Face GPU":
            self.set_classifier_param_gpu_default()
            self._set_classifier_custom_sources_visible(False)
        else:
            self._set_classifier_custom_sources_visible(True)
        self._set_workflow_message(f"Classifier preset: {mode}")
        self.update_setup_action_state()

    def refresh_classifier_mode_ui(self):
        combo = getattr(self.window, "classifier_mode_combo", None)
        if combo is None or combo.count() == 0:
            return
        self._set_classifier_option_visibility()
        if self.biomarker_type == "Spike":
            blocker = QSignalBlocker(combo)
            combo.setCurrentIndex(0)
            del blocker
            self._set_classifier_custom_sources_visible(False)
            return

        classifier_param = self.backend.get_classifier_param() if self.backend is not None and hasattr(self.backend, "get_classifier_param") else None
        target_label = "Hugging Face CPU"
        if classifier_param is not None:
            model_type = str(getattr(classifier_param, "model_type", "") or "").lower()
            has_custom_sources = any(
                bool(getattr(classifier_param, attr, ""))
                for attr in ("artifact_path", "spike_path", "ehfo_path", "artifact_card", "spike_card", "ehfo_card")
            )
            if has_custom_sources and model_type not in {"default_cpu", "default_gpu"}:
                target_label = "Custom"
            elif model_type == "default_gpu":
                target_label = "Hugging Face GPU"
            elif has_custom_sources and model_type == "":
                target_label = "Custom"
        blocker = QSignalBlocker(combo)
        index = combo.findText(target_label)
        combo.setCurrentIndex(index if index >= 0 else 0)
        del blocker
        self._set_classifier_custom_sources_visible(target_label == "Custom")
        self.update_setup_action_state()

    def refresh_detector_mode_ui(self):
        combo = getattr(self.window, "detector_mode_combo", None)
        if combo is None or combo.count() == 0:
            return
        if self.biomarker_type == "Spike":
            blocker = QSignalBlocker(combo)
            combo.setCurrentIndex(0)
            del blocker
            return
        detector_name = ""
        if self.backend is not None and getattr(self.backend, "param_detector", None) is not None:
            detector_name = str(self.backend.param_detector.detector_type).upper()
        blocker = QSignalBlocker(combo)
        index = combo.findText(detector_name) if detector_name else 0
        combo.setCurrentIndex(index if index >= 0 else 0)
        del blocker
        self.update_setup_action_state()

    def _selected_run_table_row(self):
        table = getattr(self.window, "run_table", None)
        if table is None:
            return -1
        selection_model = table.selectionModel()
        if selection_model is None:
            return -1
        selected = selection_model.selectedRows()
        if selected:
            return selected[0].row()
        return 0 if table.rowCount() > 0 else -1

    def activate_selected_run_from_popup(self):
        row = self._selected_run_table_row()
        if row < 0:
            QMessageBox.information(self.window, "No Runs", "Create a run first, then choose it in the statistics window.")
            return
        self.activate_run_from_table(row, 0)

    def accept_selected_run_from_popup(self):
        row = self._selected_run_table_row()
        if row < 0:
            QMessageBox.information(self.window, "No Runs", "Create a run first, then choose it in the statistics window.")
            return
        self.activate_run_from_table(row, 0)
        QTimer.singleShot(0, self.accept_active_run)

    def handle_run_table_click(self, row, column):
        if column != 0:
            return
        run_refs = self.window.run_table.property("run_refs") or []
        if row < 0 or row >= len(run_refs):
            return
        biomarker_type, run_id = run_refs[row]
        backend = self.case_backends.get(biomarker_type)
        if backend is None or not hasattr(backend, "set_run_visible"):
            return
        item = self.window.run_table.item(row, 0)
        if item is None:
            return
        visible = item.checkState() == Qt.Checked
        backend.set_run_visible(run_id, visible)
        self.message_handler(f"{'Showing' if visible else 'Hiding'} overlay for run {row + 1}")
        self.refresh_run_dependent_views()

    def highlight_channel_from_table(self, row, _column):
        channel_names = self.window.channel_table.property("channel_names") or []
        if row < 0 or row >= len(channel_names):
            return
        self._highlight_channel_mode = True
        if not self.highlight_waveform_channel(channel_names[row]):
            self._highlight_channel_mode = False
            self.update_waveform_toolbar_state()

    def focus_waveform_channel(self, channel_name):
        if not channel_name or not hasattr(self.window, "waveform_plot"):
            return
        self.set_channels_to_plot([channel_name], display_all=True)
        self.message_handler(f"Focused waveform review on channel: {channel_name}")
        self._set_workflow_message(f"Focused on channel {channel_name}")

    def highlight_waveform_channel(self, channel_name):
        channel_name = str(channel_name or "")
        if not channel_name or not hasattr(self.window, "waveform_plot"):
            return False

        current_channels = self._current_waveform_channel_names()
        target_channels = list(current_channels)
        restored_source_scope = False
        if channel_name not in target_channels:
            recording_channels = self._recording_waveform_channel_names()
            if channel_name not in recording_channels:
                self.message_handler(f"Channel {channel_name} is unavailable in the current recording")
                return False
            target_channels = list(recording_channels)
            restored_source_scope = target_channels != current_channels

        self._set_highlighted_waveform_channel(channel_name)

        if restored_source_scope:
            self.set_channels_to_plot(target_channels, display_all=False)

        visible_count = max(1, int(self.window.n_channel_input.value())) if hasattr(self.window, "n_channel_input") else 1
        target_index = target_channels.index(channel_name)
        max_first_channel = max(0, len(target_channels) - visible_count)
        first_channel_to_plot = max(0, min(target_index - (visible_count // 2), max_first_channel))

        self.window.waveform_plot.first_channel_to_plot = first_channel_to_plot
        self.waveform_plot_button_clicked()
        self._set_highlighted_waveform_channel(channel_name)

        if hasattr(self.window, "channel_scroll_bar"):
            blocker = QSignalBlocker(self.window.channel_scroll_bar)
            self.window.channel_scroll_bar.setValue(first_channel_to_plot)
            del blocker

        if restored_source_scope:
            self.message_handler(f"Highlighted channel {channel_name} in the source channel list")
        else:
            self.message_handler(f"Highlighted channel {channel_name}")
        self._set_workflow_message(f"Highlighted {channel_name}")
        return True

    def update_decision_overview(self):
        if not hasattr(self.window, "decision_runs_value"):
            return
        if self.backend is None or not hasattr(self.backend, "get_decision_summary"):
            self.window.decision_runs_value.setText("--")
            self.window.decision_active_value.setText("--")
            self.window.decision_accepted_value.setText("--")
            self.window.decision_channel_value.setText("--")
            return

        summary = self.backend.get_decision_summary()
        self.window.decision_runs_value.setText(str(summary.get("num_runs", 0)))
        self.window.decision_active_value.setText(summary.get("active_detector") or "--")
        self.window.decision_accepted_value.setText(summary.get("accepted_detector") or "--")
        top_channel = summary.get("top_channel")
        if top_channel:
            self.window.decision_channel_value.setText(f"{top_channel['channel_name']} ({top_channel['total_events']})")
        else:
            self.window.decision_channel_value.setText("--")

    def _run_choice_items(self):
        if self.backend is None or not hasattr(self.backend, "get_run_summaries"):
            return [], {}
        active_run_id = getattr(self.backend.analysis_session, "active_run_id", None)
        accepted_run_id = getattr(self.backend.analysis_session, "accepted_run_id", None)
        labels = []
        label_to_id = {}
        for run in self.backend.get_run_summaries():
            flags = []
            if run["run_id"] == active_run_id:
                flags.append("active")
            if run["run_id"] == accepted_run_id:
                flags.append("accepted")
            flag_text = f" [{' / '.join(flags)}]" if flags else ""
            label = f"{run['detector_name']} | {run['num_events']} events | {run['num_channels']} channels{flag_text}"
            labels.append(label)
            label_to_id[label] = run["run_id"]
        return labels, label_to_id

    def _sync_waveform_plot_with_backend(self):
        plot = getattr(self.window, "waveform_plot", None)
        if plot is None or self.backend is None or getattr(self.backend, "eeg_data", None) is None:
            return

        available_channels = [str(channel) for channel in np.array(getattr(self.backend, "channel_names", [])).tolist()]
        if not available_channels:
            return

        current_channels = []
        try:
            current_channels = [str(channel) for channel in (plot.get_channels_to_plot() or [])]
        except Exception:
            current_channels = []

        current_start = float(getattr(plot, "t_start", 0.0) or 0.0)
        current_first_channel = int(getattr(plot, "first_channel_to_plot", 0) or 0)

        if getattr(plot, "backend", None) is not self.backend:
            plot.update_backend(self.backend, False)
            plot.init_eeg_data()

        valid_channels = [channel for channel in current_channels if channel in available_channels]
        if not valid_channels:
            valid_channels = available_channels
        plot.set_channels_to_plot(valid_channels)

        plot.t_start = max(0.0, min(current_start, float(plot.get_total_time())))
        plot.first_channel_to_plot = max(0, min(current_first_channel, max(0, len(valid_channels) - 1)))

        if hasattr(self.window, "n_channel_input"):
            self.window.n_channel_input.setMaximum(len(valid_channels))
            self.window.n_channel_input.setValue(
                max(1, min(self.window.n_channel_input.value(), len(valid_channels)))
            )
        if hasattr(self.window, "channel_scroll_bar"):
            self.window.channel_scroll_bar.setMaximum(
                max(0, len(valid_channels) - max(1, int(self.window.n_channel_input.value())))
            )

    def refresh_run_dependent_views(self):
        self.set_detector_param_display()
        self.update_statistics_label()
        self.update_status_indicators()
        self.update_decision_overview()
        if hasattr(self.window, "waveform_plot"):
            self._sync_waveform_plot_with_backend()
            self.window.waveform_plot.set_plot_biomarkers(self._has_case_visible_runs())
            self.window.waveform_plot.plot(update_biomarker=True)
        self.update_waveform_toolbar_state()

    def _focus_waveform_on_event(self, channel_name, start_index, end_index):
        if self.backend is None or not hasattr(self.window, "waveform_plot"):
            return

        sample_freq = float(getattr(self.backend, "sample_freq", 0) or 0)
        if sample_freq <= 0:
            sample_freq = 1.0

        center_time = ((float(start_index) + float(end_index)) / 2.0) / sample_freq
        time_window = max(0.1, float(self.window.display_time_window_input.value()))
        total_time = float(self.window.waveform_plot.get_total_time())
        start_time = max(0.0, center_time - time_window / 2.0)
        if total_time > time_window:
            start_time = min(start_time, max(0.0, total_time - time_window))

        channels_to_plot = list(self.window.waveform_plot.get_channels_to_plot())
        if channel_name not in channels_to_plot:
            self.set_channels_to_plot([channel_name], display_all=True)
            channels_to_plot = [channel_name]

        n_visible = max(1, int(self.window.n_channel_input.value()))
        try:
            channel_index = channels_to_plot.index(channel_name)
        except ValueError:
            channel_index = 0
        max_first = max(0, len(channels_to_plot) - n_visible)
        first_channel_to_plot = max(0, min(channel_index - (n_visible // 2), max_first))

        self.window.waveform_plot.t_start = start_time
        self.window.waveform_plot.first_channel_to_plot = first_channel_to_plot
        self.waveform_plot_button_clicked()

    def _navigate_event(self, direction):
        features = self._get_active_event_features()
        total_events = self._get_feature_event_count(features)
        if total_events == 0:
            self.message_handler("No detected events available for navigation")
            self.update_waveform_toolbar_state()
            return

        if direction == "next":
            channel_name, start_index, end_index = features.get_next()
        elif direction == "prev":
            channel_name, start_index, end_index = features.get_prev()
        else:
            channel_name, start_index, end_index = features.get_current()

        self._focus_waveform_on_event(channel_name, start_index, end_index)
        self.update_waveform_toolbar_state()
        self.message_handler(
            f"Centered on {self.get_biomarker_display_name()} event {self._get_feature_current_position(features) + 1} of {total_events}"
        )

    def show_previous_event(self):
        self._navigate_event("prev")

    def center_current_event(self):
        self._navigate_event("current")

    def show_next_event(self):
        self._navigate_event("next")

    def apply_waveform_window_preset(self, seconds):
        if not hasattr(self.window, "display_time_window_input"):
            return
        self.window.display_time_window_input.setValue(float(seconds))
        self.waveform_plot_button_clicked()
        self._set_workflow_message(f"Waveform window set to {float(seconds):.2f} s")
        self.update_waveform_toolbar_state()

    def apply_visible_channel_preset(self, count):
        if self.backend is None or not hasattr(self.window, "n_channel_input"):
            return
        max_channels = max(0, int(self.window.n_channel_input.maximum()))
        if max_channels <= 0:
            return
        if count == "all":
            target = max_channels
        else:
            target = min(max_channels, max(1, int(count)))
        self.window.n_channel_input.setValue(target)
        self.waveform_plot_button_clicked()
        if target == max_channels:
            self._set_workflow_message("Showing the full current channel subset")
        else:
            self._set_workflow_message(f"Showing {target} channels at once")
        self.update_waveform_toolbar_state()

    def jump_to_pending_review(self, direction=1):
        features = self._get_active_event_features()
        if features is None:
            self.message_handler("No active review queue is available yet")
            return
        if not hasattr(features, "get_next_unannotated") or not hasattr(features, "get_prev_unannotated"):
            self.message_handler("Pending-review navigation is unavailable for the current biomarker mode")
            return

        event_tuple = features.get_next_unannotated() if int(direction) >= 0 else features.get_prev_unannotated()
        if event_tuple is None:
            self.message_handler("All visible events in the current review queue are already reviewed")
            self._set_workflow_message("Review queue complete")
            self.update_waveform_toolbar_state()
            return

        channel_name, start_index, end_index = event_tuple
        self._focus_waveform_on_event(channel_name, start_index, end_index)
        progress = features.get_review_progress() if hasattr(features, "get_review_progress") else {"remaining": 0}
        step_name = "next" if int(direction) >= 0 else "previous"
        self.message_handler(
            f"Jumped to the {step_name} pending {self.get_biomarker_display_name()} event ({progress.get('remaining', 0)} remaining)"
        )
        self._set_workflow_message("Opened a pending review target")
        self.update_waveform_toolbar_state()

    def quick_label_current_event(self, label):
        label = str(label or "").strip()
        if not label:
            return
        features = self._get_active_event_features()
        total_events = self._get_feature_event_count(features)
        if features is None or total_events == 0:
            self.message_handler("No review target is available to label yet")
            self._set_workflow_message("No review target selected")
            self.update_waveform_toolbar_state()
            return
        if not hasattr(features, "doctor_annotation"):
            self.message_handler("Quick labeling is unavailable for the current review mode")
            return

        current_event = None
        if hasattr(features, "get_current"):
            try:
                current_event = features.get_current()
            except Exception:
                current_event = None

        try:
            features.doctor_annotation(label)
            if self.backend is not None and hasattr(self.backend, "sync_active_run"):
                self.backend.sync_active_run()
        except Exception as exc:
            self.message_handler(f"Could not apply the '{label}' label: {exc}")
            return

        self.update_run_management_panel()
        self.refresh_run_dependent_views()

        current_position = min(self._get_feature_current_position(features) + 1, max(total_events, 1))
        channel_name = str(current_event[0]) if current_event and current_event[0] else ""
        label_message = f"Labeled {self.get_biomarker_display_name()} event {current_position} as {label}"
        if channel_name:
            label_message += f" on {channel_name}"
        self.message_handler(label_message)
        self._set_workflow_message(f"Quick label: {label}")

    def focus_hotspot_channels(self, limit=8):
        if self.backend is None:
            return
        channels = self._hotspot_focus_channel_names(limit=limit)
        if not channels:
            self.message_handler("No ranked channels are available for the active review run")
            return

        self.set_channels_to_plot(channels, display_all=False)
        self.window.channel_scroll_bar.setValue(0)
        if len(channels) == 1:
            self.message_handler(f"Highlighted top review channel: {channels[0]}")
            self._set_workflow_message(f"Highlighted channel {channels[0]}")
        else:
            self.message_handler(f"Highlighted {len(channels)} hotspot channels for clinical review")
            self._set_workflow_message("Highlighted hotspot channels")
        self.update_waveform_toolbar_state()

    def toggle_hotspot_channels(self, enabled, limit=8):
        if self.backend is None:
            return
        if enabled:
            self.focus_hotspot_channels(limit=limit)
        else:
            self.show_all_channels()

    def jump_to_prediction_scope(self, scope):
        scope = str(scope or "").strip()
        if not scope:
            return
        features = self._get_active_event_features()
        if features is None or not hasattr(features, "get_next_matching"):
            self.message_handler("Model triage is unavailable for the current review mode")
            return

        event_tuple = features.get_next_matching(scope, unannotated_only=True)
        suffix = "pending"
        if event_tuple is None:
            event_tuple = features.get_next_matching(scope, unannotated_only=False)
            suffix = "matching"
        if event_tuple is None:
            self.message_handler(f"No {scope} events are available in the current review queue")
            return

        channel_name, start_index, end_index = event_tuple
        self._focus_waveform_on_event(channel_name, start_index, end_index)
        self.message_handler(f"Jumped to the next {suffix} {scope} event")
        self._set_workflow_message(f"Triage scope: {scope}")
        self.update_waveform_toolbar_state()

    def toggle_waveform_cursor(self, enabled):
        if self.backend is None or not hasattr(self.window, "waveform_plot"):
            return
        if enabled and hasattr(self.window.waveform_plot, "is_measurement_enabled") and self.window.waveform_plot.is_measurement_enabled():
            self.toggle_waveform_measure(False)
        self.window.waveform_plot.set_cursor_enabled(bool(enabled))
        self._set_workflow_message("Waveform cursor on" if enabled else "Waveform cursor off")
        self.update_waveform_toolbar_state()

    def toggle_waveform_measure(self, enabled):
        if self.backend is None or not hasattr(self.window, "waveform_plot"):
            return
        if enabled and hasattr(self.window.waveform_plot, "is_cursor_enabled") and self.window.waveform_plot.is_cursor_enabled():
            self.toggle_waveform_cursor(False)
        self.window.waveform_plot.set_measurement_enabled(bool(enabled))
        if enabled:
            self._reset_waveform_measurement_state()
            self._set_workflow_message("Measure mode on")
        else:
            self._reset_waveform_measurement_state()
            self._set_workflow_message("Measure mode off")
        self.update_waveform_toolbar_state()

    def zoom_waveform(self, factor):
        if self.backend is None or not hasattr(self.window, "display_time_window_input"):
            return
        current = max(0.1, float(self.window.display_time_window_input.value()))
        total_time = max(0.1, float(self.window.waveform_plot.get_total_time()))
        new_value = min(total_time, max(0.1, current * float(factor)))
        self.window.display_time_window_input.setValue(new_value)
        self.waveform_plot_button_clicked()

    def go_to_time_position(self):
        if self.backend is None or not hasattr(self.window, "go_to_time_input"):
            return
        target = float(self.window.go_to_time_input.value())
        total_time = max(0.0, float(self.window.waveform_plot.get_total_time()))
        time_window = max(0.1, float(self.window.display_time_window_input.value()))
        if total_time > time_window:
            target = min(max(0.0, target), total_time - time_window)
        else:
            target = 0.0
        self.window.waveform_plot.t_start = target
        self.waveform_plot_button_clicked()
        self._set_workflow_message(f"Jumped to {target:.2f} s")

    def export_waveform_snapshot(self):
        if self.backend is None or not hasattr(self.window, "waveformWiget"):
            return
        default_name = "pybrain_waveform.png"
        if getattr(self.backend, "edf_param", None):
            edf_path = self.backend.edf_param.get("edf_fn", "")
            if edf_path:
                base_name = os.path.splitext(os.path.basename(edf_path))[0]
                default_name = f"{base_name}_waveform.png"
        fname, _ = QFileDialog.getSaveFileName(
            self.window,
            "Save Waveform Snapshot",
            os.path.join(os.path.expanduser("~"), default_name),
            "PNG Image (*.png)",
            options=QFileDialog.DontUseNativeDialog,
        )
        if not fname:
            return
        target = self.window.waveformWiget.grab()
        if target.save(fname):
            self.message_handler(f"Saved waveform snapshot: {os.path.basename(fname)}")
            self._set_workflow_message("Waveform snapshot saved")

    def toggle_event_channel_scope(self, enabled):
        if self.backend is None:
            return
        if enabled:
            self.focus_event_channels()
        else:
            self.show_all_channels()

    def focus_event_channels(self):
        features = self._get_active_event_features()
        ordered_channels = list(dict.fromkeys(self._get_feature_channels(features)))
        if features is None or not ordered_channels:
            self.message_handler("No detected events available to define a review channel set")
            return
        available_channels = list(
            dict.fromkeys(
                self._recording_waveform_channel_names() + self._current_waveform_channel_names()
            )
        )
        matched_channels = self._ordered_channel_subset(available_channels, ordered_channels)
        if not matched_channels:
            self.message_handler("The active run does not expose event channels that match the current recording")
            self.update_waveform_toolbar_state()
            return
        self.set_channels_to_plot(matched_channels, display_all=False)
        self.window.channel_scroll_bar.setValue(0)
        self.message_handler(f"Focused review on {len(matched_channels)} channels that contain detected events")
        self._set_workflow_message("Reviewing channels with detected events")

    def show_all_channels(self):
        if self.backend is None:
            return
        all_channels = self._recording_waveform_channel_names()
        if not all_channels:
            return
        self.set_channels_to_plot(all_channels, display_all=False)
        self.window.channel_scroll_bar.setValue(0)
        self.message_handler(f"Restored the referential source channel list ({len(all_channels)} channels)")
        self._set_workflow_message("Reviewing referential channels")

    def reset_waveform_view_state(self):
        if self.backend is None:
            return
        self._highlight_channel_mode = False
        self._set_highlighted_waveform_channel(None)
        self._clear_waveform_inspect_modes()
        self.show_all_channels()
        self.message_handler("Reset waveform view to the default referential workspace")
        self._set_workflow_message("Waveform view reset")
        self.update_waveform_toolbar_state()

    def _format_overlap_review_summary_message(self, summary):
        if not summary:
            return "Updated cross-channel overlap review for the active HFO run."
        action = summary.get("overlap_action", HFO_Feature.OVERLAP_ACTION_DISABLED)
        if action == HFO_Feature.OVERLAP_ACTION_DISABLED:
            return "Cleared cross-channel overlap review for the active HFO run."
        tagged = int(summary.get("overlap_tagged", 0))
        groups = int(summary.get("overlap_groups", 0))
        kept = int(summary.get("overlap_kept", 0))
        hidden = int(summary.get("overlap_hidden", 0))
        visible = int(summary.get("overlap_visible", 0))
        action_label = summary.get("overlap_action_label", "Tag later overlaps")
        if tagged == 0:
            return f"Applied keep-first overlap review; no later cross-channel overlaps matched the current rule."
        if action == HFO_Feature.OVERLAP_ACTION_HIDE:
            return (
                f"Applied {action_label.lower()}: kept {kept} first events across {groups} groups, "
                f"hid {hidden} later overlaps, and left {visible} visible."
            )
        return (
            f"Applied {action_label.lower()}: kept {kept} first events and tagged {tagged} later overlaps "
            f"across {groups} groups."
        )

    def open_overlap_review_dialog(self):
        if self.biomarker_type != "HFO":
            self.handle_unsupported_biomarker_mode("Cross-channel overlap review is only available for HFO runs.")
            return
        features = self._get_active_event_features()
        raw_event_count = self._get_feature_event_count(features, raw=True)
        if self.backend is None or features is None or raw_event_count == 0:
            QMessageBox.information(
                self.window,
                "No HFO Events",
                "Run HFO detection first, then review cross-channel overlaps for the active run.",
            )
            return

        current_settings = (
            self.backend.get_cross_channel_overlap_review_settings()
            if hasattr(self.backend, "get_cross_channel_overlap_review_settings")
            else HFO_Feature.default_overlap_review_settings()
        )

        dialog = QDialog(self.window)
        dialog.setWindowTitle("Cross-Channel Overlap")
        dialog.setModal(True)
        dialog.resize(460, 250)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        intro = QLabel(
            "Find HFO events whose time intervals overlap across different channels. In each overlap group, "
            "the earliest event is kept and the later overlapping events can be tagged or hidden."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(10)

        action_combo = QComboBox(dialog)
        action_combo.addItem("Disabled", HFO_Feature.OVERLAP_ACTION_DISABLED)
        action_combo.addItem("Tag later overlaps", HFO_Feature.OVERLAP_ACTION_TAG)
        action_combo.addItem("Keep first, hide later overlaps", HFO_Feature.OVERLAP_ACTION_HIDE)
        action_index = action_combo.findData(current_settings.get("action", HFO_Feature.OVERLAP_ACTION_DISABLED))
        action_combo.setCurrentIndex(action_index if action_index >= 0 else 0)
        form.addRow("Action", action_combo)

        min_overlap_spin = QDoubleSpinBox(dialog)
        min_overlap_spin.setDecimals(1)
        min_overlap_spin.setRange(0.0, 1000.0)
        min_overlap_spin.setSingleStep(5.0)
        min_overlap_spin.setSuffix(" ms")
        min_overlap_spin.setValue(float(current_settings.get("min_overlap_ms", 0.0) or 0.0))
        form.addRow("Min overlap", min_overlap_spin)

        min_channels_spin = QSpinBox(dialog)
        min_channels_spin.setRange(2, 512)
        min_channels_spin.setValue(int(current_settings.get("min_channels", 2) or 2))
        form.addRow("Min channels", min_channels_spin)

        tag_name_input = QLineEdit(dialog)
        tag_name_input.setPlaceholderText(HFO_Feature.DEFAULT_OVERLAP_TAG)
        tag_name_input.setText(str(current_settings.get("tag_name", HFO_Feature.DEFAULT_OVERLAP_TAG) or HFO_Feature.DEFAULT_OVERLAP_TAG))
        form.addRow("Tag name", tag_name_input)

        layout.addLayout(form)

        helper = QLabel(
            "Events qualify when their sample windows overlap and they belong to different channels. "
            "Connected overlap groups keep the earliest event and treat the later ones as duplicates."
        )
        helper.setWordWrap(True)
        helper.setStyleSheet("color: #667887;")
        layout.addWidget(helper)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel, dialog)
        reset_button = buttons.addButton("Reset", QDialogButtonBox.ResetRole)
        apply_button = buttons.addButton("Apply", QDialogButtonBox.AcceptRole)
        apply_button.setDefault(True)
        apply_button.setAutoDefault(True)
        reset_button.setDefault(False)
        reset_button.setAutoDefault(False)
        cancel_button = buttons.button(QDialogButtonBox.Cancel)
        if cancel_button is not None:
            cancel_button.setDefault(False)
            cancel_button.setAutoDefault(False)
        layout.addWidget(buttons)

        def sync_enabled_state():
            enabled = action_combo.currentData() != HFO_Feature.OVERLAP_ACTION_DISABLED
            min_overlap_spin.setEnabled(enabled)
            min_channels_spin.setEnabled(enabled)
            tag_name_input.setEnabled(enabled)

        def reset_form():
            defaults = HFO_Feature.default_overlap_review_settings()
            action_combo.setCurrentIndex(action_combo.findData(defaults["action"]))
            min_overlap_spin.setValue(float(defaults["min_overlap_ms"]))
            min_channels_spin.setValue(int(defaults["min_channels"]))
            tag_name_input.setText(defaults["tag_name"])
            sync_enabled_state()

        def apply_settings():
            settings = {
                "action": action_combo.currentData(),
                "min_overlap_ms": float(min_overlap_spin.value()),
                "min_channels": int(min_channels_spin.value()),
                "tag_name": tag_name_input.text().strip() or HFO_Feature.DEFAULT_OVERLAP_TAG,
            }
            summary = self.backend.apply_cross_channel_overlap_review(settings) if hasattr(self.backend, "apply_cross_channel_overlap_review") else None
            self.refresh_run_dependent_views()
            message = self._format_overlap_review_summary_message(summary)
            self.message_handler(message)
            self._set_workflow_message("Cross-channel overlap review updated")
            dialog.accept()

        action_combo.currentIndexChanged.connect(sync_enabled_state)
        reset_button.clicked.connect(reset_form)
        apply_button.clicked.connect(apply_settings)
        buttons.rejected.connect(dialog.reject)

        sync_enabled_state()
        apply_subwindow_theme(dialog)
        dialog.exec_()

    def choose_active_run(self):
        if self.backend is None or not hasattr(self.backend, "analysis_session"):
            return
        labels, label_to_id = self._run_choice_items()
        if not labels:
            QMessageBox.information(self.window, "No Runs", "Run a detector first to create saved detection runs.")
            return
        selected, ok = QInputDialog.getItem(self.window, "Switch Active Run", "Select a run to review:", labels, 0, False)
        if not ok or not selected:
            return
        run_id = label_to_id[selected]
        self.backend.activate_run(run_id)
        self.message_handler(f"Activated run: {selected}")
        self._set_workflow_message("Switched active run")
        self.refresh_run_dependent_views()

    def accept_active_run(self):
        if self.backend is None or not hasattr(self.backend, "accept_active_run"):
            return
        accepted = self.backend.accept_active_run()
        if accepted is None:
            QMessageBox.information(self.window, "No Active Run", "Run a detector first, then choose the active run to accept.")
            return
        self.message_handler(f"Accepted run for downstream export: {accepted.detector_name}")
        self._set_workflow_message("Accepted run updated")
        self.refresh_run_dependent_views()

    def show_run_comparison(self):
        if self.backend is None:
            return
        self.update_run_management_panel()
        runs = self._collect_case_run_summaries()
        if not runs:
            message = "Create a run first to open run statistics."
            self.message_handler(message)
            self._set_workflow_message(message)
            return
        if hasattr(self.window, "run_stats_dialog"):
            has_multiple_biomarkers = len({run.get("biomarker_type", "") for run in runs}) > 1
            title_prefix = "Case" if has_multiple_biomarkers else self.get_biomarker_display_name()
            self.window.run_stats_dialog.setWindowTitle(f"{title_prefix} Run Statistics")
            self.window.run_stats_dialog.show()
            if os.environ.get("QT_QPA_PLATFORM", "").lower() not in {"offscreen", "minimal"}:
                try:
                    self.window.run_stats_dialog.raise_()
                    self.window.run_stats_dialog.activateWindow()
                except Exception:
                    pass
        pairwise = self._compare_case_runs().get("pairwise_overlap", [])
        if pairwise:
            self.message_handler("Opened run statistics with run overlap")
            self._set_workflow_message("Run statistics updated")
            return
        self.message_handler("Opened run statistics")
        self._set_workflow_message("Run statistics")

    def save_to_excel(self):
        # Generate default filename based on loaded EDF file
        default_path = os.path.expanduser("~")  # Default to home directory
        if self.backend and hasattr(self.backend, 'edf_param') and self.backend.edf_param:
            edf_path = self.backend.edf_param.get("edf_fn", "")
            if edf_path and os.path.exists(edf_path):
                # Get directory and basename without extension
                directory = os.path.dirname(edf_path)
                base_name = os.path.splitext(os.path.basename(edf_path))[0]
                default_path = os.path.join(directory, f"{base_name}_clinical_summary.xlsx")
        
        # open file dialog with default path (use non-native dialog to avoid macOS freezing)
        fname, _ = QFileDialog.getSaveFileName(self.window, 'Save file', default_path, "Excel files (*.xlsx)", options=QFileDialog.DontUseNativeDialog)
        if fname:
            if hasattr(self.backend, "accept_active_run") and hasattr(self.backend, "analysis_session"):
                if self.backend.analysis_session.get_accepted_run() is None and self.backend.analysis_session.get_active_run() is not None:
                    accepted = self.backend.accept_active_run()
                    if accepted is not None:
                        self.message_handler(f"Marked active run as accepted for export: {accepted.detector_name}")
            if hasattr(self.backend, "analysis_session"):
                self._export_case_clinical_summary(fname)
                self.message_handler(f"Exported clinical summary workbook: {os.path.basename(fname)}")
            else:
                self.backend.export_excel(fname)
            self.update_statistics_label()

    def _save_to_npz(self, fname, progress_callback):
        if self.backend is None:
            raise ValueError(f"Save session is not available for biomarker mode '{self.get_biomarker_display_name()}'.")
        self.backend.export_app(fname)
        return []

    def save_to_npz(self):
        # Generate default filename based on loaded EDF file
        default_path = os.path.expanduser("~")  # Default to home directory
        if self.backend and hasattr(self.backend, 'edf_param') and self.backend.edf_param:
            edf_path = self.backend.edf_param.get("edf_fn", "")
            if edf_path and os.path.exists(edf_path):
                # Get directory and basename without extension
                directory = os.path.dirname(edf_path)
                base_name = os.path.splitext(os.path.basename(edf_path))[0]
                default_path = os.path.join(directory, f"{base_name}.pybrain")
        
        # open file dialog with default path (use non-native dialog to avoid macOS freezing)
        # print("saving to npz...",end="")
        fname, _ = QFileDialog.getSaveFileName(
            self.window,
            'Save file',
            default_path,
            "PyBrain session (*.pybrain);;Legacy NPZ (*.npz)",
            options=QFileDialog.DontUseNativeDialog,
        )
        if fname:
            # print("saving to {fname}...",end="")
            worker = Worker(self._save_to_npz, fname)
            self._connect_worker(worker, "Save state", result_handler=lambda _: 0)

    def _load_from_npz(self, fname, progress_callback):
        # Load the NPZ file and optimize the loading process
        checkpoint = load_session_checkpoint(fname)
        state_version = checkpoint_version(checkpoint)
        self.message_handler(f"Reading saved session format v{state_version}")
        checkpoint_biomarker = checkpoint_get(checkpoint, "biomarker_type", self.biomarker_type or "HFO")
        self.biomarker_type = checkpoint_biomarker
        if hasattr(self.window, "combo_box_biomarker"):
            blocker = QSignalBlocker(self.window.combo_box_biomarker)
            index = self.window.combo_box_biomarker.findText(checkpoint_biomarker)
            if index >= 0:
                self.window.combo_box_biomarker.setCurrentIndex(index)
            del blocker
        
        # Create a new backend instance
        self.backend = self._instantiate_backend_for_biomarker(self.biomarker_type)
        if self.backend is None:
            raise ValueError(f"Saved biomarker mode '{self.biomarker_type}' does not have a dedicated backend yet.")
        self.backend.load_checkpoint(checkpoint)
        self.case_backends = {self.biomarker_type: self.backend}
        self.current_recording_path = self.backend.edf_param.get("edf_fn") if getattr(self.backend, "edf_param", None) else None
        
        return []

    def load_from_npz(self):
        # open file dialog
        self.message_handler("Loading from npz...")
        self._set_workflow_message("Choose a saved session to load")
        fname = self._run_open_dialog(
            "Open Session",
            "PyBrain session (*.pybrain *.npz)",
        )
        if fname:
            self.case_backends = {}
            self.current_recording_path = None
            self.reinitialize()
            worker = Worker(self._load_from_npz, fname)
            self._set_workflow_message("Loading saved session...")
            self._connect_worker(worker, "Load session", result_handler=self.load_from_npz_finished)
        else:
            self._set_workflow_message("Open a recording to begin")
        # print(self.hfo_app.get_edf_info())

    def load_from_npz_finished(self):
        self.message_handler("Setting up UI...")
        self._set_workflow_message("Saved session loaded")
        self._sync_workspace_state()
        
        # Update backend reference in waveform plot
        self.window.waveform_plot.update_backend(self.backend)
        
        # Initialize waveform plot data (this is the main processing step)
        self.message_handler("Initializing waveform display...")
        self.window.waveform_plot.init_eeg_data()
        self._sync_waveform_channel_presentation()
        self._reset_waveform_measurement_state()
        
        # Update basic file information
        edf_info = self.backend.get_edf_info()
        edf_name = str(edf_info["edf_fn"])
        edf_name = edf_name[edf_name.rfind("/") + 1:]
        self._refresh_recording_metadata_ui([edf_name, str(edf_info["sfreq"]),
                              str(edf_info["nchan"]), str(self.backend.eeg_data.shape[1])])
        
        # Update number of jobs
        self.window.n_jobs_spinbox.setValue(self.backend.n_jobs)
        if hasattr(self.window, "go_to_time_input"):
            self.window.go_to_time_input.setMaximum(max(0.0, self.window.waveform_plot.get_total_time()))
            self.window.go_to_time_input.setValue(0.0)
        
        # Handle filtered data UI updates
        if self.backend.filtered:
            self.message_handler("Setting up filter UI...")
            self.filtering_complete()
        if getattr(self.backend, "param_filter", None) is not None:
            self._sync_filter_inputs_from_param(self.backend.param_filter)
        
        # Handle detection results UI updates
        if self.backend.detected:
            self.message_handler("Setting up detection UI...")
            self.set_detector_param_display()
            self._detect_finished()
            self.update_statistics_label()
            
            # Offer to directly open annotation window
            self._offer_direct_annotation()
            
        # Handle classification results UI updates
        if self.backend.classified:
            self.message_handler("Setting up classification UI...")
            self.set_classifier_param_display()
            self._classify_finished()
            self.update_statistics_label()
            
        self.message_handler("NPZ loading complete!")
        self._sync_workspace_state()

    def restore_loaded_backend_ui(self):
        has_recording = bool(self.backend is not None and getattr(self.backend, "eeg_data", None) is not None)
        if not has_recording:
            self.reinitialize_buttons()
            self.update_run_management_panel()
            self.update_decision_overview()
            self._sync_workspace_state()
            return

        self.window.waveform_plot.update_backend(self.backend)
        self.window.waveform_plot.init_eeg_data()
        self._sync_waveform_channel_presentation()
        self._reset_waveform_measurement_state()

        edf_info = self.backend.get_edf_info()
        edf_name = os.path.basename(str(edf_info.get("edf_fn", "No recording loaded")))
        self._refresh_recording_metadata_ui([
            edf_name,
            str(edf_info["sfreq"]),
            str(edf_info["nchan"]),
            str(self.backend.eeg_data.shape[1]),
        ])
        self.window.n_jobs_spinbox.setValue(self.backend.n_jobs)
        if hasattr(self.window, "go_to_time_input"):
            self.window.go_to_time_input.setMaximum(max(0.0, self.window.waveform_plot.get_total_time()))
            self.window.go_to_time_input.setValue(0.0)

        if getattr(self.backend, "param_filter", None) is not None:
            self._sync_filter_inputs_from_param(self.backend.param_filter)
        if self.backend.filtered and getattr(self.backend, "param_filter", None) is not None:
            self.window.is_data_filtered = True
            self.window.show_filtered = True
            self.window.waveform_plot.set_filtered(True)

        if getattr(self.backend, "param_detector", None) is not None:
            self.set_detector_param_display()
        if getattr(self.backend, "param_classifier", None) is not None:
            self.set_classifier_param_display()
        else:
            self.refresh_classifier_mode_ui()
        if getattr(self.backend, "param_detector", None) is None:
            self.refresh_detector_mode_ui()

        if self.backend.detected:
            self.window.waveform_plot.set_plot_biomarkers(self._has_case_visible_runs() or self.backend.detected)
            has_events = (
                self.backend.event_features is not None
                and self.backend.event_features.get_num_biomarker() > 0
            )
            self.window.annotation_button.setEnabled(has_events)
            self.window.save_csv_button.setEnabled(True)
            self._set_report_export_enabled(True)
            self.update_statistics_label()
        else:
            self.window.waveform_plot.set_plot_biomarkers(self._has_case_visible_runs())

        if self.backend.classified:
            self.window.detect_all_button.setEnabled(True)

        self.waveform_plot_button_clicked()
        self.update_spindle_capability_state()
        self.update_run_management_panel()
        self.update_status_indicators()
        self.update_decision_overview()
        self._sync_workspace_state()

    def _refresh_recording_metadata_ui(self, results):
        previous = self._suspend_default_configuration
        self._suspend_default_configuration = True
        try:
            self.update_edf_info(results)
        finally:
            self._suspend_default_configuration = previous

    def _can_auto_configure_defaults(self):
        if self._suspend_default_configuration or self.backend is None:
            return False
        if getattr(self.backend, "eeg_data", None) is None:
            return False
        session = getattr(self.backend, "analysis_session", None)
        has_runs = bool(session is not None and getattr(session, "runs", {}))
        return not has_runs and not self.backend.detected and not self.backend.classified

    def _apply_backend_defaults_if_needed(self):
        if not self._can_auto_configure_defaults():
            return

        if self.biomarker_type == 'HFO':
            self._apply_default_hfo_configuration()
        elif self.biomarker_type == 'Spindle':
            self._apply_default_spindle_configuration()
        elif self.biomarker_type == 'Spike':
            self.handle_unsupported_biomarker_mode(
                "Spike review mode is available, but its dedicated detection backend is not finalized yet.",
                show_dialog=False,
            )

    def _apply_default_hfo_configuration(self):
        try:
            if self.backend.param_filter is None:
                self.backend.set_filter_parameter(self._recommended_filter_param())
            self._sync_filter_inputs_from_param(self.backend.param_filter)

            if self.backend.param_detector is None:
                default_params = ParamSTE(self.backend.sample_freq)
                default_params.pass_band = int(self.backend.param_filter.fp)
                default_params.stop_band = int(self.backend.param_filter.fs)
                default_params.n_jobs = self.backend.n_jobs
                detector_params = {"detector_type": "STE", "detector_param": default_params.to_dict()}
                self.backend.set_detector(ParamDetector.from_dict(detector_params))

            if self.backend.param_classifier is None:
                self.backend.set_default_cpu_classifier()

            self.set_detector_param_display()
            self.set_classifier_param_display()
            self.window.ste_detect_button.setEnabled(True)
        except Exception as exc:
            print(f"Warning: Could not set default STE parameters: {exc}")

    def _apply_default_spindle_configuration(self):
        try:
            if self.backend.param_filter is None:
                self.backend.set_filter_parameter(self._recommended_filter_param())
            self._sync_filter_inputs_from_param(self.backend.param_filter)

            if self.backend.param_detector is None and has_yasa():
                default_params = ParamYASA(self.backend.sample_freq)
                default_params.n_jobs = self.backend.n_jobs
                detector_params = {"detector_type": "YASA", "detector_param": default_params.to_dict()}
                self.backend.set_detector(ParamDetector.from_dict(detector_params))

            self.set_detector_param_display()
            self.update_spindle_capability_state()
        except Exception as exc:
            print(f"Warning: Could not set default YASA parameters: {exc}")

    def open_channel_selection(self):
        # Check if EEG data is loaded
        if not hasattr(self.backend, 'eeg_data') or self.backend.eeg_data is None:
            QMessageBox.warning(self.window, "No Data", "Please load EEG data before selecting channels.")
            return
        self.window.channel_selection_window = ChannelSelectionWindow(self.backend, self, self.window.close_signal)
        self.window.channel_selection_window.show()

    def channel_selection_update(self):
        self.window.channel_scroll_bar.setValue(0)
        self.window.waveform_time_scroll_bar.setValue(0)
        is_empty = self.window.n_channel_input.maximum() == 0
        self.window.waveform_plot.plot(0, 0, empty=is_empty, update_biomarker=True)

    def switch_60(self):
        # get the value of the Filter60Button radio button
        filter_60 = self.window.Filter60Button.isChecked()
        # print("filtering:", filter_60)
        # if yes
        if filter_60:
            self.backend.set_filter_60()
        # if not
        else:
            self.backend.set_unfiltered_60()

        # replot
        self.window.waveform_plot.plot()
        # add a warning to the text about the HFO info saying that it is outdated now

    @pyqtSlot(str)
    def message_handler(self, s):
        s = s.replace("\n", "")
        if not s.strip():
            return
        horScrollBar = self.window.STDTextEdit.horizontalScrollBar()
        verScrollBar = self.window.STDTextEdit.verticalScrollBar()
        scrollIsAtEnd = verScrollBar.maximum() - verScrollBar.value() <= 10
        timestamped = f"[{datetime.now().strftime('%H:%M:%S')}] {s}"
        self._append_activity_entry(timestamped, s)

        contain_percentage = re.findall(r'%', s)
        contain_one_hundred_percentage = re.findall(r'100%', s)
        if contain_one_hundred_percentage:
            cursor = self.window.STDTextEdit.textCursor()
            cursor.movePosition(QTextCursor.End - 1)
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            self.window.STDTextEdit.setTextCursor(cursor)
            self.window.STDTextEdit.insertPlainText(timestamped)
        elif contain_percentage:
            cursor = self.window.STDTextEdit.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            self.window.STDTextEdit.setTextCursor(cursor)
            self.window.STDTextEdit.insertPlainText(timestamped)
        else:
            self.window.STDTextEdit.append(timestamped)

        if scrollIsAtEnd:
            verScrollBar.setValue(verScrollBar.maximum())  # Scrolls to the bottom
            horScrollBar.setValue(0)  # scroll to the left

    def _append_activity_entry(self, timestamped, raw_text):
        lowered = raw_text.lower()
        if any(token in lowered for token in ("error", "traceback", "failed", "exception")):
            color = "#b2554f"
            prefix = "Error"
        elif "warning" in lowered or "warn" in lowered:
            color = "#9a6a2b"
            prefix = "Warning"
        elif any(token in lowered for token in ("complete", "ready", "loaded", "saved", "accepted")):
            color = "#4f7a57"
            prefix = "Done"
        else:
            color = "#49667d"
            prefix = "Info"

        summary_label = getattr(self.window, "activity_summary_label", None)
        if summary_label is not None:
            summary_label.setText(f"{prefix} {timestamped}")
        level_dot = getattr(self.window, "activity_level_dot", None)
        if level_dot is not None:
            level = "info"
            if prefix == "Error":
                level = "error"
            elif prefix == "Warning":
                level = "warning"
            elif prefix == "Done":
                level = "done"
            set_dynamic_property(level_dot, "activityLevel", level)

    @pyqtSlot(object)
    def update_edf_info(self, results):
        if hasattr(self.window, "waveform_plot"):
            self.window.waveform_plot.update_backend(self.backend, False)
            self.window.waveform_plot.init_eeg_data()
            self._sync_waveform_channel_presentation()
            self._reset_waveform_measurement_state()
        # Clear the empty-state overlay as soon as the recording backend is ready,
        # even before the rest of the metadata/UI refresh finishes.
        if hasattr(self.window, "view"):
            self.window.view.set_workspace_state(True, self.get_biomarker_display_name())
        self.case_backends[self.biomarker_type] = self.backend
        if getattr(self.backend, "edf_param", None):
            self.current_recording_path = self.backend.edf_param.get("edf_fn", self.current_recording_path)
        self.window.main_filename.setText(results[0])
        self.window.main_sampfreq.setText(results[1])
        self.window.sample_freq = float(results[1])
        self._sync_filter_input_constraints()
        self.window.main_numchannels.setText(results[2])
        # print("updated")
        self.window.main_length.setText(str(round(float(results[3]) / (60 * float(results[1])), 3)) + " min")
        # self.window.waveform_plot.plot(0, update_biomarker=True)
        self.window.waveform_plot.set_plot_biomarkers(self._has_case_visible_runs())

        # print("plotted")
        # connect buttons
        safe_connect_signal_slot(self.window.waveform_time_scroll_bar.valueChanged, self.scroll_time_waveform_plot)
        safe_connect_signal_slot(self.window.channel_scroll_bar.valueChanged, self.scroll_channel_waveform_plot)

        self.window.waveform_plot_button.setEnabled(True)
        self.window.Choose_Channels_Button.setEnabled(True)
        # set the display time window spin box
        self.window.display_time_window_input.setValue(self.window.waveform_plot.get_time_window())
        self.window.display_time_window_input.setMaximum(self.window.waveform_plot.get_total_time())
        self.window.display_time_window_input.setMinimum(0.1)
        if hasattr(self.window, "go_to_time_input"):
            self.window.go_to_time_input.setMaximum(max(0.0, self.window.waveform_plot.get_total_time()))
            self.window.go_to_time_input.setValue(0.0)
        # set the n channel spin box
        self.window.n_channel_input.setValue(self.window.waveform_plot.get_n_channels_to_plot())
        self.window.n_channel_input.setMaximum(self.window.waveform_plot.get_n_channels())
        self.window.n_channel_input.setMinimum(1)
        # set the time scroll bar range
        self.window.waveform_time_scroll_bar.setMaximum(int(self.window.waveform_plot.get_total_time() / (
                self.window.waveform_plot.get_time_window() * self.window.waveform_plot.get_time_increment() / 100)))
        self.window.waveform_time_scroll_bar.setValue(0)
        # set the channel scroll bar range
        self.window.channel_scroll_bar.setMaximum(
            self.window.waveform_plot.get_n_channels() - self.window.waveform_plot.get_n_channels_to_plot())
        # enable the filter button
        self.window.overview_filter_button.setEnabled(True)
        # enable the plot out the 60Hz bandstopped signal
        self.window.Filter60Button.setEnabled(True)
        self.window.bipolar_button.setEnabled(True)
        self._set_workflow_message("Recording loaded")
        self._sync_workspace_state()
        self._apply_backend_defaults_if_needed()
        if getattr(self.backend, "param_filter", None) is not None:
            self._sync_filter_inputs_from_param(self.backend.param_filter)
        self.refresh_detector_mode_ui()
        self.refresh_classifier_mode_ui()
        self.waveform_plot_button_clicked()
        self.update_run_management_panel()
        self.update_spindle_capability_state()
        self.update_status_indicators()
        self.update_decision_overview()

    def toggle_filtered(self):
        # self.message_handler('Showing original data...')
        if self.window.is_data_filtered:
            self.window.show_filtered = bool(self.window.toggle_filtered_checkbox.isChecked())
            self.window.waveform_plot.set_filtered(self.window.show_filtered)
            self.waveform_plot_button_clicked()
        self.update_waveform_toolbar_state()

    def read_edf(self, fname, progress_callback):
        self.backend.load_edf(fname)
        filename = os.path.basename(fname)
        sample_freq = str(self.backend.sample_freq)
        num_channels = str(len(self.backend.channel_names))
        length = str(self.backend.eeg_data.shape[1])
        return [filename, sample_freq, num_channels, length]

    def _filter(self, progress_callback):
        self.backend.filter_eeg_data()
        return []

    # def open_detector(self):
    #     # Pass the function to execute, function, args, kwargs
    #     worker = Worker(self.quick_detect)
    #     self.window.threadpool.start(worker)
    #
    # def round_dict(self, d: dict, n: int):
    #     for key in d.keys():
    #         if type(d[key]) == float:
    #             d[key] = round(d[key], n)
    #     return d

    def scroll_time_waveform_plot(self, event):
        t_start = self.window.waveform_time_scroll_bar.value() * self.window.waveform_plot.get_time_window() * self.window.waveform_plot.get_time_increment() / 100
        self.window.waveform_plot.plot(t_start)

    def scroll_channel_waveform_plot(self, event):
        channel_start = self.window.channel_scroll_bar.value()
        self.window.waveform_plot.plot(first_channel_to_plot=channel_start, update_biomarker=True)

    def get_channels_to_plot(self):
        return self.window.waveform_plot.get_channels_to_plot()

    def get_channel_indices_to_plot(self):
        return self.window.waveform_plot.get_channel_indices_to_plot()

    def _run_open_dialog(self, title, name_filter, directory=""):
        dialog = QFileDialog(self.window, title, directory)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter(name_filter)
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setViewMode(QFileDialog.Detail)
        dialog.raise_()
        dialog.activateWindow()
        if dialog.exec_() == QDialog.Accepted:
            files = dialog.selectedFiles()
            if files:
                return files[0]
        return ""

    def waveform_plot_button_clicked(self):
        time_window = self.window.display_time_window_input.value()
        self.window.waveform_plot.set_time_window(time_window)
        n_channels_to_plot = self.window.n_channel_input.value()
        self.window.waveform_plot.set_n_channels_to_plot(n_channels_to_plot)
        time_increment = self.window.Time_Increment_Input.value()
        self.window.waveform_plot.set_time_increment(time_increment)
        normalize_vertical = self.window.normalize_vertical_input.isChecked()
        self.window.waveform_plot.set_normalize_vertical(normalize_vertical)
        is_empty = self.window.n_channel_input.maximum() == 0
        start = self.window.waveform_plot.t_start
        first_channel_to_plot = self.window.waveform_plot.first_channel_to_plot

        t_value = int(start // (self.window.waveform_plot.get_time_window() * self.window.waveform_plot.get_time_increment() / 100))
        self.window.waveform_time_scroll_bar.setMaximum(int(self.window.waveform_plot.get_total_time() / (
                    self.window.waveform_plot.get_time_window() * self.window.waveform_plot.get_time_increment() / 100)))
        self.window.waveform_time_scroll_bar.setValue(t_value)
        c_value = max(0, int(first_channel_to_plot))
        self.window.channel_scroll_bar.setMaximum(len(self.window.waveform_plot.get_channels_to_plot()) - n_channels_to_plot)
        self.window.channel_scroll_bar.setValue(c_value)
        self.window.waveform_plot.plot(start, first_channel_to_plot, empty=is_empty, update_biomarker=True)
        if hasattr(self.window, "go_to_time_input"):
            blocker = QSignalBlocker(self.window.go_to_time_input)
            self.window.go_to_time_input.setMaximum(max(0.0, self.window.waveform_plot.get_total_time()))
            self.window.go_to_time_input.setValue(max(0.0, float(self.window.waveform_plot.t_start)))
            del blocker
        self.update_waveform_toolbar_state()

    def open_file(self):
        self.message_handler("Opening recording browser...")
        self._set_workflow_message("Choose a recording to load")
        fname = self._run_open_dialog(
            "Open Recording",
            "Recordings Files (*.edf *.eeg *.vhdr *.vmrk *.fif *.fif.gz)",
        )
        if fname:
            self.case_backends = {}
            self.current_recording_path = fname
            self.reinitialize()
            worker = Worker(self.read_edf, fname)
            self._set_workflow_message("Loading recording...")
            self._connect_worker(worker, "Load recording", result_handler=self.update_edf_info)
        else:
            self._set_workflow_message("Open a recording to begin")

    def filtering_complete(self):
        self._end_busy_task()
        self._set_filter_controls_busy(False)
        self.message_handler('Filtering COMPLETE!')
        self._set_workflow_message("Filter ready")
        filter_60 = self.window.Filter60Button.isChecked()
        # print("filtering:", filter_60)
        # if yes
        if filter_60:
            self.backend.set_filter_60()
        # if not
        else:
            self.backend.set_unfiltered_60()

        if self.biomarker_type == 'HFO':
            self.window.STE_save_button.setEnabled(True)
            self.window.ste_detect_button.setEnabled(True)
            self.window.MNI_save_button.setEnabled(True)
            self.window.mni_detect_button.setEnabled(True)
            self.window.HIL_save_button.setEnabled(True)
            self.window.hil_detect_button.setEnabled(True)
            self.window.is_data_filtered = True
            self.window.show_filtered = True
            self._set_filtered_toggle_state(True)
            self.window.waveform_plot.set_filtered(True)
            self.window.save_npz_button.setEnabled(True)
        elif self.biomarker_type == 'Spindle':
            self.window.is_data_filtered = True
            self.window.show_filtered = True
            self._set_filtered_toggle_state(True)
            self.window.waveform_plot.set_filtered(True)
            self.window.save_npz_button.setEnabled(True)
        self.update_status_indicators()

    def detect_HFOs(self):
        print("Detecting HFOs...")
        self._set_workflow_message(f"Running {self.get_biomarker_display_name()} detection...")
        self._begin_busy_task("detection", "Detecting", self._detection_busy_buttons())
        worker = Worker(self._detect)
        self._connect_worker(worker, "Detection", result_handler=self._detect_finished)

    def detect_Spindles(self):
        print("Detecting Spindles...")
        self._set_workflow_message(f"Running {self.get_biomarker_display_name()} detection...")
        self._begin_busy_task("detection", "Detecting", self._detection_busy_buttons())
        worker = Worker(self._detect)
        self._connect_worker(worker, "Detection", result_handler=self._detect_finished)

    def _detect_finished(self):
        self._end_busy_task()
        # right now do nothing beyond message handler saying that
        # it has detected HFOs
        self.message_handler("Biomarker detected")
        self.update_statistics_label()
        self.window.waveform_plot.set_plot_biomarkers(self._has_case_visible_runs() or bool(self.backend and self.backend.detected))
        self.window.detect_all_button.setEnabled(True)
        has_events = (
            self.backend.event_features is not None
            and self.backend.event_features.get_num_biomarker() > 0
        )
        self.window.annotation_button.setEnabled(has_events)
        # Enable save as Excel button after detection is finished
        # Classification fields (artifact, spike, ehfo) will be empty/zero if not classified yet
        self.window.save_csv_button.setEnabled(True)
        self._set_report_export_enabled(True)
        self._set_workflow_message("Detection complete")
        if hasattr(self.backend, "analysis_session"):
            summaries = self.backend.get_run_summaries()
            if summaries:
                current = summaries[-1]
                self.message_handler(
                    f"Saved run '{current['detector_name']}' with {current['num_events']} events across {current['num_channels']} channels"
                )
        self.update_status_indicators()
        
        # Auto-save the detection state for future annotation work
        # Comment out the line below if you don't want automatic saving
        # self._auto_save_detection_state()

    def _auto_save_detection_state(self):
        """Automatically save the detection state to an NPZ file for future annotation"""
        try:
            # Generate a filename based on the original EDF file
            edf_info = self.backend.get_edf_info()
            original_filename = edf_info.get('edf_fn', 'unknown_file')
            base_name = os.path.splitext(os.path.basename(original_filename))[0]
            detector_type = self.backend.param_detector.detector_type if self.backend.param_detector else 'unknown'
            
            # Create a states directory if it doesn't exist
            states_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'states')
            os.makedirs(states_dir, exist_ok=True)
            
            # Generate the NPZ filename
            npz_filename = f"{base_name}_{self.biomarker_type}_{detector_type}_detected.npz"
            npz_path = os.path.join(states_dir, npz_filename)
            
            # Create enhanced checkpoint with 60Hz data for fast loading
            checkpoint = build_base_checkpoint(self.backend, self.biomarker_type)
            
            # Add biomarker-specific data
            if self.biomarker_type == 'HFO':
                checkpoint.update({
                    "HFOs": self.backend.HFOs,
                    "param_detector": self.backend.param_detector.to_dict() if self.backend.param_detector else None,
                    "event_features": self.backend.event_features.to_dict() if self.backend.event_features else None,
                    "param_classifier": self.backend.param_classifier.to_dict() if self.backend.param_classifier else None,
                    "artifact_predictions": np.array(self.backend.event_features.artifact_predictions),
                    "spike_predictions": np.array(self.backend.event_features.spike_predictions),
                    "ehfo_predictions": np.array(self.backend.event_features.ehfo_predictions),
                    "artifact_annotations": np.array(self.backend.event_features.artifact_annotations),
                    "pathological_annotations": np.array(self.backend.event_features.pathological_annotations),
                    "physiological_annotations": np.array(self.backend.event_features.physiological_annotations),
                    "annotated": np.array(self.backend.event_features.annotated),
                })
            else:  # Spindle
                checkpoint.update({
                    "Spindles": self.backend.Spindles,
                    "param_detector": self.backend.param_detector.to_dict() if self.backend.param_detector else None,
                    "Spindle_features": self.backend.event_features.to_dict() if self.backend.event_features else None,
                    "param_classifier": self.backend.param_classifier.to_dict() if self.backend.param_classifier else None,
                    "artifact_predictions": np.array(self.backend.event_features.artifact_predictions),
                    "spike_predictions": np.array(self.backend.event_features.spike_predictions),
                    "artifact_annotations": np.array(self.backend.event_features.artifact_annotations),
                    "spike_annotations": np.array(self.backend.event_features.spike_annotations),
                    "annotated": np.array(self.backend.event_features.annotated),
                })
            
            # Save using the same method as the backend
            from src.utils.utils_io import dump_to_npz
            dump_to_npz(checkpoint, npz_path)
            
            # Notify the user
            num_events = self._get_feature_event_count(self.backend.event_features)
            self.message_handler(f"Auto-saved {num_events} {self.biomarker_type} events to: {npz_filename}")
            
        except Exception as e:
            self.message_handler(f"Warning: Auto-save failed: {str(e)}")

    def _offer_direct_annotation(self):
        """Offer to directly open annotation window after loading detection state"""
        if self.backend.detected and self.backend.event_features:
            num_events = self._get_feature_event_count(self.backend.event_features)
            if num_events > 0:
                # Ensure prediction arrays are properly sized before opening annotation
                self._ensure_prediction_arrays_are_sized()
                
                # Create a message box asking if user wants to open annotation directly
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Question)
                msg.setWindowTitle("Open Annotation Window")
                msg.setText(f"Loaded {num_events} {self.biomarker_type} events successfully!")
                msg.setInformativeText("Would you like to open the annotation window directly to start annotating these events?")
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msg.setDefaultButton(QMessageBox.Yes)
                
                response = msg.exec_()
                if response == QMessageBox.Yes:
                    self.open_annotation()

    def _ensure_prediction_arrays_are_sized(self):
        """Ensure prediction arrays have the same size as the number of detected events"""
        if self.backend.detected and self.backend.event_features:
            import numpy as np
            num_events = len(self.backend.event_features.starts)
            
            # Fix spike_predictions array if it's empty
            if len(self.backend.event_features.spike_predictions) == 0 and num_events > 0:
                self.backend.event_features.spike_predictions = np.zeros(num_events)
            
            # Fix ehfo_predictions array if it's empty (for HFO type)
            if hasattr(self.backend.event_features, 'ehfo_predictions'):
                if len(self.backend.event_features.ehfo_predictions) == 0 and num_events > 0:
                    self.backend.event_features.ehfo_predictions = np.zeros(num_events)

    def _detect(self, progress_callback):
        # call detect HFO function on backend
        if self.backend is None or not self.biomarker_supports_detection():
            raise ValueError(f"Detection is not available for biomarker mode '{self.get_biomarker_display_name()}'.")
        self.backend.detect_biomarker()
        return []

    def open_quick_detection(self):
        # if we want to open multiple qd dialog
        if not self.window.quick_detect_open:
            from src.ui.quick_detection import HFOQuickDetector
            qd = HFOQuickDetector(HFO_App(), self, self.window.close_signal)
            self.window.quick_detection_window = qd
            safe_connect_signal_slot(qd.destroyed, lambda *_: self.set_quick_detect_open(False))
            # print("created new quick detector")
            qd.show()
            self.window.quick_detect_open = True

    def set_quick_detect_open(self, open):
        self.window.quick_detect_open = open
        if not open and hasattr(self.window, "quick_detection_window"):
            self.window.quick_detection_window = None

    def reinitialize_buttons(self):
        self.window.mni_detect_button.setEnabled(False)
        self.window.ste_detect_button.setEnabled(False)
        self.window.hil_detect_button.setEnabled(False)
        if hasattr(self.window, "yasa_detect_button"):
            self.window.yasa_detect_button.setEnabled(False)
        self.window.detect_all_button.setEnabled(False)
        self.window.save_csv_button.setEnabled(False)
        self._set_report_export_enabled(False)
        self.window.save_npz_button.setEnabled(False)
        self.window.switch_run_button.setEnabled(False)
        self.window.accept_run_button.setEnabled(False)
        self.window.compare_runs_button.setEnabled(False)
        self.window.STE_save_button.setEnabled(False)
        self.window.MNI_save_button.setEnabled(False)
        self.window.HIL_save_button.setEnabled(False)
        if hasattr(self.window, "YASA_save_button"):
            self.window.YASA_save_button.setEnabled(False)
        self.window.Filter60Button.setEnabled(False)

    def set_mni_input_len(self, max_len=5):
        self.window.mni_epoch_time_input.setMaxLength(max_len)
        self.window.mni_epoch_chf_input.setMaxLength(max_len)
        self.window.mni_chf_percentage_input.setMaxLength(max_len)
        self.window.mni_min_window_input.setMaxLength(max_len)
        self.window.mni_min_gap_time_input.setMaxLength(max_len)
        self.window.mni_threshold_percentage_input.setMaxLength(max_len)
        self.window.mni_baseline_window_input.setMaxLength(max_len)
        self.window.mni_baseline_shift_input.setMaxLength(max_len)
        self.window.mni_baseline_threshold_input.setMaxLength(max_len)
        self.window.mni_baseline_min_time_input.setMaxLength(max_len)

    def set_ste_input_len(self, max_len=5):
        self.window.ste_rms_window_input.setMaxLength(max_len)
        self.window.ste_min_window_input.setMaxLength(max_len)
        self.window.ste_min_gap_input.setMaxLength(max_len)
        self.window.ste_epoch_length_input.setMaxLength(max_len)
        self.window.ste_min_oscillation_input.setMaxLength(max_len)
        self.window.ste_rms_threshold_input.setMaxLength(max_len)
        self.window.ste_peak_threshold_input.setMaxLength(max_len)

    def set_hil_input_len(self, max_len=5):
        self.window.hil_sample_freq_input.setMaxLength(max_len)
        self.window.hil_pass_band_input.setMaxLength(max_len)
        self.window.hil_stop_band_input.setMaxLength(max_len)
        self.window.hil_epoch_time_input.setMaxLength(max_len)
        self.window.hil_sd_threshold_input.setMaxLength(max_len)
        self.window.hil_min_window_input.setMaxLength(max_len)

    def set_yasa_input_len(self, max_len=5):
        self.window.yasa_freq_sp_low_input.setMaxLength(max_len)
        self.window.yasa_freq_sp_high_input.setMaxLength(max_len)
        self.window.yasa_freq_broad_low_input.setMaxLength(max_len)
        self.window.yasa_freq_broad_high_input.setMaxLength(max_len)
        self.window.yasa_duration_low_input.setMaxLength(max_len)
        self.window.yasa_duration_high_input.setMaxLength(max_len)
        self.window.yasa_min_distance_input.setMaxLength(max_len)
        self.window.yasa_thresh_rel_pow_input.setMaxLength(max_len)
        self.window.yasa_thresh_corr_input.setMaxLength(max_len)
        self.window.yasa_thresh_rms_input.setMaxLength(max_len)

    def close_other_window(self):
        self.window.close_signal.emit()

    def set_n_jobs(self):
        if self.backend is None:
            return
        n_jobs = int(self.window.n_jobs_spinbox.value())
        if hasattr(self.backend, "set_n_jobs"):
            self.backend.set_n_jobs(n_jobs)
        else:
            self.backend.n_jobs = n_jobs
        self.message_handler(f"Workers set to {n_jobs}")

    def set_channels_to_plot(self, channels_to_plot, display_all=True):
        self.window.waveform_plot.set_channels_to_plot(channels_to_plot)
        # print(f"Channels to plot: {self.channels_to_plot}")
        self.window.n_channel_input.setMaximum(len(channels_to_plot))
        if display_all:
            self.window.n_channel_input.setValue(len(channels_to_plot))
        self.waveform_plot_button_clicked()
        self.update_status_indicators()

    def save_ste_params(self):
        # get filter parameters
        rms_window_raw = self.window.ste_rms_window_input.text()
        min_window_raw = self.window.ste_min_window_input.text()
        min_gap_raw = self.window.ste_min_gap_input.text()
        epoch_len_raw = self.window.ste_epoch_length_input.text()
        min_osc_raw = self.window.ste_min_oscillation_input.text()
        rms_thres_raw = self.window.ste_rms_threshold_input.text()
        peak_thres_raw = self.window.ste_peak_threshold_input.text()
        try:
            param_dict = {"sample_freq": 2000, "pass_band": 1, "stop_band": 80,
                          # these are placeholder params, will be updated later
                          "rms_window": self._parse_float_input(rms_window_raw, "STE RMS window", positive=True),
                          "min_window": self._parse_float_input(min_window_raw, "STE minimum window", positive=True),
                          "min_gap": self._parse_float_input(min_gap_raw, "STE minimum gap", non_negative=True),
                          "epoch_len": self._parse_float_input(epoch_len_raw, "STE epoch length", positive=True),
                          "min_osc": self._parse_float_input(min_osc_raw, "STE minimum oscillations", positive=True),
                          "rms_thres": self._parse_float_input(rms_thres_raw, "STE RMS threshold", positive=True),
                          "peak_thres": self._parse_float_input(peak_thres_raw, "STE peak threshold", positive=True),
                          "n_jobs": self.backend.n_jobs}
            detector_params = {"detector_type": "STE", "detector_param": param_dict}
            self.backend.set_detector(ParamDetector.from_dict(detector_params))

            # set display parameters
            self.window.ste_epoch_display.setText(epoch_len_raw)
            self.window.ste_min_window_display.setText(min_window_raw)
            self.window.ste_rms_window_display.setText(rms_window_raw)
            self.window.ste_min_gap_time_display.setText(min_gap_raw)
            self.window.ste_min_oscillations_display.setText(min_osc_raw)
            self.window.ste_peak_threshold_display.setText(peak_thres_raw)
            self.window.ste_rms_threshold_display.setText(rms_thres_raw)
            self.update_detector_tab("STE")
            # Enable the STE detect button after parameters are saved successfully
            self.window.ste_detect_button.setEnabled(True)
            return True
        except (TypeError, ValueError) as exc:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error!")
            msg.setInformativeText(f'Detector could not be constructed given the parameters. {exc}')
            msg.setWindowTitle("Detector Construction Failed")
            msg.exec_()
            self.window.ste_detect_button.setEnabled(False)
            return False

    def save_mni_params(self):
        try:
            epoch_time = self.window.mni_epoch_time_input.text()
            epo_CHF = self.window.mni_epoch_chf_input.text()
            per_CHF = self.window.mni_chf_percentage_input.text()
            min_win = self.window.mni_min_window_input.text()
            min_gap = self.window.mni_min_gap_time_input.text()
            thrd_perc = self.window.mni_threshold_percentage_input.text()
            base_seg = self.window.mni_baseline_window_input.text()
            base_shift = self.window.mni_baseline_shift_input.text()
            base_thrd = self.window.mni_baseline_threshold_input.text()
            base_min = self.window.mni_baseline_min_time_input.text()

            param_dict = {"sample_freq": 2000, "pass_band": 1, "stop_band": 80,
                          # these are placeholder params, will be updated later
                          "epoch_time": self._parse_float_input(epoch_time, "MNI epoch time", positive=True),
                          "epo_CHF": self._parse_float_input(epo_CHF, "MNI epoch CHF", positive=True),
                          "per_CHF": self._parse_float_input(per_CHF, "MNI CHF percentage", positive=True),
                          "min_win": self._parse_float_input(min_win, "MNI minimum window", positive=True),
                          "min_gap": self._parse_float_input(min_gap, "MNI minimum gap", non_negative=True),
                          "base_seg": self._parse_float_input(base_seg, "MNI baseline window", positive=True),
                          "thrd_perc": self._parse_float_input(thrd_perc, "MNI threshold percentage", positive=True) / 100,
                          "base_shift": self._parse_float_input(base_shift, "MNI baseline shift", non_negative=True),
                          "base_thrd": self._parse_float_input(base_thrd, "MNI baseline threshold", positive=True),
                          "base_min": self._parse_float_input(base_min, "MNI baseline minimum time", non_negative=True),
                          "n_jobs": self.backend.n_jobs}
            # param_dict = self.round_dict(param_dict, 3)
            detector_params = {"detector_type": "MNI", "detector_param": param_dict}
            self.backend.set_detector(ParamDetector.from_dict(detector_params))

            # set display parameters
            self.window.mni_epoch_display.setText(epoch_time)
            self.window.mni_epoch_chf_display.setText(epo_CHF)
            self.window.mni_chf_percentage_display.setText(per_CHF)
            self.window.mni_min_window_display.setText(min_win)
            self.window.mni_min_gap_time_display.setText(min_gap)
            self.window.mni_threshold_percentile_display.setText(thrd_perc)
            self.window.mni_baseline_window_display.setText(base_seg)
            self.window.mni_baseline_shift_display.setText(base_shift)
            self.window.mni_baseline_threshold_display.setText(base_thrd)
            self.window.mni_baseline_min_time_display.setText(base_min)

            self.update_detector_tab("MNI")
            # Enable the MNI detect button after parameters are saved successfully
            self.window.mni_detect_button.setEnabled(True)
            return True
        except (TypeError, ValueError) as exc:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error!")
            msg.setInformativeText(f'Detector could not be constructed given the parameters. {exc}')
            msg.setWindowTitle("Detector Construction Failed")
            msg.exec_()
            self.window.mni_detect_button.setEnabled(False)
            return False

    def save_hil_params(self):
        try:
            sample_freq = self.window.hil_sample_freq_input.text()
            pass_band = self.window.hil_pass_band_input.text()
            stop_band = self.window.hil_stop_band_input.text()
            epoch_time = self.window.hil_epoch_time_input.text()
            sd_threshold = self.window.hil_sd_threshold_input.text()
            min_window = self.window.hil_min_window_input.text()

            param_dict = {
                "sample_freq": self._parse_float_input(sample_freq, "HIL sample frequency", positive=True),
                "pass_band": self._parse_float_input(pass_band, "HIL pass band", positive=True),
                "stop_band": self._parse_float_input(stop_band, "HIL stop band", positive=True),
                "epoch_time": self._parse_float_input(epoch_time, "HIL epoch time", positive=True),
                "sd_threshold": self._parse_float_input(sd_threshold, "HIL SD threshold", positive=True),
                "min_window": self._parse_float_input(min_window, "HIL minimum window", positive=True),
                "n_jobs": self.backend.n_jobs,
            }

            detector_params = {"detector_type": "HIL", "detector_param": param_dict}
            self.backend.set_detector(ParamDetector.from_dict(detector_params))

            self.window.hil_sample_freq_display.setText(sample_freq)
            self.window.hil_pass_band_display.setText(pass_band)
            self.window.hil_stop_band_display.setText(stop_band)
            self.window.hil_epoch_time_display.setText(epoch_time)
            self.window.hil_sd_threshold_display.setText(sd_threshold)
            self.window.hil_min_window_display.setText(min_window)

            self.update_detector_tab("HIL")
            # Enable the HIL detect button after parameters are saved successfully
            self.window.hil_detect_button.setEnabled(True)
            return True

        except (ImportError, TypeError, ValueError) as exc:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error!")
            msg.setInformativeText(f'HIL Detector could not be constructed given the parameters. Error: {str(exc)}')
            msg.setWindowTitle("HIL Detector Construction Failed")
            msg.exec_()
            self.window.hil_detect_button.setEnabled(False)
            return False

    def save_yasa_params(self):
        # get filter parameters
        if not has_yasa():
            self.handle_unsupported_biomarker_mode(
                "Spindle detection requires the optional 'yasa' package. Install it to configure and run YASA detection."
            )
            return

        freq_sp_low_raw = self.window.yasa_freq_sp_low_input.text()
        freq_sp_high_raw = self.window.yasa_freq_sp_high_input.text()
        freq_broad_low_raw = self.window.yasa_freq_broad_low_input.text()
        freq_broad_high_raw = self.window.yasa_freq_broad_high_input.text()
        duration_low_raw = self.window.yasa_duration_low_input.text()
        duration_high_raw = self.window.yasa_duration_high_input.text()
        min_distance_raw = self.window.yasa_min_distance_input.text()
        thresh_rel_pow_raw = self.window.yasa_thresh_rel_pow_input.text()
        thresh_corr_raw = self.window.yasa_thresh_corr_input.text()
        thresh_rms_raw = self.window.yasa_thresh_rms_input.text()
        try:
            freq_sp = (
                self._parse_float_input(freq_sp_low_raw, "YASA spindle band low", positive=True),
                self._parse_float_input(freq_sp_high_raw, "YASA spindle band high", positive=True),
            )
            freq_broad = (
                self._parse_float_input(freq_broad_low_raw, "YASA broad band low", positive=True),
                self._parse_float_input(freq_broad_high_raw, "YASA broad band high", positive=True),
            )
            duration = (
                self._parse_float_input(duration_low_raw, "YASA duration low", positive=True),
                self._parse_float_input(duration_high_raw, "YASA duration high", positive=True),
            )
            param_dict = {"sample_freq": 2000,
                          # these are placeholder params, will be updated later
                          "freq_sp": freq_sp, "freq_broad": freq_broad,
                          "duration": duration,
                          "min_distance": self._parse_float_input(min_distance_raw, "YASA minimum distance", non_negative=True),
                          "rel_pow": self._parse_float_input(thresh_rel_pow_raw, "YASA relative power threshold", positive=True),
                          "corr": self._parse_float_input(thresh_corr_raw, "YASA correlation threshold", positive=True),
                          "rms": self._parse_float_input(thresh_rms_raw, "YASA RMS threshold", positive=True),
                          "n_jobs": self.backend.n_jobs}
            detector_params = {"detector_type": "YASA", "detector_param": param_dict}
            self.backend.set_detector(ParamDetector.from_dict(detector_params))

            # set display parameters
            self.window.yasa_freq_sp_display.setText(f"{freq_sp_low_raw} - {freq_sp_high_raw}")
            self.window.yasa_freq_broad_display.setText(f"{freq_broad_low_raw} - {freq_broad_high_raw}")
            self.window.yasa_duration_display.setText(f"{duration_low_raw} - {duration_high_raw}")
            self.window.yasa_min_distance_display.setText(min_distance_raw)
            self.window.yasa_thresh_rel_pow_display.setText(thresh_rel_pow_raw)
            self.window.yasa_thresh_corr_display.setText(thresh_corr_raw)
            self.window.yasa_thresh_rms_display.setText(thresh_rms_raw)
            # self.update_detector_tab("STE")
            self.window.yasa_detect_button.setEnabled(True)
            return True
        except (SyntaxError, ValueError, TypeError) as exc:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error!")
            msg.setInformativeText(f'Detector could not be constructed given the parameters. {exc}')
            msg.setWindowTitle("Detector Construction Failed")
            msg.exec_()
            self.window.yasa_detect_button.setEnabled(False)
            return False

    def update_spindle_capability_state(self):
        if self.biomarker_type != 'Spindle':
            return
        yasa_ready = has_yasa()
        if hasattr(self.window, "yasa_detect_button"):
            self.window.yasa_detect_button.setEnabled(yasa_ready and self.backend is not None and getattr(self.backend, "eeg_data", None) is not None)
            self.window.yasa_detect_button.setToolTip("" if yasa_ready else "Install the optional 'yasa' package to enable spindle detection.")
        if hasattr(self.window, "YASA_save_button"):
            self.window.YASA_save_button.setEnabled(yasa_ready)
            self.window.YASA_save_button.setToolTip("" if yasa_ready else "Install the optional 'yasa' package to edit and save spindle detector parameters.")

    def update_detector_tab(self, index):
        if index == "STE":
            self.window.stacked_widget_detection_param.setCurrentIndex(0)
        elif index == "MNI":
            self.window.stacked_widget_detection_param.setCurrentIndex(1)
        elif index == "HIL":
            self.window.stacked_widget_detection_param.setCurrentIndex(2)
        elif index == "YASA":
            self.window.stacked_widget_detection_param.setCurrentIndex(0)

    def reinitialize(self):
        # kill all threads in self.threadpool
        self.close_other_window()
        # self.backend = HFO_App()
        self.set_biomarker_type_and_init_backend(self.biomarker_type)
        if hasattr(self.window, "waveform_plot"):
            self.window.waveform_plot.update_backend(self.backend, False)
        self.window.main_filename.setText("")
        self.window.main_sampfreq.setText("")
        self.window.main_numchannels.setText("")
        self.window.main_length.setText("")
        self.window.statistics_label.setText("")
        if hasattr(self.window, "run_summary_label"):
            self.window.run_summary_label.setText("No detection runs yet.")
        self._set_workflow_message("Open a recording to begin")
        self.update_spindle_capability_state()
        self.update_status_indicators()
        self.update_decision_overview()
        self._sync_workspace_state()

    def start_run_mode(self, biomarker_type):
        combo = getattr(self.window, "combo_box_biomarker", None)
        if combo is not None:
            index = combo.findText(biomarker_type)
            if index >= 0:
                combo.setCurrentIndex(index)
        QTimer.singleShot(0, self._ensure_current_backend_loaded)
        self.message_handler(f"Prepared {biomarker_type} run workspace.")

    def _ensure_current_backend_loaded(self):
        if self.backend is None:
            self.update_run_management_panel()
            self.update_decision_overview()
            self._sync_workspace_state()
            return
        if getattr(self.backend, "eeg_data", None) is None and self.current_recording_path:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self.backend.load_edf(self.current_recording_path)
                self.case_backends[self.biomarker_type] = self.backend
            finally:
                QApplication.restoreOverrideCursor()
        self.restore_loaded_backend_ui()

    def update_ste_params(self, ste_params):
        rms_window = str(ste_params["rms_window"])
        min_window = str(ste_params["min_window"])
        min_gap = str(ste_params["min_gap"])
        epoch_len = str(ste_params["epoch_len"])
        min_osc = str(ste_params["min_osc"])
        rms_thres = str(ste_params["rms_thres"])
        peak_thres = str(ste_params["peak_thres"])

        self.window.ste_rms_window_input.setText(rms_window)
        self.window.ste_min_window_input.setText(min_window)
        self.window.ste_min_gap_input.setText(min_gap)
        self.window.ste_epoch_length_input.setText(epoch_len)
        self.window.ste_min_oscillation_input.setText(min_osc)
        self.window.ste_rms_threshold_input.setText(rms_thres)
        self.window.ste_peak_threshold_input.setText(peak_thres)

        # set display parameters
        self.window.ste_epoch_display.setText(epoch_len)
        self.window.ste_min_window_display.setText(min_window)
        self.window.ste_rms_window_display.setText(rms_window)
        self.window.ste_min_gap_time_display.setText(min_gap)
        self.window.ste_min_oscillations_display.setText(min_osc)
        self.window.ste_peak_threshold_display.setText(peak_thres)
        self.window.ste_rms_threshold_display.setText(rms_thres)

        self.update_detector_tab("STE")
        self.window.detector_subtabs.setCurrentIndex(0)

    def update_mni_params(self, mni_params):
        epoch_time = str(mni_params["epoch_time"])
        epo_CHF = str(mni_params["epo_CHF"])
        per_CHF = str(mni_params["per_CHF"])
        min_win = str(mni_params["min_win"])
        min_gap = str(mni_params["min_gap"])
        thrd_perc = str(mni_params["thrd_perc"])
        base_seg = str(mni_params["base_seg"])
        base_shift = str(mni_params["base_shift"])
        base_thrd = str(mni_params["base_thrd"])
        base_min = str(mni_params["base_min"])

        self.window.mni_epoch_time_input.setText(epoch_time)
        self.window.mni_epoch_chf_input.setText(epo_CHF)
        self.window.mni_chf_percentage_input.setText(per_CHF)
        self.window.mni_min_window_input.setText(min_win)
        self.window.mni_min_gap_time_input.setText(min_gap)
        self.window.mni_threshold_percentage_input.setText(thrd_perc)
        self.window.mni_baseline_window_input.setText(base_seg)
        self.window.mni_baseline_shift_input.setText(base_shift)
        self.window.mni_baseline_threshold_input.setText(base_thrd)
        self.window.mni_baseline_min_time_input.setText(base_min)

        # set display parameters
        self.window.mni_epoch_display.setText(epoch_time)
        self.window.mni_epoch_chf_display.setText(epo_CHF)
        self.window.mni_chf_percentage_display.setText(per_CHF)
        self.window.mni_min_window_display.setText(min_win)
        self.window.mni_min_gap_time_display.setText(min_gap)
        self.window.mni_threshold_percentile_display.setText(thrd_perc)
        self.window.mni_baseline_window_display.setText(base_seg)
        self.window.mni_baseline_shift_display.setText(base_shift)
        self.window.mni_baseline_threshold_display.setText(base_thrd)
        self.window.mni_baseline_min_time_display.setText(base_min)

        self.update_detector_tab("MNI")
        self.window.detector_subtabs.setCurrentIndex(1)

    def update_hil_params(self, hil_params):
        sample_freq = str(hil_params["sample_freq"])
        pass_band = str(hil_params["pass_band"])
        stop_band = str(hil_params["stop_band"])
        epoch_time = str(hil_params["epoch_time"])
        sd_threshold = str(hil_params["sd_threshold"])
        min_window = str(hil_params["min_window"])

        self.window.hil_sample_freq_input.setText(sample_freq)
        self.window.hil_pass_band_input.setText(pass_band)
        self.window.hil_stop_band_input.setText(stop_band)
        self.window.hil_epoch_time_input.setText(epoch_time)
        self.window.hil_sd_threshold_input.setText(sd_threshold)
        self.window.hil_min_window_input.setText(min_window)

        # set display parameters
        self.window.hil_sample_freq_display.setText(sample_freq)
        self.window.hil_pass_band_display.setText(pass_band)
        self.window.hil_stop_band_display.setText(stop_band)
        self.window.hil_epoch_time_display.setText(epoch_time)
        self.window.hil_sd_threshold_display.setText(sd_threshold)
        self.window.hil_min_window_display.setText(min_window)

        self.update_detector_tab("HIL")
        self.window.detector_subtabs.setCurrentIndex(2)

    def set_detector_param_display(self):
        if self.backend is None or self.backend.param_detector is None:
            self.refresh_detector_mode_ui()
            return
        detector_params = self.backend.param_detector
        detector_type = detector_params.detector_type.lower()
        if detector_type == "ste":
            self.update_ste_params(detector_params.detector_param.to_dict())
        elif detector_type == "mni":
            self.update_mni_params(detector_params.detector_param.to_dict())
        elif detector_type == "hil":
            self.update_hil_params(detector_params.detector_param.to_dict())
        elif detector_type == "yasa":
            self.window.detector_subtabs.setCurrentIndex(0)
        self.refresh_detector_mode_ui()

    def open_bipolar_channel_selection(self):
        if self.backend is None:
            self.handle_unsupported_biomarker_mode("Bipolar channel creation is not available for the current biomarker mode yet.")
            return
        self.window.bipolar_channel_selection_window = BipolarChannelSelectionWindow(self,
                                                                                     self.backend,
                                                                                     self.window,
                                                                                     self.window.close_signal,
                                                                                     self.window.waveform_plot)
        self.window.bipolar_channel_selection_window.show()

    def open_annotation(self):
        if self.backend is None:
            self.handle_unsupported_biomarker_mode("Review workspace is not available for the current biomarker mode yet.")
            return
        if self.backend.event_features is None or self.backend.event_features.get_num_biomarker() == 0:
            message = "There are no events in the active run to review yet."
            self.message_handler(message)
            self._set_workflow_message(message)
            QMessageBox.information(self.window, "No Events To Review", message)
            self.window.annotation_button.setEnabled(False)
            return
        self.window.save_csv_button.setEnabled(True)
        self._set_report_export_enabled(True)
        # Ensure prediction arrays are properly sized before opening annotation
        self._ensure_prediction_arrays_are_sized()
        annotation = Annotation(self.backend, self.window, self.window.close_signal)
        safe_connect_signal_slot(annotation.destroyed, self._annotation_window_closed)
        annotation.show()
        self._set_workflow_message("Annotation workspace opened")

    def _annotation_window_closed(self, *_args):
        if not hasattr(self, "backend"):
            return
        if not self._ui_object_is_alive(getattr(self, "window", None)):
            return
        self.update_statistics_label()

    def handle_unsupported_biomarker_mode(self, message, show_dialog=True):
        self.message_handler(message)
        self._set_workflow_message(message)
        if show_dialog:
            QMessageBox.information(self.window, "Mode In Progress", message)
