from pathlib import Path
from queue import Queue
import os
import tempfile
from datetime import datetime

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
from src.spindle_app import SpindleApp
from src.ui.channels_selection import ChannelSelectionWindow
from src.param.param_classifier import ParamClassifier
from src.param.param_detector import ParamDetector, ParamSTE, ParamMNI, ParamHIL, ParamYASA
from src.param.param_filter import ParamFilter, ParamFilterSpindle
from src.ui.bipolar_channel_selection import BipolarChannelSelectionWindow
from src.ui.annotation import Annotation
from src.utils.utils_gui import *
from src.ui.plot_waveform import *
from src.utils.app_state import build_base_checkpoint, checkpoint_array, checkpoint_get, checkpoint_version
from src.utils.session_store import load_session_checkpoint
from src.utils.utils_detector import has_yasa
from src.utils.reporting import export_analysis_report
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

    def update_status_indicators(self):
        has_file = bool(self.backend and self.backend.eeg_data is not None)
        selected_channels = 0
        if hasattr(self.window, "waveform_plot") and has_file:
            selected_channels = len(self.window.waveform_plot.get_channels_to_plot())
        has_channels = selected_channels > 0
        has_filter = bool(self.backend and hasattr(self.backend, "has_filtered_data") and self.backend.has_filtered_data())
        has_detect = bool(self.backend and getattr(self.backend, "detected", False))
        has_classify = bool(self.backend and getattr(self.backend, "classified", False))
        has_annotations = bool(
            has_detect
            and self.backend.event_features is not None
            and np.any(np.array(self.backend.event_features.annotated) > 0)
        )
        self._set_status_chip("status_loaded_chip", has_file, "Loaded" if has_file else "Load data")
        self._set_status_chip("status_channels_chip", has_channels, f"{selected_channels} channels" if has_channels else "Choose channels")
        self._set_status_chip("status_filter_chip", has_filter, "Filter ready" if has_filter else "Raw signal")
        self._set_status_chip("status_detect_chip", has_detect, "Detection complete" if has_detect else "Detect events")
        self._set_status_chip("status_classify_chip", has_classify, "Classified" if has_classify else "Classification optional")
        self._set_status_chip("status_annotate_chip", has_annotations, "Annotated" if has_annotations else "Review & annotate")
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

    def update_waveform_toolbar_state(self):
        has_recording = bool(self.backend and getattr(self.backend, "eeg_data", None) is not None)
        features = self._get_active_event_features()
        total_events = len(getattr(features, "starts", [])) if features is not None else 0

        if hasattr(self.window, "normalize_tool_button"):
            self.window.normalize_tool_button.setEnabled(has_recording)
            self._set_button_checked(self.window.normalize_tool_button, self.window.normalize_vertical_input.isChecked())
        if hasattr(self.window, "filtered_tool_button"):
            self.window.filtered_tool_button.setEnabled(self.window.toggle_filtered_checkbox.isEnabled())
            self._set_button_checked(self.window.filtered_tool_button, self.window.toggle_filtered_checkbox.isChecked())
        if hasattr(self.window, "filter60_tool_button"):
            self.window.filter60_tool_button.setEnabled(self.window.Filter60Button.isEnabled())
            self._set_button_checked(self.window.filter60_tool_button, self.window.Filter60Button.isChecked())
        if hasattr(self.window, "review_channels_button"):
            self.window.review_channels_button.setEnabled(self.window.Choose_Channels_Button.isEnabled())
        if hasattr(self.window, "montage_tool_button"):
            self.window.montage_tool_button.setEnabled(self.window.bipolar_button.isEnabled())
        if hasattr(self.window, "event_channels_button"):
            self.window.event_channels_button.setEnabled(total_events > 0)
        if hasattr(self.window, "all_channels_button"):
            self.window.all_channels_button.setEnabled(has_recording)
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
                current_index = min(int(getattr(features, "index", 0)) + 1, total_events)
                current_info = {}
                if hasattr(features, "get_current_info"):
                    try:
                        current_info = features.get_current_info() or {}
                    except Exception:
                        current_info = {}
                channel_name = current_info.get("channel_name")
                if channel_name:
                    self.window.event_position_label.setText(f"{current_index}/{total_events}  {channel_name}")
                else:
                    self.window.event_position_label.setText(f"{current_index}/{total_events}")
            else:
                self.window.event_position_label.setText("")

    def _connect_worker(self, worker, task_name, result_handler=None, finished_handler=None):
        safe_connect_signal_slot(worker.signals.error, lambda err: self.handle_worker_error(task_name, err))
        if result_handler is not None:
            safe_connect_signal_slot(worker.signals.result, result_handler)
        if finished_handler is not None:
            safe_connect_signal_slot(worker.signals.finished, finished_handler)
        self.window.threadpool.start(worker)

    def handle_worker_error(self, task_name, error_tuple):
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
            self.window.waveform_graphics_widget.ci.layout.setRowStretchFactor(0, 10)
            self.window.waveform_graphics_widget.ci.layout.setRowStretchFactor(1, 2)
        except Exception:
            pass

        main_plot_item = self.window.waveform_graphics_widget.addPlot(row=0, col=0)
        mini_plot_item = self.window.waveform_graphics_widget.addPlot(row=1, col=0)
        try:
            mini_plot_item.setMaximumHeight(76)
            mini_plot_item.setMinimumHeight(56)
        except Exception:
            pass

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
            # get filter parameters
            fp_raw = self.window.fp_input.text()
            fs_raw = self.window.fs_input.text()
            rp_raw = self.window.rp_input.text()
            rs_raw = self.window.rs_input.text()
            # self.pop_window()
            param_dict = {"fp": float(fp_raw), "fs": float(fs_raw), "rp": float(rp_raw), "rs": float(rs_raw)}
            filter_param = self._filter_param_class().from_dict(param_dict)
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
        worker = Worker(self._filter)
        self._connect_worker(worker, "Filtering", finished_handler=self.filtering_complete)

    def create_center_waveform_and_mini_plot(self):
        self.window.channels_to_plot = []
        self.window.waveform_plot = CenterWaveformAndMiniPlotController(self.window.waveform_plot_widget,
                                                                        self.window.waveform_mini_widget,
                                                                        self.backend)

        # part of “clear everything if exit”, optimize in the future
        safe_connect_signal_slot(self.window.waveform_time_scroll_bar.valueChanged, self.scroll_time_waveform_plot)
        safe_connect_signal_slot(self.window.channel_scroll_bar.valueChanged, self.scroll_channel_waveform_plot)
        self.window.waveform_time_scroll_bar.valueChanged.disconnect(self.scroll_time_waveform_plot)
        self.window.channel_scroll_bar.valueChanged.disconnect(self.scroll_channel_waveform_plot)

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
        if self.biomarker_type == 'HFO':
            default_params = ParamFilter()
            self.window.fp_input.setText(str(default_params.fp))
            self.window.fs_input.setText(str(default_params.fs))
            self.window.rp_input.setText(str(default_params.rp))
            self.window.rs_input.setText(str(default_params.rs))
        elif self.biomarker_type == 'Spindle':
            default_params = ParamFilterSpindle()
            self.window.fp_input.setText(str(default_params.fp))
            self.window.fs_input.setText(str(default_params.fs))
            self.window.rp_input.setText(str(default_params.rp))
            self.window.rs_input.setText(str(default_params.rs))

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

        safe_connect_signal_slot(self.window.save_npz_button.clicked, self.save_to_npz)
        self.window.save_npz_button.setEnabled(False)

        safe_connect_signal_slot(self.window.Filter60Button.toggled, self.switch_60)
        self.window.Filter60Button.setEnabled(False)
        safe_connect_signal_slot(self.window.toggle_filtered_checkbox.stateChanged, self.toggle_filtered)
        safe_connect_signal_slot(self.window.normalize_vertical_input.stateChanged, self.waveform_plot_button_clicked)

        safe_connect_signal_slot(self.window.bipolar_button.clicked, self.open_bipolar_channel_selection)
        self.window.bipolar_button.setEnabled(False)
        safe_connect_signal_slot(self.window.waveform_plot_button.clicked, self.waveform_plot_button_clicked)
        safe_connect_signal_slot(self.window.Choose_Channels_Button.clicked, self.open_channel_selection)
        if hasattr(self.window, "review_channels_button"):
            safe_connect_signal_slot(self.window.review_channels_button.clicked, self.open_channel_selection)
        if hasattr(self.window, "montage_tool_button"):
            safe_connect_signal_slot(self.window.montage_tool_button.clicked, self.open_bipolar_channel_selection)
        if hasattr(self.window, "event_channels_button"):
            safe_connect_signal_slot(self.window.event_channels_button.clicked, self.focus_event_channels)
        if hasattr(self.window, "all_channels_button"):
            safe_connect_signal_slot(self.window.all_channels_button.clicked, self.show_all_channels)
        if hasattr(self.window, "open_review_button"):
            safe_connect_signal_slot(self.window.open_review_button.clicked, self.open_annotation)
        if hasattr(self.window, "normalize_tool_button"):
            safe_connect_signal_slot(
                self.window.normalize_tool_button.toggled,
                lambda checked: self.window.normalize_vertical_input.setChecked(checked),
            )
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
        self.update_waveform_toolbar_state()
        self.update_setup_action_state()

    def set_classifier_param_display(self):
        if self.backend is None:
            return
        classifier_param = self.backend.get_classifier_param()
        if classifier_param is None:
            self.refresh_classifier_mode_ui()
            return

        if classifier_param.artifact_card:
            self.window.overview_artifact_path_display.setText(classifier_param.artifact_card)
        elif classifier_param.artifact_path:
            self.window.overview_artifact_path_display.setText(classifier_param.artifact_path)

        if classifier_param.spike_card:
            self.window.overview_spike_path_display.setText(classifier_param.spike_card)
        elif classifier_param.spike_path:
            self.window.overview_spike_path_display.setText(classifier_param.spike_path)

        if classifier_param.ehfo_card:
            self.window.overview_ehfo_path_display.setText(classifier_param.ehfo_card)
        elif classifier_param.ehfo_path:
            self.window.overview_ehfo_path_display.setText(classifier_param.ehfo_path)

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
            return
        # local
        artifact_path = self.window.classifier_artifact_filename.text()
        spike_path = self.window.classifier_spike_filename.text()
        ehfo_path = self.window.classifier_ehfo_filename.text()

        # hugging face
        artifact_card = self.window.classifier_artifact_card_name.text()
        spike_card = self.window.classifier_spike_card_name.text()
        ehfo_card = self.window.classifier_ehfo_card_name.text()

        use_spike = self.window.use_spike_checkbox.isChecked()
        use_ehfo = self.window.use_ehfo_checkbox.isChecked()
        device = self.window.classifier_device_input.text()
        if device == "cpu":
            model_type = "default_cpu"
        elif device == "cuda:0" and self.gpu:
            model_type = "default_gpu"
        else:
            # print("device not recognized, please set to cpu for cpu or cuda:0 for gpu")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error!")
            msg.setInformativeText('Device not recognized, please set to CPU for CPU or cuda:0 for GPU')
            msg.setWindowTitle("Device not recognized")
            msg.exec_()
            return
        batch_size = self.window.classifier_batch_size_input.text()

        classifier_param = ParamClassifier(artifact_path=artifact_path, spike_path=spike_path, ehfo_path=ehfo_path,
                                           artifact_card=artifact_card, spike_card=spike_card, ehfo_card=ehfo_card,
                                           use_spike=use_spike, use_ehfo=use_ehfo,
                                           device=device, batch_size=int(batch_size), model_type=model_type)
        self.backend.set_classifier(classifier_param)
        self.set_classifier_param_display()

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
        seconds_to_ignore_before = float(self.window.overview_ignore_before_input.text())
        seconds_to_ignore_after = float(self.window.overview_ignore_after_input.text())
        self.backend.classify_artifacts([seconds_to_ignore_before, seconds_to_ignore_after], threshold)

        use_spike = self.window.overview_use_spike_checkbox.isChecked()
        use_ehfo = self.window.overview_use_ehfo_checkbox.isChecked()
        if use_spike:
            self.backend.classify_spikes()
        if use_ehfo and hasattr(self.backend, "classify_ehfos"):
            self.backend.classify_ehfos()
        return []

    def _classify_finished(self):
        self.message_handler("Classification finished!..")
        self._set_workflow_message("Classification complete")
        self.update_statistics_label()
        self.window.waveform_plot.set_plot_biomarkers(True)
        self.window.save_csv_button.setEnabled(True)
        self._set_report_export_enabled(True)
        self.update_status_indicators()

    def classify(self):
        self.message_handler(f"Classifying {self.get_biomarker_display_name()} events...")
        self._set_workflow_message("Running classifiers...")
        worker = Worker(lambda progress_callback: self._classify())
        self._connect_worker(worker, "Classification", result_handler=self._classify_finished)

    def update_statistics_label(self):
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
        if self.backend is None:
            self.window.run_summary_label.setText("No detection runs yet.")
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

        runs = self._collect_case_run_summaries()
        session = getattr(self.backend, "analysis_session", None)
        active_run = session.get_active_run() if session is not None else None
        accepted_run = session.get_accepted_run() if session is not None else None
        if not runs:
            self.window.run_summary_label.setText("No detection runs yet.")
            self.window.run_summary_label.setToolTip("Detector settings are ready for the next run.")
            if hasattr(self.window, "analysis_summary_label"):
                self.window.analysis_summary_label.setText("Detector settings are ready for the next run")
                self.window.analysis_summary_label.setToolTip("No saved runs yet. Current detector settings will be used for the next detection.")
            self.window.switch_run_button.setEnabled(False)
            self.window.accept_run_button.setEnabled(False)
            self.window.compare_runs_button.setEnabled(True)
            if hasattr(self.window, "run_stats_activate_button"):
                self.window.run_stats_activate_button.setEnabled(False)
            if hasattr(self.window, "run_stats_accept_button"):
                self.window.run_stats_accept_button.setEnabled(False)
            if hasattr(self.window, "run_stats_export_button"):
                self.window.run_stats_export_button.setEnabled(hasattr(self.backend, "export_clinical_summary") or hasattr(self.backend, "export_excel"))
            self._set_report_export_enabled(active_run is not None or accepted_run is not None)
            self._sync_active_run_selector([], active_run)
            self.update_active_run_panel(active_run, accepted_run)
            self.populate_decision_tables([], [], [])
            return

        visible_count = sum(1 for run in runs if run.get("visible"))
        summary_parts = [f"{len(runs)} runs", f"{visible_count} visible"]
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
        self.window.switch_run_button.setEnabled(len(runs) > 1)
        self.window.accept_run_button.setEnabled(active_run is not None)
        self.window.compare_runs_button.setEnabled(True)
        if hasattr(self.window, "run_stats_activate_button"):
            self.window.run_stats_activate_button.setEnabled(bool(runs))
        if hasattr(self.window, "run_stats_accept_button"):
            self.window.run_stats_accept_button.setEnabled(active_run is not None)
        if hasattr(self.window, "run_stats_export_button"):
            self.window.run_stats_export_button.setEnabled(hasattr(self.backend, "export_clinical_summary") or hasattr(self.backend, "export_excel"))
        self._set_report_export_enabled(active_run is not None or accepted_run is not None)
        self._sync_active_run_selector(runs, active_run)
        self.update_active_run_panel(active_run, accepted_run)
        comparison_rows = []
        if hasattr(self.backend, "compare_runs"):
            comparison_rows = self.backend.compare_runs().get("pairwise_overlap", [])
        self.populate_decision_tables(runs, ranking if 'ranking' in locals() else [], comparison_rows)

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

    def _collect_case_run_summaries(self):
        summaries = []
        for biomarker_type, backend in self.case_backends.items():
            if backend is None or not hasattr(backend, "get_run_summaries"):
                continue
            for run in backend.get_run_summaries():
                enriched = dict(run)
                enriched["biomarker_type"] = biomarker_type
                summaries.append(enriched)
        summaries.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return summaries

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
                    item_row.get("left_detector", ""),
                    item_row.get("right_detector", ""),
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
                classifier_run.setText("Run Classification")
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
        handler()
        after_type = ""
        if getattr(self.backend, "param_detector", None) is not None:
            after_type = str(self.backend.param_detector.detector_type).upper()
        success = after_type == detector_name and bool(after_type)
        if success and show_feedback:
            self.message_handler(f"{detector_name} parameters applied")
            self._set_workflow_message(f"{detector_name} settings ready")
        elif not success and show_feedback and before_type == after_type and not after_type:
            self._set_workflow_message("Detector settings need attention")
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
        mode = combo.currentText() if combo is not None and combo.count() > 0 else "Default CPU"
        try:
            if mode == "Custom":
                self.set_custom_classifier_param()
            else:
                self._sync_classifier_execution_inputs()
                classifier_param = self.backend.get_classifier_param() if hasattr(self.backend, "get_classifier_param") else None
                if classifier_param is None:
                    if mode == "Default GPU":
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

    def apply_classifier_mode(self, index):
        combo = getattr(self.window, "classifier_mode_combo", None)
        if combo is None or index < 0:
            return
        mode = combo.currentText()
        if self.biomarker_type == "Spike":
            self._set_classifier_custom_sources_visible(False)
            self.window.detect_all_button.setEnabled(False)
            return
        if mode == "Default CPU":
            self.set_classifier_param_cpu_default()
            self._set_classifier_custom_sources_visible(False)
        elif mode == "Default GPU":
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
        target_label = "Default CPU"
        if classifier_param is not None:
            model_type = str(getattr(classifier_param, "model_type", "") or "").lower()
            has_custom_sources = any(
                bool(getattr(classifier_param, attr, ""))
                for attr in ("artifact_path", "spike_path", "ehfo_path", "artifact_card", "spike_card", "ehfo_card")
            )
            if has_custom_sources and model_type not in {"default_cpu", "default_gpu"}:
                target_label = "Custom"
            elif model_type == "default_gpu":
                target_label = "Default GPU"
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
        channel_name = channel_names[row]
        if hasattr(self.window, "waveform_plot"):
            self.set_channels_to_plot([channel_name], display_all=True)
            self.message_handler(f"Focused waveform review on channel: {channel_name}")
            self._set_workflow_message(f"Focused on channel {channel_name}")

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
            self.window.waveform_plot.set_plot_biomarkers(bool(self.backend and self.backend.detected))
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
        total_events = len(getattr(features, "starts", [])) if features is not None else 0
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
        self.message_handler(f"Centered on {self.get_biomarker_display_name()} event {features.index + 1} of {total_events}")

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

    def focus_event_channels(self):
        features = self._get_active_event_features()
        if features is None or len(getattr(features, "channel_names", [])) == 0:
            self.message_handler("No detected events available to define a review channel set")
            return
        ordered_channels = list(dict.fromkeys([str(channel) for channel in features.channel_names]))
        if not ordered_channels:
            self.message_handler("No detected events available to define a review channel set")
            return
        self.set_channels_to_plot(ordered_channels, display_all=False)
        self.window.channel_scroll_bar.setValue(0)
        self.message_handler(f"Focused review on {len(ordered_channels)} channels that contain detected events")
        self._set_workflow_message("Reviewing channels with detected events")

    def show_all_channels(self):
        if self.backend is None or getattr(self.backend, "channel_names", None) is None:
            return
        all_channels = [str(channel) for channel in self.backend.channel_names]
        if not all_channels:
            return
        self.set_channels_to_plot(all_channels, display_all=False)
        self.window.channel_scroll_bar.setValue(0)
        self.message_handler(f"Restored the full review channel list ({len(all_channels)} channels)")
        self._set_workflow_message("Reviewing the full channel list")

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
        if hasattr(self.window, "run_stats_dialog"):
            self.window.run_stats_dialog.show()
            self.window.run_stats_dialog.raise_()
            self.window.run_stats_dialog.activateWindow()
        if hasattr(self.backend, "compare_runs"):
            comparison = self.backend.compare_runs()
            pairwise = comparison.get("pairwise_overlap", [])
            if pairwise:
                self.message_handler("Opened run statistics with detector agreement")
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
            if hasattr(self.backend, "export_clinical_summary"):
                self.backend.export_clinical_summary(fname)
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
            filter_param = self.backend.param_filter
            # Update filter params
            self.window.fp_input.setText(str(filter_param.fp))
            self.window.fs_input.setText(str(filter_param.fs))
            self.window.rp_input.setText(str(filter_param.rp))
            self.window.rs_input.setText(str(filter_param.rs))
        
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

        if self.backend.filtered and getattr(self.backend, "param_filter", None) is not None:
            filter_param = self.backend.param_filter
            self.window.fp_input.setText(str(filter_param.fp))
            self.window.fs_input.setText(str(filter_param.fs))
            self.window.rp_input.setText(str(filter_param.rp))
            self.window.rs_input.setText(str(filter_param.rs))
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
            self.window.waveform_plot.set_plot_biomarkers(True)
            has_events = (
                self.backend.event_features is not None
                and self.backend.event_features.get_num_biomarker() > 0
            )
            self.window.annotation_button.setEnabled(has_events)
            self.window.save_csv_button.setEnabled(True)
            self._set_report_export_enabled(True)
            self.update_statistics_label()
        else:
            self.window.waveform_plot.set_plot_biomarkers(False)

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
                nyquist = max(2.0, float(self.backend.sample_freq) / 2.0)
                stop_band = min(500.0, max(20.0, nyquist - 1.0))
                pass_band = min(80.0, max(10.0, stop_band - max(10.0, nyquist * 0.15)))
                if pass_band >= stop_band:
                    pass_band = max(1.0, stop_band - 5.0)
                self.backend.set_filter_parameter(
                    ParamFilter(
                        fp=pass_band,
                        fs=stop_band,
                        sample_freq=self.backend.sample_freq,
                    )
                )

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
                self.backend.set_filter_parameter(ParamFilterSpindle())

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
            level_dot.setStyleSheet(f"background: {color}; border-radius: 5px;")

    @pyqtSlot(object)
    def update_edf_info(self, results):
        if hasattr(self.window, "waveform_plot"):
            self.window.waveform_plot.update_backend(self.backend, False)
            self.window.waveform_plot.init_eeg_data()
        self.case_backends[self.biomarker_type] = self.backend
        if getattr(self.backend, "edf_param", None):
            self.current_recording_path = self.backend.edf_param.get("edf_fn", self.current_recording_path)
        self.window.main_filename.setText(results[0])
        self.window.main_sampfreq.setText(results[1])
        self.window.sample_freq = float(results[1])
        self.window.main_numchannels.setText(results[2])
        # print("updated")
        self.window.main_length.setText(str(round(float(results[3]) / (60 * float(results[1])), 3)) + " min")
        # self.window.waveform_plot.plot(0, update_biomarker=True)
        self.window.waveform_plot.set_plot_biomarkers(False)

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
            self.window.show_filtered = not self.window.show_filtered
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
        c_value = self.window.channel_scroll_bar.value()
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
            self.window.waveform_plot.set_filtered(True)
            self.window.save_npz_button.setEnabled(True)
        elif self.biomarker_type == 'Spindle':
            self.window.is_data_filtered = True
            self.window.show_filtered = True
            self.window.waveform_plot.set_filtered(True)
            self.window.save_npz_button.setEnabled(True)
        self.update_status_indicators()

    def detect_HFOs(self):
        print("Detecting HFOs...")
        self._set_workflow_message(f"Running {self.get_biomarker_display_name()} detection...")
        worker = Worker(self._detect)
        self._connect_worker(worker, "Detection", result_handler=self._detect_finished)

    def detect_Spindles(self):
        print("Detecting Spindles...")
        self._set_workflow_message(f"Running {self.get_biomarker_display_name()} detection...")
        worker = Worker(self._detect)
        self._connect_worker(worker, "Detection", result_handler=self._detect_finished)

    def _detect_finished(self):
        # right now do nothing beyond message handler saying that
        # it has detected HFOs
        self.message_handler("Biomarker detected")
        self.update_statistics_label()
        self.window.waveform_plot.set_plot_biomarkers(True)
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
            num_events = len(self.backend.event_features.starts) if self.backend.event_features else 0
            self.message_handler(f"Auto-saved {num_events} {self.biomarker_type} events to: {npz_filename}")
            
        except Exception as e:
            self.message_handler(f"Warning: Auto-save failed: {str(e)}")

    def _offer_direct_annotation(self):
        """Offer to directly open annotation window after loading detection state"""
        if self.backend.detected and self.backend.event_features:
            num_events = len(self.backend.event_features.starts)
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
                          "rms_window": float(rms_window_raw), "min_window": float(min_window_raw),
                          "min_gap": float(min_gap_raw),
                          "epoch_len": float(epoch_len_raw), "min_osc": float(min_osc_raw),
                          "rms_thres": float(rms_thres_raw),
                          "peak_thres": float(peak_thres_raw), "n_jobs": self.backend.n_jobs}
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
        except (TypeError, ValueError) as exc:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error!")
            msg.setInformativeText(f'Detector could not be constructed given the parameters. {exc}')
            msg.setWindowTitle("Detector Construction Failed")
            msg.exec_()

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
                          "epoch_time": float(epoch_time), "epo_CHF": float(epo_CHF), "per_CHF": float(per_CHF),
                          "min_win": float(min_win), "min_gap": float(min_gap), "base_seg": float(base_seg),
                          "thrd_perc": float(thrd_perc) / 100,
                          "base_shift": float(base_shift), "base_thrd": float(base_thrd), "base_min": float(base_min),
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
        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error!")
            msg.setInformativeText('Detector could not be constructed given the parameters')
            msg.setWindowTitle("Detector Construction Failed")
            msg.exec_()

    def save_hil_params(self):
        try:
            sample_freq = self.window.hil_sample_freq_input.text()
            pass_band = self.window.hil_pass_band_input.text()
            stop_band = self.window.hil_stop_band_input.text()
            epoch_time = self.window.hil_epoch_time_input.text()
            sd_threshold = self.window.hil_sd_threshold_input.text()
            min_window = self.window.hil_min_window_input.text()

            param_dict = {
                "sample_freq": float(sample_freq),
                "pass_band": float(pass_band),
                "stop_band": float(stop_band),
                "epoch_time": float(epoch_time),
                "sd_threshold": float(sd_threshold),
                "min_window": float(min_window),
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

        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error!")
            msg.setInformativeText(f'HIL Detector could not be constructed given the parameters. Error: {str(e)}')
            msg.setWindowTitle("HIL Detector Construction Failed")
            msg.exec_()

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
            freq_sp = (float(freq_sp_low_raw), float(freq_sp_high_raw))
            freq_broad = (float(freq_broad_low_raw), float(freq_broad_high_raw))
            duration = (float(duration_low_raw), float(duration_high_raw))
            param_dict = {"sample_freq": 2000,
                          # these are placeholder params, will be updated later
                          "freq_sp": freq_sp, "freq_broad": freq_broad,
                          "duration": duration,
                          "min_distance": float(min_distance_raw), "rel_pow": float(thresh_rel_pow_raw),
                          "corr": float(thresh_corr_raw),
                          "rms": float(thresh_rms_raw), "n_jobs": self.backend.n_jobs}
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
        except (SyntaxError, ValueError, TypeError) as exc:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error!")
            msg.setInformativeText(f'Detector could not be constructed given the parameters. {exc}')
            msg.setWindowTitle("Detector Construction Failed")
            msg.exec_()

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
        safe_connect_signal_slot(annotation.destroyed, lambda *_: self.update_statistics_label())
        annotation.show()
        self._set_workflow_message("Annotation workspace opened")

    def handle_unsupported_biomarker_mode(self, message, show_dialog=True):
        self.message_handler(message)
        self._set_workflow_message(message)
        if show_dialog:
            QMessageBox.information(self.window, "Mode In Progress", message)
