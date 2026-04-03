from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5 import uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import sys
import os
from pathlib import Path
from src.hfo_app import HFO_App
from src.param.param_classifier import ParamClassifier
from src.param.param_detector import ParamDetector, ParamSTE, ParamMNI, ParamHIL
from src.param.param_filter import ParamFilter
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
        self.ui = uic.loadUi(os.path.join(ROOT_DIR, 'quick_detection.ui'), self)
        self.setWindowTitle("HFO Quick Detector")
        self.setWindowIcon(QtGui.QIcon(os.path.join(ROOT_DIR, 'images/icon1.png')))
        self.filename = None
        self.threadpool = QThreadPool()
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

        #set n_jobs min and max
        self.n_jobs_spinbox.setMinimum(1)
        self.n_jobs_spinbox.setMaximum(mp.cpu_count())

        #set default n_jobs
        self.n_jobs_spinbox.setValue(self.backend.n_jobs)

        self.main_window = main_window
        self._run_result = None
        self._run_failed = False

        #classifier default buttons
        safe_connect_signal_slot(self.default_cpu_button.clicked, self.set_classifier_param_cpu_default)
        safe_connect_signal_slot(self.default_gpu_button.clicked, self.set_classifier_param_gpu_default)
        if torch is None or not torch.cuda.is_available():
            self.default_gpu_button.setEnabled(False)
        if self.backend.get_classifier_param() is None:
            self.backend.set_default_cpu_classifier()
        self.set_classifier_param_display()
        self.qd_npz_checkbox.setChecked(True)

        safe_connect_signal_slot(self.cancel_button.clicked, self.close)
        self.running = False

        self.close_signal = close_signal
        if self.close_signal is not None:
            safe_connect_signal_slot(self.close_signal, self.close)

        # Disable the full configuration area while the worker is active.
        self.controls_to_disable = [self.scrollArea]
        self._apply_dialog_theme()

    def _apply_dialog_theme(self):
        self.resize(940, 820)
        self.qd_filename.setWordWrap(True)
        for label, width in (
            (self.qd_filename, 320),
            (self.qd_sampfreq, 140),
            (self.qd_numchannels, 140),
            (self.qd_length, 140),
        ):
            style_value_badge(label, min_width=width, selectable=True)
        self.qd_filename.setText("No recording loaded")
        self.qd_sampfreq.setText("--")
        self.qd_numchannels.setText("--")
        self.qd_length.setText("--")
        set_accent_button(self.run_button)
        cancel_button = self.cancel_button.button(QtWidgets.QDialogButtonBox.Cancel)
        if cancel_button is not None:
            cancel_button.setText("Close")
        apply_subwindow_theme(self)

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
        self.qd_filename.setText(results[0])
        self.qd_sampfreq.setText(results[1])
        self.qd_numchannels.setText(results[2])
        self.qd_length.setText(results[3])
        self.run_button.setEnabled(True)


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

        # filter stuff
    def init_default_filter_input_params(self):
        default_params=ParamFilter()
        self.qd_fp_input.setText(str(default_params.fp))
        self.qd_fs_input.setText(str(default_params.fs))
        self.qd_rp_input.setText(str(default_params.rp))
        self.qd_rs_input.setText(str(default_params.rs))

    def get_filter_param(self):
        fp = self.qd_fp_input.text()
        fs = self.qd_fs_input.text()
        rp = self.qd_rp_input.text()
        rs = self.qd_rs_input.text()
        return ParamFilter().from_dict({"fp": float(fp), "fs": float(fs), "rp": float(rp), "rs": float(rs)})

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
                    "epoch_time":float(epoch_time), "epo_CHF":float(epo_CHF), "per_CHF":float(per_CHF),
                    "min_win":float(min_win), "min_gap":float(min_gap), "base_seg":float(base_seg),
                    "thrd_perc":float(thrd_perc)/100,
                    "base_shift":float(base_shift), "base_thrd":float(base_thrd), "base_min":float(base_min),
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
                    "rms_window":float(rms_window_raw), "min_window":float(min_window_raw), "min_gap":float(min_gap_raw),
                    "epoch_len":float(epoch_len_raw), "min_osc":float(min_osc_raw), "rms_thres":float(rms_thres_raw),
                    "peak_thres":float(peak_thres_raw),"n_jobs":self.backend.n_jobs}
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
            "sample_freq": float(sample_freq_raw),
            "pass_band": float(pass_band_raw),
            "stop_band": float(stop_band_raw),
            "epoch_time": float(epoch_time_raw),
            "sd_threshold": float(sd_threshold_raw),
            "min_window": float(min_window_raw),
            "n_jobs": self.backend.n_jobs
        }
        detector_params = {"detector_type": "HIL", "detector_param": param_dict}
        return ParamDetector.from_dict(detector_params)
        
    def get_classifier_param(self):
        artifact_path = self.qd_classifier_artifact_filename_display.text()
        spike_path = self.qd_classifier_spike_filename_display.text()
        ehfo_path = self.qd_classifier_ehfo_filename_display.text()
        use_spike = self.qd_use_spikes_checkbox.isChecked()
        use_ehfo = self.qd_use_ehfo_checkbox.isChecked()
        device, model_type = self.normalize_classifier_device(self.qd_classifier_device_input.text())
        batch_size = int(self.qd_classifier_batch_size_input.text())

        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")
        if not artifact_path:
            raise ValueError("Artifact model file is required when classifier is enabled.")
        if use_spike and not spike_path:
            raise ValueError("Spike model file is required when spkHFO is enabled.")
        if use_ehfo and not ehfo_path:
            raise ValueError("eHFO model file is required when eHFO is enabled.")

        classifier_param = ParamClassifier(artifact_path=artifact_path, spike_path=spike_path, ehfo_path=ehfo_path,
                                           use_spike=use_spike, use_ehfo=use_ehfo,
                                           device=device, batch_size=batch_size, model_type=model_type)

        seconds_before = float(self.qd_ignore_sec_before_input.text())
        seconds_after = float(self.qd_ignore_sec_after_input.text())
        if seconds_before < 0 or seconds_after < 0:
            raise ValueError("Ignore windows must be non-negative.")
        return {"classifier_param": classifier_param, "use_spike": use_spike, "use_ehfo": use_ehfo,
                "seconds_before": seconds_before, "seconds_after": seconds_after}
    
    def set_classifier_param_display(self):
        classifier_param = self.backend.get_classifier_param()

        #set also the input fields
        self.qd_classifier_artifact_filename_display.setText(classifier_param.artifact_path or "")
        self.qd_classifier_spike_filename_display.setText(classifier_param.spike_path or "")
        self.qd_classifier_ehfo_filename_display.setText(classifier_param.ehfo_path or "")
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
        if model_type == "artifact":
            self.qd_classifier_artifact_filename_display.setText(fname)
        elif model_type == "spike":
            self.qd_classifier_spike_filename_display.setText(fname)
        elif model_type == "ehfo":
            self.qd_classifier_ehfo_filename_display.setText(fname)

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
        return str(recording_path.with_name(f"{self.get_output_stem()}{extension}"))

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
        safe_connect_signal_slot(worker.signals.result, self._run_finished)
        safe_connect_signal_slot(worker.signals.error, self._run_error)
        safe_connect_signal_slot(worker.signals.finished, self._run_cleanup)
        self.threadpool.start(worker)

    def _run_finished(self, result=None):
        self._run_result = result or {}
        self._run_failed = False
        self.setWindowTitle("Quick Detection - Complete")

    def _run_error(self, error_tuple):
        self._run_result = None
        self._run_failed = True
        self.setWindowTitle("Quick Detection - Failed")
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
