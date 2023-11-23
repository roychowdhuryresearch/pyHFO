from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5 import uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import sys
import re
import os
from pathlib import Path
from src.hfo_app import HFO_App
from src.param.param_classifier import ParamClassifier
from src.param.param_detector import ParamDetector, ParamSTE, ParamMNI
from src.param.param_filter import ParamFilter
from src.utils.utils_gui import *

import multiprocessing as mp
import torch

ROOT_DIR = Path(__file__).parent


class HFOQuickDetector(QtWidgets.QDialog):
    def __init__(self, hfo_app=None, main_window=None, close_signal = None):
        super(HFOQuickDetector, self).__init__()
        # print("initializing HFOQuickDetector")
        self.ui = uic.loadUi(os.path.join(ROOT_DIR, 'quick_detection.ui'), self)
        self.setWindowTitle("HFO Quick Detector")
        self.setWindowIcon(QtGui.QIcon(os.path.join(ROOT_DIR, 'images/icon.png')))
        # print("loaded ui")
        self.filename = None
        self.threadpool = QThreadPool()
        self.detectionTypeComboBox.currentIndexChanged['int'].connect(
            lambda: self.update_detector_tab(self.detectionTypeComboBox.currentText()))  # type: ignore
        self.detectionTypeComboBox.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(self)
        # self.qd_loadEDF_button.clicked.connect(hfoMainWindow.Ui_MainWindow.openFile)
        self.qd_loadEDF_button.clicked.connect(self.open_file)
        #print("hfo_app: ", hfo_app)
        if hfo_app is None:
            #print("hfo_app is None creating new HFO_App")
            self.hfo_app = HFO_App()
        else:
            #print("hfo_app is not None")
            self.hfo_app = hfo_app
        self.init_default_filter_input_params()
        self.init_default_mni_input_params()
        self.init_default_ste_input_params()
        self.qd_choose_artifact_model_button.clicked.connect(lambda: self.choose_model_file("artifact"))
        self.qd_choose_spike_model_button.clicked.connect(lambda: self.choose_model_file("spike"))

        self.run_button.clicked.connect(self.run)
        self.run_button.setEnabled(False)

        #set n_jobs min and max
        self.n_jobs_spinbox.setMinimum(1)
        self.n_jobs_spinbox.setMaximum(mp.cpu_count())

        #set default n_jobs
        self.n_jobs_spinbox.setValue(self.hfo_app.n_jobs)

        self.main_window = main_window
        self.stdout = Queue()
        self.stderr = Queue()
        # sys.stdout = WriteStream(self.stdout)
        # sys.stderr = WriteStream(self.stderr)
        self.thread_stdout = STDOutReceiver(self.stdout)
        self.thread_stdout.std_received_signal.connect(self.main_window.message_handler)
        self.thread_stdout.start()
        # print("not here 2")
        self.thread_stderr = STDErrReceiver(self.stderr)
        self.thread_stderr.std_received_signal.connect(self.main_window.message_handler)
        self.thread_stderr.start()

        #classifier default buttons
        self.default_cpu_button.clicked.connect(self.set_classifier_param_cpu_default)
        self.default_gpu_button.clicked.connect(self.set_classifier_param_gpu_default)
        if not torch.cuda.is_available():
            self.default_gpu_button.setEnabled(False)

        self.cancel_button.clicked.connect(self.close)
        self.running = False
        # self.setWindowFlags( QtCore.Qt.CustomizeWindowHint )

        self.close_signal = close_signal
        self.close_signal.connect(self.close)

    def open_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open File", "", "EDF Files (*.edf)")
        if fname:
            worker = Worker(self.read_edf, fname)
            worker.signals.result.connect(self.update_edf_info)
            # worker.signals.finished.connect(lambda: self.message_handler('Open File thread COMPLETE!'))
            # worker.signals.progress.connect(self.progress_fn)
            # Execute
            self.threadpool.start(worker)

    def read_edf(self, fname, progress_callback):
        self.fname = fname
        self.hfo_app.load_edf(fname)
        eeg_data,channel_names=self.hfo_app.get_eeg_data()
        edf_info=self.hfo_app.get_edf_info()
        filename = os.path.basename(fname)
        self.filename = filename
        sample_freq = str(self.hfo_app.sample_freq)
        num_channels = str(len(self.hfo_app.channel_names))
        length = str(self.hfo_app.eeg_data.shape[1])
        return [filename, sample_freq, num_channels, length]
    
    @pyqtSlot(list)
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
                    "n_jobs":self.hfo_app.n_jobs}
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
                    "peak_thres":float(peak_thres_raw),"n_jobs":self.hfo_app.n_jobs}
        detector_params={"detector_type":"STE", "detector_param":param_dict}
        return ParamDetector.from_dict(detector_params)
    
    def get_classifier_param(self):
        artifact_path = self.qd_classifier_artifact_filename_display.text()
        spike_path = self.qd_classifier_spike_filename_display.text()
        use_spike = self.qd_use_spikes_checkbox.isChecked()
        device = self.qd_classifier_device_input.text()
        batch_size = self.qd_classifier_batch_size_input.text()

        classifier_param = ParamClassifier(artifact_path=artifact_path, spike_path=spike_path, use_spike=use_spike,
                                          device=device, batch_size=int(batch_size))
        
        seconds_before = float(self.qd_ignore_sec_before_input.text())
        seconds_after = float(self.qd_ignore_sec_after_input.text())
        return {"classifier_param":classifier_param,"use_spike":use_spike, "seconds_before":seconds_before, "seconds_after":seconds_after}
    
    def set_classifier_param_display(self):
        classifier_param = self.hfo_app.get_classifier_param()

        #set also the input fields
        self.qd_classifier_artifact_filename_display.setText(classifier_param.artifact_path)
        self.qd_classifier_spike_filename_display.setText(classifier_param.spike_path)
        self.qd_use_spikes_checkbox.setChecked(classifier_param.use_spike)
        self.qd_classifier_device_input.setText(str(classifier_param.device))
        self.qd_classifier_batch_size_input.setText(str(classifier_param.batch_size))

    def set_classifier_param_gpu_default(self):
        self.hfo_app.set_default_gpu_classifier()
        self.set_classifier_param_display()
    
    def set_classifier_param_cpu_default(self):
        self.hfo_app.set_default_cpu_classifier()
        self.set_classifier_param_display()

    def choose_model_file(self, model_type):
        fname,_  = QFileDialog.getOpenFileName(self, 'Open file', "", ".tar files (*.tar)")
        if model_type == "artifact":
            self.qd_classifier_artifact_filename_display.setText(fname)
        elif model_type == "spike":
            self.qd_classifier_spike_filename_display.setText(fname)
    
    def _detect(self, progress_callback):
        #call detect HFO function on backend
        self.hfo_app.detect_HFO()
        return []
    
    def detect_HFOs(self):
        # print("Detecting HFOs...")
        worker=Worker(self._detect)
        worker.signals.result.connect(self._detect_finished)
        self.threadpool.start(worker)

    # def _detect_finished(self):
    #     #right now do nothing beyond message handler saying that 
    #     # it has detected HFOs
    #     # self.message_handler("HFOs detected")

    def filter_data(self):
        # print("Filtering data...")
        worker=Worker(self._filter)
        worker.signals.finished.connect(self.filtering_complete)
        self.threadpool.start(worker)

    def _filter(self, progress_callback):
        self.hfo_app.filter_eeg_data(self.filter_params)

    # def filtering_complete(self):
    #     self.message_handler('Filtering COMPLETE!')

    def _classify(self,classify_spikes,seconds_to_ignore_before=0,seconds_to_ignore_after=0):
        self.hfo_app.classify_artifacts([seconds_to_ignore_before,seconds_to_ignore_after])
        if classify_spikes:
            self.hfo_app.classify_spikes()
        return []

    # def _classify_finished(self):
    #     self.message_handler("Classification finished!..")

    def classify(self,params):
        #set the parameters
        self.hfo_app.set_classifier(params["classifier_param"])
        seconds_to_ignore_before = params["seconds_before"]
        seconds_to_ignore_after = params["seconds_after"]
        self._classify(params["use_spike"],seconds_to_ignore_before,seconds_to_ignore_after)
        # worker.signals.result.connect(self._classify_finished)
        # self.threadpool.start(worker)


    def _run(self, progress_callback):
        self.run_button.setEnabled(False)
        self.hfo_app.n_jobs = int(self.n_jobs_spinbox.value())
        # get the filter parameters
        filter_param = self.get_filter_param()
        # get the detector parameters
        if self.detector == "MNI":
            detector_param = self.get_mni_params()
        elif self.detector == "STE":
            detector_param = self.get_ste_params()
        #print("filter_param: ", filter_param.to_dict())
        #print("detector_param: ", detector_param.to_dict())
        # get the classifier parameters
        #if we use classifier, run the classifier
        use_classifier = self.qd_use_classifier_checkbox.isChecked()
        if use_classifier:
            classifier_param = self.get_classifier_param()
        save_as_excel = self.qd_excel_checkbox.isChecked()
        save_as_npz = self.qd_npz_checkbox.isChecked()
        #if neither excel nor npz is selected, then don't do anything
        if not save_as_excel and not save_as_npz:
            # print("No output selected. Please select at least one output format!")
            return []

        # run the filter
        self.hfo_app.filter_eeg_data(filter_param)
        # print("Filtering COMPLETE!")     
        #run the detector
        self.hfo_app.set_detector(detector_param)
        self.hfo_app.detect_HFO()
        # print("HFOs DETECTED!")
        #if we use classifier, run the classifier
        use_classifier = self.qd_use_classifier_checkbox.isChecked()
        if use_classifier:
            self.classify(classifier_param)

        # print("Classification FINISH!")
        if save_as_excel:
            fname = self.fname.split(".")[0]+".xlsx"    
            self.hfo_app.export_excel(fname)
        if save_as_npz:
            fname = self.fname.split(".")[0]+".npz"
            self.hfo_app.export_app(fname)
        # print(f"Exporting {fname} FINISH!")
        return []
    
    def run(self):
        worker=Worker(self._run)
        self.running = True
        #disable cancel button
        self.cancel_button.setEnabled(False)
        worker.signals.result.connect(self._run_finished)
        self.threadpool.start(worker)

    def _run_finished(self):
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        self.running = False
        # print("run finished")
        
    # def close(self):
    #     print("closing")
    #     self.main_window.set_quick_detect_open(False)
    #     self.reject()
    #     # if self.running:
    #     #     print("cannot close while running detection")
    #     #     return
    #     # else:
    #     #     super().close()
    def reject(self):
        # print("rejecting")
        # super().reject()
        # #if running a detection do not allow to close
        if self.running:
            # print("cannot close while running detection")
            return
        else:
            self.main_window.set_quick_detect_open(False)
            # self.thread_stderr.stop()
            self.thread_stderr.terminate()
            self.thread_stderr.wait(5)
            # self.thread_stdout.stop()
            self.thread_stdout.terminate()
            self.thread_stdout.wait(5)
            # self.threadpool
            super().reject()
        
        


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = HFOQuickDetector()
    mainWindow.show()
    sys.exit(app.exec_())

