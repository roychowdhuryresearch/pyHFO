from pathlib import Path
from queue import Queue

from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import pyqtSignal
import ast
import multiprocessing as mp
import torch
from src.hfo_app import HFO_App
from src.spindle_app import SpindleApp
from src.ui.quick_detection import HFOQuickDetector
from src.ui.channels_selection import ChannelSelectionWindow
from src.param.param_classifier import ParamClassifier
from src.param.param_detector import ParamDetector, ParamSTE, ParamMNI, ParamHIL, ParamYASA
from src.param.param_filter import ParamFilter, ParamFilterSpindle
from src.ui.bipolar_channel_selection import BipolarChannelSelectionWindow
from src.ui.annotation import Annotation
from src.utils.utils_gui import *
from src.ui.plot_waveform import *


class MainWindowModel(QObject):
    def __init__(self, main_window):
        super(MainWindowModel, self).__init__()
        self.window = main_window
        self.backend = None
        self.biomarker_type = None

    def set_biomarker_type_and_init_backend(self, bio_type):
        self.biomarker_type = bio_type
        if bio_type == 'HFO':
            self.backend = HFO_App()
        elif bio_type == 'Spindle':
            self.backend = SpindleApp()
        elif bio_type == 'Spike':
            self.backend = HFO_App()

    def init_error_terminal_display(self):
        self.window.stdout = Queue()
        self.window.stderr = Queue()
        sys.stdout = WriteStream(self.window.stdout)
        sys.stderr = WriteStream(self.window.stderr)
        self.window.thread_stdout = STDOutReceiver(self.window.stdout)
        self.window.thread_stdout.std_received_signal.connect(self.message_handler)
        self.window.thread_stdout.start()

        self.window.thread_stderr = STDErrReceiver(self.window.stderr)
        self.window.thread_stderr.std_received_signal.connect(self.message_handler)
        self.window.thread_stderr.start()

    def init_menu_bar(self):
        self.window.action_Open_EDF.triggered.connect(self.open_file)
        self.window.actionQuick_Detection.triggered.connect(self.open_quick_detection)
        self.window.action_Load_Detection.triggered.connect(self.load_from_npz)

        ## top toolbar buttoms
        self.window.actionOpen_EDF_toolbar.triggered.connect(self.open_file)
        self.window.actionQuick_Detection_toolbar.triggered.connect(self.open_quick_detection)
        self.window.actionLoad_Detection_toolbar.triggered.connect(self.load_from_npz)

    def init_waveform_display(self):
        # waveform display widget
        self.window.waveform_plot_widget = pg.PlotWidget()
        self.window.waveform_mini_widget = pg.PlotWidget()
        self.window.widget.layout().addWidget(self.window.waveform_plot_widget, 0, 1)
        self.window.widget.layout().addWidget(self.window.waveform_mini_widget, 1, 1)
        self.window.widget.layout().setRowStretch(0, 9)
        self.window.widget.layout().setRowStretch(1, 1)

    def set_backend(self, backend):
        self.backend = backend

    def filter_data(self):
        self.message_handler("Filtering data...")
        try:
            # get filter parameters
            fp_raw = self.window.fp_input.text()
            fs_raw = self.window.fs_input.text()
            rp_raw = self.window.rp_input.text()
            rs_raw = self.window.rs_input.text()
            # self.pop_window()
            param_dict = {"fp": float(fp_raw), "fs": float(fs_raw), "rp": float(rp_raw), "rs": float(rs_raw)}
            filter_param = ParamFilter.from_dict(param_dict)
            self.backend.set_filter_parameter(filter_param)
        except:
            # there is error of the filter machine
            # therefore pop up window to show that filter failed
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('Filter could not be constructed with the given parameters')
            msg.setWindowTitle("Filter Construction Error")
            msg.exec_()
            return
        worker = Worker(self._filter)
        worker.signals.finished.connect(self.filtering_complete)
        self.window.threadpool.start(worker)

    def create_center_waveform_and_mini_plot(self):
        self.window.channels_to_plot = []
        self.window.waveform_plot = CenterWaveformAndMiniPlotController(self.window.waveform_plot_widget,
                                                                        self.window.waveform_mini_widget,
                                                                        self.backend)

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
        default_params = ParamHIL(2000)  # 初始化默认参数，假设采样率是 2000
        self.window.hil_sample_freq_input.setText(str(default_params.sample_freq))
        self.window.hil_pass_band_input.setText(str(default_params.pass_band))
        self.window.hil_stop_band_input.setText(str(default_params.stop_band))
        self.window.hil_epoch_time_input.setText(str(default_params.epoch_time))
        self.window.hil_sd_threshold_input.setText(str(default_params.sd_threshold))
        self.window.hil_min_window_input.setText(str(default_params.min_window))

    def init_default_yasa_input_params(self):
        default_params = ParamYASA(2000)
        self.window.yasa_freq_sp_input.setText(str(default_params.freq_sp))
        self.window.yasa_freq_broad_input.setText(str(default_params.freq_broad))
        self.window.yasa_duration_input.setText(str(default_params.duration))
        self.window.yasa_min_distance_input.setText(str(default_params.min_distance))
        self.window.yasa_thresh_rel_pow_input.setText(str(default_params.rel_pow))
        self.window.yasa_thresh_corr_input.setText(str(default_params.corr))
        self.window.yasa_thresh_rms_input.setText(str(default_params.rms))

    def connect_signal_and_slot(self, biomarker_type='HFO'):
        # classifier default buttons
        self.window.default_cpu_button.clicked.connect(self.set_classifier_param_cpu_default)
        self.window.default_gpu_button.clicked.connect(self.set_classifier_param_gpu_default)

        # choose model files connection
        self.window.choose_artifact_model_button.clicked.connect(lambda: self.choose_model_file("artifact"))
        self.window.choose_spike_model_button.clicked.connect(lambda: self.choose_model_file("spike"))

        # custom model param connection
        self.window.classifier_save_button.clicked.connect(self.set_custom_classifier_param)

        # detect_all_button
        self.window.detect_all_button.clicked.connect(lambda: self.classify(True))
        self.window.detect_all_button.setEnabled(False)
        # self.detect_artifacts_button.clicked.connect(lambda : self.classify(False))

        self.window.save_csv_button.clicked.connect(self.save_to_excel)
        self.window.save_csv_button.setEnabled(False)

        # set n_jobs min and max
        self.window.n_jobs_spinbox.setMinimum(1)
        self.window.n_jobs_spinbox.setMaximum(mp.cpu_count())

        # set default n_jobs
        self.window.n_jobs_spinbox.setValue(self.backend.n_jobs)
        self.window.n_jobs_ok_button.clicked.connect(self.set_n_jobs)

        self.window.save_npz_button.clicked.connect(self.save_to_npz)
        self.window.save_npz_button.setEnabled(False)

        self.window.Filter60Button.toggled.connect(self.switch_60)
        self.window.Filter60Button.setEnabled(False)

        self.window.bipolar_button.clicked.connect(self.open_bipolar_channel_selection)
        self.window.bipolar_button.setEnabled(False)

        # annotation button
        self.window.annotation_button.clicked.connect(self.open_annotation)
        self.window.annotation_button.setEnabled(False)

        self.window.Choose_Channels_Button.setEnabled(False)
        self.window.waveform_plot_button.setEnabled(False)

        # check if gpu is available
        self.gpu = torch.cuda.is_available()
        # print(f"GPU available: {self.gpu}")
        if not self.gpu:
            # disable gpu buttons
            self.window.default_gpu_button.setEnabled(False)

        if biomarker_type == 'HFO':
            self.window.overview_filter_button.clicked.connect(self.filter_data)
            # set filter button to be disabled by default
            self.window.overview_filter_button.setEnabled(False)
            # self.show_original_button.clicked.connect(self.toggle_filtered)

            self.window.mni_detect_button.clicked.connect(self.detect_HFOs)
            self.window.mni_detect_button.setEnabled(False)
            self.window.ste_detect_button.clicked.connect(self.detect_HFOs)
            self.window.ste_detect_button.setEnabled(False)
            self.window.hil_detect_button.clicked.connect(self.detect_HFOs)
            self.window.hil_detect_button.setEnabled(False)

            self.window.STE_save_button.clicked.connect(self.save_ste_params)
            self.window.MNI_save_button.clicked.connect(self.save_mni_params)
            self.window.HIL_save_button.clicked.connect(self.save_hil_params)
            self.window.STE_save_button.setEnabled(False)
            self.window.MNI_save_button.setEnabled(False)
            self.window.HIL_save_button.setEnabled(False)
        elif biomarker_type == 'Spindle':
            self.window.overview_filter_button.clicked.connect(self.filter_data)
            # set filter button to be disabled by default
            self.window.overview_filter_button.setEnabled(False)

            self.window.yasa_detect_button.clicked.connect(self.detect_Spindles)
            self.window.yasa_detect_button.setEnabled(False)

            self.window.YASA_save_button.clicked.connect(self.save_yasa_params)
            # self.window.YASA_save_button.setEnabled(False)

    def set_classifier_param_display(self):
        classifier_param = self.backend.get_classifier_param()

        self.window.overview_artifact_path_display.setText(classifier_param.artifact_path)
        self.window.overview_spike_path_display.setText(classifier_param.spike_path)
        self.window.overview_use_spike_checkbox.setChecked(classifier_param.use_spike)
        self.window.overview_device_display.setText(str(classifier_param.device))
        self.window.overview_batch_size_display.setText(str(classifier_param.batch_size))

        # set also the input fields
        self.window.classifier_artifact_filename.setText(classifier_param.artifact_path)
        self.window.classifier_spike_filename.setText(classifier_param.spike_path)
        self.window.use_spike_checkbox.setChecked(classifier_param.use_spike)
        self.window.classifier_device_input.setText(str(classifier_param.device))
        self.window.classifier_batch_size_input.setText(str(classifier_param.batch_size))

    def set_classifier_param_gpu_default(self):
        self.backend.set_default_gpu_classifier()
        self.set_classifier_param_display()

    def set_classifier_param_cpu_default(self):
        self.backend.set_default_cpu_classifier()
        self.set_classifier_param_display()

    def set_custom_classifier_param(self):
        artifact_path = self.window.classifier_artifact_filename.text()
        spike_path = self.window.classifier_spike_filename.text()
        use_spike = self.window.use_spike_checkbox.isChecked()
        device = self.window.classifier_device_input.text()
        if device == "cpu":
            model_type = "default_cpu"
        elif device == "cuda:0" and self.window.gpu:
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

        classifier_param = ParamClassifier(artifact_path=artifact_path, spike_path=spike_path, use_spike=use_spike,
                                           device=device, batch_size=int(batch_size), model_type=model_type)
        self.backend.set_classifier(classifier_param)
        self.set_classifier_param_display()

    def choose_model_file(self, model_type):
        fname, _ = QFileDialog.getOpenFileName(self.window, 'Open file', "", ".tar files (*.tar)")
        if model_type == "artifact":
            self.window.classifier_artifact_filename.setText(fname)
        elif model_type == "spike":
            self.window.classifier_spike_filename.setText(fname)

    def _classify(self, artifact_only=False):
        threshold = 0.5
        seconds_to_ignore_before = float(self.window.overview_ignore_before_input.text())
        seconds_to_ignore_after = float(self.window.overview_ignore_after_input.text())
        self.backend.classify_artifacts([seconds_to_ignore_before, seconds_to_ignore_after], threshold)
        if not artifact_only:
            self.backend.classify_spikes()
        return []

    def _classify_finished(self):
        self.message_handler("Classification finished!..")
        self.update_statistics_label()
        self.window.waveform_plot.set_plot_biomarkers(True)
        self.window.save_csv_button.setEnabled(True)

    def classify(self, check_spike=True):
        self.message_handler("Classifying HFOs...")
        if check_spike:
            use_spike = self.window.overview_use_spike_checkbox.isChecked()
        else:
            use_spike = False
        worker = Worker(lambda progress_callback: self._classify((not use_spike)))
        worker.signals.result.connect(self._classify_finished)
        self.window.threadpool.start(worker)

    def update_statistics_label(self):
        if self.biomarker_type == 'HFO':
            num_HFO = self.backend.event_features.get_num_biomarker()
            num_artifact = self.backend.event_features.get_num_artifact()
            num_spike = self.backend.event_features.get_num_spike()
            num_real = self.backend.event_features.get_num_real()

            self.window.statistics_label.setText(" Number of HFOs: " + str(num_HFO) + \
                                          "\n Number of artifacts: " + str(num_artifact) + \
                                          "\n Number of spikes: " + str(num_spike) + \
                                          "\n Number of real HFOs: " + str(num_real))
        elif self.biomarker_type == 'Spindle':
            num_spindle = self.backend.event_features.get_num_biomarker()
            num_artifact = self.backend.event_features.get_num_artifact()
            num_spike = self.backend.event_features.get_num_spike()
            num_real = self.backend.event_features.get_num_real()

            self.window.statistics_label.setText(" Number of Spindles: " + str(num_spindle) + \
                                                 "\n Number of artifacts: " + str(num_artifact) + \
                                                 "\n Number of spikes: " + str(num_spike) + \
                                                 "\n Number of real Spindles: " + str(num_real))
        elif self.biomarker_type == 'Spike':
            num_spindle = self.backend.event_features.get_num_biomarker()
            num_artifact = self.backend.event_features.get_num_artifact()
            num_spike = self.backend.event_features.get_num_spike()
            num_real = self.backend.event_features.get_num_real()

            self.window.statistics_label.setText(" Number of Spindles: " + str(num_spindle) + \
                                                 "\n Number of artifacts: " + str(num_artifact) + \
                                                 "\n Number of spikes: " + str(num_spike) + \
                                                 "\n Number of real Spindles: " + str(num_real))

    def save_to_excel(self):
        # open file dialog
        fname, _ = QFileDialog.getSaveFileName(self.window, 'Save file', "", ".xlsx files (*.xlsx)")
        if fname:
            self.backend.export_excel(fname)

    def _save_to_npz(self, fname, progress_callback):
        self.backend.export_app(fname)
        return []

    def save_to_npz(self):
        # open file dialog
        # print("saving to npz...",end="")
        fname, _ = QFileDialog.getSaveFileName(self.window, 'Save file', "", ".npz files (*.npz)")
        if fname:
            # print("saving to {fname}...",end="")
            worker = Worker(self._save_to_npz, fname)
            worker.signals.result.connect(lambda: 0)
            self.window.threadpool.start(worker)

    def _load_from_npz(self, fname, progress_callback):
        self.backend = self.backend.import_app(fname)
        return []

    def load_from_npz(self):
        # open file dialog
        fname, _ = QFileDialog.getOpenFileName(self.window, 'Open file', "", ".npz files (*.npz)")
        self.message_handler("Loading from npz...")
        if fname:
            self.reinitialize()
            worker = Worker(self._load_from_npz, fname)
            worker.signals.result.connect(self.load_from_npz_finished)
            self.window.threadpool.start(worker)
        # print(self.hfo_app.get_edf_info())

    def load_from_npz_finished(self):
        edf_info = self.backend.get_edf_info()
        self.window.waveform_plot.update_backend(self.backend)
        self.window.waveform_plot.init_eeg_data()
        edf_name = str(edf_info["edf_fn"])
        edf_name = edf_name[edf_name.rfind("/") + 1:]
        self.update_edf_info([edf_name, str(edf_info["sfreq"]),
                              str(edf_info["nchan"]), str(self.backend.eeg_data.shape[1])])
        # update number of jobs
        self.window.n_jobs_spinbox.setValue(self.backend.n_jobs)
        if self.backend.filtered:
            self.filtering_complete()
            filter_param = self.backend.param_filter
            # update filter params
            self.window.fp_input.setText(str(filter_param.fp))
            self.window.fs_input.setText(str(filter_param.fs))
            self.window.rp_input.setText(str(filter_param.rp))
            self.window.rs_input.setText(str(filter_param.rs))
        # update the detector parameters:
        if self.backend.detected:
            self.set_detector_param_display()
            self._detect_finished()
            self.update_statistics_label()
        # update classifier param
        if self.backend.classified:
            self.set_classifier_param_display()
            self._classify_finished()
            self.update_statistics_label()

    def open_channel_selection(self):
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
        horScrollBar = self.window.STDTextEdit.horizontalScrollBar()
        verScrollBar = self.window.STDTextEdit.verticalScrollBar()
        scrollIsAtEnd = verScrollBar.maximum() - verScrollBar.value() <= 10

        contain_percentage = re.findall(r'%', s)
        contain_one_hundred_percentage = re.findall(r'100%', s)
        if contain_one_hundred_percentage:
            cursor = self.window.STDTextEdit.textCursor()
            cursor.movePosition(QTextCursor.End - 1)
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            self.window.STDTextEdit.setTextCursor(cursor)
            self.window.STDTextEdit.insertPlainText(s)
        elif contain_percentage:
            cursor = self.window.STDTextEdit.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            self.window.STDTextEdit.setTextCursor(cursor)
            self.window.STDTextEdit.insertPlainText(s)
        else:
            self.window.STDTextEdit.append(s)

        if scrollIsAtEnd:
            verScrollBar.setValue(verScrollBar.maximum())  # Scrolls to the bottom
            horScrollBar.setValue(0)  # scroll to the left

    @pyqtSlot(list)
    def update_edf_info(self, results):
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
        self.window.waveform_time_scroll_bar.valueChanged.connect(self.scroll_time_waveform_plot)
        self.window.channel_scroll_bar.valueChanged.connect(self.scroll_channel_waveform_plot)
        self.window.waveform_plot_button.clicked.connect(self.waveform_plot_button_clicked)
        self.window.waveform_plot_button.setEnabled(True)
        self.window.Choose_Channels_Button.clicked.connect(self.open_channel_selection)
        self.window.Choose_Channels_Button.setEnabled(True)
        # set the display time window spin box
        self.window.display_time_window_input.setValue(self.window.waveform_plot.get_time_window())
        self.window.display_time_window_input.setMaximum(self.window.waveform_plot.get_total_time())
        self.window.display_time_window_input.setMinimum(0.1)
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
        self.window.toggle_filtered_checkbox.stateChanged.connect(self.toggle_filtered)
        self.window.normalize_vertical_input.stateChanged.connect(self.waveform_plot_button_clicked)
        # enable the plot out the 60Hz bandstopped signal
        self.window.Filter60Button.setEnabled(True)
        self.window.bipolar_button.setEnabled(True)
        # print("EDF file loaded")

    def toggle_filtered(self):
        # self.message_handler('Showing original data...')
        if self.window.is_data_filtered:
            self.window.show_filtered = not self.window.show_filtered
            self.window.waveform_plot.set_filtered(self.window.show_filtered)
            self.waveform_plot_button_clicked()

    def read_edf(self, fname, progress_callback):
        self.reinitialize()
        self.backend.load_edf(fname)
        eeg_data, channel_names = self.backend.get_eeg_data()
        edf_info = self.backend.get_edf_info()
        self.window.waveform_plot.init_eeg_data()
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

    def open_file(self):
        # reinitialize the app
        self.set_biomarker_type_and_init_backend(self.biomarker_type)
        fname, _ = QFileDialog.getOpenFileName(self.window, "Open File", "", "Recordings Files (*.edf *.eeg *.vhdr *.vmrk)")
        if fname:
            worker = Worker(self.read_edf, fname)
            worker.signals.result.connect(self.update_edf_info)
            self.window.threadpool.start(worker)

    def filtering_complete(self):
        self.message_handler('Filtering COMPLETE!')
        filter_60 = self.window.Filter60Button.isChecked()
        # print("filtering:", filter_60)
        # if yes
        if filter_60:
            self.backend.set_filter_60()
        # if not
        else:
            self.backend.set_unfiltered_60()

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

    def detect_HFOs(self):
        print("Detecting HFOs...")
        worker = Worker(self._detect)
        worker.signals.result.connect(self._detect_finished)
        self.window.threadpool.start(worker)

    def detect_Spindles(self):
        print("Detecting Spindles...")
        worker = Worker(self._detect)
        worker.signals.result.connect(self._detect_finished)
        self.window.threadpool.start(worker)

    def _detect_finished(self):
        # right now do nothing beyond message handler saying that
        # it has detected HFOs
        self.message_handler("Biomarker detected")
        self.update_statistics_label()
        self.window.waveform_plot.set_plot_biomarkers(True)
        self.window.detect_all_button.setEnabled(True)
        self.window.annotation_button.setEnabled(True)

    def _detect(self, progress_callback):
        # call detect HFO function on backend
        self.backend.detect_biomarker()
        return []

    def open_quick_detection(self):
        # if we want to open multiple qd dialog
        if not self.window.quick_detect_open:
            qd = HFOQuickDetector(HFO_App(), self, self.window.close_signal)
            # print("created new quick detector")
            qd.show()
            self.window.quick_detect_open = True

    def set_quick_detect_open(self, open):
        self.window.quick_detect_open = open

    def reinitialize_buttons(self):
        self.window.mni_detect_button.setEnabled(False)
        self.window.ste_detect_button.setEnabled(False)
        self.window.hil_detect_button.setEnabled(False)
        self.window.detect_all_button.setEnabled(False)
        self.window.save_csv_button.setEnabled(False)
        self.window.save_npz_button.setEnabled(False)
        self.window.STE_save_button.setEnabled(False)
        self.window.MNI_save_button.setEnabled(False)
        self.window.HIL_save_button.setEnabled(False)
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
        self.window.yasa_freq_sp_input.setMaxLength(max_len)
        self.window.yasa_freq_broad_input.setMaxLength(max_len)
        self.window.yasa_duration_input.setMaxLength(max_len)
        self.window.yasa_min_distance_input.setMaxLength(max_len)
        self.window.yasa_thresh_rel_pow_input.setMaxLength(max_len)
        self.window.yasa_thresh_corr_input.setMaxLength(max_len)
        self.window.yasa_thresh_rms_input.setMaxLength(max_len)

    def close_other_window(self):
        self.window.close_signal.emit()

    def set_n_jobs(self):
        self.backend.n_jobs = int(self.window.n_jobs_spinbox.value())
        # print(f"n_jobs set to {self.hfo_app.n_jobs}")

    def set_channels_to_plot(self, channels_to_plot, display_all=True):
        self.window.waveform_plot.set_channels_to_plot(channels_to_plot)
        # print(f"Channels to plot: {self.channels_to_plot}")
        self.window.n_channel_input.setMaximum(len(channels_to_plot))
        if display_all:
            self.window.n_channel_input.setValue(len(channels_to_plot))
        self.waveform_plot_button_clicked()

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
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error!")
            msg.setInformativeText('Detector could not be constructed given the parameters')
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

        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error!")
            msg.setInformativeText(f'HIL Detector could not be constructed given the parameters. Error: {str(e)}')
            msg.setWindowTitle("HIL Detector Construction Failed")
            msg.exec_()

    def save_yasa_params(self):
        # get filter parameters

        freq_sp_raw = self.window.yasa_freq_sp_input.text()
        freq_broad_raw = self.window.yasa_freq_broad_input.text()
        duration_raw = self.window.yasa_duration_input.text()
        min_distance_raw = self.window.yasa_min_distance_input.text()
        thresh_rel_pow_raw = self.window.yasa_thresh_rel_pow_input.text()
        thresh_corr_raw = self.window.yasa_thresh_corr_input.text()
        thresh_rms_raw = self.window.yasa_thresh_rms_input.text()
        try:
            param_dict = {"sample_freq": 2000,
                          # these are placeholder params, will be updated later
                          "freq_sp": ast.literal_eval(freq_sp_raw), "freq_broad": ast.literal_eval(freq_broad_raw),
                          "duration": ast.literal_eval(duration_raw),
                          "min_distance": float(min_distance_raw), "rel_pow": float(thresh_rel_pow_raw),
                          "corr": float(thresh_corr_raw),
                          "rms": float(thresh_rms_raw), "n_jobs": self.backend.n_jobs}
            detector_params = {"detector_type": "YASA", "detector_param": param_dict}
            self.backend.set_detector(ParamDetector.from_dict(detector_params))

            # set display parameters
            self.window.yasa_freq_sp_display.setText(freq_sp_raw)
            self.window.yasa_freq_broad_display.setText(freq_broad_raw)
            self.window.yasa_duration_display.setText(duration_raw)
            self.window.yasa_min_distance_display.setText(min_distance_raw)
            self.window.yasa_thresh_rel_pow_display.setText(thresh_rel_pow_raw)
            self.window.yasa_thresh_corr_display.setText(thresh_corr_raw)
            self.window.yasa_thresh_rms_display.setText(thresh_rms_raw)
            # self.update_detector_tab("STE")
            self.window.yasa_detect_button.setEnabled(True)
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error!")
            msg.setInformativeText('Detector could not be constructed given the parameters')
            msg.setWindowTitle("Detector Construction Failed")
            msg.exec_()

    def update_detector_tab(self, index):
        if index == "STE":
            self.window.stacked_widget_detection_param.setCurrentIndex(0)
        elif index == "MNI":
            self.window.stacked_widget_detection_param.setCurrentIndex(1)
        elif index == "HIL":
            self.window.stacked_widget_detection_param.setCurrentIndex(2)

    def reinitialize(self):
        # kill all threads in self.threadpool
        self.close_other_window()
        # self.backend = HFO_App()
        self.set_biomarker_type_and_init_backend(self.biomarker_type)
        self.window.waveform_plot.update_backend(self.backend, False)
        self.window.main_filename.setText("")
        self.window.main_sampfreq.setText("")
        self.window.main_numchannels.setText("")
        self.window.main_length.setText("")
        self.window.statistics_label.setText("")

    def close_other_window(self):
        self.window.close_signal.emit()

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
        detector_params = self.backend.param_detector
        detector_type = detector_params.detector_type.lower()
        if detector_type == "ste":
            self.update_ste_params(detector_params.detector_param.to_dict())
        elif detector_type == "mni":
            self.update_mni_params(detector_params.detector_param.to_dict())
        elif detector_type == "hil":
            self.update_hil_params(detector_params.detector_param.to_dict())

    def open_bipolar_channel_selection(self):
        self.window.bipolar_channel_selection_window = BipolarChannelSelectionWindow(self.backend,
                                                                                     self.window,
                                                                                     self.window.close_signal,
                                                                                     self.window.waveform_plot)
        self.window.bipolar_channel_selection_window.show()

    def open_annotation(self):
        self.window.save_csv_button.setEnabled(True)
        annotation = Annotation(self.backend, self.window, self.window.close_signal)
        annotation.show()