import os
import re
import sys
import traceback
from pathlib import Path
from queue import Queue

from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QMessageBox


from src.hfo_app import HFO_App
from src.param.param_classifier import ParamClassifier
from src.param.param_detector import ParamDetector, ParamSTE, ParamMNI
from src.param.param_filter import ParamFilter
from src.ui.quick_detection import HFOQuickDetector
from src.ui.channels_selection import ChannelSelectionWindow
from src.ui.bipolar_channel_selection import BipolarChannelSelectionWindow
from src.ui.annotation import HFOAnnotation
from src.utils.utils_gui import *
from src.ui.plot_waveform import *
from PyQt5.QtCore import pyqtSignal
# import tkinter as tk
# from tkinter import *
# from tkinter import messagebox
import threading
import time
  
import multiprocessing as mp
import torch

import warnings
warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).parent


class HFOMainWindow(QMainWindow):
    close_signal = pyqtSignal()
    def __init__(self):
        super(HFOMainWindow, self).__init__()
        self.ui = uic.loadUi(os.path.join(ROOT_DIR, 'src/ui/main_window.ui'), self)
        self.setWindowIcon(QtGui.QIcon(os.path.join(ROOT_DIR, 'src/ui/images/icon1.png')))
        self.setWindowTitle("pyHFO")
        self.hfo_app = HFO_App()
        self.threadpool = QThreadPool()
        self.replace_last_line = False
        self.stdout = Queue()
        self.stderr = Queue()
        sys.stdout = WriteStream(self.stdout)
        sys.stderr = WriteStream(self.stderr)

        self.thread_stdout = STDOutReceiver(self.stdout)
        self.thread_stdout.std_received_signal.connect(self.message_handler)
        self.thread_stdout.start()

        self.thread_stderr = STDErrReceiver(self.stderr)
        self.thread_stderr.std_received_signal.connect(self.message_handler)
        self.thread_stderr.start()

        self.action_Open_EDF.triggered.connect(self.open_file)
        self.actionQuick_Detection.triggered.connect(self.open_quick_detection)
        self.action_Load_Detection.triggered.connect(self.load_from_npz)
        self.overview_filter_button.clicked.connect(self.filter_data)
        # set filter button to be disabled by default
        self.overview_filter_button.setEnabled(False)
        # self.show_original_button.clicked.connect(self.toggle_filtered)

        self.is_data_filtered = False

        self.waveform_plot_widget = pg.PlotWidget()
        self.waveform_mini_widget = pg.PlotWidget()
        self.widget.layout().addWidget(self.waveform_plot_widget, 0, 1)
        self.widget.layout().addWidget(self.waveform_mini_widget, 1, 1)
        self.widget.layout().setRowStretch(0, 9)
        self.widget.layout().setRowStretch(1, 1)
        self.waveform_plot = PlotWaveform(self.waveform_plot_widget, self.waveform_mini_widget, self.hfo_app)

        ## top toolbar buttoms
        self.actionOpen_EDF_toolbar.triggered.connect(self.open_file)
        self.actionQuick_Detection_toolbar.triggered.connect(self.open_quick_detection)
        self.actionLoad_Detection_toolbar.triggered.connect(self.load_from_npz)
        
        self.mni_detect_button.clicked.connect(self.detect_HFOs)
        self.mni_detect_button.setEnabled(False)
        self.ste_detect_button.clicked.connect(self.detect_HFOs)
        self.ste_detect_button.setEnabled(False)

        #classifier tab buttons
        self.classifier_param = ParamClassifier()
        #self.classifier_save_button.clicked.connect(self.hfo_app.set_classifier())

        #init inputs
        self.init_default_filter_input_params()
        self.init_default_ste_input_params()
        self.init_default_mni_input_params()
    
        #classifier default buttons
        self.default_cpu_button.clicked.connect(self.set_classifier_param_cpu_default)
        self.default_gpu_button.clicked.connect(self.set_classifier_param_gpu_default)

        #choose model files connection
        self.choose_artifact_model_button.clicked.connect(lambda : self.choose_model_file("artifact"))
        self.choose_spike_model_button.clicked.connect(lambda : self.choose_model_file("spike"))
        
        #custom model param connection
        self.classifier_save_button.clicked.connect(self.set_custom_classifier_param)
        
        #detect_all_button
        self.detect_all_button.clicked.connect(lambda: self.classify(True))
        self.detect_all_button.setEnabled(False)
        # self.detect_artifacts_button.clicked.connect(lambda : self.classify(False))
        
        self.save_csv_button.clicked.connect(self.save_to_excel)
        self.save_csv_button.setEnabled(False)

        #set n_jobs min and max
        self.n_jobs_spinbox.setMinimum(1)
        self.n_jobs_spinbox.setMaximum(mp.cpu_count())

        #set default n_jobs
        self.n_jobs_spinbox.setValue(self.hfo_app.n_jobs)
        self.n_jobs_ok_button.clicked.connect(self.set_n_jobs)

        self.STE_save_button.clicked.connect(self.save_ste_params)
        self.MNI_save_button.clicked.connect(self.save_mni_params)
        self.STE_save_button.setEnabled(False)
        self.MNI_save_button.setEnabled(False)

        self.save_npz_button.clicked.connect(self.save_to_npz)
        self.save_npz_button.setEnabled(False)

        self.Filter60Button.toggled.connect(self.switch_60)
        self.Filter60Button.setEnabled(False)

        self.bipolar_button.clicked.connect(self.open_bipolar_channel_selection)
        self.bipolar_button.setEnabled(False)

        #annotation button
        self.annotation_button.clicked.connect(self.open_annotation)
        self.annotation_button.setEnabled(False)

        self.Choose_Channels_Button.setEnabled(False)
        self.waveform_plot_button.setEnabled(False)

        self.channels_to_plot = []

        #check if gpu is available
        self.gpu = torch.cuda.is_available()
        # print(f"GPU available: {self.gpu}")
        if not self.gpu:
            #disable gpu buttons
            self.default_gpu_button.setEnabled(False)

        self.quick_detect_open = False
        self.set_mni_input_len(8)
        self.set_ste_input_len(8)
       
        #close window signal
        
    def reinitialize_buttons(self):
        self.mni_detect_button.setEnabled(False)
        self.ste_detect_button.setEnabled(False)
        self.detect_all_button.setEnabled(False)
        self.save_csv_button.setEnabled(False)
        self.save_npz_button.setEnabled(False)
        self.STE_save_button.setEnabled(False)
        self.MNI_save_button.setEnabled(False)
        self.Filter60Button.setEnabled(False)

    def set_mni_input_len(self,max_len = 5):
        self.mni_epoch_time_input.setMaxLength(max_len)
        self.mni_epoch_chf_input.setMaxLength(max_len) 
        self.mni_chf_percentage_input.setMaxLength(max_len)
        self.mni_min_window_input.setMaxLength(max_len)
        self.mni_min_gap_time_input.setMaxLength(max_len)
        self.mni_threshold_percentage_input.setMaxLength(max_len)
        self.mni_baseline_window_input.setMaxLength(max_len)
        self.mni_baseline_shift_input.setMaxLength(max_len)
        self.mni_baseline_threshold_input.setMaxLength(max_len)
        self.mni_baseline_min_time_input.setMaxLength(max_len)

    def set_ste_input_len(self,max_len = 5):
        self.ste_rms_window_input.setMaxLength(max_len)
        self.ste_min_window_input.setMaxLength(max_len)
        self.ste_min_gap_input.setMaxLength(max_len)
        self.ste_epoch_length_input.setMaxLength(max_len)
        self.ste_min_oscillation_input.setMaxLength(max_len)
        self.ste_rms_threshold_input.setMaxLength(max_len)
        self.ste_peak_threshold_input.setMaxLength(max_len)


    def close_other_window(self):
        self.close_signal.emit() 

    def set_n_jobs(self):
        self.hfo_app.n_jobs = int(self.n_jobs_spinbox.value())
        # print(f"n_jobs set to {self.hfo_app.n_jobs}")

    def set_channels_to_plot(self, channels_to_plot, display_all = True):
        self.waveform_plot.set_channels_to_plot(channels_to_plot)
        # print(f"Channels to plot: {self.channels_to_plot}")
        self.n_channel_input.setMaximum(len(channels_to_plot))
        if display_all:
            self.n_channel_input.setValue(len(channels_to_plot))
        self.waveform_plot_button_clicked()
    
    def open_channel_selection(self):
        self.channel_selection_window = ChannelSelectionWindow(self.hfo_app, self, self.close_signal)
        self.channel_selection_window.show()
    
    def channel_selection_update(self):
        self.channel_scroll_bar.setValue(0)
        self.waveform_time_scroll_bar.setValue(0)
        is_empty = self.n_channel_input.maximum() == 0
        self.waveform_plot.plot(0,0,empty=is_empty,update_hfo=True)

    def switch_60(self):
        #get the value of the Filter60Button radio button
        filter_60 = self.Filter60Button.isChecked()
        # print("filtering:", filter_60)
        #if yes
        if filter_60:
            self.hfo_app.set_filter_60()
        #if not
        else:
            self.hfo_app.set_unfiltered_60()

        #replot 
        self.waveform_plot.plot()
        #add a warning to the text about the HFO info saying that it is outdated now

    @pyqtSlot(str)
    def message_handler(self, s):
        s = s.replace("\n", "")
        horScrollBar = self.STDTextEdit.horizontalScrollBar()
        verScrollBar = self.STDTextEdit.verticalScrollBar()
        scrollIsAtEnd = verScrollBar.maximum() - verScrollBar.value() <= 10
        
        contain_percentage = re.findall(r'%', s)
        contain_one_hundred_percentage = re.findall(r'100%', s)
        if contain_one_hundred_percentage:
            cursor = self.STDTextEdit.textCursor()
            cursor.movePosition(QTextCursor.End - 1)
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            self.STDTextEdit.setTextCursor(cursor)
            self.STDTextEdit.insertPlainText(s)
        elif contain_percentage:
            cursor = self.STDTextEdit.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            self.STDTextEdit.setTextCursor(cursor)
            self.STDTextEdit.insertPlainText(s)
        else:
            self.STDTextEdit.append(s)
        
        if scrollIsAtEnd:
            verScrollBar.setValue(verScrollBar.maximum()) # Scrolls to the bottom
            horScrollBar.setValue(0) # scroll to the left

    def reinitialize(self):
        #kill all threads in self.threadpool
        self.close_other_window()
        self.hfo_app = HFO_App()
        self.waveform_plot.update_backend(self.hfo_app, False)
        self.main_filename.setText("")
        self.main_sampfreq.setText("")
        self.main_numchannels.setText("")
        self.main_length.setText("")
        self.statistics_label.setText("")


    @pyqtSlot(list)
    def update_edf_info(self, results):
        self.main_filename.setText(results[0])
        self.main_sampfreq.setText(results[1])
        self.sample_freq = float(results[1])
        self.main_numchannels.setText(results[2])
        # print("updated")
        self.main_length.setText(str(round(float(results[3])/(60*float(results[1])),3))+" min")
        self.waveform_plot.plot(0, update_hfo=True)
        # print("plotted")
        #connect buttons
        self.waveform_time_scroll_bar.valueChanged.connect(self.scroll_time_waveform_plot)
        self.channel_scroll_bar.valueChanged.connect(self.scroll_channel_waveform_plot)
        self.waveform_plot_button.clicked.connect(self.waveform_plot_button_clicked)
        self.waveform_plot_button.setEnabled(True)
        self.Choose_Channels_Button.clicked.connect(self.open_channel_selection)
        self.Choose_Channels_Button.setEnabled(True)
        #set the display time window spin box
        self.display_time_window_input.setValue(self.waveform_plot.get_time_window())
        self.display_time_window_input.setMaximum(self.waveform_plot.get_total_time())
        self.display_time_window_input.setMinimum(0.1)
        #set the n channel spin box
        self.n_channel_input.setValue(self.waveform_plot.get_n_channels_to_plot())
        self.n_channel_input.setMaximum(self.waveform_plot.get_n_channels())
        self.n_channel_input.setMinimum(1)
        #set the time scroll bar range
        self.waveform_time_scroll_bar.setMaximum(int(self.waveform_plot.get_total_time()/(self.waveform_plot.get_time_window()*self.waveform_plot.get_time_increment()/100)))
        self.waveform_time_scroll_bar.setValue(0)
        #set the channel scroll bar range
        self.channel_scroll_bar.setMaximum(self.waveform_plot.get_n_channels()-self.waveform_plot.get_n_channels_to_plot())
        #enable the filter button
        self.overview_filter_button.setEnabled(True)
        self.toggle_filtered_checkbox.stateChanged.connect(self.toggle_filtered)
        self.normalize_vertical_input.stateChanged.connect(self.waveform_plot_button_clicked)
        #enable the plot out the 60Hz bandstopped signal
        self.Filter60Button.setEnabled(True)
        self.bipolar_button.setEnabled(True)
        #print("EDF file loaded")


    def init_default_filter_input_params(self):
        default_params=ParamFilter()
        self.fp_input.setText(str(default_params.fp))
        self.fs_input.setText(str(default_params.fs))
        self.rp_input.setText(str(default_params.rp))
        self.rs_input.setText(str(default_params.rs))

    def init_default_ste_input_params(self):
        default_params=ParamSTE(2000)
        self.ste_rms_window_input.setText(str(default_params.rms_window))
        self.ste_rms_threshold_input.setText(str(default_params.rms_thres))
        self.ste_min_window_input.setText(str(default_params.min_window))
        self.ste_epoch_length_input.setText(str(default_params.epoch_len))
        self.ste_min_gap_input.setText(str(default_params.min_gap))
        self.ste_min_oscillation_input.setText(str(default_params.min_osc))
        self.ste_peak_threshold_input.setText(str(default_params.peak_thres))

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
        self.mni_epoch_time_input.setText(str(default_params.epoch_time))
        self.mni_epoch_chf_input.setText(str(default_params.epo_CHF))
        self.mni_chf_percentage_input.setText(str(default_params.per_CHF))
        self.mni_min_window_input.setText(str(default_params.min_win))
        self.mni_min_gap_time_input.setText(str(default_params.min_gap))
        self.mni_threshold_percentage_input.setText(str(default_params.thrd_perc*100))
        self.mni_baseline_window_input.setText(str(default_params.base_seg))
        self.mni_baseline_shift_input.setText(str(default_params.base_shift))
        self.mni_baseline_threshold_input.setText(str(default_params.base_thrd))
        self.mni_baseline_min_time_input.setText(str(default_params.base_min))

    def scroll_time_waveform_plot(self, event):
        t_start=self.waveform_time_scroll_bar.value()*self.waveform_plot.get_time_window()*self.waveform_plot.get_time_increment()/100
        self.waveform_plot.plot(t_start)
    
    def scroll_channel_waveform_plot(self, event):
        channel_start=self.channel_scroll_bar.value()
        self.waveform_plot.plot(first_channel_to_plot=channel_start, update_hfo=True)

    def get_channels_to_plot(self):
        return self.waveform_plot.get_channels_to_plot()
    
    def get_channel_indices_to_plot(self):
        return self.waveform_plot.get_channel_indices_to_plot()

    def waveform_plot_button_clicked(self):
        time_window=self.display_time_window_input.value()
        self.waveform_plot.set_time_window(time_window)
        n_channels_to_plot=self.n_channel_input.value()
        self.waveform_plot.set_n_channels_to_plot(n_channels_to_plot)
        time_increment = self.Time_Increment_Input.value()
        self.waveform_plot.set_time_increment(time_increment)
        normalize_vertical = self.normalize_vertical_input.isChecked()
        self.waveform_plot.set_normalize_vertical(normalize_vertical)
        is_empty = self.n_channel_input.maximum() == 0
        start = self.waveform_plot.t_start
        first_channel_to_plot = self.waveform_plot.first_channel_to_plot
        
        t_value = int(start//(self.waveform_plot.get_time_window()*self.waveform_plot.get_time_increment()/100))
        self.waveform_time_scroll_bar.setMaximum(int(self.waveform_plot.get_total_time()/(self.waveform_plot.get_time_window()*self.waveform_plot.get_time_increment()/100)))
        self.waveform_time_scroll_bar.setValue(t_value)
        c_value = self.channel_scroll_bar.value()
        self.channel_scroll_bar.setMaximum(len(self.waveform_plot.get_channels_to_plot())-n_channels_to_plot)
        self.channel_scroll_bar.setValue(c_value)
        self.waveform_plot.plot(start,first_channel_to_plot,empty=is_empty,update_hfo=True)

    def open_file(self):
        #reinitialize the app
        self.hfo_app = HFO_App()
        fname, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Recordings Files (*.edf *.eeg *.vhdr *.vmrk)")
        if fname:
            worker = Worker(self.read_edf, fname)
            worker.signals.result.connect(self.update_edf_info)
            self.threadpool.start(worker)

    def filtering_complete(self):
        self.message_handler('Filtering COMPLETE!')
        filter_60 = self.Filter60Button.isChecked()
        # print("filtering:", filter_60)
        #if yes
        if filter_60:
            self.hfo_app.set_filter_60()
        #if not
        else:
            self.hfo_app.set_unfiltered_60()
            
        self.STE_save_button.setEnabled(True)
        self.ste_detect_button.setEnabled(True)
        self.MNI_save_button.setEnabled(True)
        self.mni_detect_button.setEnabled(True)
        self.is_data_filtered = True
        self.show_filtered = True
        self.waveform_plot.set_filtered(True)
        self.save_npz_button.setEnabled(True)

    def filter_data(self):
        self.message_handler("Filtering data...")
        try: 
            #get filter parameters
            fp_raw = self.fp_input.text()
            fs_raw = self.fs_input.text()
            rp_raw = self.rp_input.text()
            rs_raw = self.rs_input.text()
            #self.pop_window()
            param_dict={"fp":float(fp_raw), "fs":float(fs_raw), "rp":float(rp_raw), "rs":float(rs_raw)}
            filter_param = ParamFilter.from_dict(param_dict)
            self.hfo_app.set_filter_parameter(filter_param)
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
        worker=Worker(self._filter)
        worker.signals.finished.connect(self.filtering_complete)
        self.threadpool.start(worker)

    def toggle_filtered(self):
        # self.message_handler('Showing original data...')
        if self.is_data_filtered:
            self.show_filtered = not self.show_filtered
            self.waveform_plot.set_filtered(self.show_filtered)
            self.waveform_plot_button_clicked()

    def read_edf(self, fname, progress_callback):
        self.reinitialize()
        self.hfo_app.load_edf(fname)
        eeg_data,channel_names=self.hfo_app.get_eeg_data()
        edf_info=self.hfo_app.get_edf_info()
        self.waveform_plot.init_eeg_data()
        filename = os.path.basename(fname)
        sample_freq = str(self.hfo_app.sample_freq)
        num_channels = str(len(self.hfo_app.channel_names))
        length = str(self.hfo_app.eeg_data.shape[1])
        return [filename, sample_freq, num_channels, length]


    def _filter(self, progress_callback):
        self.hfo_app.filter_eeg_data()
        return []


    def open_detector(self):
        # Pass the function to execute, function, args, kwargs
        worker = Worker(self.quick_detect)
        self.threadpool.start(worker)

    def round_dict(self, d:dict, n:int):
        for key in d.keys():
            if type(d[key]) == float:
                d[key] = round(d[key], n)
        return d
    
    def save_ste_params(self):
        #get filter parameters
        rms_window_raw = self.ste_rms_window_input.text()
        min_window_raw = self.ste_min_window_input.text()
        min_gap_raw = self.ste_min_gap_input.text()
        epoch_len_raw = self.ste_epoch_length_input.text()
        min_osc_raw = self.ste_min_oscillation_input.text()
        rms_thres_raw = self.ste_rms_threshold_input.text()
        peak_thres_raw = self.ste_peak_threshold_input.text()
        try:
            param_dict = {"sample_freq":2000,"pass_band":1, "stop_band":80, #these are placeholder params, will be updated later
                        "rms_window":float(rms_window_raw), "min_window":float(min_window_raw), "min_gap":float(min_gap_raw),
                        "epoch_len":float(epoch_len_raw), "min_osc":float(min_osc_raw), "rms_thres":float(rms_thres_raw),
                        "peak_thres":float(peak_thres_raw),"n_jobs":self.hfo_app.n_jobs}
            detector_params = {"detector_type":"STE", "detector_param":param_dict}
            self.hfo_app.set_detector(ParamDetector.from_dict(detector_params))
            
            #set display parameters
            self.ste_epoch_display.setText(epoch_len_raw)
            self.ste_min_window_display.setText(min_window_raw)
            self.ste_rms_window_display.setText(rms_window_raw)
            self.ste_min_gap_time_display.setText(min_gap_raw)
            self.ste_min_oscillations_display.setText(min_osc_raw)
            self.ste_peak_threshold_display.setText(peak_thres_raw)
            self.ste_rms_threshold_display.setText(rms_thres_raw)
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
            epoch_time = self.mni_epoch_time_input.text()
            epo_CHF = self.mni_epoch_chf_input.text() 
            per_CHF = self.mni_chf_percentage_input.text()
            min_win = self.mni_min_window_input.text()
            min_gap = self.mni_min_gap_time_input.text()
            thrd_perc = self.mni_threshold_percentage_input.text()
            base_seg = self.mni_baseline_window_input.text()
            base_shift = self.mni_baseline_shift_input.text()
            base_thrd = self.mni_baseline_threshold_input.text()
            base_min = self.mni_baseline_min_time_input.text()

            param_dict = {"sample_freq":2000,"pass_band":1, "stop_band":80, #these are placeholder params, will be updated later
                        "epoch_time":float(epoch_time), "epo_CHF":float(epo_CHF), "per_CHF":float(per_CHF),
                        "min_win":float(min_win), "min_gap":float(min_gap), "base_seg":float(base_seg),
                        "thrd_perc":float(thrd_perc)/100,
                        "base_shift":float(base_shift), "base_thrd":float(base_thrd), "base_min":float(base_min),
                        "n_jobs":self.hfo_app.n_jobs}
            # param_dict = self.round_dict(param_dict, 3)
            detector_params = {"detector_type":"MNI", "detector_param":param_dict}
            self.hfo_app.set_detector(ParamDetector.from_dict(detector_params))

            #set display parameters
            self.mni_epoch_display.setText(epoch_time)
            self.mni_epoch_chf_display.setText(epo_CHF)
            self.mni_chf_percentage_display.setText(per_CHF)
            self.mni_min_window_display.setText(min_win)
            self.mni_min_gap_time_display.setText(min_gap)
            self.mni_threshold_percentile_display.setText(thrd_perc)
            self.mni_baseline_window_display.setText(base_seg)
            self.mni_baseline_shift_display.setText(base_shift)
            self.mni_baseline_threshold_display.setText(base_thrd)
            self.mni_baseline_min_time_display.setText(base_min)

            self.update_detector_tab("MNI")
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error!")
            msg.setInformativeText('Detector could not be constructed given the parameters')
            msg.setWindowTitle("Detector Construction Failed")
            msg.exec_()

    def detect_HFOs(self):
        print("Detecting HFOs...")
        worker=Worker(self._detect)
        worker.signals.result.connect(self._detect_finished)
        self.threadpool.start(worker)

    def _detect_finished(self):
        #right now do nothing beyond message handler saying that 
        # it has detected HFOs
        self.message_handler("HFOs detected")
        self.update_statistics_label()
        self.waveform_plot.set_plot_HFOs(True)
        self.detect_all_button.setEnabled(True)
        self.annotation_button.setEnabled(True)

    def _detect(self, progress_callback):
        #call detect HFO function on backend
        self.hfo_app.detect_HFO()
        return []
    
    def open_quick_detection(self):
        # if we want to open multiple qd dialog
        if not self.quick_detect_open:
            qd = HFOQuickDetector(HFO_App(), self, self.close_signal)
            # print("created new quick detector")
            qd.show()
            self.quick_detect_open = True
    
    def set_quick_detect_open(self, open):
        self.quick_detect_open = open

    def update_detector_tab(self, index):
        if index == "MNI":
            self.stackedWidget.setCurrentIndex(0)
        elif index == "STE":
            self.stackedWidget.setCurrentIndex(1)

    def set_classifier_param_display(self):
        classifier_param = self.hfo_app.get_classifier_param()

        self.overview_artifact_path_display.setText(classifier_param.artifact_path)
        self.overview_spike_path_display.setText(classifier_param.spike_path)
        self.overview_use_spike_checkbox.setChecked(classifier_param.use_spike)
        self.overview_device_display.setText(str(classifier_param.device))
        self.overview_batch_size_display.setText(str(classifier_param.batch_size))

        #set also the input fields
        self.classifier_artifact_filename.setText(classifier_param.artifact_path)
        self.classifier_spike_filename.setText(classifier_param.spike_path)
        self.use_spike_checkbox.setChecked(classifier_param.use_spike)
        self.classifier_device_input.setText(str(classifier_param.device))
        self.classifier_batch_size_input.setText(str(classifier_param.batch_size))

    def set_classifier_param_gpu_default(self):
        self.hfo_app.set_default_gpu_classifier()
        self.set_classifier_param_display()
    
    def set_classifier_param_cpu_default(self):
        self.hfo_app.set_default_cpu_classifier()
        self.set_classifier_param_display()

    def set_custom_classifier_param(self):
        artifact_path = self.classifier_artifact_filename.text()
        spike_path = self.classifier_spike_filename.text()
        use_spike = self.use_spike_checkbox.isChecked()
        device = self.classifier_device_input.text()
        if device=="cpu":
            model_type = "default_cpu"
        elif device=="cuda:0" and self.gpu:
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
        batch_size = self.classifier_batch_size_input.text()

        classifier_param = ParamClassifier(artifact_path=artifact_path, spike_path=spike_path, use_spike=use_spike,
                                          device=device, batch_size=int(batch_size), model_type=model_type)
        self.hfo_app.set_classifier(classifier_param)
        self.set_classifier_param_display()

    def choose_model_file(self, model_type):
        fname,_  = QFileDialog.getOpenFileName(self, 'Open file', "", ".tar files (*.tar)")
        if model_type == "artifact":
            self.classifier_artifact_filename.setText(fname)
        elif model_type == "spike":
            self.classifier_spike_filename.setText(fname)

    def _classify(self,artifact_only=False):
        threshold = 0.5
        seconds_to_ignore_before=float(self.overview_ignore_before_input.text())
        seconds_to_ignore_after=float(self.overview_ignore_after_input.text())
        self.hfo_app.classify_artifacts([seconds_to_ignore_before,seconds_to_ignore_after], threshold)
        if not artifact_only:
            self.hfo_app.classify_spikes()
        return []

    def _classify_finished(self):
        self.message_handler("Classification finished!..")
        self.update_statistics_label()
        self.waveform_plot.set_plot_HFOs(True)
        self.save_csv_button.setEnabled(True)

    def classify(self,check_spike=True):
        self.message_handler("Classifying HFOs...")
        if check_spike:
            use_spike=self.overview_use_spike_checkbox.isChecked()
        else:
            use_spike=False
        worker=Worker(lambda progress_callback: self._classify((not use_spike)))
        worker.signals.result.connect(self._classify_finished)
        self.threadpool.start(worker)

    def update_statistics_label(self):
        num_HFO = self.hfo_app.hfo_features.get_num_HFO()
        num_artifact = self.hfo_app.hfo_features.get_num_artifact()
        num_spike = self.hfo_app.hfo_features.get_num_spike()
        num_real = self.hfo_app.hfo_features.get_num_real()

        self.statistics_label.setText(" Number of HFOs: " + str(num_HFO) +\
                                      "\n Number of artifacts: " + str(num_artifact) +\
                                        "\n Number of spikes: " + str(num_spike) +\
                                        "\n Number of real HFOs: " + str(num_real))
    
    def save_to_excel(self):
        #open file dialog
        fname,_  = QFileDialog.getSaveFileName(self, 'Save file', "", ".xlsx files (*.xlsx)")
        if fname:
            self.hfo_app.export_excel(fname)

    def _save_to_npz(self,fname,progress_callback):
        self.hfo_app.export_app(fname)
        return []

    def save_to_npz(self):
        #open file dialog
        # print("saving to npz...",end="")
        fname,_  = QFileDialog.getSaveFileName(self, 'Save file', "", ".npz files (*.npz)")
        if fname:
            # print("saving to {fname}...",end="")
            worker = Worker(self._save_to_npz, fname)
            worker.signals.result.connect(lambda: 0)
            self.threadpool.start(worker)

    def _load_from_npz(self,fname,progress_callback):
        self.hfo_app = self.hfo_app.import_app(fname)
        return []

    def load_from_npz(self):
        #open file dialog
        fname,_  = QFileDialog.getOpenFileName(self, 'Open file', "", ".npz files (*.npz)")
        self.message_handler("Loading from npz...")
        if fname:
            self.reinitialize()
            worker = Worker(self._load_from_npz, fname)
            worker.signals.result.connect(self.load_from_npz_finished)
            self.threadpool.start(worker)
        # print(self.hfo_app.get_edf_info())

    def load_from_npz_finished(self):
        edf_info = self.hfo_app.get_edf_info()
        self.waveform_plot.update_backend(self.hfo_app)
        self.waveform_plot.init_eeg_data()
        edf_name=str(edf_info["edf_fn"])
        edf_name=edf_name[edf_name.rfind("/")+1:]
        self.update_edf_info([edf_name, str(edf_info["sfreq"]), 
                              str(edf_info["nchan"]), str(self.hfo_app.eeg_data.shape[1])])
        #update number of jobs
        self.n_jobs_spinbox.setValue(self.hfo_app.n_jobs)
        if self.hfo_app.filtered:
            self.filtering_complete()
            filter_param = self.hfo_app.param_filter
            #update filter params
            self.fp_input.setText(str(filter_param.fp))
            self.fs_input.setText(str(filter_param.fs))
            self.rp_input.setText(str(filter_param.rp))
            self.rs_input.setText(str(filter_param.rs))
        #update the detector parameters:
        if self.hfo_app.detected:
            self.set_detector_param_display()
            self._detect_finished()
            self.update_statistics_label()
        #update classifier param
        if self.hfo_app.classified:
            self.set_classifier_param_display()
            self._classify_finished()
            self.update_statistics_label()

    def update_ste_params(self,ste_params):
        rms_window = str(ste_params["rms_window"])
        min_window = str(ste_params["min_window"])
        min_gap = str(ste_params["min_gap"])
        epoch_len = str(ste_params["epoch_len"])
        min_osc = str(ste_params["min_osc"])
        rms_thres = str(ste_params["rms_thres"])
        peak_thres = str(ste_params["peak_thres"])

        self.ste_rms_window_input.setText(rms_window)
        self.ste_min_window_input.setText(min_window)
        self.ste_min_gap_input.setText(min_gap)
        self.ste_epoch_length_input.setText(epoch_len)
        self.ste_min_oscillation_input.setText(min_osc)
        self.ste_rms_threshold_input.setText(rms_thres)
        self.ste_peak_threshold_input.setText(peak_thres)
        
        #set display parameters
        self.ste_epoch_display.setText(epoch_len)
        self.ste_min_window_display.setText(min_window)
        self.ste_rms_window_display.setText(rms_window)
        self.ste_min_gap_time_display.setText(min_gap)
        self.ste_min_oscillations_display.setText(min_osc)
        self.ste_peak_threshold_display.setText(peak_thres)
        self.ste_rms_threshold_display.setText(rms_thres)

        self.update_detector_tab("STE")
        self.detector_subtabs.setCurrentIndex(0)

    def update_mni_params(self,mni_params):
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

        self.mni_epoch_time_input.setText(epoch_time)
        self.mni_epoch_chf_input.setText(epo_CHF)
        self.mni_chf_percentage_input.setText(per_CHF)
        self.mni_min_window_input.setText(min_win)
        self.mni_min_gap_time_input.setText(min_gap)
        self.mni_threshold_percentage_input.setText(thrd_perc)
        self.mni_baseline_window_input.setText(base_seg)
        self.mni_baseline_shift_input.setText(base_shift)
        self.mni_baseline_threshold_input.setText(base_thrd)
        self.mni_baseline_min_time_input.setText(base_min)

        #set display parameters
        self.mni_epoch_display.setText(epoch_time)
        self.mni_epoch_chf_display.setText(epo_CHF)
        self.mni_chf_percentage_display.setText(per_CHF)
        self.mni_min_window_display.setText(min_win)
        self.mni_min_gap_time_display.setText(min_gap)
        self.mni_threshold_percentile_display.setText(thrd_perc)
        self.mni_baseline_window_display.setText(base_seg)
        self.mni_baseline_shift_display.setText(base_shift)
        self.mni_baseline_threshold_display.setText(base_thrd)
        self.mni_baseline_min_time_display.setText(base_min)

        self.update_detector_tab("MNI")
        self.detector_subtabs.setCurrentIndex(1)

    def set_detector_param_display(self):
        detector_params = self.hfo_app.param_detector
        detector_type = detector_params.detector_type.lower()
        if detector_type == "ste":
            self.update_ste_params(detector_params.detector_param.to_dict())
        elif detector_type == "mni":
            self.update_mni_params(detector_params.detector_param.to_dict())
    
    def open_bipolar_channel_selection(self):
        self.bipolar_channel_selection_window = BipolarChannelSelectionWindow(self.hfo_app, self, self.close_signal,self.waveform_plot)
        self.bipolar_channel_selection_window.show()

    def open_annotation(self):
        self.save_csv_button.setEnabled(True)
        annotation = HFOAnnotation(self.hfo_app, self, self.close_signal)
        annotation.show()


def closeAllWindows():
    QApplication.instance().closeAllWindows()


if __name__ == '__main__':
    mp.freeze_support()
    app = QApplication(sys.argv)
    mainWindow = HFOMainWindow()
    mainWindow.show()
    app.aboutToQuit.connect(closeAllWindows)
    sys.exit(app.exec_())
    
