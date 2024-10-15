import numpy as np
from pathlib import Path
from PyQt5 import uic
from PyQt5 import QtCore, QtGui, QtWidgets
from src.utils.utils_gui import *

ROOT_DIR = Path(__file__).parent.parent.parent


class MainWindowView(QObject):
    def __init__(self, window):
        super(MainWindowView, self).__init__()
        self.window = window
        # self._init_plot_widget(plot_widget)

    def init_general_window(self):
        self.window.ui = uic.loadUi(os.path.join(ROOT_DIR, 'src/ui/main_window.ui'), self.window)
        self.window.setWindowIcon(QtGui.QIcon(os.path.join(ROOT_DIR, 'src/ui/images/icon1.png'))) 
        self.window.setWindowTitle("pyBrain")

        self.window.threadpool = QThreadPool()
        self.window.replace_last_line = False

    def get_biomarker_type(self):
        return self.window.combo_box_biomarker.currentText()

    def create_stacked_widget_detection_param(self, biomarker_type='HFO'):
        if biomarker_type == 'HFO':
            clear_stacked_widget(self.window.stacked_widget_detection_param)
            page_ste = self.create_detection_parameter_page_ste('Detection Parameters (STE)')
            page_mni = self.create_detection_parameter_page_mni('Detection Parameters (MNI)')
            page_hil = self.create_detection_parameter_page_hil('Detection Parameters (HIL)')
            self.window.stacked_widget_detection_param.addWidget(page_ste)
            self.window.stacked_widget_detection_param.addWidget(page_mni)
            self.window.stacked_widget_detection_param.addWidget(page_hil)

            self.window.detector_subtabs.clear()
            tab_ste = self.create_detection_parameter_tab_ste()
            tab_mni = self.create_detection_parameter_tab_mni()
            tab_hil = self.create_detection_parameter_tab_hil()
            self.window.detector_subtabs.addTab(tab_ste, 'STE')
            self.window.detector_subtabs.addTab(tab_mni, 'MNI')
            self.window.detector_subtabs.addTab(tab_hil, 'HIL')

        elif biomarker_type == 'Spindle':
            clear_stacked_widget(self.window.stacked_widget_detection_param)
            page_yasa = self.create_detection_parameter_page_yasa('Detection Parameters (YASA)')
            self.window.stacked_widget_detection_param.addWidget(page_yasa)

            self.window.detector_subtabs.clear()
            tab_yasa = self.create_detection_parameter_tab_yasa()
            self.window.detector_subtabs.addTab(tab_yasa, 'YASA')

    def create_frame_biomarker(self, biomarker_type='HFO'):
        if biomarker_type == 'HFO':
            clear_layout(self.window.frame_biomarker_layout)
            self.create_frame_biomarker_hfo()
        elif biomarker_type == 'Spindle':
            clear_layout(self.window.frame_biomarker_layout)
            self.create_frame_biomarker_spindle()

    def create_detection_parameter_page_ste(self, groupbox_title):
        page = QWidget()
        layout = QGridLayout()

        detection_groupbox_ste = QGroupBox(groupbox_title)
        ste_parameter_layout = QGridLayout(detection_groupbox_ste)

        clear_layout(ste_parameter_layout)
        # Create widgets
        text_font = QFont('Arial', 11)
        label1 = QLabel('Epoch (s)')
        label2 = QLabel('Min Window (s)')
        label3 = QLabel('RMS Window (s)')
        label4 = QLabel('Min Gap Time (s)')
        label5 = QLabel('Min Oscillations')
        label6 = QLabel('Peak Threshold')
        label7 = QLabel('RMS Threshold')

        self.window.ste_epoch_display = QLabel()
        self.window.ste_epoch_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.ste_epoch_display.setFont(text_font)
        self.window.ste_min_window_display = QLabel()
        self.window.ste_min_window_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.ste_min_window_display.setFont(text_font)
        self.window.ste_rms_window_display = QLabel()
        self.window.ste_rms_window_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.ste_rms_window_display.setFont(text_font)
        self.window.ste_min_gap_time_display = QLabel()
        self.window.ste_min_gap_time_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.ste_min_gap_time_display.setFont(text_font)
        self.window.ste_min_oscillations_display = QLabel()
        self.window.ste_min_oscillations_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.ste_min_oscillations_display.setFont(text_font)
        self.window.ste_peak_threshold_display = QLabel()
        self.window.ste_peak_threshold_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.ste_peak_threshold_display.setFont(text_font)
        self.window.ste_rms_threshold_display = QLabel()
        self.window.ste_rms_threshold_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.ste_rms_threshold_display.setFont(text_font)
        self.window.ste_detect_button = QPushButton('Detect')

        # Add widgets to the grid layout
        ste_parameter_layout.addWidget(label1, 0, 0)  # Row 0, Column 0
        ste_parameter_layout.addWidget(label2, 0, 1)  # Row 0, Column 1
        ste_parameter_layout.addWidget(self.window.ste_epoch_display, 1, 0)  # Row 1, Column 0
        ste_parameter_layout.addWidget(self.window.ste_min_window_display, 1, 1)  # Row 1, Column 1
        ste_parameter_layout.addWidget(label3, 2, 0)
        ste_parameter_layout.addWidget(label4, 2, 1)
        ste_parameter_layout.addWidget(self.window.ste_rms_window_display, 3, 0)
        ste_parameter_layout.addWidget(self.window.ste_min_gap_time_display, 3, 1)
        ste_parameter_layout.addWidget(label5, 4, 0)
        ste_parameter_layout.addWidget(label6, 4, 1)
        ste_parameter_layout.addWidget(self.window.ste_min_oscillations_display, 5, 0)
        ste_parameter_layout.addWidget(self.window.ste_peak_threshold_display, 5, 1)
        ste_parameter_layout.addWidget(label7, 6, 0)
        ste_parameter_layout.addWidget(self.window.ste_rms_threshold_display, 7, 0)
        ste_parameter_layout.addWidget(self.window.ste_detect_button, 7, 1)

        # Set the layout for the page
        layout.addWidget(detection_groupbox_ste)
        page.setLayout(layout)
        return page

    def create_detection_parameter_page_mni(self, groupbox_title):
        page = QWidget()
        layout = QGridLayout()

        detection_groupbox_mni = QGroupBox(groupbox_title)
        mni_parameter_layout = QGridLayout(detection_groupbox_mni)

        clear_layout(mni_parameter_layout)
        # self.detection_groupbox_mni.setTitle("Detection Parameters (MNI)")

        # Create widgets
        text_font = QFont('Arial', 11)
        label1 = QLabel('Epoch (s)')
        label2 = QLabel('Min Window (s)')
        label3 = QLabel('Epoch CHF (s)')
        label4 = QLabel('Min Gap Time (s)')
        label5 = QLabel('CHF Percentage')
        label6 = QLabel('Threshold Percentile')
        label7 = QLabel('Window (s)')
        label8 = QLabel('Shift')
        label9 = QLabel('Threshold')
        label10 = QLabel('Min Time')

        self.window.mni_epoch_display = QLabel()
        self.window.mni_epoch_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.mni_epoch_display.setFont(text_font)
        self.window.mni_min_window_display = QLabel()
        self.window.mni_min_window_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.mni_min_window_display.setFont(text_font)
        self.window.mni_epoch_chf_display = QLabel()
        self.window.mni_epoch_chf_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.mni_epoch_chf_display.setFont(text_font)
        self.window.mni_min_gap_time_display = QLabel()
        self.window.mni_min_gap_time_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.mni_min_gap_time_display.setFont(text_font)
        self.window.mni_chf_percentage_display = QLabel()
        self.window.mni_chf_percentage_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.mni_chf_percentage_display.setFont(text_font)
        self.window.mni_threshold_percentile_display = QLabel()
        self.window.mni_threshold_percentile_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.mni_threshold_percentile_display.setFont(text_font)
        self.window.mni_baseline_window_display = QLabel()
        self.window.mni_baseline_window_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.mni_baseline_window_display.setFont(text_font)
        self.window.mni_baseline_shift_display = QLabel()
        self.window.mni_baseline_shift_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.mni_baseline_shift_display.setFont(text_font)
        self.window.mni_baseline_threshold_display = QLabel()
        self.window.mni_baseline_threshold_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.mni_baseline_threshold_display.setFont(text_font)
        self.window.mni_baseline_min_time_display = QLabel()
        self.window.mni_baseline_min_time_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.mni_baseline_min_time_display.setFont(text_font)
        self.window.mni_detect_button = QPushButton('Detect')

        # Add widgets to the grid layout
        mni_parameter_layout.addWidget(label1, 0, 0)  # Row 0, Column 0
        mni_parameter_layout.addWidget(label2, 0, 1)  # Row 0, Column 1
        mni_parameter_layout.addWidget(self.window.mni_epoch_display, 1, 0)  # Row 1, Column 0
        mni_parameter_layout.addWidget(self.window.mni_min_window_display, 1, 1)  # Row 1, Column 1
        mni_parameter_layout.addWidget(label3, 2, 0)
        mni_parameter_layout.addWidget(label4, 2, 1)
        mni_parameter_layout.addWidget(self.window.mni_epoch_chf_display, 3, 0)
        mni_parameter_layout.addWidget(self.window.mni_min_gap_time_display, 3, 1)
        mni_parameter_layout.addWidget(label5, 4, 0)
        mni_parameter_layout.addWidget(label6, 4, 1)
        mni_parameter_layout.addWidget(self.window.mni_chf_percentage_display, 5, 0)
        mni_parameter_layout.addWidget(self.window.mni_threshold_percentile_display, 5, 1)

        group_box = QGroupBox('Baseline')
        baseline_parameter_layout = QVBoxLayout(group_box)
        baseline_parameter_layout.addWidget(label7)
        baseline_parameter_layout.addWidget(self.window.mni_baseline_window_display)
        baseline_parameter_layout.addWidget(label8)
        baseline_parameter_layout.addWidget(self.window.mni_baseline_shift_display)
        baseline_parameter_layout.addWidget(label9)
        baseline_parameter_layout.addWidget(self.window.mni_baseline_threshold_display)
        baseline_parameter_layout.addWidget(label10)
        baseline_parameter_layout.addWidget(self.window.mni_baseline_min_time_display)

        mni_parameter_layout.addWidget(group_box, 0, 2, 6, 1)  # Row 0, Column 2, span 1 row, 6 columns
        mni_parameter_layout.addWidget(self.window.mni_detect_button, 6, 2)

        # Set the layout for the page
        layout.addWidget(detection_groupbox_mni)
        page.setLayout(layout)
        return page

    def create_detection_parameter_page_hil(self, groupbox_title):
        page = QWidget()
        layout = QGridLayout()

        detection_groupbox_hil = QGroupBox(groupbox_title)
        hil_parameter_layout = QGridLayout(detection_groupbox_hil)

        clear_layout(hil_parameter_layout)
        # self.detection_groupbox_hil.setTitle("Detection Parameters (HIL)")

        # Create widgets
        text_font = QFont('Arial', 11)
        label1 = QLabel('Epoch (s)')
        label2 = QLabel('Min Window (s)')
        label3 = QLabel('Pass Band (Hz)')
        label4 = QLabel('Stop Band (Hz)')
        label5 = QLabel('Sample Frequency')
        label6 = QLabel('Sliding Window')
        label7 = QLabel('Number of Jobs')

        self.window.hil_epoch_time_display = QLabel()
        self.window.hil_epoch_time_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.hil_epoch_time_display.setFont(text_font)
        self.window.hil_min_window_display = QLabel()
        self.window.hil_min_window_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.hil_min_window_display.setFont(text_font)
        self.window.hil_pass_band_display = QLabel()
        self.window.hil_pass_band_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.hil_pass_band_display.setFont(text_font)
        self.window.hil_stop_band_display = QLabel()
        self.window.hil_stop_band_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.hil_stop_band_display.setFont(text_font)
        self.window.hil_sample_freq_display = QLabel()
        self.window.hil_sample_freq_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.hil_sample_freq_display.setFont(text_font)
        self.window.hil_sliding_window_display = QLabel()
        self.window.hil_sliding_window_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.hil_sliding_window_display.setFont(text_font)
        self.window.hil_n_jobs_display = QLabel()
        self.window.hil_n_jobs_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.hil_n_jobs_display.setFont(text_font)
        self.window.hil_detect_button = QPushButton('Detect')

        # Add widgets to the grid layout
        hil_parameter_layout.addWidget(label1, 0, 0)  # Row 0, Column 0
        hil_parameter_layout.addWidget(label2, 0, 1)  # Row 0, Column 1
        hil_parameter_layout.addWidget(self.window.hil_epoch_time_display, 1, 0)  # Row 1, Column 0
        hil_parameter_layout.addWidget(self.window.hil_min_window_display, 1, 1)  # Row 1, Column 1
        hil_parameter_layout.addWidget(label3, 2, 0)
        hil_parameter_layout.addWidget(label4, 2, 1)
        hil_parameter_layout.addWidget(self.window.hil_pass_band_display, 3, 0)
        hil_parameter_layout.addWidget(self.window.hil_stop_band_display, 3, 1)
        hil_parameter_layout.addWidget(label5, 4, 0)
        hil_parameter_layout.addWidget(label6, 4, 1)
        hil_parameter_layout.addWidget(self.window.hil_sample_freq_display, 5, 0)
        hil_parameter_layout.addWidget(self.window.hil_sliding_window_display, 5, 1)
        hil_parameter_layout.addWidget(label7, 6, 0)
        hil_parameter_layout.addWidget(self.window.hil_n_jobs_display, 7, 0)
        hil_parameter_layout.addWidget(self.window.hil_detect_button, 7, 1)

        # Set the layout for the page
        layout.addWidget(detection_groupbox_hil)
        page.setLayout(layout)
        return page

    def create_detection_parameter_page_yasa(self, groupbox_title):
        page = QWidget()
        layout = QGridLayout()

        detection_groupbox_yasa = QGroupBox(groupbox_title)
        yasa_parameter_layout = QGridLayout(detection_groupbox_yasa)

        clear_layout(yasa_parameter_layout)
        # self.detection_groupbox_hil.setTitle("Detection Parameters (HIL)")

        # Create widgets
        text_font = QFont('Arial', 11)
        label1 = QLabel('Freq Spindle (Hz)')
        label2 = QLabel('Freq Broad (Hz)')
        label3 = QLabel('Duration (s)')
        label4 = QLabel('Min Distance (ms)')
        label5 = QLabel('rel_pow')
        label6 = QLabel('corr')
        label7 = QLabel('rms')

        self.window.yasa_freq_sp_display = QLabel()
        self.window.yasa_freq_sp_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.yasa_freq_sp_display.setFont(text_font)
        self.window.yasa_freq_broad_display = QLabel()
        self.window.yasa_freq_broad_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.yasa_freq_broad_display.setFont(text_font)
        self.window.yasa_duration_display = QLabel()
        self.window.yasa_duration_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.yasa_duration_display.setFont(text_font)
        self.window.yasa_min_distance_display = QLabel()
        self.window.yasa_min_distance_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.yasa_min_distance_display.setFont(text_font)
        self.window.yasa_thresh_rel_pow_display = QLabel()
        self.window.yasa_thresh_rel_pow_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.yasa_thresh_rel_pow_display.setFont(text_font)
        self.window.yasa_thresh_corr_display = QLabel()
        self.window.yasa_thresh_corr_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.yasa_thresh_corr_display.setFont(text_font)
        self.window.yasa_thresh_rms_display = QLabel()
        self.window.yasa_thresh_rms_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.window.yasa_thresh_rms_display.setFont(text_font)

        self.window.yasa_detect_button = QPushButton('Detect')

        # Add widgets to the grid layout
        yasa_parameter_layout.addWidget(label1, 0, 0)  # Row 0, Column 0
        yasa_parameter_layout.addWidget(label2, 0, 1)  # Row 0, Column 1
        yasa_parameter_layout.addWidget(self.window.yasa_freq_sp_display, 1, 0)  # Row 1, Column 0
        yasa_parameter_layout.addWidget(self.window.yasa_freq_broad_display, 1, 1)  # Row 1, Column 1
        yasa_parameter_layout.addWidget(label3, 2, 0)
        yasa_parameter_layout.addWidget(label4, 2, 1)
        yasa_parameter_layout.addWidget(self.window.yasa_duration_display, 3, 0)
        yasa_parameter_layout.addWidget(self.window.yasa_min_distance_display, 3, 1)

        group_box = QGroupBox('thresh')
        thresh_parameter_layout = QVBoxLayout(group_box)
        thresh_parameter_layout.addWidget(label5)
        thresh_parameter_layout.addWidget(self.window.yasa_thresh_rel_pow_display)
        thresh_parameter_layout.addWidget(label6)
        thresh_parameter_layout.addWidget(self.window.yasa_thresh_corr_display)
        thresh_parameter_layout.addWidget(label7)
        thresh_parameter_layout.addWidget(self.window.yasa_thresh_rms_display)

        yasa_parameter_layout.addWidget(group_box, 0, 2, 4, 1)  # Row 0, Column 2, span 1 row, 6 columns
        yasa_parameter_layout.addWidget(self.window.mni_detect_button, 4, 2)

        # Set the layout for the page
        layout.addWidget(detection_groupbox_yasa)
        page.setLayout(layout)
        return page

    def create_detection_parameter_tab_ste(self):
        tab = QWidget()
        layout = QGridLayout()

        detection_groupbox = QGroupBox('Detection Parameters')
        parameter_layout = QGridLayout(detection_groupbox)

        clear_layout(parameter_layout)

        # Create widgets
        text_font = QFont('Arial', 11)
        label1 = QLabel('RMS Window')
        label2 = QLabel('Min Window')
        label3 = QLabel('Min Gap')
        label4 = QLabel('Epoch Length')
        label5 = QLabel('Min Oscillations')
        label6 = QLabel('RMS Threshold')
        label7 = QLabel('Peak Threshold')
        label8 = QLabel('sec')
        label9 = QLabel('sec')
        label10 = QLabel('sec')
        label11 = QLabel('sec')

        self.window.ste_rms_window_input = QLineEdit()
        self.window.ste_rms_window_input.setFont(text_font)
        self.window.ste_min_window_input = QLineEdit()
        self.window.ste_min_window_input.setFont(text_font)
        self.window.ste_min_gap_input = QLineEdit()
        self.window.ste_min_gap_input.setFont(text_font)
        self.window.ste_epoch_length_input = QLineEdit()
        self.window.ste_epoch_length_input.setFont(text_font)
        self.window.ste_min_oscillation_input = QLineEdit()
        self.window.ste_min_oscillation_input.setFont(text_font)
        self.window.ste_rms_threshold_input = QLineEdit()
        self.window.ste_rms_threshold_input.setFont(text_font)
        self.window.ste_peak_threshold_input = QLineEdit()
        self.window.ste_peak_threshold_input.setFont(text_font)
        self.window.STE_save_button = QPushButton('Save')

        # Add widgets to the grid layout
        parameter_layout.addWidget(label1, 0, 0)  # Row 0, Column 0
        parameter_layout.addWidget(self.window.ste_rms_window_input, 0, 1)  # Row 0, Column 1
        parameter_layout.addWidget(label8, 0, 2)
        parameter_layout.addWidget(label2, 1, 0)
        parameter_layout.addWidget(self.window.ste_min_window_input, 1, 1)
        parameter_layout.addWidget(label9, 1, 2)
        parameter_layout.addWidget(label3, 2, 0)
        parameter_layout.addWidget(self.window.ste_min_gap_input, 2, 1)
        parameter_layout.addWidget(label10, 2, 2)
        parameter_layout.addWidget(label4, 3, 0)
        parameter_layout.addWidget(self.window.ste_epoch_length_input, 3, 1)
        parameter_layout.addWidget(label11, 3, 2)

        parameter_layout.addWidget(label5, 4, 0)
        parameter_layout.addWidget(self.window.ste_min_oscillation_input, 4, 1)
        parameter_layout.addWidget(label6, 5, 0)
        parameter_layout.addWidget(self.window.ste_rms_threshold_input, 5, 1)
        parameter_layout.addWidget(label7, 6, 0)
        parameter_layout.addWidget(self.window.ste_peak_threshold_input, 6, 1)

        parameter_layout.addWidget(self.window.STE_save_button, 7, 2)

        # Set the layout for the page
        layout.addWidget(detection_groupbox)
        tab.setLayout(layout)
        return tab

    def create_detection_parameter_tab_mni(self):
        tab = QWidget()
        layout = QGridLayout()

        detection_groupbox = QGroupBox('Detection Parameters')
        parameter_layout = QGridLayout(detection_groupbox)

        clear_layout(parameter_layout)

        # Create widgets
        text_font = QFont('Arial', 11)
        label1 = QLabel('Epoch Time')
        label2 = QLabel('Epoch CHF')
        label3 = QLabel('CHF Percentage')
        label4 = QLabel('Min Window')
        label5 = QLabel('Min Gap Time')
        label6 = QLabel('Threshold Percentage')
        label7 = QLabel('Baseline Window')
        label8 = QLabel('Baseline Shift')
        label9 = QLabel('Baseline Threshold')
        label10 = QLabel('Baseline Minimum Time')
        label11 = QLabel('sec')
        label12 = QLabel('sec')
        label13 = QLabel('sec')
        label14 = QLabel('sec')
        label15 = QLabel('sec')
        label16 = QLabel('%')
        label17 = QLabel('%')

        self.window.mni_epoch_time_input = QLineEdit()
        self.window.mni_epoch_time_input.setFont(text_font)
        self.window.mni_epoch_chf_input = QLineEdit()
        self.window.mni_epoch_chf_input.setFont(text_font)
        self.window.mni_chf_percentage_input = QLineEdit()
        self.window.mni_chf_percentage_input.setFont(text_font)
        self.window.mni_min_window_input = QLineEdit()
        self.window.mni_min_window_input.setFont(text_font)
        self.window.mni_min_gap_time_input = QLineEdit()
        self.window.mni_min_gap_time_input.setFont(text_font)
        self.window.mni_threshold_percentage_input = QLineEdit()
        self.window.mni_threshold_percentage_input.setFont(text_font)
        self.window.mni_baseline_window_input = QLineEdit()
        self.window.mni_baseline_window_input.setFont(text_font)
        self.window.mni_baseline_shift_input = QLineEdit()
        self.window.mni_baseline_shift_input.setFont(text_font)
        self.window.mni_baseline_threshold_input = QLineEdit()
        self.window.mni_baseline_threshold_input.setFont(text_font)
        self.window.mni_baseline_min_time_input = QLineEdit()
        self.window.mni_baseline_min_time_input.setFont(text_font)
        self.window.MNI_save_button = QPushButton('Save')

        # Add widgets to the grid layout
        parameter_layout.addWidget(label1, 0, 0)  # Row 0, Column 0
        parameter_layout.addWidget(self.window.mni_epoch_time_input, 0, 1)  # Row 0, Column 1
        parameter_layout.addWidget(label11, 0, 2)
        parameter_layout.addWidget(label2, 1, 0)
        parameter_layout.addWidget(self.window.mni_epoch_chf_input, 1, 1)
        parameter_layout.addWidget(label12, 1, 2)
        parameter_layout.addWidget(label3, 2, 0)
        parameter_layout.addWidget(self.window.mni_chf_percentage_input, 2, 1)
        parameter_layout.addWidget(label16, 2, 2)
        parameter_layout.addWidget(label4, 3, 0)
        parameter_layout.addWidget(self.window.mni_min_window_input, 3, 1)
        parameter_layout.addWidget(label13, 3, 2)
        parameter_layout.addWidget(label5, 4, 0)
        parameter_layout.addWidget(self.window.mni_min_gap_time_input, 4, 1)
        parameter_layout.addWidget(label14, 4, 2)
        parameter_layout.addWidget(label6, 5, 0)
        parameter_layout.addWidget(self.window.mni_threshold_percentage_input, 5, 1)
        parameter_layout.addWidget(label17, 5, 2)
        parameter_layout.addWidget(label7, 6, 0)
        parameter_layout.addWidget(self.window.mni_baseline_window_input, 6, 1)
        parameter_layout.addWidget(label15, 6, 2)

        parameter_layout.addWidget(label8, 7, 0)
        parameter_layout.addWidget(self.window.mni_baseline_shift_input, 7, 1)
        parameter_layout.addWidget(label9, 8, 0)
        parameter_layout.addWidget(self.window.mni_baseline_threshold_input, 8, 1)
        parameter_layout.addWidget(label10, 9, 0)
        parameter_layout.addWidget(self.window.mni_baseline_min_time_input, 9, 1)

        parameter_layout.addWidget(self.window.MNI_save_button, 10, 2)

        # Set the layout for the page
        layout.addWidget(detection_groupbox)
        tab.setLayout(layout)
        return tab

    def create_detection_parameter_tab_hil(self):
        tab = QWidget()
        layout = QGridLayout()

        detection_groupbox = QGroupBox('Detection Parameters')
        parameter_layout = QGridLayout(detection_groupbox)

        clear_layout(parameter_layout)

        # Create widgets
        text_font = QFont('Arial', 11)
        label1 = QLabel('Sample Frequency')
        label2 = QLabel('Pass Band')
        label3 = QLabel('Stop Band')
        label4 = QLabel('Epoch Time')
        label5 = QLabel('Sliding Window')
        label6 = QLabel('Min Window')
        label7 = QLabel('Number of Jobs')
        label8 = QLabel('sec')
        label9 = QLabel('sec')
        label10 = QLabel('sec')
        label11 = QLabel('Hz')
        label12 = QLabel('Hz')
        label13 = QLabel('Hz')

        self.window.hil_sample_freq_input = QLineEdit()
        self.window.hil_sample_freq_input.setFont(text_font)
        self.window.hil_pass_band_input = QLineEdit()
        self.window.hil_pass_band_input.setFont(text_font)
        self.window.hil_stop_band_input = QLineEdit()
        self.window.hil_stop_band_input.setFont(text_font)
        self.window.hil_epoch_time_input = QLineEdit()
        self.window.hil_epoch_time_input.setFont(text_font)
        self.window.hil_sliding_window_input = QLineEdit()
        self.window.hil_sliding_window_input.setFont(text_font)
        self.window.hil_min_window_input = QLineEdit()
        self.window.hil_min_window_input.setFont(text_font)
        self.window.hil_n_jobs_input = QLineEdit()
        self.window.hil_n_jobs_input.setFont(text_font)
        self.window.HIL_save_button = QPushButton('Save')

        # Add widgets to the grid layout
        parameter_layout.addWidget(label1, 0, 0)  # Row 0, Column 0
        parameter_layout.addWidget(self.window.hil_sample_freq_input, 0, 1)  # Row 0, Column 1
        parameter_layout.addWidget(label11, 0, 2)
        parameter_layout.addWidget(label2, 1, 0)
        parameter_layout.addWidget(self.window.hil_pass_band_input, 1, 1)
        parameter_layout.addWidget(label12, 1, 2)
        parameter_layout.addWidget(label3, 2, 0)
        parameter_layout.addWidget(self.window.hil_stop_band_input, 2, 1)
        parameter_layout.addWidget(label13, 2, 2)
        parameter_layout.addWidget(label4, 3, 0)
        parameter_layout.addWidget(self.window.hil_epoch_time_input, 3, 1)
        parameter_layout.addWidget(label8, 3, 2)
        parameter_layout.addWidget(label5, 4, 0)
        parameter_layout.addWidget(self.window.hil_sliding_window_input, 4, 1)
        parameter_layout.addWidget(label9, 4, 2)
        parameter_layout.addWidget(label6, 5, 0)
        parameter_layout.addWidget(self.window.hil_min_window_input, 5, 1)
        parameter_layout.addWidget(label10, 5, 2)

        parameter_layout.addWidget(label7, 6, 0)
        parameter_layout.addWidget(self.window.hil_n_jobs_input, 6, 1)

        parameter_layout.addWidget(self.window.HIL_save_button, 7, 2)

        # Set the layout for the page
        layout.addWidget(detection_groupbox)
        tab.setLayout(layout)
        return tab

    def create_detection_parameter_tab_yasa(self):
        tab = QWidget()
        layout = QGridLayout()

        detection_groupbox = QGroupBox('Detection Parameters')
        parameter_layout = QGridLayout(detection_groupbox)

        clear_layout(parameter_layout)

        # Create widgets
        text_font = QFont('Arial', 11)
        label1 = QLabel('Freq Spindle')
        label2 = QLabel('Freq Broad')
        label3 = QLabel('Duration')
        label4 = QLabel('Min Distance')
        label5 = QLabel('Thresh-rel_pow')
        label6 = QLabel('Thresh-corr')
        label7 = QLabel('Thresh-rms')
        label8 = QLabel('Hz')
        label9 = QLabel('Hz')
        label10 = QLabel('sec')
        label11 = QLabel('ms')

        self.window.yasa_freq_sp_input = QLineEdit()
        self.window.yasa_freq_sp_input.setFont(text_font)
        self.window.yasa_freq_broad_input = QLineEdit()
        self.window.yasa_freq_broad_input.setFont(text_font)
        self.window.yasa_duration_input = QLineEdit()
        self.window.yasa_duration_input.setFont(text_font)
        self.window.yasa_min_distance_input = QLineEdit()
        self.window.yasa_min_distance_input.setFont(text_font)
        self.window.yasa_thresh_rel_pow_input = QLineEdit()
        self.window.yasa_thresh_rel_pow_input.setFont(text_font)
        self.window.yasa_thresh_corr_input = QLineEdit()
        self.window.yasa_thresh_corr_input.setFont(text_font)
        self.window.yasa_thresh_rms_input = QLineEdit()
        self.window.yasa_thresh_rms_input.setFont(text_font)
        self.window.YASA_save_button = QPushButton('Save')

        # Add widgets to the grid layout
        parameter_layout.addWidget(label1, 0, 0)  # Row 0, Column 0
        parameter_layout.addWidget(self.window.yasa_freq_sp_input, 0, 1)  # Row 0, Column 1
        parameter_layout.addWidget(label8, 0, 2)
        parameter_layout.addWidget(label2, 1, 0)
        parameter_layout.addWidget(self.window.yasa_freq_broad_input, 1, 1)
        parameter_layout.addWidget(label9, 1, 2)
        parameter_layout.addWidget(label3, 2, 0)
        parameter_layout.addWidget(self.window.yasa_duration_input, 2, 1)
        parameter_layout.addWidget(label10, 2, 2)
        parameter_layout.addWidget(label4, 3, 0)
        parameter_layout.addWidget(self.window.yasa_min_distance_input, 3, 1)
        parameter_layout.addWidget(label11, 3, 2)

        parameter_layout.addWidget(label5, 4, 0)
        parameter_layout.addWidget(self.window.yasa_thresh_rel_pow_input, 4, 1)
        parameter_layout.addWidget(label6, 5, 0)
        parameter_layout.addWidget(self.window.yasa_thresh_corr_input, 5, 1)
        parameter_layout.addWidget(label7, 6, 0)
        parameter_layout.addWidget(self.window.yasa_thresh_rms_input, 6, 1)

        parameter_layout.addWidget(self.window.YASA_save_button, 7, 2)

        # Set the layout for the page
        layout.addWidget(detection_groupbox)
        tab.setLayout(layout)
        return tab

    def create_frame_biomarker_hfo(self):
        # self.frame_biomarker_layout = QHBoxLayout(self.frame_biomarker_type)
        self.window.frame_biomarker_layout.addStretch(1)

        # Add three QLabel widgets to the QFrame
        label_type1 = QLabel("Artifact")
        label_type1.setFixedWidth(150)
        label_type2 = QLabel("spk-HFO")
        label_type2.setFixedWidth(150)
        label_type3 = QLabel("HFO")
        label_type3.setFixedWidth(150)

        line_type1 = QLineEdit()
        line_type1.setReadOnly(True)
        line_type1.setFrame(True)
        line_type1.setFixedWidth(50)
        line_type1.setStyleSheet("background-color: orange;")
        line_type2 = QLineEdit()
        line_type2.setReadOnly(True)
        line_type2.setFrame(True)
        line_type2.setFixedWidth(50)
        line_type2.setStyleSheet("background-color: purple;")
        line_type3 = QLineEdit()
        line_type3.setReadOnly(True)
        line_type3.setFrame(True)
        line_type3.setFixedWidth(50)
        line_type3.setStyleSheet("background-color: green;")

        # Add labels to the layout
        self.window.frame_biomarker_layout.addWidget(line_type1)
        self.window.frame_biomarker_layout.addWidget(label_type1)
        self.window.frame_biomarker_layout.addWidget(line_type2)
        self.window.frame_biomarker_layout.addWidget(label_type2)
        self.window.frame_biomarker_layout.addWidget(line_type3)
        self.window.frame_biomarker_layout.addWidget(label_type3)
        self.window.frame_biomarker_layout.addStretch(1)

    def create_frame_biomarker_spindle(self):
        # self.frame_biomarker_layout = QHBoxLayout(self.frame_biomarker_type)
        self.window.frame_biomarker_layout.addStretch(1)

        # Add three QLabel widgets to the QFrame
        label_type1 = QLabel("Artifact")
        label_type1.setFixedWidth(150)
        label_type2 = QLabel("spk-Spindle")
        label_type2.setFixedWidth(150)
        label_type3 = QLabel("Spindle")
        label_type3.setFixedWidth(150)

        line_type1 = QLineEdit()
        line_type1.setReadOnly(True)
        line_type1.setFrame(True)
        line_type1.setFixedWidth(50)
        line_type1.setStyleSheet("background-color: orange;")
        line_type2 = QLineEdit()
        line_type2.setReadOnly(True)
        line_type2.setFrame(True)
        line_type2.setFixedWidth(50)
        line_type2.setStyleSheet("background-color: purple;")
        line_type3 = QLineEdit()
        line_type3.setReadOnly(True)
        line_type3.setFrame(True)
        line_type3.setFixedWidth(50)
        line_type3.setStyleSheet("background-color: green;")

        # Add labels to the layout
        self.window.frame_biomarker_layout.addWidget(line_type1)
        self.window.frame_biomarker_layout.addWidget(label_type1)
        self.window.frame_biomarker_layout.addWidget(line_type2)
        self.window.frame_biomarker_layout.addWidget(label_type2)
        self.window.frame_biomarker_layout.addWidget(line_type3)
        self.window.frame_biomarker_layout.addWidget(label_type3)
        self.window.frame_biomarker_layout.addStretch(1)

    def add_widget(self, layout, widget):
        attr = getattr(self.window.window_widget, layout)
        method = getattr(attr, 'addWidget')
        method(widget)