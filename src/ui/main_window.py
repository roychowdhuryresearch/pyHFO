import os
import re
import sys
import traceback
from queue import Queue
from PyQt5.QtWidgets import QMessageBox
from src.hfo_app import HFO_App
from src.controllers.main_window_controller import MainWindowController
from src.models.main_window_model import MainWindowModel
from src.views.main_window_view import MainWindowView
from src.utils.utils_gui import *
from PyQt5.QtCore import pyqtSignal


class MainWindow(QMainWindow):
    close_signal = pyqtSignal()

    def __init__(self):
        super(MainWindow, self).__init__()

        self.backend = None
        self.model = MainWindowModel(self, self.backend)
        self.view = MainWindowView(self)
        self.main_window_controller = MainWindowController(self.view, self.model)

        # initialize general UI
        self.main_window_controller.init_general_window()

        # initialize biomarker type
        self.main_window_controller.init_biomarker_type()

        # initialize biomarker specific UI
        biomarker = self.main_window_controller.get_biomarker_type()
        self.main_window_controller.init_biomarker_window(biomarker)

    def create_spindle_window(self):
        clear_stacked_widget(self.stacked_widget_detection_param)
        page_yasa = self.create_detection_parameter_page_yasa('Detection Parameters (YASA)')
        self.stacked_widget_detection_param.addWidget(page_yasa)

        self.detector_subtabs.clear()
        tab_yasa = self.create_detection_parameter_tab_yasa()
        self.detector_subtabs.addTab(tab_yasa, 'YASA')

        # create biomarker type
        clear_layout(self.frame_biomarker_layout)
        self.create_frame_biomarker_spindle()

        self.overview_filter_button.clicked.connect(self.filter_data)
        # set filter button to be disabled by default
        self.overview_filter_button.setEnabled(False)
        # self.show_original_button.clicked.connect(self.toggle_filtered)

        self.is_data_filtered = False

        self.waveform_plot = CenterWaveformAndMiniPlotController(self.waveform_plot_widget, self.waveform_mini_widget,
                                                                 self.hfo_app)

    def create_hypsarrhythmia_window(self):
        print('not implemented yet')

    def create_frame_biomarker_spindle(self):
        # self.frame_biomarker_layout = QHBoxLayout(self.frame_biomarker_type)
        self.frame_biomarker_layout.addStretch(1)

        # Add three QLabel widgets to the QFrame
        self.label_type1 = QLabel("Artifact")
        self.label_type1.setFixedWidth(150)
        self.label_type2 = QLabel("spk-Spindle")
        self.label_type2.setFixedWidth(150)
        self.label_type3 = QLabel("Spindle")
        self.label_type3.setFixedWidth(150)

        self.line_type1 = QLineEdit()
        self.line_type1.setReadOnly(True)
        self.line_type1.setFrame(True)
        self.line_type1.setFixedWidth(50)
        self.line_type1.setStyleSheet("background-color: orange;")
        self.line_type2 = QLineEdit()
        self.line_type2.setReadOnly(True)
        self.line_type2.setFrame(True)
        self.line_type2.setFixedWidth(50)
        self.line_type2.setStyleSheet("background-color: purple;")
        self.line_type3 = QLineEdit()
        self.line_type3.setReadOnly(True)
        self.line_type3.setFrame(True)
        self.line_type3.setFixedWidth(50)
        self.line_type3.setStyleSheet("background-color: green;")

        # Add labels to the layout
        self.frame_biomarker_layout.addWidget(self.line_type1)
        self.frame_biomarker_layout.addWidget(self.label_type1)
        self.frame_biomarker_layout.addWidget(self.line_type2)
        self.frame_biomarker_layout.addWidget(self.label_type2)
        self.frame_biomarker_layout.addWidget(self.line_type3)
        self.frame_biomarker_layout.addWidget(self.label_type3)
        self.frame_biomarker_layout.addStretch(1)

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

        self.yasa_freq_sp_display = QLabel()
        self.yasa_freq_sp_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.yasa_freq_sp_display.setFont(text_font)
        self.yasa_freq_broad_display = QLabel()
        self.yasa_freq_broad_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.yasa_freq_broad_display.setFont(text_font)
        self.yasa_duration_display = QLabel()
        self.yasa_duration_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.yasa_duration_display.setFont(text_font)
        self.yasa_min_distance_display = QLabel()
        self.yasa_min_distance_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.yasa_min_distance_display.setFont(text_font)
        self.yasa_thresh_rel_pow_display = QLabel()
        self.yasa_thresh_rel_pow_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.yasa_thresh_rel_pow_display.setFont(text_font)
        self.yasa_thresh_corr_display = QLabel()
        self.yasa_thresh_corr_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.yasa_thresh_corr_display.setFont(text_font)
        self.yasa_thresh_rms_display = QLabel()
        self.yasa_thresh_rms_display.setStyleSheet("background-color: rgb(235, 235, 235);")
        self.yasa_thresh_rms_display.setFont(text_font)

        self.yasa_detect_button = QPushButton('Detect')

        # Add widgets to the grid layout
        yasa_parameter_layout.addWidget(label1, 0, 0)  # Row 0, Column 0
        yasa_parameter_layout.addWidget(label2, 0, 1)  # Row 0, Column 1
        yasa_parameter_layout.addWidget(self.yasa_freq_sp_display, 1, 0)  # Row 1, Column 0
        yasa_parameter_layout.addWidget(self.yasa_freq_broad_display, 1, 1)  # Row 1, Column 1
        yasa_parameter_layout.addWidget(label3, 2, 0)
        yasa_parameter_layout.addWidget(label4, 2, 1)
        yasa_parameter_layout.addWidget(self.yasa_duration_display, 3, 0)
        yasa_parameter_layout.addWidget(self.yasa_min_distance_display, 3, 1)

        group_box = QGroupBox('thresh')
        thresh_parameter_layout = QVBoxLayout(group_box)
        thresh_parameter_layout.addWidget(label5)
        thresh_parameter_layout.addWidget(self.yasa_thresh_rel_pow_display)
        thresh_parameter_layout.addWidget(label6)
        thresh_parameter_layout.addWidget(self.yasa_thresh_corr_display)
        thresh_parameter_layout.addWidget(label7)
        thresh_parameter_layout.addWidget(self.yasa_thresh_rms_display)

        yasa_parameter_layout.addWidget(group_box, 0, 2, 4, 1)  # Row 0, Column 2, span 1 row, 6 columns
        yasa_parameter_layout.addWidget(self.mni_detect_button, 4, 2)

        # Set the layout for the page
        layout.addWidget(detection_groupbox_yasa)
        page.setLayout(layout)
        return page

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

        self.yasa_freq_sp_input = QLineEdit()
        self.yasa_freq_sp_input.setFont(text_font)
        self.yasa_freq_broad_input = QLineEdit()
        self.yasa_freq_broad_input.setFont(text_font)
        self.yasa_duration_input = QLineEdit()
        self.yasa_duration_input.setFont(text_font)
        self.yasa_min_distance_input = QLineEdit()
        self.yasa_min_distance_input.setFont(text_font)
        self.yasa_thresh_rel_pow_input = QLineEdit()
        self.yasa_thresh_rel_pow_input.setFont(text_font)
        self.yasa_thresh_corr_input = QLineEdit()
        self.yasa_thresh_corr_input.setFont(text_font)
        self.yasa_thresh_rms_input = QLineEdit()
        self.yasa_thresh_rms_input.setFont(text_font)
        self.YASA_save_button = QPushButton('Save')

        # Add widgets to the grid layout
        parameter_layout.addWidget(label1, 0, 0)  # Row 0, Column 0
        parameter_layout.addWidget(self.yasa_freq_sp_input, 0, 1)  # Row 0, Column 1
        parameter_layout.addWidget(label8, 0, 2)
        parameter_layout.addWidget(label2, 1, 0)
        parameter_layout.addWidget(self.yasa_freq_broad_input, 1, 1)
        parameter_layout.addWidget(label9, 1, 2)
        parameter_layout.addWidget(label3, 2, 0)
        parameter_layout.addWidget(self.yasa_duration_input, 2, 1)
        parameter_layout.addWidget(label10, 2, 2)
        parameter_layout.addWidget(label4, 3, 0)
        parameter_layout.addWidget(self.yasa_min_distance_input, 3, 1)
        parameter_layout.addWidget(label11, 3, 2)

        parameter_layout.addWidget(label5, 4, 0)
        parameter_layout.addWidget(self.yasa_thresh_rel_pow_input, 4, 1)
        parameter_layout.addWidget(label6, 5, 0)
        parameter_layout.addWidget(self.yasa_thresh_corr_input, 5, 1)
        parameter_layout.addWidget(label7, 6, 0)
        parameter_layout.addWidget(self.yasa_thresh_rms_input, 6, 1)

        parameter_layout.addWidget(self.YASA_save_button, 7, 2)

        # Set the layout for the page
        layout.addWidget(detection_groupbox)
        tab.setLayout(layout)
        return tab