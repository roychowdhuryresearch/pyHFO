from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5 import uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import sys
import re
import os
import numpy as np
from pathlib import Path
from src.hfo_app import HFO_App
from src.param.param_classifier import ParamClassifier
from src.param.param_detector import ParamDetector, ParamSTE, ParamMNI
from src.param.param_filter import ParamFilter
from src.utils.utils_gui import *

import multiprocessing as mp

ROOT_DIR = Path(__file__).parent


class ChannelSelectionWindow(QtWidgets.QDialog):
    def __init__(self, backend=None, main_window_model=None, close_signal = None):
        super(ChannelSelectionWindow, self).__init__()
        
        self.backend = backend
        self.main_window_model = main_window_model
        self.layout = QGridLayout()
        self.layout.setContentsMargins(18, 18, 18, 18)
        self.layout.setHorizontalSpacing(12)
        self.layout.setVerticalSpacing(12)
        self.setWindowTitle("Channel Selection")
        self.setWindowIcon(QtGui.QIcon(os.path.join(ROOT_DIR, 'images/icon1.png')))
        fit_window_to_screen(
            self,
            default_width=760,
            default_height=560,
            min_width=580,
            min_height=420,
            width_ratio=0.72,
            height_ratio=0.74,
        )
        self.setLayout(self.layout)

        self.title_label = QtWidgets.QLabel("Choose visible channels")
        self.title_label.setProperty("dialogTitle", True)
        self.layout.addWidget(self.title_label, 0, 0, 1, 2)

        self.helper_label = QtWidgets.QLabel("Select the channels to keep in the waveform workspace. The list also shows the peak-to-peak amplitude for quick review.")
        self.helper_label.setProperty("mutedText", True)
        self.helper_label.setWordWrap(True)
        self.layout.addWidget(self.helper_label, 1, 0, 1, 2)
        
        #create the rest of the checkboxes in a scroll area
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(QtWidgets.QWidget())
        self.scroll_area.widget().setObjectName("DialogScrollContent")
        # self.scroll_area.setFixedHeight(300)
        self.layout.addWidget(self.scroll_area, 2, 0, 1, 2)
        self.scroll_layout = QGridLayout()
        self.scroll_layout.setContentsMargins(14, 14, 14, 14)
        self.scroll_layout.setHorizontalSpacing(12)
        self.scroll_layout.setVerticalSpacing(10)
        self.scroll_area.widget().setLayout(self.scroll_layout)
        #disable horizontal scroll bar
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.set_channels()
        
        #add ok and cancel buttons
        self.ok_button = QtWidgets.QPushButton("OK")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.layout.addWidget(self.ok_button, 3, 0)
        self.layout.addWidget(self.cancel_button, 3, 1)
        set_accent_button(self.ok_button)
        self.ok_button.setDefault(True)
        self.ok_button.setAutoDefault(True)
        self.cancel_button.setDefault(False)
        self.cancel_button.setAutoDefault(False)

        #connect cancel button to close window
        safe_connect_signal_slot(self.cancel_button.clicked, self.close)
        #conncet ok button to get channels to show
        safe_connect_signal_slot(self.ok_button.clicked, self.get_channels_to_show)

        self.close_signal = close_signal
        if self.close_signal is not None:
            safe_connect_signal_slot(self.close_signal, self.close_me)
        apply_subwindow_theme(self)

    def set_channels(self):
        eeg_data,channels = self.backend.get_eeg_data()
        
        # Handle case where no data is loaded
        if channels is None:
            build_themed_message_box(
                self,
                icon=QMessageBox.Warning,
                title="No Data",
                text="No EEG data available.",
                informative_text="Please load data first.",
            ).exec_()
            self.close()
            return
            
        channels_indexes_to_plot = self.main_window_model.get_channel_indices_to_plot()
        self.channel_checkboxes = {}
        self.n_channels = len(channels)
        self.channels = channels
        # TODO group checkbox
        # add select none and select all buttons
        self.check_box_none = QtWidgets.QCheckBox('Select None')
        self.check_box_all = QtWidgets.QCheckBox('Select All')
        self.check_box_none.setTristate(True)
        self.check_box_all.setTristate(True)
        self.check_box_none.setCheckState(Qt.Unchecked)
        self.check_box_all.setCheckState(Qt.Checked)
        safe_connect_signal_slot(self.check_box_none.stateChanged, lambda: self.select_channels(False))
        safe_connect_signal_slot(self.check_box_all.stateChanged, lambda: self.select_channels(True))
        self.scroll_layout.addWidget(self.check_box_none, 0, 0)
        self.scroll_layout.addWidget(self.check_box_all, 0, 1)
        for i,channel in enumerate(channels):
            amplitude = round(np.ptp(eeg_data[i]), 3)
            checkbox = QtWidgets.QCheckBox(f"{channel} | amplitude: {amplitude} uV")
            checkbox.setObjectName(f"channel_{i}")
            checkbox.setToolTip(f"Peak-to-peak amplitude: {amplitude} uV")
            self.channel_checkboxes[channel]=checkbox
            self.__dict__[f"channel_{i}"] = checkbox
            safe_connect_signal_slot(checkbox.stateChanged, self.channel_clicked)
            self.scroll_layout.addWidget(checkbox, i//2 + 1, i % 2)

        for i in range(self.n_channels):
            if i in channels_indexes_to_plot:
                self.scroll_layout.itemAtPosition(i//2 +1 ,i%2).widget().setChecked(True)
        self.check_channel_state()
   
    def channel_clicked(self):
        self.check_channel_state()
    
    def select_channels(self, state):
        for i in range(self.n_channels):
            self.scroll_layout.itemAtPosition(1+i//2, i % 2).widget().blockSignals(True)
            self.scroll_layout.itemAtPosition(1+i//2, i % 2).widget().setChecked(state)
            self.scroll_layout.itemAtPosition(1 + i // 2, i % 2).widget().blockSignals(False)
        self.check_channel_state()

    def check_channel_state(self):
        states = [self.scroll_layout.itemAtPosition(1+i//2, i % 2).widget().isChecked() for i in range(self.n_channels)]
        checked_count = sum(1 for state in states if state)
        self.check_box_none.blockSignals(True)
        self.check_box_all.blockSignals(True)
        if checked_count == 0:
            self.check_box_none.setCheckState(Qt.Checked)
            self.check_box_all.setCheckState(Qt.Unchecked)
        elif checked_count == self.n_channels:
            self.check_box_none.setCheckState(Qt.Unchecked)
            self.check_box_all.setCheckState(Qt.Checked)
        else:
            self.check_box_none.setCheckState(Qt.PartiallyChecked)
            self.check_box_all.setCheckState(Qt.PartiallyChecked)
        self.check_box_none.blockSignals(False)
        self.check_box_all.blockSignals(False)

    def get_channels_to_show(self):
        channels_to_show = []

        for i in range(self.n_channels):
            if self.scroll_layout.itemAtPosition(1+i//2,i%2).widget().isChecked():
                channels_to_show.append(self.channels[i])
        
        if self.main_window_model is not None:
            self.main_window_model.set_channels_to_plot(channels_to_show)
        # else:
        #     print("main window is none")
        #     print(channels_to_show)
        self.main_window_model.channel_selection_update()
        self.close()


    def close_me(self):
        self.close()
