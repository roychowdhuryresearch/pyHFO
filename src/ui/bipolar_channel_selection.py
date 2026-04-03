from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5 import uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox

import os
import numpy as np
from pathlib import Path
from src.hfo_app import HFO_App
from src.param.param_classifier import ParamClassifier
from src.param.param_detector import ParamDetector, ParamSTE, ParamMNI
from src.param.param_filter import ParamFilter
from src.utils.utils_gui import *

ROOT_DIR = Path(__file__).parent


class BipolarChannelSelectionWindow(QtWidgets.QDialog):
    def __init__(self, main_window_model=None, backend=None, main_window=None, close_signal = None,waveform_plot = None):
        # print(ROOT_DIR)
        super(BipolarChannelSelectionWindow, self).__init__()
        self.ui = uic.loadUi(os.path.join(ROOT_DIR, 'bipolar_channel_selection.ui'), self)

        self.main_window_model = main_window_model
        self.backend = backend
        self.main_window = main_window
        self.setWindowTitle("Bipolar Channel Selection")
        self.setWindowIcon(QtGui.QIcon(os.path.join(ROOT_DIR, 'images/icon1.png')))
        self.resize(760, 240)

        self.label_3.setText("Create a derived channel by subtracting Channel 2 from Channel 1.")
        self.label_3.setProperty("mutedText", True)
        self.label_3.setWordWrap(True)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.ok_button.setText("Create Channel")
        set_accent_button(self.ok_button)

        eeg_data,channel_names = self.backend.get_eeg_data()

        for channel in channel_names:
        #check if channel is not already in the list, then concat
            if "#-#" not in channel:
                self.ch_1_dropdown.addItem((channel))
                self.ch_2_dropdown.addItem((channel))
        self.ch_1_dropdown.setMinimumWidth(260)
        self.ch_2_dropdown.setMinimumWidth(260)
        apply_subwindow_theme(self)

        #connect cancel button to close window
        safe_connect_signal_slot(self.cancel_button.clicked, self.close)
        #conncet ok button to get channels to show
        safe_connect_signal_slot(self.ok_button.clicked, self.check_channels)
        self.waveform_plot = waveform_plot
        self.close_signal = close_signal
        if self.close_signal is not None:
            safe_connect_signal_slot(self.close_signal, self.close)

    def check_channels(self):

        self.channel_1 = self.ch_1_dropdown.currentText()
        self.channel_2 = self.ch_2_dropdown.currentText()
        
        #remove
        # print(self.channel_1)
        # print(self.channel_2)

        if str(self.channel_1) != str(self.channel_2):
            if f"{self.channel_1}#-#{self.channel_2}" not in self.backend.channel_names:
                #create bipolar channel and add to data, channel_name lists
                self.backend.add_bipolar_channel(self.channel_1,self.channel_2)
                self.waveform_plot.update_channel_names(self.backend.channel_names)
                self.main_window_model.set_channels_to_plot(self.backend.channel_names, display_all=False)
            self.close()
        else:
            msg = build_themed_message_box(
                self,
                icon=QMessageBox.Critical,
                title="Channel Selection Error",
                text="Choose two different channels.",
                informative_text="The two dropdowns cannot point to the same source channel.",
            )
            msg.exec_()

        # self.ch_1_dropdown 
