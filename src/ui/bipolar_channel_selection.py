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
    def __init__(self, hfo_app=None, main_window=None, close_signal = None,waveform_plot = None):
        # print(ROOT_DIR)
        super(BipolarChannelSelectionWindow, self).__init__()
        self.ui = uic.loadUi(os.path.join(ROOT_DIR, 'bipolar_channel_selection.ui'), self)

        self.hfo_app = hfo_app
        self.main_window = main_window
        self.setWindowTitle("Bipolar Channel Selection")
        self.setWindowIcon(QtGui.QIcon(os.path.join(ROOT_DIR, 'images/icon.png')))

        eeg_data,channel_names = self.hfo_app.get_eeg_data()

        for channel in channel_names:
        #check if channel is not already in the list, then concat
            if "#-#" not in channel:
                self.ch_1_dropdown.addItem((channel))
                self.ch_2_dropdown.addItem((channel))

        #connect cancel button to close window
        self.cancel_button.clicked.connect(self.close)
        #conncet ok button to get channels to show
        self.ok_button.clicked.connect(self.check_channels)
        self.waveform_plot = waveform_plot
        self.close_signal = close_signal
        self.close_signal.connect(self.close)

    def check_channels(self):

        self.channel_1 = self.ch_1_dropdown.currentText()
        self.channel_2 = self.ch_2_dropdown.currentText()
        
        #remove
        # print(self.channel_1)
        # print(self.channel_2)

        if str(self.channel_1) != str(self.channel_2):
            if f"{self.channel_1}#-#{self.channel_2}" not in self.hfo_app.channel_names:
                #create bipolar channel and add to data, channel_name lists
                self.hfo_app.add_bipolar_channel(self.channel_1,self.channel_2)
                self.waveform_plot.update_channel_names(self.hfo_app.channel_names)
            self.close()
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('Please select two different channels.')
            msg.setWindowTitle("Channel Selection Error")
            msg.exec_()

        # self.ch_1_dropdown 
