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
import torch

ROOT_DIR = Path(__file__).parent


class ChannelSelectionWindow(QtWidgets.QDialog):
    def __init__(self, backend=None, main_window_model=None, close_signal = None):
        super(ChannelSelectionWindow, self).__init__()
        
        self.backend = backend
        self.main_window_model = main_window_model
        self.layout = QGridLayout()
        self.setWindowTitle("Channel Selection")
        self.setWindowIcon(QtGui.QIcon(os.path.join(ROOT_DIR, 'images/icon1.png')))
        self.setLayout(self.layout)
        
        #create the rest of the checkboxes in a scroll area
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(QtWidgets.QWidget())
        # self.scroll_area.setFixedHeight(300)
        self.layout.addWidget(self.scroll_area, 0, 0, 1, 2)
        self.scroll_layout = QGridLayout()
        self.scroll_area.widget().setLayout(self.scroll_layout)
        #disable horizontal scroll bar
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.set_channels()
        
        #add ok and cancel buttons
        self.ok_button = QtWidgets.QPushButton("OK")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.layout.addWidget(self.ok_button, 1, 0)
        self.layout.addWidget(self.cancel_button, 1, 1)

        #connect cancel button to close window
        self.cancel_button.clicked.connect(self.close)
        #conncet ok button to get channels to show
        self.ok_button.clicked.connect(self.get_channels_to_show)
        
        self.close_signal = close_signal
        self.close_signal.connect(self.close_me)

    def set_channels(self):
        eeg_data,channels = self.backend.get_eeg_data()
        channels_indexes_to_plot = self.main_window_model.get_channel_indices_to_plot()
        self.channel_checkboxes = {}
        self.n_channels = len(channels)
        self.channels = channels
        # TODO group checkbox
        # add select none and select all buttons
        self.check_box_none = QtWidgets.QCheckBox('Select None')
        self.check_box_all = QtWidgets.QCheckBox('Select All')
        self.check_box_none.setCheckState(Qt.Unchecked)
        self.check_box_all.setCheckState(Qt.Checked)
        self.check_box_none.stateChanged.connect(lambda: self.select_channels(False))
        self.check_box_all.stateChanged.connect(lambda: self.select_channels(True))
        self.scroll_layout.addWidget(self.check_box_none, 0, 0)
        self.scroll_layout.addWidget(self.check_box_all, 0, 1)
        for i,channel in enumerate(channels):
            # print(channel)
            #add checkbox
            checkbox = QtWidgets.QCheckBox(f"{channel}")
            checkbox.setObjectName(f"channel_{i}")
            self.channel_checkboxes[channel]=checkbox
            self.__dict__[f"channel_{i}"] = checkbox
            checkbox.stateChanged.connect(self.channel_clicked)
            # checkbox.setChecked(True)
            self.scroll_layout.addWidget(QtWidgets.QCheckBox(f"{channel}, amplitude: {round(np.ptp(eeg_data[i]),3)} uV"),
                                  i//2 + 1, i % 2)

        for i in range(self.n_channels):
            if i in channels_indexes_to_plot:
                self.scroll_layout.itemAtPosition(i//2 +1 ,i%2).widget().setChecked(True)
                self.scroll_layout.itemAtPosition(i // 2 + 1, i % 2).widget().stateChanged.connect(self.check_channel_state)
   
    def channel_clicked(self):
        #print("clicked")
        pass
    
    def select_channels(self, state):
        for i in range(self.n_channels):
            self.scroll_layout.itemAtPosition(1+i//2, i % 2).widget().blockSignals(True)
            self.scroll_layout.itemAtPosition(1+i//2, i % 2).widget().setChecked(state)
            self.scroll_layout.itemAtPosition(1 + i // 2, i % 2).widget().blockSignals(False)
        self.check_channel_state()

    def check_channel_state(self):
        states = [self.scroll_layout.itemAtPosition(1+i//2, i % 2).widget().isChecked() for i in range(self.n_channels)]
        self.check_box_none.blockSignals(True)
        self.check_box_all.blockSignals(True)
        self.check_box_none.setCheckState(Qt.Unchecked if not any(states) else Qt.PartiallyChecked)
        self.check_box_all.setCheckState(Qt.Checked if all(states) else Qt.PartiallyChecked)
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
