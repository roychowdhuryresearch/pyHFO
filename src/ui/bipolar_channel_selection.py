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
        fit_window_to_screen(
            self,
            default_width=760,
            default_height=240,
            min_width=620,
            min_height=220,
            width_ratio=0.68,
            height_ratio=0.4,
        )

        self.label_3.setText("Create a derived channel by subtracting Channel 2 from Channel 1.")
        self.label_3.setProperty("helperText", True)
        self.label_3.setWordWrap(True)
        self.label_3.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.ok_button.setText("Create Channel")
        set_accent_button(self.ok_button)
        self.ok_button.setDefault(True)
        self.ok_button.setAutoDefault(True)
        self.cancel_button.setDefault(False)
        self.cancel_button.setAutoDefault(False)

        eeg_data,channel_names = self.backend.get_eeg_data()

        for channel in channel_names:
        #check if channel is not already in the list, then concat
            if "#-#" not in channel:
                self.ch_1_dropdown.addItem((channel))
                self.ch_2_dropdown.addItem((channel))
        if self.ch_2_dropdown.count() > 1:
            self.ch_2_dropdown.setCurrentIndex(1)
        self.ch_1_dropdown.setMinimumWidth(260)
        self.ch_2_dropdown.setMinimumWidth(260)
        self._rebuild_dialog_shell()
        apply_subwindow_theme(self)

        #connect cancel button to close window
        safe_connect_signal_slot(self.cancel_button.clicked, self.close)
        #conncet ok button to get channels to show
        safe_connect_signal_slot(self.ok_button.clicked, self.check_channels)
        safe_connect_signal_slot(self.ch_1_dropdown.currentIndexChanged, self._update_action_state)
        safe_connect_signal_slot(self.ch_2_dropdown.currentIndexChanged, self._update_action_state)
        self.waveform_plot = waveform_plot
        self.close_signal = close_signal
        if self.close_signal is not None:
            safe_connect_signal_slot(self.close_signal, self.close)
        self._update_action_state()

    def _rebuild_dialog_shell(self):
        root_layout = self.layout()
        if root_layout is None:
            return

        while root_layout.count():
            root_layout.takeAt(0)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setHorizontalSpacing(0)
        root_layout.setVerticalSpacing(12)

        title = QtWidgets.QLabel("Create bipolar channel", self)
        title.setProperty("dialogTitle", True)
        root_layout.addWidget(title, 0, 0, 1, 2)
        root_layout.addWidget(self.label_3, 1, 0, 1, 2)

        form_card = QtWidgets.QFrame(self)
        form_card.setProperty("surfaceCard", True)
        form_layout = QtWidgets.QGridLayout(form_card)
        form_layout.setContentsMargins(14, 14, 14, 14)
        form_layout.setHorizontalSpacing(12)
        form_layout.setVerticalSpacing(8)

        self.label.setText("Channel 1")
        self.label.setProperty("fieldLabel", True)
        self.label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.label_2.setText("Channel 2")
        self.label_2.setProperty("fieldLabel", True)
        self.label_2.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.ch_1_dropdown.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.ch_2_dropdown.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        hint = QtWidgets.QLabel("New channel = Channel 1 - Channel 2", form_card)
        hint.setProperty("helperText", True)
        hint.setAlignment(QtCore.Qt.AlignCenter)
        self.selection_status_label = QtWidgets.QLabel("", form_card)
        self.selection_status_label.setProperty("helperText", True)
        self.selection_status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.selection_status_label.setWordWrap(True)

        form_layout.addWidget(self.label, 0, 0)
        form_layout.addWidget(self.label_2, 0, 1)
        form_layout.addWidget(self.ch_1_dropdown, 1, 0)
        form_layout.addWidget(self.ch_2_dropdown, 1, 1)
        form_layout.addWidget(hint, 2, 0, 1, 2)
        form_layout.addWidget(self.selection_status_label, 3, 0, 1, 2)
        root_layout.addWidget(form_card, 2, 0, 1, 2)

        self.ok_button.setMinimumWidth(160)
        self.cancel_button.setMinimumWidth(120)
        self.ok_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.cancel_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        root_layout.addWidget(self.ok_button, 3, 0)
        root_layout.addWidget(self.cancel_button, 3, 1)

    def _update_action_state(self, *_args):
        channel_1 = self.ch_1_dropdown.currentText().strip()
        channel_2 = self.ch_2_dropdown.currentText().strip()
        can_create = bool(channel_1 and channel_2 and channel_1 != channel_2)
        self.ok_button.setEnabled(can_create)
        if self.ch_1_dropdown.count() < 2:
            status_text = "At least two source channels are required to create a bipolar pair."
        elif can_create:
            status_text = "Ready to create the derived bipolar channel from the selected pair."
        else:
            status_text = "Choose two different source channels to enable Create Channel."
        self.selection_status_label.setText(status_text)
        if can_create:
            self.ok_button.setToolTip("Create the derived bipolar channel from the selected pair.")
        else:
            self.ok_button.setToolTip("Choose two different source channels to create a bipolar pair.")

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
