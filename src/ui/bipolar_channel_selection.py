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
    def __init__(self, main_window_model=None, backend=None, main_window=None, close_signal=None, waveform_plot=None):
        super(BipolarChannelSelectionWindow, self).__init__()
        self.ui = uic.loadUi(os.path.join(ROOT_DIR, 'bipolar_channel_selection.ui'), self)

        self.main_window_model = main_window_model
        self.backend = backend
        self.main_window = main_window
        self.waveform_plot = waveform_plot
        self.close_signal = close_signal

        self.setWindowTitle("Bipolar Channel Selection")
        self.setWindowIcon(QtGui.QIcon(os.path.join(ROOT_DIR, 'images/icon1.png')))
        fit_window_to_screen(
            self,
            default_width=760,
            default_height=420,
            min_width=620,
            min_height=320,
            width_ratio=0.68,
            height_ratio=0.6,
        )

        self.label_3.setText("Create a derived channel by subtracting Channel 2 from Channel 1.")
        self.label_3.setProperty("helperText", True)
        self.label_3.setWordWrap(True)
        self.label_3.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.ok_button.setText("Create Channel")
        set_accent_button(self.ok_button)
        self.ok_button.setDefault(True)
        self.ok_button.setAutoDefault(True)
        self.cancel_button.setText("Close")
        self.cancel_button.setDefault(False)
        self.cancel_button.setAutoDefault(False)

        eeg_data, channel_names = self.backend.get_eeg_data()

        for channel in channel_names:
            if "#-#" not in channel:
                self.ch_1_dropdown.addItem((channel))
                self.ch_2_dropdown.addItem((channel))
        if self.ch_2_dropdown.count() > 1:
            self.ch_2_dropdown.setCurrentIndex(1)
        self.ch_1_dropdown.setMinimumWidth(260)
        self.ch_2_dropdown.setMinimumWidth(260)
        self._rebuild_dialog_shell()
        apply_subwindow_theme(self)

        safe_connect_signal_slot(self.cancel_button.clicked, self.close)
        safe_connect_signal_slot(self.ok_button.clicked, self.check_channels)
        safe_connect_signal_slot(self.ch_1_dropdown.currentIndexChanged, self._update_action_state)
        safe_connect_signal_slot(self.ch_2_dropdown.currentIndexChanged, self._update_action_state)

        if self.close_signal is not None:
            safe_connect_signal_slot(self.close_signal, self.close)
        self._update_action_state()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_derived_channels(self):
        """Return channel names that are derived (contain #-#)."""
        if self.backend is None or self.backend.channel_names is None:
            return []
        return [str(n) for n in self.backend.channel_names if "#-#" in str(n)]

    def _format_bipolar_label(self, channel_name):
        """Turn 'FP1#-#FP2' into 'FP1 − FP2'."""
        parts = str(channel_name).split("#-#", 1)
        if len(parts) == 2:
            return f"{parts[0]} − {parts[1]}"
        return channel_name

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _rebuild_dialog_shell(self):
        root_layout = self.layout()
        if root_layout is None:
            return

        while root_layout.count():
            root_layout.takeAt(0)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setHorizontalSpacing(0)
        root_layout.setVerticalSpacing(12)

        title = QtWidgets.QLabel("Bipolar Channels", self)
        title.setProperty("dialogTitle", True)
        root_layout.addWidget(title, 0, 0, 1, 2)

        # ---- Existing bipolar channels card ----
        existing_card = QtWidgets.QFrame(self)
        existing_card.setProperty("surfaceCard", True)
        existing_layout = QtWidgets.QVBoxLayout(existing_card)
        existing_layout.setContentsMargins(14, 14, 14, 14)
        existing_layout.setSpacing(8)

        existing_header = QtWidgets.QLabel("Current bipolar channels", existing_card)
        existing_header.setProperty("fieldLabel", True)
        existing_layout.addWidget(existing_header)

        self._bipolar_list = QtWidgets.QListWidget(existing_card)
        self._bipolar_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._bipolar_list.setMinimumHeight(80)
        self._bipolar_list.setMaximumHeight(160)
        existing_layout.addWidget(self._bipolar_list)

        remove_row = QtWidgets.QHBoxLayout()
        remove_row.setSpacing(8)
        self._remove_button = QtWidgets.QPushButton("Remove Selected", existing_card)
        self._remove_button.setEnabled(False)
        remove_row.addStretch()
        remove_row.addWidget(self._remove_button)
        existing_layout.addLayout(remove_row)

        root_layout.addWidget(existing_card, 1, 0, 1, 2)

        # ---- Create new bipolar channel section ----
        create_header = QtWidgets.QLabel("Add new bipolar channel", self)
        create_header.setProperty("fieldLabel", True)
        root_layout.addWidget(create_header, 2, 0, 1, 2)

        root_layout.addWidget(self.label_3, 3, 0, 1, 2)

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
        root_layout.addWidget(form_card, 4, 0, 1, 2)

        self.ok_button.setMinimumWidth(160)
        self.cancel_button.setMinimumWidth(120)
        self.ok_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.cancel_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        root_layout.addWidget(self.ok_button, 5, 0)
        root_layout.addWidget(self.cancel_button, 5, 1)

        # Wire up the list and remove button
        safe_connect_signal_slot(self._bipolar_list.itemSelectionChanged, self._on_list_selection_changed)
        safe_connect_signal_slot(self._remove_button.clicked, self._on_remove_selected)

        self._refresh_bipolar_list()

    def _refresh_bipolar_list(self):
        self._bipolar_list.clear()
        derived = self._get_derived_channels()
        if derived:
            for ch in derived:
                item = QtWidgets.QListWidgetItem(self._format_bipolar_label(ch))
                item.setData(QtCore.Qt.UserRole, ch)
                self._bipolar_list.addItem(item)
        else:
            placeholder = QtWidgets.QListWidgetItem("No bipolar channels added yet.")
            placeholder.setFlags(placeholder.flags() & ~QtCore.Qt.ItemIsSelectable)
            placeholder.setData(QtCore.Qt.UserRole, None)
            self._bipolar_list.addItem(placeholder)
        self._on_list_selection_changed()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_list_selection_changed(self):
        selected = self._bipolar_list.selectedItems()
        can_remove = bool(selected and selected[0].data(QtCore.Qt.UserRole) is not None)
        self._remove_button.setEnabled(can_remove)

    def _on_remove_selected(self):
        selected = self._bipolar_list.selectedItems()
        if not selected:
            return
        channel_name = selected[0].data(QtCore.Qt.UserRole)
        if not channel_name:
            return
        removed = self.backend.remove_bipolar_channel(channel_name)
        if not removed:
            return
        self._refresh_bipolar_list()
        if self.waveform_plot is not None:
            self.waveform_plot.update_channel_names(self.backend.channel_names)
        if self.main_window_model is not None:
            self.main_window_model.set_channels_to_plot(
                list(self.backend.channel_names), display_all=False
            )
            self.main_window_model.update_waveform_toolbar_state()
            if hasattr(self.main_window_model, '_start_60hz_background'):
                self.main_window_model._start_60hz_background()

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

        if str(self.channel_1) != str(self.channel_2):
            if f"{self.channel_1}#-#{self.channel_2}" not in self.backend.channel_names:
                self.backend.add_bipolar_channel(self.channel_1, self.channel_2)
                if self.waveform_plot is not None:
                    self.waveform_plot.update_channel_names(self.backend.channel_names)
                self.main_window_model.set_channels_to_plot(self.backend.channel_names, display_all=False)
                if hasattr(self.main_window_model, "update_waveform_toolbar_state"):
                    self.main_window_model.update_waveform_toolbar_state()
            self._refresh_bipolar_list()
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
