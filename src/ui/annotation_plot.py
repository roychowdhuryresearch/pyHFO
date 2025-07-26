from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5 import uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import matplotlib.pyplot as plt

import pyqtgraph as pg
import matplotlib.ticker as ticker
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg # We will try using pyqtgraph for plotting
import time
import mne
# from superqt import QDoubleRangeSlider
from tqdm import tqdm
import os
from src.hfo_app import HFO_App
from src.hfo_feature import HFO_Feature

from src.utils.utils_annotation import *
from src.utils.utils_gui import *

import random
import scipy.fft as fft #FFT plot (5)
from scipy.signal import periodogram, welch
import numpy as np

import re
from pathlib import Path
from src.hfo_app import HFO_App
from src.param.param_classifier import ParamClassifier
from src.param.param_detector import ParamDetector, ParamSTE, ParamMNI
from src.param.param_filter import ParamFilter
from src.utils.utils_gui import *
# from src.ui.plot_waveform import *
# import FormatStrFormatter
from matplotlib.ticker import FormatStrFormatter


# from src.ui.plot_annotation_waveform import *
# from src.ui.a_channel_selection import AnnotationChannelSelection 

# from src.plot_time_frequency import PlotTimeFrequencyNoLabel
from src.utils.utils_plotting import *
# from src.plot_time_frequency import MainWindow

import multiprocessing as mp
import torch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
#import fft 
import scipy.fft as fft



def custom_formatter(x, pos):
    # if number >1000, then use scientific notation but still fix the width to 5
    max_width = 5
    if abs(x) > 1000:
        return f'{x:.0e}'
    # 4 digits + 1 for potential negative sign
    formatted_number = f' {x:.0f}' if x >= 0 else f'{x:.0f}'
    return f'{formatted_number:>{max_width}}'


class AnnotationPlot(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=10, height=4, dpi=100, backend=None):
        fig,self.axs = plt.subplots(3,1,figsize=(width, height), dpi=dpi)
        super(AnnotationPlot, self).__init__(fig)
        self.backend = backend
        self.interval = (1.0, 1.0, 1.0)

        FigureCanvasQTAgg.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)
        # self.setParent(parent)
        # self.plot()

    def set_current_interval(self, interval, ax_idx):
        list_interval = list(self.interval)
        list_interval[ax_idx] = interval
        self.interval = tuple(list_interval)

    def configure_plot_axes(self, ax, title, ylabel, yformatter, xlim=None, ylim=None):
        """
        Configure a matplotlib axis with title, label, formatters, and limits.
        
        Args:
            ax (matplotlib.axes.Axes): Axis to configure.
            title (str): Title of the plot.
            ylabel (str): Y-axis label.
            yformatter (Formatter): Y-axis formatter.
            xlim (tuple, optional): X-axis limits.
            ylim (tuple, optional): Y-axis limits.
        """
        ax.set_title(title)
        ax.set_ylabel(ylabel, rotation=90, labelpad=6)
        ax.set_xlabel("Time (s)")
        ax.yaxis.set_major_formatter(yformatter)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
        ax.yaxis.set_label_position("right")
        if ylim:
            ax.set_ylim(ylim)
        if xlim:
            ax.set_xlim(xlim)
    
    def compute_tf_global_minmax(self, event_start_index, event_end_index, channel):
        fs = self.backend.sample_freq
        # Use the default window for the event
        unfiltered_eeg_data, channel_names = self.backend.get_eeg_data(event_start_index, event_end_index)
        unfiltered = unfiltered_eeg_data[channel_names == channel, :][0]
        tf_data = calculate_time_frequency(unfiltered, fs, freq_min=10, freq_max=500)
        self.tf_vmin = np.min(tf_data)
        self.tf_vmax = np.max(tf_data)

    def plot(self, event_start_index: int = None, event_end_index: int = None, channel: str = None, xlims: list = None, ylims: list = None):
        for ax in self.axs:
            ax.cla()

        if event_start_index is None or event_start_index < 0:
            return

        fs = self.backend.sample_freq
        channel_name = channel
        prediction = self.backend.event_features.get_current_info()["prediction"]
        event_color = COLOR_MAP.get(prediction, COLOR_MAP['HFO'])
        signal_color = COLOR_MAP['waveform']
        win_len = int(fs * self.interval[0])
        length = self.backend.get_eeg_data_shape()[1]

        boundary_template = {
            "window_start_time": 0,
            "window_end_time": 0,
            "window_start_index": 0,
            "window_end_index": 0,
            "relative_start_index": 0,
            "relative_end_index": 0
        }

        boundaries = [boundary_template.copy() for _ in range(3)]

        # Determine window in time
        if xlims is not None:
            for i in range(3):
                boundaries[i]["window_start_time"] = xlims[i][0]
                boundaries[i]["window_end_time"] = xlims[i][1]
                boundaries[i]["window_start_index"] = int(boundaries[i]["window_start_time"] * fs)
                boundaries[i]["window_end_index"] = int(boundaries[i]["window_end_time"] * fs)
                boundaries[i]["relative_start_index"] = max(0, event_start_index - boundaries[i]["window_start_index"])
                boundaries[i]["relative_end_index"] = max(0, event_end_index - boundaries[i]["window_start_index"])
        else:
            # Use same default boundary for all 3 subplots
            ws_idx, we_idx, rs_idx, re_idx = calculate_default_boundary(
                event_start_index, event_end_index, length, win_len=win_len
            )
            for i in range(3):
                boundaries[i]["window_start_index"] = ws_idx
                boundaries[i]["window_end_index"] = we_idx
                boundaries[i]["relative_start_index"] = rs_idx
                boundaries[i]["relative_end_index"] = re_idx
                boundaries[i]["window_start_time"] = ws_idx / fs
                boundaries[i]["window_end_time"] = we_idx / fs

        # Get EEG data
        # === Get unfiltered EEG data for Plot 0 ===
        unfiltered_eeg_data, self.channel_names = self.backend.get_eeg_data(
            boundaries[0]["window_start_index"],
            boundaries[0]["window_end_index"]
        )
    
        # === Get filtered EEG data for Plot 1 ===
        filtered_eeg_data, _ = self.backend.get_eeg_data(
            boundaries[1]["window_start_index"],
            boundaries[1]["window_end_index"],
            filtered=True
        )

        unfiltered_eeg_data_2, self.channel_names = self.backend.get_eeg_data(
            boundaries[2]["window_start_index"],
            boundaries[2]["window_end_index"]
        )

        # === Slice channels ===
        unfiltered = unfiltered_eeg_data[self.channel_names == channel_name, :][0]
        filtered = filtered_eeg_data[self.channel_names == channel_name, :][0]
        unfiltered_2 = unfiltered_eeg_data_2[self.channel_names == channel_name, :][0]

        # === Time-to-display for each subplot ===
        time0 = np.arange(unfiltered.shape[0]) / fs + boundaries[0]["window_start_time"]
        time1 = np.arange(filtered.shape[0]) / fs + boundaries[1]["window_start_time"]
        time2 = np.arange(unfiltered_2.shape[0]) / fs + boundaries[2]["window_start_time"]  # Use same signal for TF

        # === Default xlims ===
        default_xlim_0 = (time0[0], time0[-1])
        default_xlim_1 = (time1[0], time1[-1])
        default_xlim_2 = (time2[0], time2[-1])

        # === Plot 1: Unfiltered ===
        y0lim = ylims[0] if ylims is not None and ylims[0] is not None else (unfiltered.min(), unfiltered.max())
        x0lim = xlims[0] if xlims is not None and xlims[0] is not None else default_xlim_0
        self.axs[0].plot(time0, unfiltered, color=signal_color)
        start0 = boundaries[0]["relative_start_index"]
        end0 = boundaries[0]["relative_end_index"]
        self.axs[0].plot(time0[start0:end0], unfiltered[start0:end0], color=event_color)
        self.configure_plot_axes(self.axs[0], "EEG Tracing", 'Amplitude (uV)', ticker.FuncFormatter(custom_formatter), x0lim, y0lim)

        # === Plot 2: Filtered ===
        y1lim = ylims[1] if ylims is not None and ylims[1] is not None else (filtered.min(), filtered.max())
        x1lim = xlims[1] if xlims is not None and xlims[1] is not None else default_xlim_1
        self.axs[1].plot(time1, filtered, color=signal_color)
        start1 = boundaries[1]["relative_start_index"]
        end1 = boundaries[1]["relative_end_index"]
        self.axs[1].plot(time1[start1:end1], filtered[start1:end1], color=event_color)
        self.configure_plot_axes(self.axs[1], "Filtered Tracing", 'Amplitude (uV)', ticker.FuncFormatter(custom_formatter), x1lim, y1lim)

        # === Plot 3: Time-Frequency ===
        x2lim = xlims[2] if xlims is not None and xlims[2] is not None else default_xlim_2
        y2lim = ylims[2] if ylims is not None and ylims[2] is not None else (10, 500)
        tf_data = calculate_time_frequency(unfiltered_2, fs, freq_min=y2lim[0], freq_max=y2lim[1])
        self.compute_tf_global_minmax(event_start_index, event_end_index, channel)
        self.axs[2].imshow(tf_data, extent=[time2[0], time2[-1], y2lim[0], y2lim[1]], aspect='auto', cmap='jet', vmin = self.tf_vmin, vmax = self.tf_vmax)
        self.axs[2].set_xticks(np.linspace(time2[0], time2[-1], 5))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
        self.axs[2].set_xticklabels(np.round(np.linspace(time2[0], time2[-1], 5), 1))
        self.axs[2].set_yticks(np.linspace(y2lim[0], y2lim[1], 5).astype(int))
        self.configure_plot_axes(self.axs[2], "Time Frequency", 'Frequency (Hz)', ticker.FuncFormatter(custom_formatter), x2lim, y2lim)
        self.axs[2].set_xlabel('Time (s)')

        plt.tight_layout()
        self.draw()

        #print(time2[0])


    def get_active_axes_index(self, event):
        """
        Determine which subplot the mouse is currently over (plotting area only).
        Returns:
            int: Index of self.axs the mouse is over, or None if not over any.
        """
        for i, ax in enumerate(self.axs):
            bbox = ax.get_position()  # in figure coordinates
            fig_w, fig_h = self.figure.get_size_inches() * self.figure.dpi  # in pixels
            left = bbox.x0 * fig_w
            right = bbox.x1 * fig_w
            bottom = (1 - bbox.y1) * fig_h  # flip because (0,0) is top-left in Qt coords
            top = (1 - bbox.y0) * fig_h

            if left <= event.x() <= right and bottom <= event.y() <= top:
                return i
        return None
    
    def wheelEvent(self, event):
        
        if not self.backend:
            return
        
        ax_idx = self.get_active_axes_index(event)
        #print(f"Wheel event on axis index: {ax_idx}")
        if ax_idx is None:
            return
        
        event_info = self.backend.event_features.get_current_info()
        start_index = int(event_info["start_index"])
        end_index = int(event_info["end_index"])
        channel = event_info["channel_name"]
        fs = self.backend.sample_freq

        # Initialize full xlims and ylims for all axes
        xlims = []
        ylims = []
        for ax in self.axs:
            xlims.append(ax.get_xlim())
            ylims.append(ax.get_ylim())

        is_zoom_in = event.angleDelta().y() > 0

        # Zoom factor
        zoom_amount = 1.3
        zoom_factor = 1/zoom_amount if is_zoom_in else zoom_amount
        new_interval = self.interval[ax_idx] * zoom_factor
        # Allow zoom factor to change even when hitting boundaries, but keep reasonable limits
        if new_interval < 0.01 or new_interval > 100.0:
            print(f"New interval {new_interval} is out of reasonable bounds")
            return

        # Get mouse position in data coords
        qt_x = event.pos().x()
        qt_y = event.pos().y()
        widget_h = self.height()

        mpl_x = qt_x
        mpl_y = widget_h - qt_y  # flip y-axis

        inv = self.axs[ax_idx].transData.inverted()
        try:
            xdata, ydata = inv.transform((mpl_x, mpl_y))
        except Exception as e:
            print(f"Transform failed: {e}")
            return

        # Determine relative mouse position on the axes
        y_min, y_max = self.axs[ax_idx].get_ylim()
        axes_bbox = self.axs[ax_idx].get_window_extent()
        y_range = y_max - y_min
        y_frac = (ydata - y_min) / y_range
        x_frac = (qt_x - axes_bbox.x0) / axes_bbox.width

        # Calculate new window start so that xdata stays at the same relative position
        new_window_start = xdata - new_interval * x_frac
        new_window_end = new_window_start + new_interval
        new_y_range = y_range * zoom_factor
        new_y_min = ydata - new_y_range * y_frac
        new_y_max = new_y_min + new_y_range

        # Always update the interval to maintain consistent zoom behavior
        self.set_current_interval(new_interval, ax_idx)
        
        # Get data bounds
        total_samples = self.backend.get_eeg_data_shape()[1]  # shape is (channels, samples)
        total_duration = total_samples / fs
        
        # Clamp x-axis window to data bounds while maintaining zoom factor
        clamped_window_start = max(0, min(new_window_start, total_duration - new_interval))
        clamped_window_end = clamped_window_start + new_interval
        
        # Clamp y-axis for time-frequency plot (axis 2)
        if ax_idx == 2:
            max_freq = fs / 2
            # First clamp y_min to be >= 0
            clamped_y_min = max(0, new_y_min)
            # Then clamp y_max to be <= max_freq
            clamped_y_max = min(max_freq, new_y_max)
            # If the range is too large, adjust to fit within bounds
            if clamped_y_max - clamped_y_min > max_freq:
                clamped_y_min = 0
                clamped_y_max = max_freq
            print(f"Clamped y min from {new_y_min} to {clamped_y_min} and y max from {new_y_max} to {clamped_y_max}")
        else:
            clamped_y_min = new_y_min
            clamped_y_max = new_y_max
        
        xlims[ax_idx] = (clamped_window_start, clamped_window_end)
        ylims[ax_idx] = (clamped_y_min, clamped_y_max)
        
        self.plot(event_start_index=start_index, event_end_index=end_index, channel=channel, xlims=xlims, ylims=ylims)
        


class FFTPlot(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100, backend=None):
        fig,self.axs = plt.subplots(1,1,figsize=(width, height), dpi=dpi)
        super(FFTPlot, self).__init__(fig)
        self.backend = backend
        self.min_freq = 10
        self.max_freq = 500
        self.interval = 1.0

        FigureCanvasQTAgg.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

    def set_current_freq_limit(self, min_freq, max_freq):
        self.min_freq = min_freq
        self.max_freq = max_freq

    def set_current_interval(self, interval):
        self.interval = interval

    def plot(self, start_index: int = None, end_index: int = None, channel: str = None):
        self.axs.cla()
        start_index = int(start_index)
        fs = self.backend.sample_freq
        middle_index = (start_index + end_index) // 2
        half_interval_samples = int((self.interval * fs) // 2)
        plot_start_index = int(max(0, middle_index - half_interval_samples))
        plot_end_index = int(min(self.backend.get_eeg_data_shape()[1], middle_index + half_interval_samples))

        unfiltered_eeg_data, channel_names = self.backend.get_eeg_data(plot_start_index, plot_end_index, filtered=False)
        unfiltered_eeg_data = unfiltered_eeg_data[channel_names == channel, :][0]

        # Compute the FFT
        unfiltered_eeg_data -= np.mean(unfiltered_eeg_data) # Remove DC offset
        window = np.hanning(len(unfiltered_eeg_data))
        frequencies, psd = periodogram(unfiltered_eeg_data, fs, window=window)
        # frequencies, psd = welch(unfiltered_eeg_data, fs=fs, window='hann', nperseg=1000, noverlap=500)
        valid_indices = (frequencies >= self.min_freq) & (frequencies <= self.max_freq)
        filtered_freqs = frequencies[valid_indices]
        filtered_psd = psd[valid_indices]
        psd_percent = (filtered_psd / np.sum(filtered_psd)) * 100  # Normalize to sum to 100%

        # Plotting the FFT
        # self.axs.semilogy(f, Pxx_den, color=COLOR_MAP['waveform'])
        self.axs.plot(filtered_freqs, psd_percent, color=COLOR_MAP['waveform'])
        self.axs.set_xlabel('Frequency (Hz)')
        # self.axs.set_ylabel(r"PSD (V$^2$/Hz)")
        self.axs.set_ylabel("PSD (Bandwidth %)")

        # self.axs.set_ylim([1e-7, 1e3])
        # self.axs.set_xlim([min(f), max(f)])
        self.axs.set_xlim([self.min_freq, self.max_freq])
        self.axs.grid(True)
        plt.tight_layout()
        self.draw()