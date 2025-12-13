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
import math
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


def decimals_for_interval(width: float, base_decimals: int = 2, max_decimals: int = 6) -> int:
    """Choose decimals based on the displayed interval width.

    Rule: floor(-log10(width)) + base_decimals, clamped to [0, max_decimals].
    Examples: width=0.831 -> 2 decimals, width=0.0031 -> 4 decimals.
    """
    if width <= 0 or not math.isfinite(width):
        return base_decimals
    dynamic = int(max(0, math.floor(-math.log10(width))))
    return int(min(max_decimals, base_decimals + dynamic))

def format_time_without_trailing_zeros(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f}"
    return text.rstrip('0').rstrip('.')

class AnnotationPlot(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=10, height=6, dpi=100, backend=None):
        # Use fixed subplot positioning to prevent window shifting during zoom
        fig, self.axs = plt.subplots(
            3,
            1,
            figsize=(width, height),
            dpi=dpi
        )
        # Set fixed subplot spacing to prevent shifting and text overlap
        fig.subplots_adjust(left=0.1, right=0.85, top=0.90, bottom=0.15, hspace=0.75)
        super(AnnotationPlot, self).__init__(fig)
        self.backend = backend
        self.interval = [1.0, 1.0, 1.0]
        
        # Cache for time-frequency data per event
        self.tf_cache = {}  # {event_key: tf_data}
        self.tf_cache_vmin_vmax = {}  # {event_key: (vmin, vmax)} for consistent color scaling
        self.current_event_key = None
        self.max_cached_events = 5  # Limit cache size to prevent excessive memory usage

        FigureCanvasQTAgg.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

        self.boundary_template = {
            "window_start_time": 0,
            "window_end_time": 0,
            "window_start_index": 0,
            "window_end_index": 0,
            "relative_start_index": 0,
            "relative_end_index": 0
        }

        self.zoom_max = 5 # Seconds before and after the event
        self.is_dragging = False

    def set_current_interval(self, interval, ax_idx):
        self.interval[ax_idx] = interval

    def reset_intervals_to_default(self, default_interval):
        """Reset all axis intervals to the default value (e.g., from dropdown)."""
        for ax_idx in range(3):
            self.interval[ax_idx] = default_interval

    def get_event_key(self, channel_name, event_start_index, event_end_index):
        return f"{channel_name}_{event_start_index}_{event_end_index}"

    def get_cached_tf_data(self, channel_name, event_start_index, event_end_index, index_range, ylim=None):
        fs = self.backend.sample_freq
        event_key = self.get_event_key(channel_name, event_start_index, event_end_index)

        zoom_max_samples = int(self.zoom_max * fs)
        full_start_index = max(0, event_start_index - zoom_max_samples)
        full_end_index = min(self.backend.get_eeg_data_shape()[1], event_end_index + zoom_max_samples)
        
        if event_key not in self.tf_cache:
            # Manage cache size
            if len(self.tf_cache) >= self.max_cached_events:
                oldest_key = next(iter(self.tf_cache))
                del self.tf_cache[oldest_key]
                if oldest_key in self.tf_cache_vmin_vmax:
                    del self.tf_cache_vmin_vmax[oldest_key]
            
            # Get full channel data for the zoom range and calculate time-frequency
            full_eeg_data, channel_names = self.backend.get_eeg_data(full_start_index, full_end_index)
            channel_data = full_eeg_data[channel_names == channel_name, :][0]
            tf_data = calculate_time_frequency(channel_data, fs)
            self.tf_cache[event_key] = tf_data
            
            # Compute vmin/vmax from default interval window (same as before zoom was implemented)
            # Use the default interval for the time-frequency plot (ax_idx=2)
            default_interval = self.interval[2]
            length = self.backend.get_eeg_data_shape()[1]
            win_len = int(fs * default_interval)
            ws_idx, we_idx, _, _ = calculate_default_boundary(
                event_start_index, event_end_index, length, win_len=win_len
            )
            # Convert to relative indices within the cached data
            default_relative_start = max(0, ws_idx - full_start_index)
            default_relative_end = min(tf_data.shape[1], we_idx - full_start_index)
            # Extract default interval portion and compute min/max
            default_interval_data = tf_data[:, default_relative_start:default_relative_end]
            if default_interval_data.size > 0:
                self.tf_cache_vmin_vmax[event_key] = (np.min(default_interval_data), np.max(default_interval_data))
            else:
                # Fallback to full data if default interval extraction fails
                self.tf_cache_vmin_vmax[event_key] = (np.min(tf_data), np.max(tf_data))
        
        # Slice the cached data based on the requested index range
        cached_data = self.tf_cache[event_key]
        
        # Adjust the index range to be relative to the cached data
        relative_start = index_range[0] - full_start_index
        relative_end = index_range[1] - full_start_index
        
        # Ensure indices are within bounds
        relative_start = max(0, min(relative_start, cached_data.shape[1]))
        relative_end = max(relative_start, min(relative_end, cached_data.shape[1]))
        
        # Slice time dimension
        time_sliced_data = cached_data[:, relative_start:relative_end]
        
        # Slice frequency dimension if ylim is provided
        if ylim is not None:
            # The compute_spectrum function creates frequency bins from 500 Hz (top) to 10 Hz (bottom)
            # So the frequency bins are in reverse order: [500, 499, ..., 11, 10]
            freq_bins = np.linspace(500, 10, time_sliced_data.shape[0])
            
            # Find frequency indices within the ylim range
            freq_mask = (freq_bins >= ylim[0]) & (freq_bins <= ylim[1])
            freq_indices = np.where(freq_mask)[0]
            
            if len(freq_indices) > 0:
                freq_start = freq_indices[0]
                freq_end = freq_indices[-1] + 1
                return time_sliced_data[freq_start:freq_end, :]
            else:
                # If no frequencies in range, return empty array
                return np.empty((0, time_sliced_data.shape[1]))
        
        return time_sliced_data

    def clear_tf_cache(self):
        self.tf_cache.clear()
        self.tf_cache_vmin_vmax.clear()
        self.current_event_key = None

    def configure_plot_axes(self, ax, title, ylabel, yformatter, x_decimals: int, xlim=None, ylim=None):
        ax.set_title(title)
        ax.set_ylabel(ylabel, rotation=90, labelpad=6)
        ax.set_xlabel("Time (s)")
        ax.yaxis.set_major_formatter(yformatter)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: format_time_without_trailing_zeros(x, x_decimals))
        )
        ax.yaxis.set_label_position("right")
        if ylim:
            ax.set_ylim(ylim)
        if xlim:
            ax.set_xlim(xlim)

    def plot_all_axes(self, event_start_index: int = None, event_end_index: int = None, channel: str = None):
        for ax_idx in range(3):
            self.plot(ax_idx, event_start_index, event_end_index, channel)

    def plot(self, ax_idx: int, event_start_index: int = None, event_end_index: int = None, channel: str = None, xlim: tuple = None, ylim: tuple = None):
        
        self.axs[ax_idx].cla()

        if event_start_index is None or event_start_index < 0:
            print (f"Invalid event start index: {event_start_index}")
            return
        if xlim is not None and xlim[1] < xlim[0]:
            print (f"Invalid xlim: {xlim}")
            return
        if ylim is not None and ylim[1] < ylim[0]:
            print (f"Invalid ylim: {ylim}")
            return

        fs = self.backend.sample_freq
        channel_name = channel
        
        # Track event changes for cache management
        current_event_key = self.get_event_key(channel_name, event_start_index, event_end_index)
        if current_event_key != self.current_event_key:
            self.current_event_key = current_event_key
        prediction = self.backend.event_features.get_current_info()["prediction"]
        event_color = COLOR_MAP.get(prediction, COLOR_MAP['HFO'])
        signal_color = COLOR_MAP['waveform']
        win_len = int(fs * self.interval[ax_idx])
        length = self.backend.get_eeg_data_shape()[1]

        boundaries = self.boundary_template.copy()

        # Determine window in time
        if xlim is not None:
            boundaries["window_start_time"] = xlim[0]
            boundaries["window_end_time"] = xlim[1]
            boundaries["window_start_index"] = int(boundaries["window_start_time"] * fs)
            boundaries["window_end_index"] = int(boundaries["window_end_time"] * fs)
            boundaries["relative_start_index"] = max(0, event_start_index - boundaries["window_start_index"])
            boundaries["relative_end_index"] = max(0, event_end_index - boundaries["window_start_index"])
        else:
            # Use same default boundary for all 3 subplots
            ws_idx, we_idx, rs_idx, re_idx = calculate_default_boundary(
                event_start_index, event_end_index, length, win_len=win_len
            )
            boundaries["window_start_index"] = ws_idx
            boundaries["window_end_index"] = we_idx
            boundaries["relative_start_index"] = rs_idx
            boundaries["relative_end_index"] = re_idx
            boundaries["window_start_time"] = ws_idx / fs
            boundaries["window_end_time"] = we_idx / fs

        # Get EEG data
        # === Get unfiltered EEG data for Plot 0 and 2 ===
        if ax_idx == 0 or ax_idx == 2:
            eeg_data, self.channel_names = self.backend.get_eeg_data(
                boundaries["window_start_index"],
                boundaries["window_end_index"]
            )
    
        # === Get filtered EEG data for Plot 1 ===
        if ax_idx == 1:
            eeg_data, _ = self.backend.get_eeg_data(
                boundaries["window_start_index"],
                boundaries["window_end_index"],
                filtered=True
            )

        # === Slice channels ===
        eeg_data_to_display = eeg_data[self.channel_names == channel_name, :][0]

        # === Time-to-display for each subplot ===
        time = np.arange(eeg_data_to_display.shape[0]) / fs + boundaries["window_start_time"]

        # === Limits ===
        # Use the image's right edge (last pixel edge) for consistent alignment with imshow extent
        dt = 1.0 / fs
        right_edge_time = time[0] + len(time) * dt
        default_xlim = (time[0], right_edge_time)
        # Determine x tick precision from current interval width with a max of 6 decimals
        x_decimals = decimals_for_interval(self.interval[ax_idx], base_decimals=2, max_decimals=6)
        if ax_idx == 0 or ax_idx == 1:
            default_ylim = (eeg_data_to_display.min(), eeg_data_to_display.max())
        else:
            default_ylim = (10, 500)
        ylim_final = ylim if ylim is not None else default_ylim
        xlim_final = xlim if xlim is not None else default_xlim
        start = boundaries["relative_start_index"]
        end = boundaries["relative_end_index"]

        # === Plot 1: Unfiltered ===
        if ax_idx == 0:
            self.axs[0].plot(time, eeg_data_to_display, color=signal_color)
            self.axs[0].plot(time[start:end], eeg_data_to_display[start:end], color=event_color)
            self.configure_plot_axes(self.axs[0], "EEG Tracing", 'Amplitude (uV)', ticker.FuncFormatter(custom_formatter), x_decimals, xlim_final, ylim_final)

        # === Plot 2: Filtered ===
        if ax_idx == 1:
            self.axs[1].plot(time, eeg_data_to_display, color=signal_color)
            self.axs[1].plot(time[start:end], eeg_data_to_display[start:end], color=event_color)
            self.configure_plot_axes(self.axs[1], "Filtered Tracing", 'Amplitude (uV)', ticker.FuncFormatter(custom_formatter), x_decimals, xlim_final, ylim_final)

        # === Plot 3: Time-Frequency ===
        if ax_idx == 2:
            # Use cached time-frequency data
            index_range = (boundaries["window_start_index"], boundaries["window_end_index"])
            tf_data = self.get_cached_tf_data(channel_name, event_start_index, event_end_index, index_range, ylim_final)
            # Get vmin/vmax from full cached data for consistent color scaling
            event_key = self.get_event_key(channel_name, event_start_index, event_end_index)
            if event_key in self.tf_cache_vmin_vmax:
                vmin, vmax = self.tf_cache_vmin_vmax[event_key]
            else:
                # Fallback: compute from current data if cache entry missing
                vmin, vmax = np.min(tf_data), np.max(tf_data)
            # Align the image extent directly with axis limits to prevent aspect ratio distortion
            # Use xlim_final directly instead of calculating from data to avoid rounding errors
            x_left, x_right = xlim_final
            dt = 1.0 / fs
            epsilon = dt * 0.5  # Slightly overdraw right edge to prevent subpixel white gap
            self.axs[2].imshow(
                tf_data,
                extent=[x_left, x_right + epsilon, ylim_final[0], ylim_final[1]],
                aspect='auto', cmap='jet', vmin=vmin, vmax=vmax
            ) 
            # Use current x limits for ticks to keep them aligned with the displayed window
            x_left, x_right = xlim_final
            self.axs[2].set_xticks(np.linspace(x_left, x_right, 5))
            # y ticks remain integers (Hz)
            self.axs[2].set_yticks(np.linspace(ylim_final[0], ylim_final[1], 5).astype(int))
            self.configure_plot_axes(self.axs[2], "Time Frequency", 'Frequency (Hz)', ticker.FuncFormatter(custom_formatter), x_decimals, xlim_final, ylim_final)
            self.axs[2].set_xlabel('Time (s)')

        self.draw()

       


    def get_active_axes_index(self, event):
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

    def get_mouse_data_position(self, event, ax_idx):
        qt_x = event.pos().x()
        qt_y = event.pos().y()
        widget_h = self.height()
        mpl_x = qt_x
        mpl_y = widget_h - qt_y  # flip y-axis
        
        # Use matplotlib's built-in coordinate transformation
        inv = self.axs[ax_idx].transData.inverted()
        try:
            xdata, ydata = inv.transform((mpl_x, mpl_y))
            # Check for valid coordinates
            if not (np.isfinite(xdata) and np.isfinite(ydata)):
                return None
            return xdata, ydata
        except Exception as e:
            print(f"Transform failed: {e}")
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


        is_zoom_in = event.angleDelta().y() > 0

        # Zoom factor
        zoom_amount = 1.3
        zoom_factor = 1/zoom_amount if is_zoom_in else zoom_amount
        desired_x_range = self.interval[ax_idx] * zoom_factor  # seconds
        
        if desired_x_range <= 0.01:
            return 

        xdata, ydata = self.get_mouse_data_position(event, ax_idx)

        # Determine relative mouse position on the axes
        y_min, y_max = self.axs[ax_idx].get_ylim()
        axes_bbox = self.axs[ax_idx].get_window_extent()
        y_range = max(y_max - y_min, 1e-12)
        y_frac = (ydata - y_min) / y_range
        x_frac = (event.pos().x() - axes_bbox.x0) / axes_bbox.width
        x_frac = np.clip(x_frac, 0.0, 1.0)
        y_frac = np.clip(y_frac, 0.0, 1.0)

        # Desired isotropic zoom (keep aspect ratio of data ranges)
        desired_y_range = y_range * zoom_factor
            
        # Get limits for x and y
        total_samples = self.backend.get_eeg_data_shape()[1]  # shape is (channels, samples)
        
        xlim_max = min(end_index / fs + self.zoom_max, total_samples / fs)
        xlim_min = max(start_index / fs - self.zoom_max, 0)
        if ax_idx == 2:
            ylim_min = 0
            ylim_max = fs/2
        else:
            ylim_min = -np.inf
            ylim_max = np.inf

        # Compute maximum allowed sizes
        max_x_range = xlim_max - xlim_min
        max_y_range = ylim_max - ylim_min

        # Scale desired window to fit within allowed bounds while keeping aspect ratio
        scale_x = max_x_range / desired_x_range if desired_x_range > 0 else 1.0
        scale_y = max_y_range / desired_y_range if (np.isfinite(max_y_range) and desired_y_range > 0) else 1.0
        overall_scale = min(1.0, scale_x, scale_y)
        new_x_range = desired_x_range * overall_scale
        new_y_range = desired_y_range * overall_scale

        # Calculate window starts aiming to keep mouse-relative position, allow drift if clamped
        new_x_start = xdata - new_x_range * x_frac
        new_y_start = ydata - new_y_range * y_frac

        # Clamp starts to ensure window stays within bounds
        new_x_start = min(max(new_x_start, xlim_min), xlim_max - new_x_range)
        if np.isfinite(ylim_min) and np.isfinite(ylim_max):
            new_y_start = min(max(new_y_start, ylim_min), ylim_max - new_y_range)

        xlim = (new_x_start, new_x_start + new_x_range)
        ylim = (new_y_start, new_y_start + new_y_range)

        # Update stored interval (seconds) for this axis
        self.set_current_interval(new_x_range, ax_idx)
        self.plot(ax_idx=ax_idx, event_start_index=start_index, event_end_index=end_index, channel=channel, xlim=xlim, ylim=ylim)
    
    def mousePressEvent(self, event):
        self.is_dragging = True
        ax_idx = self.get_active_axes_index(event)
        if ax_idx is None:
            return
        
        # Store initial mouse position in screen/pixel coordinates
        self.drag_start_screen_x = event.pos().x()
        self.drag_start_screen_y = event.pos().y()
        
        # Store initial data position for reference
        xdata, ydata = self.get_mouse_data_position(event, ax_idx)
        self.drag_start_data_x = xdata
        self.drag_start_data_y = ydata
        
        # Store initial limits when drag starts
        self.drag_start_xlim = self.axs[ax_idx].get_xlim()
        self.drag_start_ylim = self.axs[ax_idx].get_ylim()
    
    def mouseMoveEvent(self, event):
        if not self.is_dragging:
            return
        
        ax_idx = self.get_active_axes_index(event)
        if ax_idx is None:
            return
        
        # Calculate screen coordinate delta
        screen_delta_x = event.pos().x() - self.drag_start_screen_x
        screen_delta_y = event.pos().y() - self.drag_start_screen_y
        
        # Convert screen delta to data coordinate delta
        # Get the current axis limits to calculate the scale
        current_xlim = self.axs[ax_idx].get_xlim()
        current_ylim = self.axs[ax_idx].get_ylim()
        
        # Get axis bounding box in screen coordinates
        axes_bbox = self.axs[ax_idx].get_window_extent()
        
        # Calculate data coordinate delta based on screen delta and current scale
        x_range = current_xlim[1] - current_xlim[0]
        y_range = current_ylim[1] - current_ylim[0]
        
        data_delta_x = -(screen_delta_x / axes_bbox.width) * x_range
        data_delta_y = (screen_delta_y / axes_bbox.height) * y_range  # Note: y is flipped
        
        # Calculate new limits by applying the data delta to the initial limits
        new_xlim = (self.drag_start_xlim[0] + data_delta_x, self.drag_start_xlim[1] + data_delta_x)
        new_ylim = (self.drag_start_ylim[0] + data_delta_y, self.drag_start_ylim[1] + data_delta_y)
        
        # Apply boundary constraints similar to wheelEvent
        event_info = self.backend.event_features.get_current_info()
        start_index = int(event_info["start_index"])
        end_index = int(event_info["end_index"])
        channel = event_info["channel_name"]
        fs = self.backend.sample_freq
        
        total_samples = self.backend.get_eeg_data_shape()[1]
        xlim_max = min(end_index / fs + self.zoom_max, total_samples / fs)
        xlim_min = max(start_index / fs - self.zoom_max, 0)
        
        if ax_idx == 2:
            ylim_min = 0
            ylim_max = fs/2
        else:
            ylim_min = -np.inf
            ylim_max = np.inf
        
        # Clamp x limits - use simpler clamping to avoid oscillations
        x_range = new_xlim[1] - new_xlim[0]
        if new_xlim[0] < xlim_min:
            new_xlim = (xlim_min, xlim_min + x_range)
        elif new_xlim[1] > xlim_max:
            new_xlim = (xlim_max - x_range, xlim_max)
        
        # Clamp y limits if they are finite - use simpler clamping
        if np.isfinite(ylim_min) and np.isfinite(ylim_max):
            y_range = new_ylim[1] - new_ylim[0]
            if new_ylim[0] < ylim_min:
                new_ylim = (ylim_min, ylim_min + y_range)
            elif new_ylim[1] > ylim_max:
                new_ylim = (ylim_max - y_range, ylim_max)
        
        # Update the plot with new limits
        self.plot(ax_idx=ax_idx, event_start_index=start_index, event_end_index=end_index, channel=channel, xlim=new_xlim, ylim=new_ylim)
        
        
    def mouseReleaseEvent(self, event):
        self.is_dragging = False
    


class FFTPlot(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100, backend=None):
        fig,self.axs = plt.subplots(1,1,figsize=(width, height), dpi=dpi)
        # Set fixed subplot spacing to prevent label cutoff
        fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.25)
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
        self.draw()