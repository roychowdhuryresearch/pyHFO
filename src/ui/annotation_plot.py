from PyQt5 import QtWidgets
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from src.utils.utils_annotation import calculate_default_boundary
from src.utils.utils_gui import COLOR_MAP
from src.utils.utils_plotting import calculate_time_frequency
from scipy.signal import periodogram


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
        fig, self.axs = plt.subplots(3, 1, figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(left=0.1, right=0.85, top=0.90, bottom=0.15, hspace=0.75)
        super(AnnotationPlot, self).__init__(fig)
        
        self.backend = backend
        self.interval = [1.0, 1.0, 1.0]  # Default interval for each axis
        self.zoom_max = 1  # Seconds before and after the event (max zoom range)
        self.is_dragging = False
        self.sync_views = False
        
        # Track if data has been plotted (for update_view validation)
        self.data_plotted = [False, False, False]
        
        FigureCanvasQTAgg.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

    def set_current_interval(self, interval, ax_idx):
        self.interval[ax_idx] = interval

    def reset_intervals_to_default(self, default_interval):
        """Reset all axis intervals to the default value (e.g., from dropdown)."""
        for ax_idx in range(3):
            self.interval[ax_idx] = default_interval

    def set_sync_views(self, enabled):
        """Enable or disable syncing of view movements across all subplots."""
        self.sync_views = enabled

    def get_axis_limits(self, ax_idx, start_index, end_index, fs):
        """Get the x and y limits for a given axis index."""
        total_samples = self.backend.get_eeg_data_shape()[1]
        xlim_max = min(end_index / fs + self.zoom_max, total_samples / fs)
        xlim_min = max(start_index / fs - self.zoom_max, 0)
        if ax_idx == 2:
            ylim_min, ylim_max = 0, fs / 2
        else:
            ylim_min, ylim_max = -np.inf, np.inf
        return xlim_min, xlim_max, ylim_min, ylim_max

    def configure_plot_axes(self, ax, title, ylabel, yformatter, x_decimals, xlim=None, ylim=None):
        """Configure axis labels, formatters, and limits."""
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

    def plot_all_axes(self, event_start_index: int, event_end_index: int, channel: str):
        """Plot all axes with full data for the zoom range. Called when event changes."""
        for ax_idx in range(3):
            self.plot_full_data(ax_idx, event_start_index, event_end_index, channel, skip_draw=True)
        self.draw()

    def plot_full_data(self, ax_idx: int, event_start_index: int, event_end_index: int, channel: str, skip_draw: bool = False):
        """Plot the full zoom range data for an axis. Called once when event changes."""
        self.axs[ax_idx].cla()
        
        if event_start_index is None or event_start_index < 0:
            print(f"Invalid event start index: {event_start_index}")
            return
        
        fs = self.backend.sample_freq
        
        # Get event info for colors
        prediction = self.backend.event_features.get_current_info()["prediction"]
        event_color = COLOR_MAP.get(prediction, COLOR_MAP['HFO'])
        signal_color = COLOR_MAP['waveform']
        
        # Calculate the FULL zoom range
        total_samples = self.backend.get_eeg_data_shape()[1]
        zoom_max_samples = int(self.zoom_max * fs)
        full_start_index = max(0, event_start_index - zoom_max_samples)
        full_end_index = min(total_samples, event_end_index + zoom_max_samples)
        
        # Get EEG data for the full zoom range
        if ax_idx == 1:
            eeg_data, channel_names = self.backend.get_eeg_data(full_start_index, full_end_index, filtered=True)
        else:
            eeg_data, channel_names = self.backend.get_eeg_data(full_start_index, full_end_index)
        
        eeg_data_to_display = eeg_data[channel_names == channel, :][0]
        time = np.arange(len(eeg_data_to_display)) / fs + full_start_index / fs
        
        # Event indices relative to the plotted data
        event_rel_start = event_start_index - full_start_index
        event_rel_end = event_end_index - full_start_index
        
        # Calculate default view limits
        win_len = int(fs * self.interval[ax_idx])
        ws_idx, we_idx, _, _ = calculate_default_boundary(event_start_index, event_end_index, total_samples, win_len=win_len)
        default_xlim = (ws_idx / fs, we_idx / fs)
        x_decimals = decimals_for_interval(self.interval[ax_idx])
        
        if ax_idx in (0, 1):
            # EEG / Filtered plot
            default_ylim = (eeg_data_to_display.min(), eeg_data_to_display.max())
            
            self.axs[ax_idx].plot(time, eeg_data_to_display, color=signal_color)
            self.axs[ax_idx].plot(
                time[event_rel_start:event_rel_end],
                eeg_data_to_display[event_rel_start:event_rel_end],
                color=event_color
            )
            
            title = "EEG Tracing" if ax_idx == 0 else "Filtered Tracing"
            self.configure_plot_axes(
                self.axs[ax_idx], title, 'Amplitude (uV)',
                ticker.FuncFormatter(custom_formatter), x_decimals,
                default_xlim, default_ylim
            )
        
        else:  # ax_idx == 2: Time-Frequency
            # Use 1 Hz as min freq to avoid division by zero in wavelet transform
            min_freq = 1
            max_freq = fs / 2
            default_ylim = (10, 500)
            
            # Compute time-frequency data
            tf_data = calculate_time_frequency(eeg_data_to_display, fs, freq_min=min_freq, freq_max=max_freq)
            
            # Compute vmin/vmax from the default view window for consistent coloring
            view_start = max(0, ws_idx - full_start_index)
            view_end = min(tf_data.shape[1], we_idx - full_start_index)
            view_data = tf_data[:, view_start:view_end]
            vmin, vmax = (np.min(view_data), np.max(view_data)) if view_data.size > 0 else (np.min(tf_data), np.max(tf_data))
            
            # Plot the image
            extent = [full_start_index / fs, full_end_index / fs, min_freq, max_freq]
            self.axs[ax_idx].imshow(tf_data, extent=extent, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
            
            self.axs[ax_idx].set_xlim(default_xlim)
            self.axs[ax_idx].set_ylim(default_ylim)
            self.axs[ax_idx].set_xticks(np.linspace(default_xlim[0], default_xlim[1], 5))
            self.axs[ax_idx].set_yticks(np.linspace(default_ylim[0], default_ylim[1], 5).astype(int))
            
            self.configure_plot_axes(
                self.axs[ax_idx], "Time Frequency", 'Frequency (Hz)',
                ticker.FuncFormatter(custom_formatter), x_decimals,
                default_xlim, default_ylim
            )
        
        self.data_plotted[ax_idx] = True
        
        if not skip_draw:
            self.draw()

    def update_view(self, ax_idx: int, xlim: tuple, ylim: tuple = None, skip_draw: bool = False):
        """Update only the view limits without replotting data. Fast for zoom/pan."""
        if not self.data_plotted[ax_idx]:
            return
        
        self.axs[ax_idx].set_xlim(xlim)
        if ylim is not None:
            self.axs[ax_idx].set_ylim(ylim)
        
        # Update tick formatting
        x_decimals = decimals_for_interval(xlim[1] - xlim[0])
        self.axs[ax_idx].xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _, d=x_decimals: format_time_without_trailing_zeros(x, d))
        )
        
        if ax_idx == 2:
            self.axs[ax_idx].set_xticks(np.linspace(xlim[0], xlim[1], 5))
            if ylim is not None:
                self.axs[ax_idx].set_yticks(np.linspace(ylim[0], ylim[1], 5).astype(int))
        
        if not skip_draw:
            self.draw()

    def get_active_axes_index(self, event):
        """Get the index of the axis under the mouse cursor."""
        for i, ax in enumerate(self.axs):
            bbox = ax.get_position()
            fig_w, fig_h = self.figure.get_size_inches() * self.figure.dpi
            left = bbox.x0 * fig_w
            right = bbox.x1 * fig_w
            bottom = (1 - bbox.y1) * fig_h
            top = (1 - bbox.y0) * fig_h
            if left <= event.x() <= right and bottom <= event.y() <= top:
                return i
        return None

    def get_mouse_data_position(self, event, ax_idx):
        """Convert mouse position to data coordinates."""
        mpl_x = event.pos().x()
        mpl_y = self.height() - event.pos().y()
        inv = self.axs[ax_idx].transData.inverted()
        try:
            xdata, ydata = inv.transform((mpl_x, mpl_y))
            if np.isfinite(xdata) and np.isfinite(ydata):
                return xdata, ydata
        except Exception:
            pass
        return None

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        if not self.backend:
            return
        
        ax_idx = self.get_active_axes_index(event)
        if ax_idx is None:
            return
        
        event_info = self.backend.event_features.get_current_info()
        start_index = int(event_info["start_index"])
        end_index = int(event_info["end_index"])
        fs = self.backend.sample_freq
        
        # Calculate zoom
        is_zoom_in = event.angleDelta().y() > 0
        zoom_factor = 1/1.3 if is_zoom_in else 1.3
        desired_x_range = self.interval[ax_idx] * zoom_factor
        
        if desired_x_range <= 0.01:
            return
        
        mouse_pos = self.get_mouse_data_position(event, ax_idx)
        if mouse_pos is None:
            return
        xdata, ydata = mouse_pos
        
        # Get current limits and calculate fractions
        y_min, y_max = self.axs[ax_idx].get_ylim()
        axes_bbox = self.axs[ax_idx].get_window_extent()
        y_range = max(y_max - y_min, 1e-12)
        y_frac = np.clip((ydata - y_min) / y_range, 0.0, 1.0)
        x_frac = np.clip((event.pos().x() - axes_bbox.x0) / axes_bbox.width, 0.0, 1.0)
        
        desired_y_range = y_range * zoom_factor
        
        # Get axis limits
        xlim_min, xlim_max, ylim_min, ylim_max = self.get_axis_limits(ax_idx, start_index, end_index, fs)
        max_x_range = xlim_max - xlim_min
        max_y_range = ylim_max - ylim_min
        
        # Scale to fit within bounds
        scale_x = max_x_range / desired_x_range if desired_x_range > 0 else 1.0
        scale_y = max_y_range / desired_y_range if np.isfinite(max_y_range) and desired_y_range > 0 else 1.0
        overall_scale = min(1.0, scale_x, scale_y)
        new_x_range = desired_x_range * overall_scale
        new_y_range = desired_y_range * overall_scale
        
        # Calculate new limits
        new_x_start = np.clip(xdata - new_x_range * x_frac, xlim_min, xlim_max - new_x_range)
        new_y_start = ydata - new_y_range * y_frac
        if np.isfinite(ylim_min) and np.isfinite(ylim_max):
            new_y_start = np.clip(new_y_start, ylim_min, ylim_max - new_y_range)
        
        xlim = (new_x_start, new_x_start + new_x_range)
        ylim = (new_y_start, new_y_start + new_y_range)
        
        self.set_current_interval(new_x_range, ax_idx)
        
        if self.sync_views:
            self._sync_zoom(start_index, end_index, fs, zoom_factor, x_frac)
        else:
            self.update_view(ax_idx, xlim, ylim)

    def _sync_zoom(self, start_index, end_index, fs, zoom_factor, x_frac):
        """Apply synchronized zoom to all axes."""
        new_xlims, new_ylims = [], []
        
        for ax_idx in range(3):
            xlim_min, xlim_max, ylim_min, ylim_max = self.get_axis_limits(ax_idx, start_index, end_index, fs)
            current_xlim = self.axs[ax_idx].get_xlim()
            current_ylim = self.axs[ax_idx].get_ylim()
            current_x_range = current_xlim[1] - current_xlim[0]
            current_y_range = current_ylim[1] - current_ylim[0]
            
            # Check bounds
            max_x_range = xlim_max - xlim_min
            max_y_range = ylim_max - ylim_min
            if zoom_factor > 1.0 and current_x_range >= max_x_range * 0.999:
                return  # Can't zoom out further
            
            desired_x_range = current_x_range * zoom_factor
            desired_y_range = current_y_range * zoom_factor
            
            if desired_x_range <= 0.01:
                return
            
            scale_x = max_x_range / desired_x_range if desired_x_range > 0 else 1.0
            scale_y = max_y_range / desired_y_range if np.isfinite(max_y_range) and desired_y_range > 0 else 1.0
            overall_scale = min(1.0, scale_x, scale_y)
            
            new_x_range = desired_x_range * overall_scale
            new_y_range = desired_y_range * overall_scale
            
            x_center = current_xlim[0] + current_x_range * x_frac
            new_x_start = np.clip(x_center - new_x_range * x_frac, xlim_min, xlim_max - new_x_range)
            
            y_center = (current_ylim[0] + current_ylim[1]) / 2.0
            new_y_start = y_center - new_y_range / 2.0
            if np.isfinite(ylim_min) and np.isfinite(ylim_max):
                new_y_start = np.clip(new_y_start, ylim_min, ylim_max - new_y_range)
            
            new_xlims.append((new_x_start, new_x_start + new_x_range))
            new_ylims.append((new_y_start, new_y_start + new_y_range))
        
        # Apply to all axes
        for ax_idx in range(3):
            self.set_current_interval(new_xlims[ax_idx][1] - new_xlims[ax_idx][0], ax_idx)
            self.update_view(ax_idx, new_xlims[ax_idx], new_ylims[ax_idx], skip_draw=True)
        self.draw()

    def mousePressEvent(self, event):
        """Handle mouse press for panning."""
        self.is_dragging = True
        ax_idx = self.get_active_axes_index(event)
        if ax_idx is None:
            return
        
        self.drag_start_screen_x = event.pos().x()
        self.drag_start_screen_y = event.pos().y()
        self.drag_start_xlim = self.axs[ax_idx].get_xlim()
        self.drag_start_ylim = self.axs[ax_idx].get_ylim()
        
        if self.sync_views:
            self.drag_start_xlims = [self.axs[i].get_xlim() for i in range(3)]
            self.drag_start_ylims = [self.axs[i].get_ylim() for i in range(3)]

    def mouseMoveEvent(self, event):
        """Handle mouse move for panning."""
        if not self.is_dragging:
            return
        
        ax_idx = self.get_active_axes_index(event)
        if ax_idx is None:
            return
        
        # Calculate delta in data coordinates
        axes_bbox = self.axs[ax_idx].get_window_extent()
        x_range = self.drag_start_xlim[1] - self.drag_start_xlim[0]
        y_range = self.drag_start_ylim[1] - self.drag_start_ylim[0]
        
        data_delta_x = -(event.pos().x() - self.drag_start_screen_x) / axes_bbox.width * x_range
        data_delta_y = (event.pos().y() - self.drag_start_screen_y) / axes_bbox.height * y_range
        
        new_xlim = (self.drag_start_xlim[0] + data_delta_x, self.drag_start_xlim[1] + data_delta_x)
        new_ylim = (self.drag_start_ylim[0] + data_delta_y, self.drag_start_ylim[1] + data_delta_y)
        
        event_info = self.backend.event_features.get_current_info()
        start_index = int(event_info["start_index"])
        end_index = int(event_info["end_index"])
        fs = self.backend.sample_freq
        
        if self.sync_views:
            self._sync_pan(ax_idx, start_index, end_index, fs, data_delta_x, data_delta_y)
        else:
            xlim_min, xlim_max, ylim_min, ylim_max = self.get_axis_limits(ax_idx, start_index, end_index, fs)
            new_xlim = self._clamp_limits(new_xlim, xlim_min, xlim_max)
            if np.isfinite(ylim_min) and np.isfinite(ylim_max):
                new_ylim = self._clamp_limits(new_ylim, ylim_min, ylim_max)
            self.update_view(ax_idx, new_xlim, new_ylim)

    def _clamp_limits(self, lim, lim_min, lim_max):
        """Clamp limits to valid range."""
        lim_range = lim[1] - lim[0]
        if lim[0] < lim_min:
            return (lim_min, lim_min + lim_range)
        elif lim[1] > lim_max:
            return (lim_max - lim_range, lim_max)
        return lim

    def _sync_pan(self, active_ax_idx, start_index, end_index, fs, data_delta_x, data_delta_y):
        """Apply synchronized panning to all axes."""
        new_xlims, new_ylims = [], []
        
        for ax_idx in range(3):
            xlim_min, xlim_max, ylim_min, ylim_max = self.get_axis_limits(ax_idx, start_index, end_index, fs)
            
            # Apply x-delta
            new_xlim = (self.drag_start_xlims[ax_idx][0] + data_delta_x, self.drag_start_xlims[ax_idx][1] + data_delta_x)
            
            # Check bounds
            if new_xlim[0] < xlim_min or new_xlim[1] > xlim_max:
                return  # Can't pan further
            
            # Y-axis: only sync within same group
            if (active_ax_idx == 2 and ax_idx == 2) or (active_ax_idx != 2 and ax_idx != 2):
                active_y_range = self.drag_start_ylim[1] - self.drag_start_ylim[0]
                relative_y_delta = data_delta_y / active_y_range if active_y_range > 0 else 0
                sync_y_range = self.drag_start_ylims[ax_idx][1] - self.drag_start_ylims[ax_idx][0]
                sync_delta_y = relative_y_delta * sync_y_range
                new_ylim = (self.drag_start_ylims[ax_idx][0] + sync_delta_y, self.drag_start_ylims[ax_idx][1] + sync_delta_y)
                
                if np.isfinite(ylim_min) and np.isfinite(ylim_max):
                    new_ylim = self._clamp_limits(new_ylim, ylim_min, ylim_max)
            else:
                new_ylim = self.drag_start_ylims[ax_idx]
            
            new_xlims.append(new_xlim)
            new_ylims.append(new_ylim)
        
        for ax_idx in range(3):
            self.update_view(ax_idx, new_xlims[ax_idx], new_ylims[ax_idx], skip_draw=True)
        self.draw()

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        self.is_dragging = False


class FFTPlot(FigureCanvasQTAgg):
    """Plot widget for FFT/Power Spectral Density."""
    
    def __init__(self, parent=None, width=5, height=3, dpi=100, backend=None):
        fig, self.axs = plt.subplots(1, 1, figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(left=0.18, right=0.95, top=0.92, bottom=0.22)
        super(FFTPlot, self).__init__(fig)
        
        self.backend = backend
        self.min_freq = 10
        self.max_freq = 500
        self.interval = 1.0
        
        FigureCanvasQTAgg.setSizePolicy(self, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        FigureCanvasQTAgg.updateGeometry(self)

    def set_current_freq_limit(self, min_freq, max_freq):
        """Set frequency display range."""
        self.min_freq = min_freq
        self.max_freq = max_freq

    def set_current_interval(self, interval):
        """Set time interval for FFT computation."""
        self.interval = interval

    def plot(self, start_index: int, end_index: int, channel: str):
        """Compute and plot the FFT."""
        self.axs.cla()
        
        fs = self.backend.sample_freq
        middle_index = (int(start_index) + int(end_index)) // 2
        half_interval_samples = int((self.interval * fs) // 2)
        plot_start = max(0, middle_index - half_interval_samples)
        plot_end = min(self.backend.get_eeg_data_shape()[1], middle_index + half_interval_samples)
        
        eeg_data, channel_names = self.backend.get_eeg_data(plot_start, plot_end, filtered=False)
        eeg_data = eeg_data[channel_names == channel, :][0]
        
        # Compute FFT
        eeg_data = eeg_data - np.mean(eeg_data)
        window = np.hanning(len(eeg_data))
        frequencies, psd = periodogram(eeg_data, fs, window=window)
        
        # Filter to display range
        valid = (frequencies >= self.min_freq) & (frequencies <= self.max_freq)
        filtered_freqs = frequencies[valid]
        filtered_psd = psd[valid]
        psd_percent = (filtered_psd / np.sum(filtered_psd)) * 100
        
        # Plot
        self.axs.plot(filtered_freqs, psd_percent, color=COLOR_MAP['waveform'])
        self.axs.set_xlabel('Frequency (Hz)')
        self.axs.set_ylabel("PSD (Bandwidth %)")
        self.axs.set_xlim([self.min_freq, self.max_freq])
        self.axs.grid(True)
        self.draw()
