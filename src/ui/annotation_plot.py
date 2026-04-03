from PyQt5 import QtCore, QtWidgets
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.patches import Rectangle
from scipy.signal import periodogram

from src.utils.utils_annotation import calculate_default_boundary
from src.utils.utils_gui import COLOR_MAP
from src.utils.utils_plotting import calculate_time_frequency


def custom_formatter(x, pos):
    max_width = 5
    if abs(x) > 1000:
        return f"{x:.0e}"
    formatted_number = f" {x:.0f}" if x >= 0 else f"{x:.0f}"
    return f"{formatted_number:>{max_width}}"


def decimals_for_interval(width: float, base_decimals: int = 2, max_decimals: int = 6) -> int:
    if width <= 0 or not math.isfinite(width):
        return base_decimals
    dynamic = int(max(0, math.floor(-math.log10(width))))
    return int(min(max_decimals, base_decimals + dynamic))


def format_time_without_trailing_zeros(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f}"
    return text.rstrip("0").rstrip(".")


class AnnotationPlot(FigureCanvasQTAgg):
    hover_text_changed = QtCore.pyqtSignal(str)
    fft_window_selected = QtCore.pyqtSignal(object)
    history_state_changed = QtCore.pyqtSignal(bool, bool)

    def __init__(self, parent=None, width=10, height=6, dpi=100, backend=None):
        fig, self.axs = plt.subplots(3, 1, figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(left=0.1, right=0.85, top=0.90, bottom=0.15, hspace=0.75)
        super().__init__(fig)

        self.backend = backend
        self.interval = [1.0, 1.0, 1.0]
        self.zoom_max = 1
        self.sync_views = False
        self.manual_tf_freq_limit = None
        self.selected_fft_window = None
        self.preview_fft_window = None

        self.data_plotted = [False, False, False]
        self.axis_data = [None, None, None]
        self.vertical_guides = [None, None, None]
        self.horizontal_guides = [None, None, None]
        self.selected_window_patches = [None, None, None]
        self.preview_window_patches = [None, None, None]

        self.current_channel = None
        self.current_event_start = None
        self.current_event_end = None

        self.view_history = []
        self.view_history_index = -1

        self.interaction_mode = None
        self.drag_ax_idx = None
        self.drag_start_screen_x = None
        self.drag_start_screen_y = None
        self.drag_start_xlim = None
        self.drag_start_ylim = None
        self.drag_start_xlims = None
        self.drag_start_ylims = None
        self.box_zoom_start = None
        self.box_zoom_rect = None
        self.fft_select_start = None

        FigureCanvasQTAgg.setSizePolicy(
            self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        FigureCanvasQTAgg.updateGeometry(self)
        self.setMouseTracking(True)

    def _has_filtered_signal(self) -> bool:
        data = getattr(self.backend, "filter_data", None)
        return data is not None and np.size(data) > 0

    def _ensure_filtered_signal(self) -> bool:
        if self._has_filtered_signal():
            return True

        try:
            if hasattr(self.backend, "filter_eeg_data"):
                if getattr(self.backend, "param_filter", None) is not None:
                    self.backend.filter_eeg_data(self.backend.param_filter)
                else:
                    self.backend.filter_eeg_data()
        except Exception:
            return False

        return self._has_filtered_signal()

    def _get_tf_frequency_range(self, use_filtered_signal: bool, fs: float) -> tuple:
        if self.manual_tf_freq_limit is not None:
            return self.manual_tf_freq_limit

        nyquist = fs / 2
        min_freq, max_freq = 1.0, max(1.0, nyquist)

        if use_filtered_signal:
            param_filter = getattr(self.backend, "param_filter", None)
            if param_filter is not None:
                try:
                    band_min = float(param_filter.fp)
                    band_max = float(param_filter.fs)
                    if band_max < band_min:
                        band_min, band_max = band_max, band_min
                    min_freq = max(1.0, band_min)
                    max_freq = min(nyquist, max(min_freq + 1.0, band_max))
                except (TypeError, ValueError):
                    pass

        return min_freq, max_freq

    def set_manual_tf_freq_limit(self, min_freq, max_freq):
        self.manual_tf_freq_limit = (float(min_freq), float(max_freq))

    def set_current_interval(self, interval, ax_idx):
        self.interval[ax_idx] = interval

    def reset_intervals_to_default(self, default_interval):
        for ax_idx in range(3):
            self.interval[ax_idx] = default_interval

    def set_sync_views(self, enabled):
        self.sync_views = enabled

    def can_go_back(self):
        return self.view_history_index > 0

    def can_go_forward(self):
        return 0 <= self.view_history_index < len(self.view_history) - 1

    def go_back_view(self):
        if not self.can_go_back():
            return False
        self.view_history_index -= 1
        self._restore_view_state(self.view_history[self.view_history_index])
        return True

    def go_forward_view(self):
        if not self.can_go_forward():
            return False
        self.view_history_index += 1
        self._restore_view_state(self.view_history[self.view_history_index])
        return True

    def _capture_view_state(self):
        return {
            "xlims": [tuple(ax.get_xlim()) for ax in self.axs],
            "ylims": [tuple(ax.get_ylim()) for ax in self.axs],
            "interval": list(self.interval),
        }

    def _same_view_state(self, left, right):
        if left is None or right is None:
            return False
        if any(not np.allclose(a, b) for a, b in zip(left["xlims"], right["xlims"])):
            return False
        if any(not np.allclose(a, b) for a, b in zip(left["ylims"], right["ylims"])):
            return False
        return np.allclose(left["interval"], right["interval"])

    def _record_view_state(self, reset_history=False):
        state = self._capture_view_state()
        if reset_history:
            self.view_history = [state]
            self.view_history_index = 0
            self.history_state_changed.emit(self.can_go_back(), self.can_go_forward())
            return
        if self.view_history and self._same_view_state(self.view_history[self.view_history_index], state):
            return
        self.view_history = self.view_history[: self.view_history_index + 1]
        self.view_history.append(state)
        self.view_history_index = len(self.view_history) - 1
        self.history_state_changed.emit(self.can_go_back(), self.can_go_forward())

    def _restore_view_state(self, state):
        self.interval = list(state["interval"])
        for ax_idx in range(3):
            self.update_view(
                ax_idx,
                state["xlims"][ax_idx],
                state["ylims"][ax_idx],
                skip_draw=True,
            )
        self.draw()
        self.history_state_changed.emit(self.can_go_back(), self.can_go_forward())

    def get_axis_limits(self, ax_idx, start_index, end_index, fs):
        total_samples = self.backend.get_eeg_data_shape()[1]
        xlim_max = min(end_index / fs + self.zoom_max, total_samples / fs)
        xlim_min = max(start_index / fs - self.zoom_max, 0)
        if ax_idx == 2:
            min_freq, max_freq = self._get_tf_frequency_range(self._has_filtered_signal(), fs)
            return xlim_min, xlim_max, min_freq, max_freq
        return xlim_min, xlim_max, -np.inf, np.inf

    def configure_plot_axes(self, ax, title, ylabel, yformatter, x_decimals, xlim=None, ylim=None):
        ax.set_title(title)
        ax.set_ylabel(ylabel, rotation=90, labelpad=6)
        ax.set_xlabel("Time (s)")
        ax.yaxis.set_major_formatter(yformatter)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: format_time_without_trailing_zeros(x, x_decimals))
        )
        ax.yaxis.set_label_position("right")
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)

    def _compute_signal_ylim(self, ax_idx, xlim):
        axis_data = self.axis_data[ax_idx]
        if axis_data is None or axis_data.get("signal") is None:
            return self.axs[ax_idx].get_ylim()
        time = axis_data["time"]
        signal = axis_data["signal"]
        mask = (time >= xlim[0]) & (time <= xlim[1])
        if not np.any(mask):
            return self.axs[ax_idx].get_ylim()
        view_data = signal[mask]
        y_min = float(np.min(view_data))
        y_max = float(np.max(view_data))
        spread = y_max - y_min
        margin = spread * 0.1 if spread > 0 else max(abs(y_min), abs(y_max), 1.0) * 0.1
        return y_min - margin, y_max + margin

    def _install_guides(self, ax_idx):
        self.vertical_guides[ax_idx] = self.axs[ax_idx].axvline(
            0, color="#17324d", linestyle="--", linewidth=0.8, alpha=0.45, visible=False
        )
        self.horizontal_guides[ax_idx] = self.axs[ax_idx].axhline(
            0, color="#17324d", linestyle=":", linewidth=0.8, alpha=0.35, visible=False
        )

    def _draw_event_overlay(self, ax_idx, event_start_time, event_end_time, event_color):
        if ax_idx not in (0, 1, 2):
            return
        ax = self.axs[ax_idx]
        ax.axvspan(event_start_time, event_end_time, color=event_color, alpha=0.10)
        ax.axvline(event_start_time, color=event_color, linestyle="--", linewidth=0.9, alpha=0.75)
        ax.axvline(event_end_time, color=event_color, linestyle="--", linewidth=0.9, alpha=0.75)

    def _remove_patch_ref(self, patches, ax_idx):
        patch = patches[ax_idx]
        if patch is not None:
            try:
                patch.remove()
            except ValueError:
                pass
            patches[ax_idx] = None

    def _apply_window_overlay(self, window, patches, color, alpha):
        for ax_idx in range(3):
            self._remove_patch_ref(patches, ax_idx)
            if window is None:
                continue
            ax = self.axs[ax_idx]
            patches[ax_idx] = ax.axvspan(window[0], window[1], color=color, alpha=alpha)

    def _refresh_window_overlays(self, redraw=True):
        self._apply_window_overlay(self.selected_fft_window, self.selected_window_patches, "#17324d", 0.10)
        self._apply_window_overlay(self.preview_fft_window, self.preview_window_patches, "#58748a", 0.10)
        if redraw:
            self.draw_idle()

    def set_selected_fft_window(self, time_window, emit_signal=True):
        if time_window is None:
            self.selected_fft_window = None
        else:
            start, end = sorted((float(time_window[0]), float(time_window[1])))
            if end - start < 1e-6:
                return
            self.selected_fft_window = (start, end)
        self.preview_fft_window = None
        self._refresh_window_overlays()
        if emit_signal:
            self.fft_window_selected.emit(self.selected_fft_window)

    def clear_selected_fft_window(self, emit_signal=True, redraw=True):
        self.selected_fft_window = None
        self.preview_fft_window = None
        self._refresh_window_overlays(redraw=redraw)
        if emit_signal:
            self.fft_window_selected.emit(None)

    def plot_all_axes(self, event_start_index: int, event_end_index: int, channel: str):
        self.current_event_start = int(event_start_index)
        self.current_event_end = int(event_end_index)
        self.current_channel = channel
        for ax_idx in range(3):
            self.plot_full_data(ax_idx, event_start_index, event_end_index, channel, skip_draw=True)
        self.draw()
        self._record_view_state(reset_history=True)

    def reset_to_default_view(self, event_start_index: int, event_end_index: int):
        if not self.backend:
            return
        fs = self.backend.sample_freq
        total_samples = self.backend.get_eeg_data_shape()[1]
        event_start_index = int(event_start_index)
        event_end_index = int(event_end_index)

        for ax_idx in range(3):
            if not self.data_plotted[ax_idx]:
                continue
            win_len = int(fs * self.interval[ax_idx])
            ws_idx, we_idx, _, _ = calculate_default_boundary(
                event_start_index, event_end_index, total_samples, win_len=win_len
            )
            default_xlim = (ws_idx / fs, we_idx / fs)
            if ax_idx == 2:
                default_ylim = self._get_tf_frequency_range(self._has_filtered_signal(), fs)
            else:
                default_ylim = self._compute_signal_ylim(ax_idx, default_xlim)
            self.update_view(ax_idx, default_xlim, default_ylim, skip_draw=True)

        self.draw()
        self._record_view_state()

    def _apply_shared_xlim(self, xlim, skip_draw=True):
        width = xlim[1] - xlim[0]
        for ax_idx in range(3):
            self.set_current_interval(width, ax_idx)
            self.update_view(ax_idx, xlim, None, skip_draw=skip_draw)

    def zoom_by_factor(self, zoom_factor):
        if not self.backend:
            return

        event_info = self.backend.event_features.get_current_info()
        start_index = int(event_info["start_index"])
        end_index = int(event_info["end_index"])
        fs = self.backend.sample_freq

        for ax_idx in range(3):
            if not self.data_plotted[ax_idx]:
                continue

            current_xlim = self.axs[ax_idx].get_xlim()
            current_ylim = self.axs[ax_idx].get_ylim()
            current_x_range = current_xlim[1] - current_xlim[0]
            current_y_range = current_ylim[1] - current_ylim[0]

            new_x_range = np.clip(
                current_x_range * zoom_factor,
                0.05,
                self.get_axis_limits(ax_idx, start_index, end_index, fs)[1]
                - self.get_axis_limits(ax_idx, start_index, end_index, fs)[0],
            )

            x_center = (current_xlim[0] + current_xlim[1]) / 2.0
            new_x_start = x_center - new_x_range / 2.0
            new_x_end = x_center + new_x_range / 2.0
            xlim_min, xlim_max, ylim_min, ylim_max = self.get_axis_limits(ax_idx, start_index, end_index, fs)
            if new_x_start < xlim_min:
                new_x_start = xlim_min
                new_x_end = new_x_start + new_x_range
            if new_x_end > xlim_max:
                new_x_end = xlim_max
                new_x_start = new_x_end - new_x_range

            if ax_idx == 2:
                self.update_view(ax_idx, (new_x_start, new_x_end), None, skip_draw=True)
            else:
                new_y_range = current_y_range * zoom_factor
                y_center = (current_ylim[0] + current_ylim[1]) / 2.0
                new_ylim = (y_center - new_y_range / 2.0, y_center + new_y_range / 2.0)
                if np.isfinite(ylim_min) and np.isfinite(ylim_max):
                    new_ylim = self._clamp_limits(new_ylim, ylim_min, ylim_max)
                self.update_view(ax_idx, (new_x_start, new_x_end), new_ylim, skip_draw=True)
            self.set_current_interval(new_x_range, ax_idx)

        self.draw()
        self._record_view_state()

    def pan_horizontal(self, fraction):
        if not self.backend:
            return
        event_info = self.backend.event_features.get_current_info()
        start_index = int(event_info["start_index"])
        end_index = int(event_info["end_index"])
        fs = self.backend.sample_freq

        for ax_idx in range(3):
            if not self.data_plotted[ax_idx]:
                continue
            current_xlim = self.axs[ax_idx].get_xlim()
            current_x_range = current_xlim[1] - current_xlim[0]
            delta_x = current_x_range * fraction
            xlim_min, xlim_max, _, _ = self.get_axis_limits(ax_idx, start_index, end_index, fs)
            new_xlim = self._clamp_limits(
                (current_xlim[0] + delta_x, current_xlim[1] + delta_x), xlim_min, xlim_max
            )
            self.update_view(ax_idx, new_xlim, None, skip_draw=True)

        self.draw()
        self._record_view_state()

    def pan_vertical(self, fraction):
        if not self.backend:
            return

        for ax_idx in [0, 1]:
            if not self.data_plotted[ax_idx]:
                continue
            current_ylim = self.axs[ax_idx].get_ylim()
            current_y_range = current_ylim[1] - current_ylim[0]
            delta_y = current_y_range * fraction
            self.update_view(
                ax_idx,
                None,
                (current_ylim[0] + delta_y, current_ylim[1] + delta_y),
                skip_draw=True,
            )

        self.draw()
        self._record_view_state()

    def plot_full_data(self, ax_idx: int, event_start_index: int, event_end_index: int, channel: str, skip_draw: bool = False):
        self.axs[ax_idx].cla()

        if event_start_index is None or event_start_index < 0:
            return

        event_start_index = int(event_start_index)
        event_end_index = int(event_end_index)
        fs = self.backend.sample_freq
        prediction = self.backend.event_features.get_current_info()["prediction"]
        event_color = COLOR_MAP.get(prediction, COLOR_MAP["HFO"])
        signal_color = COLOR_MAP["waveform"]
        total_samples = self.backend.get_eeg_data_shape()[1]
        zoom_max_samples = int(self.zoom_max * fs)
        full_start_index = max(0, event_start_index - zoom_max_samples)
        full_end_index = min(total_samples, event_end_index + zoom_max_samples)
        event_start_time = event_start_index / fs
        event_end_time = event_end_index / fs

        use_filtered_signal = ax_idx in (1, 2) and self._ensure_filtered_signal()
        if use_filtered_signal:
            eeg_data, channel_names = self.backend.get_eeg_data(full_start_index, full_end_index, filtered=True)
        else:
            eeg_data, channel_names = self.backend.get_eeg_data(full_start_index, full_end_index)

        eeg_data_to_display = eeg_data[channel_names == channel, :][0]
        time = np.arange(len(eeg_data_to_display)) / fs + full_start_index / fs

        event_rel_start = max(0, event_start_index - full_start_index)
        event_rel_end = min(len(eeg_data_to_display), event_end_index - full_start_index)

        win_len = int(fs * self.interval[ax_idx])
        ws_idx, we_idx, _, _ = calculate_default_boundary(
            event_start_index, event_end_index, total_samples, win_len=win_len
        )
        default_xlim = (ws_idx / fs, we_idx / fs)
        x_decimals = decimals_for_interval(self.interval[ax_idx])

        if ax_idx in (0, 1):
            default_ylim = self._compute_signal_default_ylim_for_data(
                eeg_data_to_display, time, default_xlim
            )
            self.axs[ax_idx].plot(time, eeg_data_to_display, color=signal_color, linewidth=1.0)
            self.axs[ax_idx].plot(
                time[event_rel_start:event_rel_end],
                eeg_data_to_display[event_rel_start:event_rel_end],
                color=event_color,
                linewidth=1.3,
            )
            title = "EEG Tracing" if ax_idx == 0 else "Filtered Tracing"
            self.configure_plot_axes(
                self.axs[ax_idx],
                title,
                "Amplitude (uV)",
                ticker.FuncFormatter(custom_formatter),
                x_decimals,
                default_xlim,
                default_ylim,
            )
            self.axis_data[ax_idx] = {"time": time, "signal": eeg_data_to_display}
        else:
            min_freq, max_freq = self._get_tf_frequency_range(use_filtered_signal, fs)
            tf_data = calculate_time_frequency(
                eeg_data_to_display, fs, freq_min=min_freq, freq_max=max_freq
            )
            view_start = max(0, ws_idx - full_start_index)
            view_end = min(tf_data.shape[1], we_idx - full_start_index)
            view_data = tf_data[:, view_start:view_end]
            vmin, vmax = (
                (np.min(view_data), np.max(view_data))
                if view_data.size > 0
                else (np.min(tf_data), np.max(tf_data))
            )
            extent = [full_start_index / fs, full_end_index / fs, min_freq, max_freq]
            self.axs[ax_idx].imshow(
                tf_data,
                extent=extent,
                aspect="auto",
                cmap="jet",
                vmin=vmin,
                vmax=vmax,
                origin="lower",
            )
            default_ylim = (min_freq, max_freq)
            tf_title = "Time Frequency (Filtered)" if use_filtered_signal else "Time Frequency (Raw)"
            self.configure_plot_axes(
                self.axs[ax_idx],
                tf_title,
                "Frequency (Hz)",
                ticker.FuncFormatter(custom_formatter),
                x_decimals,
                default_xlim,
                default_ylim,
            )
            self.axs[ax_idx].set_xticks(np.linspace(default_xlim[0], default_xlim[1], 5))
            self.axs[ax_idx].set_yticks(np.linspace(default_ylim[0], default_ylim[1], 5).astype(int))
            self.axis_data[ax_idx] = {"time": time, "signal": None}

        self._draw_event_overlay(ax_idx, event_start_time, event_end_time, event_color)
        self._install_guides(ax_idx)
        self.data_plotted[ax_idx] = True
        self._refresh_window_overlays(redraw=False)

        if not skip_draw:
            self.draw()

    def _compute_signal_default_ylim_for_data(self, signal, time, xlim):
        mask = (time >= xlim[0]) & (time <= xlim[1])
        view_data = signal[mask] if np.any(mask) else signal
        y_min = float(np.min(view_data))
        y_max = float(np.max(view_data))
        spread = y_max - y_min
        margin = spread * 0.1 if spread > 0 else max(abs(y_min), abs(y_max), 1.0) * 0.1
        return y_min - margin, y_max + margin

    def update_view(self, ax_idx: int, xlim: tuple = None, ylim: tuple = None, skip_draw: bool = False):
        if not self.data_plotted[ax_idx]:
            return

        if xlim is not None:
            self.axs[ax_idx].set_xlim(xlim)
            x_decimals = decimals_for_interval(xlim[1] - xlim[0])
            self.axs[ax_idx].xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _, d=x_decimals: format_time_without_trailing_zeros(x, d))
            )

        if ylim is not None:
            self.axs[ax_idx].set_ylim(ylim)

        if ax_idx == 2:
            current_xlim = self.axs[ax_idx].get_xlim()
            current_ylim = self.axs[ax_idx].get_ylim()
            self.axs[ax_idx].set_xticks(np.linspace(current_xlim[0], current_xlim[1], 5))
            self.axs[ax_idx].set_yticks(np.linspace(current_ylim[0], current_ylim[1], 5).astype(int))

        if not skip_draw:
            self.draw()

    def get_active_axes_index(self, event):
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

    def _lookup_signal_value(self, ax_idx, xdata):
        axis_data = self.axis_data[ax_idx]
        if axis_data is None or axis_data.get("signal") is None:
            return None
        time = axis_data["time"]
        signal = axis_data["signal"]
        if len(time) == 0:
            return None
        position = int(np.clip(np.searchsorted(time, xdata), 0, len(time) - 1))
        return float(signal[position])

    def _hide_hover_guides(self):
        for line in self.vertical_guides + self.horizontal_guides:
            if line is not None:
                line.set_visible(False)
        self.hover_text_changed.emit("")
        self.draw_idle()

    def _update_hover_display(self, ax_idx, xdata, ydata):
        for idx, line in enumerate(self.vertical_guides):
            if line is not None:
                line.set_xdata([xdata, xdata])
                line.set_visible(True)
        for idx, line in enumerate(self.horizontal_guides):
            if line is not None:
                visible = idx == ax_idx
                line.set_visible(visible)
                if visible:
                    line.set_ydata([ydata, ydata])

        raw_value = self._lookup_signal_value(0, xdata)
        filtered_value = self._lookup_signal_value(1, xdata)
        parts = [f"t={xdata:.4f}s"]
        if raw_value is not None:
            parts.append(f"raw={raw_value:.2f}")
        if filtered_value is not None:
            parts.append(f"filtered={filtered_value:.2f}")
        if ax_idx == 2:
            parts.append(f"freq={ydata:.1f}Hz")
        else:
            parts.append(f"y={ydata:.2f}")
        if self.selected_fft_window is not None:
            parts.append(
                f"fft-roi={self.selected_fft_window[0]:.4f}-{self.selected_fft_window[1]:.4f}s"
            )
        self.hover_text_changed.emit(" | ".join(parts))
        self.draw_idle()

    def wheelEvent(self, event):
        if not self.backend:
            return

        ax_idx = self.get_active_axes_index(event)
        if ax_idx is None:
            return super().wheelEvent(event)

        mouse_pos = self.get_mouse_data_position(event, ax_idx)
        if mouse_pos is None:
            return super().wheelEvent(event)
        xdata, ydata = mouse_pos

        event_info = self.backend.event_features.get_current_info()
        start_index = int(event_info["start_index"])
        end_index = int(event_info["end_index"])
        fs = self.backend.sample_freq
        current_xlim = self.axs[ax_idx].get_xlim()
        current_ylim = self.axs[ax_idx].get_ylim()
        current_x_range = current_xlim[1] - current_xlim[0]
        current_y_range = current_ylim[1] - current_ylim[0]

        zoom_factor = 1 / 1.3 if event.angleDelta().y() > 0 else 1.3
        xlim_min, xlim_max, ylim_min, ylim_max = self.get_axis_limits(ax_idx, start_index, end_index, fs)
        new_x_range = np.clip(current_x_range * zoom_factor, 0.05, xlim_max - xlim_min)
        x_frac = np.clip((xdata - current_xlim[0]) / max(current_x_range, 1e-6), 0.1, 0.9)
        new_x_start = xdata - new_x_range * x_frac
        new_x_end = new_x_start + new_x_range
        new_xlim = self._clamp_limits((new_x_start, new_x_end), xlim_min, xlim_max)

        if ax_idx == 2:
            new_y_range = max(current_y_range * zoom_factor, 1.0)
            y_frac = np.clip((ydata - current_ylim[0]) / max(current_y_range, 1e-6), 0.1, 0.9)
            new_y_start = ydata - new_y_range * y_frac
            new_y_end = new_y_start + new_y_range
            new_ylim = self._clamp_limits((new_y_start, new_y_end), ylim_min, ylim_max)
        else:
            new_y_range = current_y_range * zoom_factor
            y_center = (current_ylim[0] + current_ylim[1]) / 2.0
            new_ylim = (y_center - new_y_range / 2.0, y_center + new_y_range / 2.0)

        if self.sync_views:
            self._apply_shared_xlim(new_xlim, skip_draw=True)
            self.update_view(ax_idx, None, new_ylim, skip_draw=True)
        else:
            self.update_view(ax_idx, new_xlim, new_ylim, skip_draw=True)
            self.set_current_interval(new_x_range, ax_idx)

        self.draw()
        self._record_view_state()
        super().wheelEvent(event)

    def mousePressEvent(self, event):
        ax_idx = self.get_active_axes_index(event)
        if ax_idx is None:
            return super().mousePressEvent(event)

        data_pos = self.get_mouse_data_position(event, ax_idx)
        if data_pos is None:
            return super().mousePressEvent(event)

        modifiers = event.modifiers()
        self.drag_ax_idx = ax_idx
        if modifiers & QtCore.Qt.ShiftModifier:
            self.interaction_mode = "box_zoom"
            self.box_zoom_start = data_pos
            self.box_zoom_rect = Rectangle(
                data_pos,
                0,
                0,
                fill=False,
                edgecolor="#17324d",
                linewidth=1.0,
                linestyle="--",
            )
            self.axs[ax_idx].add_patch(self.box_zoom_rect)
            self.draw_idle()
        elif modifiers & QtCore.Qt.AltModifier:
            self.interaction_mode = "fft_select"
            self.fft_select_start = data_pos[0]
            self.preview_fft_window = (data_pos[0], data_pos[0])
            self._refresh_window_overlays(redraw=True)
        else:
            self.interaction_mode = "pan"
            self.drag_start_screen_x = event.pos().x()
            self.drag_start_screen_y = event.pos().y()
            self.drag_start_xlim = self.axs[ax_idx].get_xlim()
            self.drag_start_ylim = self.axs[ax_idx].get_ylim()
            self.drag_start_xlims = [ax.get_xlim() for ax in self.axs]
            self.drag_start_ylims = [ax.get_ylim() for ax in self.axs]

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        ax_idx = self.get_active_axes_index(event)
        if self.interaction_mode == "pan" and ax_idx is not None and self.drag_ax_idx is not None:
            axes_bbox = self.axs[self.drag_ax_idx].get_window_extent()
            x_range = self.drag_start_xlim[1] - self.drag_start_xlim[0]
            y_range = self.drag_start_ylim[1] - self.drag_start_ylim[0]
            data_delta_x = -(event.pos().x() - self.drag_start_screen_x) / axes_bbox.width * x_range
            data_delta_y = (event.pos().y() - self.drag_start_screen_y) / axes_bbox.height * y_range

            event_info = self.backend.event_features.get_current_info()
            start_index = int(event_info["start_index"])
            end_index = int(event_info["end_index"])
            fs = self.backend.sample_freq
            xlim_min, xlim_max, ylim_min, ylim_max = self.get_axis_limits(
                self.drag_ax_idx, start_index, end_index, fs
            )
            new_xlim = self._clamp_limits(
                (self.drag_start_xlim[0] + data_delta_x, self.drag_start_xlim[1] + data_delta_x),
                xlim_min,
                xlim_max,
            )
            new_ylim = (self.drag_start_ylim[0] + data_delta_y, self.drag_start_ylim[1] + data_delta_y)
            if np.isfinite(ylim_min) and np.isfinite(ylim_max):
                new_ylim = self._clamp_limits(new_ylim, ylim_min, ylim_max)

            if self.sync_views:
                self._apply_shared_xlim(new_xlim, skip_draw=True)
                self.update_view(self.drag_ax_idx, None, new_ylim, skip_draw=True)
            else:
                self.update_view(self.drag_ax_idx, new_xlim, new_ylim, skip_draw=True)
            self.draw_idle()
        elif self.interaction_mode == "box_zoom" and self.box_zoom_rect is not None and ax_idx == self.drag_ax_idx:
            data_pos = self.get_mouse_data_position(event, self.drag_ax_idx)
            if data_pos is not None:
                x0, y0 = self.box_zoom_start
                x1, y1 = data_pos
                self.box_zoom_rect.set_x(min(x0, x1))
                self.box_zoom_rect.set_y(min(y0, y1))
                self.box_zoom_rect.set_width(abs(x1 - x0))
                self.box_zoom_rect.set_height(abs(y1 - y0))
                self.draw_idle()
        elif self.interaction_mode == "fft_select" and ax_idx is not None:
            data_pos = self.get_mouse_data_position(event, ax_idx)
            if data_pos is not None:
                self.preview_fft_window = tuple(sorted((self.fft_select_start, data_pos[0])))
                self._refresh_window_overlays(redraw=True)
        else:
            if ax_idx is None:
                self._hide_hover_guides()
            else:
                data_pos = self.get_mouse_data_position(event, ax_idx)
                if data_pos is None:
                    self._hide_hover_guides()
                else:
                    self._update_hover_display(ax_idx, data_pos[0], data_pos[1])

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.interaction_mode == "pan":
            self._record_view_state()
        elif self.interaction_mode == "box_zoom" and self.drag_ax_idx is not None:
            data_pos = self.get_mouse_data_position(event, self.drag_ax_idx)
            if data_pos is not None:
                x0, y0 = self.box_zoom_start
                x1, y1 = data_pos
                if abs(x1 - x0) > 1e-4 and abs(y1 - y0) > 1e-4:
                    new_xlim = tuple(sorted((x0, x1)))
                    new_ylim = tuple(sorted((y0, y1)))
                    event_info = self.backend.event_features.get_current_info()
                    start_index = int(event_info["start_index"])
                    end_index = int(event_info["end_index"])
                    fs = self.backend.sample_freq
                    xlim_min, xlim_max, ylim_min, ylim_max = self.get_axis_limits(
                        self.drag_ax_idx, start_index, end_index, fs
                    )
                    new_xlim = self._clamp_limits(new_xlim, xlim_min, xlim_max)
                    if np.isfinite(ylim_min) and np.isfinite(ylim_max):
                        new_ylim = self._clamp_limits(new_ylim, ylim_min, ylim_max)
                    if self.sync_views:
                        self._apply_shared_xlim(new_xlim, skip_draw=True)
                        self.update_view(self.drag_ax_idx, None, new_ylim, skip_draw=True)
                    else:
                        self.update_view(self.drag_ax_idx, new_xlim, new_ylim, skip_draw=True)
                    self.draw()
                    self._record_view_state()
            if self.box_zoom_rect is not None:
                try:
                    self.box_zoom_rect.remove()
                except ValueError:
                    pass
        elif self.interaction_mode == "fft_select":
            if self.preview_fft_window is not None and self.preview_fft_window[1] - self.preview_fft_window[0] > 1e-4:
                self.set_selected_fft_window(self.preview_fft_window, emit_signal=True)
            else:
                self.preview_fft_window = None
                self._refresh_window_overlays(redraw=True)

        self.interaction_mode = None
        self.drag_ax_idx = None
        self.box_zoom_start = None
        self.box_zoom_rect = None
        self.fft_select_start = None
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self._hide_hover_guides()
        super().leaveEvent(event)

    def _clamp_limits(self, lim, lim_min, lim_max):
        lim_range = lim[1] - lim[0]
        if lim_range >= lim_max - lim_min:
            return lim_min, lim_max
        if lim[0] < lim_min:
            return lim_min, lim_min + lim_range
        if lim[1] > lim_max:
            return lim_max - lim_range, lim_max
        return lim


class FFTPlot(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=3, dpi=100, backend=None):
        fig, self.axs = plt.subplots(1, 1, figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(left=0.18, right=0.95, top=0.92, bottom=0.22)
        super().__init__(fig)

        self.backend = backend
        self.min_freq = 10
        self.max_freq = 500
        self.interval = 1.0
        self.selected_time_window = None

        FigureCanvasQTAgg.setSizePolicy(
            self, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        FigureCanvasQTAgg.updateGeometry(self)

    def set_current_freq_limit(self, min_freq, max_freq):
        self.min_freq = min_freq
        self.max_freq = max_freq

    def set_current_interval(self, interval):
        self.interval = interval

    def set_selected_time_window(self, time_window):
        if time_window is None:
            self.selected_time_window = None
            return
        start, end = sorted((float(time_window[0]), float(time_window[1])))
        if end - start < 1e-6:
            return
        self.selected_time_window = (start, end)

    def clear_selected_time_window(self):
        self.selected_time_window = None

    def plot(self, start_index: int, end_index: int, channel: str):
        self.axs.cla()

        fs = self.backend.sample_freq
        total_samples = self.backend.get_eeg_data_shape()[1]

        if self.selected_time_window is not None:
            plot_start = max(0, int(self.selected_time_window[0] * fs))
            plot_end = min(total_samples, int(self.selected_time_window[1] * fs))
        else:
            middle_index = (int(start_index) + int(end_index)) // 2
            half_interval_samples = int((self.interval * fs) // 2)
            plot_start = max(0, middle_index - half_interval_samples)
            plot_end = min(total_samples, middle_index + half_interval_samples)

        eeg_data, channel_names = self.backend.get_eeg_data(plot_start, plot_end, filtered=False)
        eeg_data = eeg_data[channel_names == channel, :][0]

        eeg_data = eeg_data - np.mean(eeg_data)
        if len(eeg_data) < 4:
            self.axs.set_title("FFT")
            self.axs.set_xlabel("Frequency (Hz)")
            self.axs.set_ylabel("PSD (Bandwidth %)")
            self.axs.set_xlim([self.min_freq, self.max_freq])
            self.axs.text(0.5, 0.5, "Window too small", transform=self.axs.transAxes, ha="center", va="center")
            self.draw()
            return

        window = np.hanning(len(eeg_data))
        frequencies, psd = periodogram(eeg_data, fs, window=window)

        valid = (frequencies >= self.min_freq) & (frequencies <= self.max_freq)
        filtered_freqs = frequencies[valid]
        filtered_psd = psd[valid]
        total_power = float(np.sum(filtered_psd))
        if total_power <= 0 or not np.isfinite(total_power):
            psd_percent = np.zeros_like(filtered_psd)
        else:
            psd_percent = (filtered_psd / total_power) * 100

        self.axs.plot(filtered_freqs, psd_percent, color=COLOR_MAP["waveform"])
        self.axs.set_xlabel("Frequency (Hz)")
        self.axs.set_ylabel("PSD (Bandwidth %)")
        self.axs.set_xlim([self.min_freq, self.max_freq])
        self.axs.grid(True)
        if self.selected_time_window is not None:
            self.axs.set_title(
                f"FFT ROI {self.selected_time_window[0]:.3f}-{self.selected_time_window[1]:.3f}s"
            )
        else:
            self.axs.set_title("FFT")
        self.draw()
