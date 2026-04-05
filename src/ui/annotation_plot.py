from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from scipy.signal import periodogram

from src.utils.utils_annotation import calculate_default_boundary
from src.utils.utils_gui import COLOR_MAP
from src.utils.utils_plotting import calculate_time_frequency


PLOT_THEME = {
    "figure_bg": "#f4f7fb",
    "axes_bg": "#ffffff",
    "title": "#203344",
    "label": "#43586b",
    "tick": "#5c7082",
    "spine": "#d8e1e8",
    "grid": "#e4eaf0",
    "grid_soft": "#f1f4f7",
    "waveform_raw": "#6a8194",
    "waveform_filtered": "#50697e",
    "guide": "#2f4960",
    "empty": "#8493a0",
    "fft_fill": "#9eb5c8",
}
SPECTROGRAM_CMAP = "magma"


def _mk_pen(color: str, width: float = 1.0, alpha: float = 1.0, style=QtCore.Qt.SolidLine):
    qcolor = QtGui.QColor(color)
    qcolor.setAlphaF(float(np.clip(alpha, 0.0, 1.0)))
    return pg.mkPen(qcolor, width=width, style=style)


def _mk_brush(color: str, alpha: float = 1.0):
    qcolor = QtGui.QColor(color)
    qcolor.setAlphaF(float(np.clip(alpha, 0.0, 1.0)))
    return pg.mkBrush(qcolor)


def blend_color(color: str, target: str = "#ffffff", amount: float = 0.5) -> str:
    amount = float(np.clip(amount, 0.0, 1.0))
    base = QtGui.QColor(color)
    dest = QtGui.QColor(target)
    red = round(base.red() * (1.0 - amount) + dest.red() * amount)
    green = round(base.green() * (1.0 - amount) + dest.green() * amount)
    blue = round(base.blue() * (1.0 - amount) + dest.blue() * amount)
    return QtGui.QColor(red, green, blue).name()


def custom_formatter(value):
    if abs(value) > 1000:
        return f"{value:.0e}"
    if abs(value - int(value)) < 1e-6:
        return f"{int(value)}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def decimals_for_interval(width: float, base_decimals: int = 2, max_decimals: int = 6) -> int:
    if width <= 0 or not math.isfinite(width):
        return base_decimals
    dynamic = int(max(0, math.floor(-math.log10(width))))
    return int(min(max_decimals, base_decimals + dynamic))


def format_time_without_trailing_zeros(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f}"
    return text.rstrip("0").rstrip(".")


def percentile_limits(data, lower: float = 5.0, upper: float = 99.5) -> tuple:
    values = np.asarray(data)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0, 1.0

    lower_value = float(np.nanpercentile(finite_values, lower))
    upper_value = float(np.nanpercentile(finite_values, upper))
    if math.isclose(lower_value, upper_value):
        data_min = float(np.nanmin(finite_values))
        data_max = float(np.nanmax(finite_values))
        magnitude = max(abs(data_min), abs(data_max), 1.0)
        return data_min - magnitude * 0.1, data_max + magnitude * 0.1
    return lower_value, upper_value


class FormattedAxisItem(pg.AxisItem):
    def __init__(self, orientation: str, formatter: Callable[[float], str] | None = None):
        super().__init__(orientation=orientation)
        self._formatter = formatter or (lambda value: str(value))

    def set_formatter(self, formatter: Callable[[float], str]):
        self._formatter = formatter
        self.picture = None
        self.update()

    def tickStrings(self, values, scale, spacing):
        return [self._formatter(float(value)) for value in values]


class AnnotationPlot(pg.GraphicsLayoutWidget):
    hover_text_changed = QtCore.pyqtSignal(str)
    fft_window_selected = QtCore.pyqtSignal(object)
    history_state_changed = QtCore.pyqtSignal(bool, bool)

    def __init__(self, parent=None, width=10, height=6, dpi=100, backend=None):
        super().__init__(parent=parent)
        self.setBackground(PLOT_THEME["figure_bg"])
        self.setViewportMargins(6, 6, 6, 6)
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
        self.selected_window_regions = [None, None, None]
        self.preview_window_regions = [None, None, None]
        self.event_regions = [None, None, None]
        self.event_boundary_lines = [[], [], []]
        self.tf_image_items = [None, None, None]
        self._spectrogram_metadata = {}

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
        self.box_zoom_start = None
        self.box_zoom_origin = None
        self.fft_select_start = None
        self.box_zoom_rubber_band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)

        self._build_plots()
        self.setMouseTracking(True)

    def _build_plots(self):
        self.ci.layout.setContentsMargins(8, 8, 8, 8)
        self.ci.layout.setHorizontalSpacing(0)
        self.ci.layout.setVerticalSpacing(12)
        self.axs = []
        self._bottom_axes = []
        self._left_axes = []

        left_formatters = [
            lambda value: custom_formatter(value),
            lambda value: custom_formatter(value),
            lambda value: custom_formatter(value),
        ]

        for ax_idx in range(3):
            bottom_axis = FormattedAxisItem("bottom", formatter=lambda value: format_time_without_trailing_zeros(value, 2))
            left_axis = FormattedAxisItem("left", formatter=left_formatters[ax_idx])
            plot = pg.PlotItem(axisItems={"bottom": bottom_axis, "left": left_axis})
            self.ci.addItem(plot, row=ax_idx, col=0)
            self.ci.layout.setRowStretchFactor(ax_idx, 118 if ax_idx == 2 else 100)
            self._style_plot_item(plot, show_x_grid=ax_idx != 2, show_y_grid=ax_idx != 2)
            self.axs.append(plot)
            self._bottom_axes.append(bottom_axis)
            self._left_axes.append(left_axis)

    def _style_plot_item(self, plot, *, show_x_grid=True, show_y_grid=True):
        plot.hideButtons()
        plot.setMenuEnabled(False)
        plot.setClipToView(True)
        plot.getViewBox().setMouseEnabled(x=False, y=False)
        plot.getViewBox().setBackgroundColor(PLOT_THEME["axes_bg"])
        plot.showGrid(x=show_x_grid, y=show_y_grid, alpha=0.32)
        for axis_name in ("left", "bottom"):
            axis = plot.getAxis(axis_name)
            axis.setPen(_mk_pen(PLOT_THEME["spine"], width=1.0))
            axis.setTextPen(_mk_pen(PLOT_THEME["tick"], width=1.0))
            axis.setStyle(
                autoExpandTextSpace=True,
                hideOverlappingLabels=False,
                tickTextOffset=6,
            )

    def draw(self):
        self.viewport().update()

    def draw_idle(self):
        self.viewport().update()

    def get_label_bounding_rects(self):
        rects = []
        for plot in self.axs:
            items = [
                plot.titleLabel,
                plot.getAxis("left").label,
                plot.getAxis("bottom").label,
            ]
            for item in items:
                if item is None or not item.isVisible():
                    continue
                scene_rect = item.mapRectToScene(item.boundingRect())
                top_left = self.mapFromScene(scene_rect.topLeft())
                bottom_right = self.mapFromScene(scene_rect.bottomRight())
                rects.append(QtCore.QRectF(QtCore.QPointF(top_left), QtCore.QPointF(bottom_right)).normalized())
        return rects

    def get_spectrogram_metadata(self):
        return dict(self._spectrogram_metadata)

    def get_axis_ranges(self):
        return [(tuple(plot.viewRange()[0]), tuple(plot.viewRange()[1])) for plot in self.axs]

    def _set_plot_title(self, plot, title: str):
        plot.setTitle(f"<span style='color:{PLOT_THEME['title']}; font-size:17px; font-weight:600'>{title}</span>")

    def _set_axis_labels(self, plot, ylabel: str, show_xlabel: bool):
        plot.setLabel("left", ylabel, color=PLOT_THEME["label"], size="12pt")
        plot.setLabel("bottom", "Time (s)" if show_xlabel else "", color=PLOT_THEME["label"], size="12pt")

    def _set_time_formatter(self, ax_idx: int, width: float):
        decimals = decimals_for_interval(width)
        self._bottom_axes[ax_idx].set_formatter(
            lambda value, d=decimals: format_time_without_trailing_zeros(value, d)
        )

    def _scene_point_to_data(self, event_pos, ax_idx):
        scene_pos = self.mapToScene(event_pos)
        if not self.axs[ax_idx].getViewBox().sceneBoundingRect().contains(scene_pos):
            return None
        mapped = self.axs[ax_idx].getViewBox().mapSceneToView(scene_pos)
        return float(mapped.x()), float(mapped.y())

    def _view_box_widget_rect(self, ax_idx):
        rect = self.axs[ax_idx].getViewBox().sceneBoundingRect()
        top_left = self.mapFromScene(rect.topLeft())
        bottom_right = self.mapFromScene(rect.bottomRight())
        return QtCore.QRect(QtCore.QPoint(top_left.x(), top_left.y()), QtCore.QPoint(bottom_right.x(), bottom_right.y())).normalized()

    def _get_lut(self):
        cmap = pg.colormap.get(SPECTROGRAM_CMAP)
        return cmap.getLookupTable(0.0, 1.0, 256)

    def _create_region(self, window, color: str, alpha: float):
        region = pg.LinearRegionItem(values=window, orientation=pg.LinearRegionItem.Vertical, movable=False)
        region.setBrush(_mk_brush(color, alpha))
        region.setHoverBrush(_mk_brush(color, alpha))
        region.setZValue(8)
        for line in region.lines:
            line.setPen(_mk_pen(color, width=1.0, alpha=max(alpha, 0.55)))
            line.setHoverPen(_mk_pen(color, width=1.0, alpha=max(alpha, 0.55)))
            line.setMovable(False)
        return region

    def _install_guides(self, ax_idx):
        self.vertical_guides[ax_idx] = pg.InfiniteLine(angle=90, movable=False, pen=_mk_pen(PLOT_THEME["guide"], width=1.0, alpha=0.45, style=QtCore.Qt.DashLine))
        self.horizontal_guides[ax_idx] = pg.InfiniteLine(angle=0, movable=False, pen=_mk_pen(PLOT_THEME["guide"], width=1.0, alpha=0.32, style=QtCore.Qt.DotLine))
        self.vertical_guides[ax_idx].setVisible(False)
        self.horizontal_guides[ax_idx].setVisible(False)
        self.axs[ax_idx].addItem(self.vertical_guides[ax_idx], ignoreBounds=True)
        self.axs[ax_idx].addItem(self.horizontal_guides[ax_idx], ignoreBounds=True)

    def _draw_event_overlay(self, ax_idx, event_start_time, event_end_time, event_color):
        plot = self.axs[ax_idx]
        region_color = blend_color(event_color, PLOT_THEME["axes_bg"], 0.5)
        boundary_color = blend_color(event_color, "#14202a", 0.12)
        self.event_regions[ax_idx] = self._create_region((event_start_time, event_end_time), region_color, 0.18 if ax_idx != 2 else 0.10)
        self.event_regions[ax_idx].setZValue(3)
        plot.addItem(self.event_regions[ax_idx], ignoreBounds=True)
        self.event_boundary_lines[ax_idx] = []
        for boundary in (event_start_time, event_end_time):
            line = pg.InfiniteLine(pos=boundary, angle=90, movable=False, pen=_mk_pen(boundary_color, width=1.2, alpha=0.8))
            line.setZValue(4)
            plot.addItem(line, ignoreBounds=True)
            self.event_boundary_lines[ax_idx].append(line)

    def _refresh_window_overlays(self):
        selected_color = PLOT_THEME["guide"]
        preview_color = PLOT_THEME["waveform_raw"]
        for ax_idx in range(3):
            if self.selected_window_regions[ax_idx] is not None:
                self.axs[ax_idx].removeItem(self.selected_window_regions[ax_idx])
                self.selected_window_regions[ax_idx] = None
            if self.preview_window_regions[ax_idx] is not None:
                self.axs[ax_idx].removeItem(self.preview_window_regions[ax_idx])
                self.preview_window_regions[ax_idx] = None
            if self.selected_fft_window is not None:
                self.selected_window_regions[ax_idx] = self._create_region(self.selected_fft_window, selected_color, 0.10)
                self.selected_window_regions[ax_idx].setZValue(9)
                self.axs[ax_idx].addItem(self.selected_window_regions[ax_idx], ignoreBounds=True)
            if self.preview_fft_window is not None:
                self.preview_window_regions[ax_idx] = self._create_region(self.preview_fft_window, preview_color, 0.08)
                self.preview_window_regions[ax_idx].setZValue(8.5)
                self.axs[ax_idx].addItem(self.preview_window_regions[ax_idx], ignoreBounds=True)

    def _capture_view_state(self):
        return {
            "xlims": [tuple(plot.viewRange()[0]) for plot in self.axs],
            "ylims": [tuple(plot.viewRange()[1]) for plot in self.axs],
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
            self.update_view(ax_idx, state["xlims"][ax_idx], state["ylims"][ax_idx], skip_draw=True)
        self.draw()
        self.history_state_changed.emit(self.can_go_back(), self.can_go_forward())

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

    def get_axis_limits(self, ax_idx, start_index, end_index, fs):
        total_samples = self.backend.get_eeg_data_shape()[1]
        xlim_max = min(end_index / fs + self.zoom_max, total_samples / fs)
        xlim_min = max(start_index / fs - self.zoom_max, 0)
        if ax_idx == 2:
            min_freq, max_freq = self._get_tf_frequency_range(self._has_filtered_signal(), fs)
            return xlim_min, xlim_max, min_freq, max_freq
        return xlim_min, xlim_max, -np.inf, np.inf

    def _compute_signal_ylim(self, ax_idx, xlim):
        axis_data = self.axis_data[ax_idx]
        if axis_data is None or axis_data.get("signal") is None:
            return self.axs[ax_idx].viewRange()[1]
        time = axis_data["time"]
        signal = axis_data["signal"]
        mask = (time >= xlim[0]) & (time <= xlim[1])
        if not np.any(mask):
            return self.axs[ax_idx].viewRange()[1]
        view_data = signal[mask]
        y_min = float(np.min(view_data))
        y_max = float(np.max(view_data))
        spread = y_max - y_min
        margin = spread * 0.1 if spread > 0 else max(abs(y_min), abs(y_max), 1.0) * 0.1
        return y_min - margin, y_max + margin

    def _compute_signal_default_ylim_for_data(self, signal, time, xlim):
        mask = (time >= xlim[0]) & (time <= xlim[1])
        view_data = signal[mask] if np.any(mask) else signal
        y_min = float(np.min(view_data))
        y_max = float(np.max(view_data))
        spread = y_max - y_min
        margin = spread * 0.1 if spread > 0 else max(abs(y_min), abs(y_max), 1.0) * 0.1
        return y_min - margin, y_max + margin

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

            current_xlim, current_ylim = self.get_axis_ranges()[ax_idx]
            current_x_range = current_xlim[1] - current_xlim[0]
            current_y_range = current_ylim[1] - current_ylim[0]

            limits = self.get_axis_limits(ax_idx, start_index, end_index, fs)
            new_x_range = np.clip(current_x_range * zoom_factor, 0.05, limits[1] - limits[0])

            x_center = (current_xlim[0] + current_xlim[1]) / 2.0
            new_x_start = x_center - new_x_range / 2.0
            new_x_end = x_center + new_x_range / 2.0
            new_xlim = self._clamp_limits((new_x_start, new_x_end), limits[0], limits[1])

            if ax_idx == 2:
                self.update_view(ax_idx, new_xlim, None, skip_draw=True)
            else:
                new_y_range = current_y_range * zoom_factor
                y_center = (current_ylim[0] + current_ylim[1]) / 2.0
                new_ylim = (y_center - new_y_range / 2.0, y_center + new_y_range / 2.0)
                self.update_view(ax_idx, new_xlim, new_ylim, skip_draw=True)
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
            current_xlim, _ = self.get_axis_ranges()[ax_idx]
            current_x_range = current_xlim[1] - current_xlim[0]
            delta_x = current_x_range * fraction
            limits = self.get_axis_limits(ax_idx, start_index, end_index, fs)
            new_xlim = self._clamp_limits((current_xlim[0] + delta_x, current_xlim[1] + delta_x), limits[0], limits[1])
            self.update_view(ax_idx, new_xlim, None, skip_draw=True)

        self.draw()
        self._record_view_state()

    def pan_vertical(self, fraction):
        if not self.backend:
            return

        for ax_idx in [0, 1]:
            if not self.data_plotted[ax_idx]:
                continue
            _, current_ylim = self.get_axis_ranges()[ax_idx]
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
        plot = self.axs[ax_idx]
        plot.clear()
        if event_start_index is None or event_start_index < 0:
            return

        self._style_plot_item(plot, show_x_grid=ax_idx != 2, show_y_grid=ax_idx != 2)

        event_start_index = int(event_start_index)
        event_end_index = int(event_end_index)
        fs = self.backend.sample_freq
        prediction = self.backend.event_features.get_current_info()["prediction"]
        event_color = COLOR_MAP.get(prediction, COLOR_MAP["HFO"])
        signal_color = PLOT_THEME["waveform_filtered"] if ax_idx == 1 else PLOT_THEME["waveform_raw"]
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
        self._set_time_formatter(ax_idx, default_xlim[1] - default_xlim[0])

        if ax_idx in (0, 1):
            default_ylim = self._compute_signal_default_ylim_for_data(eeg_data_to_display, time, default_xlim)
            curve = pg.PlotDataItem(time, eeg_data_to_display, pen=_mk_pen(signal_color, width=1.4, alpha=0.95))
            plot.addItem(curve)
            if event_rel_end > event_rel_start:
                event_time = time[event_rel_start:event_rel_end]
                event_signal = eeg_data_to_display[event_rel_start:event_rel_end]
                plot.addItem(
                    pg.PlotDataItem(
                        event_time,
                        event_signal,
                        pen=_mk_pen(blend_color(event_color, "#ffffff", 0.08), width=4.2 if ax_idx == 0 else 3.8, alpha=0.18),
                    )
                )
                plot.addItem(
                    pg.PlotDataItem(
                        event_time,
                        event_signal,
                        pen=_mk_pen(blend_color(event_color, "#14202a", 0.12), width=1.9 if ax_idx == 0 else 1.8, alpha=0.98),
                    )
                )
            self._set_plot_title(plot, "EEG Tracing" if ax_idx == 0 else "Filtered Tracing")
            self._set_axis_labels(plot, "Amplitude (uV)", ax_idx == 2)
            self.axis_data[ax_idx] = {"time": time, "signal": eeg_data_to_display}
        else:
            min_freq, max_freq = self._get_tf_frequency_range(use_filtered_signal, fs)
            tf_data = calculate_time_frequency(eeg_data_to_display, fs, freq_min=min_freq, freq_max=max_freq)
            tf_display = np.log10(np.maximum(tf_data, np.finfo(float).tiny))
            tf_display = tf_display[::-1, :]
            view_start = max(0, ws_idx - full_start_index)
            view_end = min(tf_display.shape[1], we_idx - full_start_index)
            view_data = tf_display[:, view_start:view_end]
            if view_data.size > 0:
                vmin, vmax = percentile_limits(view_data, lower=8.0, upper=99.7)
            else:
                vmin, vmax = percentile_limits(tf_display, lower=8.0, upper=99.7)
            image_item = pg.ImageItem(tf_display)
            image_item.setLookupTable(self._get_lut())
            image_item.setLevels((vmin, vmax))
            image_item.setRect(QtCore.QRectF(full_start_index / fs, min_freq, (full_end_index - full_start_index) / fs, max_freq - min_freq))
            image_item.setZValue(-10)
            plot.addItem(image_item)
            self.tf_image_items[ax_idx] = image_item
            self._spectrogram_metadata = {
                "cmap": SPECTROGRAM_CMAP,
                "high_frequency_at_top": True,
            }
            default_ylim = (min_freq, max_freq)
            self._set_plot_title(plot, "Time Frequency (Filtered)" if use_filtered_signal else "Time Frequency (Raw)")
            self._set_axis_labels(plot, "Frequency (Hz)", True)
            self._style_plot_item(plot, show_x_grid=False, show_y_grid=False)
            self.axis_data[ax_idx] = {"time": time, "signal": None}

        self._draw_event_overlay(ax_idx, event_start_time, event_end_time, event_color)
        self._install_guides(ax_idx)
        self._refresh_window_overlays()
        self.data_plotted[ax_idx] = True
        self.update_view(ax_idx, default_xlim, default_ylim, skip_draw=True)

        if not skip_draw:
            self.draw()

    def update_view(self, ax_idx: int, xlim: tuple = None, ylim: tuple = None, skip_draw: bool = False):
        if not self.data_plotted[ax_idx]:
            return

        plot = self.axs[ax_idx]
        if xlim is not None:
            plot.setXRange(xlim[0], xlim[1], padding=0)
            self._set_time_formatter(ax_idx, xlim[1] - xlim[0])

        if ylim is not None:
            plot.setYRange(ylim[0], ylim[1], padding=0)

        if not skip_draw:
            self.draw()

    def get_active_axes_index(self, event):
        scene_pos = self.mapToScene(event.pos())
        for index, plot in enumerate(self.axs):
            if plot.getViewBox().sceneBoundingRect().contains(scene_pos):
                return index
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
                line.setVisible(False)
        self.hover_text_changed.emit("")
        self.draw_idle()

    def _update_hover_display(self, ax_idx, xdata, ydata):
        for index, line in enumerate(self.vertical_guides):
            if line is not None:
                line.setPos(xdata)
                line.setVisible(True)
        for index, line in enumerate(self.horizontal_guides):
            if line is not None:
                line.setVisible(index == ax_idx)
                if index == ax_idx:
                    line.setPos(ydata)

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
            parts.append(f"fft-roi={self.selected_fft_window[0]:.4f}-{self.selected_fft_window[1]:.4f}s")
        self.hover_text_changed.emit(" | ".join(parts))
        self.draw_idle()

    def wheelEvent(self, event):
        if not self.backend:
            return super().wheelEvent(event)

        ax_idx = self.get_active_axes_index(event)
        if ax_idx is None:
            return super().wheelEvent(event)

        mouse_pos = self._scene_point_to_data(event.pos(), ax_idx)
        if mouse_pos is None:
            return super().wheelEvent(event)
        xdata, ydata = mouse_pos

        event_info = self.backend.event_features.get_current_info()
        start_index = int(event_info["start_index"])
        end_index = int(event_info["end_index"])
        fs = self.backend.sample_freq
        current_xlim, current_ylim = self.get_axis_ranges()[ax_idx]
        current_x_range = current_xlim[1] - current_xlim[0]
        current_y_range = current_ylim[1] - current_ylim[0]

        zoom_factor = 1 / 1.3 if event.angleDelta().y() > 0 else 1.3
        limits = self.get_axis_limits(ax_idx, start_index, end_index, fs)
        new_x_range = np.clip(current_x_range * zoom_factor, 0.05, limits[1] - limits[0])
        x_frac = np.clip((xdata - current_xlim[0]) / max(current_x_range, 1e-6), 0.1, 0.9)
        new_x_start = xdata - new_x_range * x_frac
        new_x_end = new_x_start + new_x_range
        new_xlim = self._clamp_limits((new_x_start, new_x_end), limits[0], limits[1])

        if ax_idx == 2:
            new_y_range = max(current_y_range * zoom_factor, 1.0)
            y_frac = np.clip((ydata - current_ylim[0]) / max(current_y_range, 1e-6), 0.1, 0.9)
            new_y_start = ydata - new_y_range * y_frac
            new_y_end = new_y_start + new_y_range
            new_ylim = self._clamp_limits((new_y_start, new_y_end), limits[2], limits[3])
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

        data_pos = self._scene_point_to_data(event.pos(), ax_idx)
        if data_pos is None:
            return super().mousePressEvent(event)

        modifiers = event.modifiers()
        self.drag_ax_idx = ax_idx
        if modifiers & QtCore.Qt.ShiftModifier:
            self.interaction_mode = "box_zoom"
            self.box_zoom_start = data_pos
            self.box_zoom_origin = event.pos()
            self.box_zoom_rubber_band.setGeometry(QtCore.QRect(event.pos(), QtCore.QSize()))
            self.box_zoom_rubber_band.show()
        elif modifiers & QtCore.Qt.AltModifier:
            self.interaction_mode = "fft_select"
            self.fft_select_start = data_pos[0]
            self.preview_fft_window = (data_pos[0], data_pos[0])
            self._refresh_window_overlays()
            self.draw_idle()
        else:
            self.interaction_mode = "pan"
            self.drag_start_screen_x = event.pos().x()
            self.drag_start_screen_y = event.pos().y()
            self.drag_start_xlim, self.drag_start_ylim = self.get_axis_ranges()[ax_idx]

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        ax_idx = self.get_active_axes_index(event)
        if self.interaction_mode == "pan" and self.drag_ax_idx is not None:
            view_rect = self._view_box_widget_rect(self.drag_ax_idx)
            x_range = self.drag_start_xlim[1] - self.drag_start_xlim[0]
            y_range = self.drag_start_ylim[1] - self.drag_start_ylim[0]
            data_delta_x = -(event.pos().x() - self.drag_start_screen_x) / max(view_rect.width(), 1) * x_range
            data_delta_y = (event.pos().y() - self.drag_start_screen_y) / max(view_rect.height(), 1) * y_range

            event_info = self.backend.event_features.get_current_info()
            start_index = int(event_info["start_index"])
            end_index = int(event_info["end_index"])
            fs = self.backend.sample_freq
            limits = self.get_axis_limits(self.drag_ax_idx, start_index, end_index, fs)
            new_xlim = self._clamp_limits(
                (self.drag_start_xlim[0] + data_delta_x, self.drag_start_xlim[1] + data_delta_x),
                limits[0],
                limits[1],
            )
            new_ylim = (self.drag_start_ylim[0] + data_delta_y, self.drag_start_ylim[1] + data_delta_y)
            if np.isfinite(limits[2]) and np.isfinite(limits[3]):
                new_ylim = self._clamp_limits(new_ylim, limits[2], limits[3])

            if self.sync_views:
                self._apply_shared_xlim(new_xlim, skip_draw=True)
                self.update_view(self.drag_ax_idx, None, new_ylim, skip_draw=True)
            else:
                self.update_view(self.drag_ax_idx, new_xlim, new_ylim, skip_draw=True)
            self.draw_idle()
        elif self.interaction_mode == "box_zoom" and self.drag_ax_idx is not None:
            self.box_zoom_rubber_band.setGeometry(QtCore.QRect(self.box_zoom_origin, event.pos()).normalized())
        elif self.interaction_mode == "fft_select" and self.drag_ax_idx is not None:
            data_pos = self._scene_point_to_data(event.pos(), self.drag_ax_idx)
            if data_pos is not None:
                self.preview_fft_window = tuple(sorted((self.fft_select_start, data_pos[0])))
                self._refresh_window_overlays()
                self.draw_idle()
        else:
            if ax_idx is None:
                self._hide_hover_guides()
            else:
                data_pos = self._scene_point_to_data(event.pos(), ax_idx)
                if data_pos is None:
                    self._hide_hover_guides()
                else:
                    self._update_hover_display(ax_idx, data_pos[0], data_pos[1])

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.interaction_mode == "pan":
            self._record_view_state()
        elif self.interaction_mode == "box_zoom" and self.drag_ax_idx is not None:
            data_pos = self._scene_point_to_data(event.pos(), self.drag_ax_idx)
            if data_pos is not None and self.box_zoom_start is not None:
                x0, y0 = self.box_zoom_start
                x1, y1 = data_pos
                if abs(x1 - x0) > 1e-4 and abs(y1 - y0) > 1e-4:
                    new_xlim = tuple(sorted((x0, x1)))
                    new_ylim = tuple(sorted((y0, y1)))
                    event_info = self.backend.event_features.get_current_info()
                    start_index = int(event_info["start_index"])
                    end_index = int(event_info["end_index"])
                    fs = self.backend.sample_freq
                    limits = self.get_axis_limits(self.drag_ax_idx, start_index, end_index, fs)
                    new_xlim = self._clamp_limits(new_xlim, limits[0], limits[1])
                    if np.isfinite(limits[2]) and np.isfinite(limits[3]):
                        new_ylim = self._clamp_limits(new_ylim, limits[2], limits[3])
                    if self.sync_views:
                        self._apply_shared_xlim(new_xlim, skip_draw=True)
                        self.update_view(self.drag_ax_idx, None, new_ylim, skip_draw=True)
                    else:
                        self.update_view(self.drag_ax_idx, new_xlim, new_ylim, skip_draw=True)
                    self.draw()
                    self._record_view_state()
            self.box_zoom_rubber_band.hide()
        elif self.interaction_mode == "fft_select":
            if self.preview_fft_window is not None and self.preview_fft_window[1] - self.preview_fft_window[0] > 1e-4:
                self.set_selected_fft_window(self.preview_fft_window, emit_signal=True)
            else:
                self.preview_fft_window = None
                self._refresh_window_overlays()
                self.draw_idle()

        self.interaction_mode = None
        self.drag_ax_idx = None
        self.box_zoom_start = None
        self.box_zoom_origin = None
        self.fft_select_start = None
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self._hide_hover_guides()
        super().leaveEvent(event)

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
        self.draw_idle()
        if emit_signal:
            self.fft_window_selected.emit(self.selected_fft_window)

    def clear_selected_fft_window(self, emit_signal=True, redraw=True):
        self.selected_fft_window = None
        self.preview_fft_window = None
        self._refresh_window_overlays()
        if redraw:
            self.draw_idle()
        if emit_signal:
            self.fft_window_selected.emit(None)

    def _clamp_limits(self, lim, lim_min, lim_max):
        lim_range = lim[1] - lim[0]
        if lim_range >= lim_max - lim_min:
            return lim_min, lim_max
        if lim[0] < lim_min:
            return lim_min, lim_min + lim_range
        if lim[1] > lim_max:
            return lim_max - lim_range, lim_max
        return lim


class FFTPlot(pg.PlotWidget):
    def __init__(self, parent=None, width=5, height=3, dpi=100, backend=None):
        self.bottom_axis = FormattedAxisItem("bottom", formatter=lambda value: custom_formatter(value))
        self.left_axis = FormattedAxisItem("left", formatter=lambda value: custom_formatter(value))
        super().__init__(parent=parent, axisItems={"bottom": self.bottom_axis, "left": self.left_axis})
        self.setBackground(PLOT_THEME["axes_bg"])
        self.setViewportMargins(6, 6, 6, 6)
        self.backend = backend
        self.min_freq = 10
        self.max_freq = 500
        self.interval = 1.0
        self.selected_time_window = None
        self.last_curve_data = (np.array([]), np.array([]))
        self.peak_badge_text = ""
        self.peak_marker = None
        self.peak_badge = None
        self.area_fill = None
        self.curve_item = None
        self._empty_label = None

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumHeight(156)
        self._style_axis()

    @property
    def axs(self):
        return self.getPlotItem()

    def draw(self):
        self.viewport().update()

    def draw_idle(self):
        self.viewport().update()

    def get_label_bounding_rects(self):
        rects = []
        plot = self.getPlotItem()
        items = [plot.titleLabel, plot.getAxis("left").label, plot.getAxis("bottom").label]
        for item in items:
            if item is None or not item.isVisible():
                continue
            scene_rect = item.mapRectToScene(item.boundingRect())
            top_left = self.mapFromScene(scene_rect.topLeft())
            bottom_right = self.mapFromScene(scene_rect.bottomRight())
            rects.append(QtCore.QRectF(QtCore.QPointF(top_left), QtCore.QPointF(bottom_right)).normalized())
        return rects

    def get_curve_data(self):
        return self.last_curve_data

    def has_peak_badge(self):
        return bool(self.peak_badge_text)

    def has_peak_marker(self):
        return self.peak_marker is not None

    def _set_title(self, title: str):
        self.getPlotItem().setTitle(f"<span style='color:{PLOT_THEME['title']}; font-size:16px; font-weight:600'>{title}</span>")

    def _style_axis(self):
        plot = self.getPlotItem()
        plot.hideButtons()
        plot.setMenuEnabled(False)
        plot.getViewBox().setMouseEnabled(x=False, y=False)
        plot.getViewBox().setBackgroundColor(PLOT_THEME["axes_bg"])
        # Reserve enough room for the bottom axis title inside the plot widget.
        plot.layout.setContentsMargins(0, 0, 0, 18)
        plot.showGrid(x=True, y=True, alpha=0.32)
        for axis_name in ("left", "bottom"):
            axis = plot.getAxis(axis_name)
            axis.setPen(_mk_pen(PLOT_THEME["spine"], width=1.0))
            axis.setTextPen(_mk_pen(PLOT_THEME["tick"], width=1.0))
            axis.setStyle(autoExpandTextSpace=True, hideOverlappingLabels=False, tickTextOffset=6)
        plot.setLabel("left", "PSD (%)", color=PLOT_THEME["label"], size="12pt")
        plot.setLabel("bottom", "Frequency (Hz)", color=PLOT_THEME["label"], size="12pt")

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
        plot = self.getPlotItem()
        plot.clear()
        self._style_axis()
        self.peak_marker = None
        self.peak_badge = None
        self.peak_badge_text = ""
        self._empty_label = None

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
            self.last_curve_data = (np.array([]), np.array([]))
            self._set_title("FFT")
            plot.setXRange(self.min_freq, self.max_freq, padding=0)
            plot.setYRange(0, 1, padding=0)
            self._empty_label = pg.TextItem("Window too small", color=PLOT_THEME["empty"], anchor=(0.5, 0.5))
            self._empty_label.setPos((self.min_freq + self.max_freq) / 2.0, 0.5)
            plot.addItem(self._empty_label)
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
        self.last_curve_data = (filtered_freqs, psd_percent)

        line_color = PLOT_THEME["waveform_filtered"]
        zero_curve = pg.PlotDataItem(filtered_freqs, np.zeros_like(psd_percent), pen=None)
        self.curve_item = pg.PlotDataItem(filtered_freqs, psd_percent, pen=_mk_pen(line_color, width=1.8))
        plot.addItem(zero_curve)
        plot.addItem(self.curve_item)
        self.area_fill = pg.FillBetweenItem(self.curve_item, zero_curve, brush=_mk_brush(PLOT_THEME["fft_fill"], 0.32))
        plot.addItem(self.area_fill)

        y_max = max(float(np.max(psd_percent)) * 1.15, 1.0) if psd_percent.size else 1.0
        if filtered_freqs.size > 0:
            peak_index = int(np.argmax(psd_percent))
            peak_freq = float(filtered_freqs[peak_index])
            peak_power = float(psd_percent[peak_index])
            if np.isfinite(peak_freq) and np.isfinite(peak_power) and peak_power > 0:
                plot.addItem(pg.InfiniteLine(pos=peak_freq, angle=90, movable=False, pen=_mk_pen(blend_color(line_color, "#ffffff", 0.25), width=1.0, alpha=0.4)))
                self.peak_marker = pg.ScatterPlotItem([peak_freq], [peak_power], size=8, brush=_mk_brush(line_color), pen=_mk_pen(line_color))
                plot.addItem(self.peak_marker)
                self.peak_badge_text = f"Peak {peak_freq:.0f} Hz"
                self.peak_badge = pg.TextItem(
                    text=self.peak_badge_text,
                    color=PLOT_THEME["label"],
                    fill=_mk_brush("#ffffff", 0.9),
                    border=_mk_pen(PLOT_THEME["spine"], width=0.8, alpha=0.95),
                    anchor=(1.0, 1.0),
                )
                self.peak_badge.setPos(self.max_freq - ((self.max_freq - self.min_freq) * 0.02), y_max * 0.94)
                plot.addItem(self.peak_badge)

        plot.setXRange(self.min_freq, self.max_freq, padding=0)
        plot.setYRange(0, y_max, padding=0)
        if self.selected_time_window is not None:
            self._set_title(f"FFT ROI {self.selected_time_window[0]:.3f}-{self.selected_time_window[1]:.3f}s")
        else:
            self._set_title("FFT")
        self.draw()
