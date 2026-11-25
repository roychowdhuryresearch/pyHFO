from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
from PyQt5 import QtGui
import numpy as np
import math
from src.ui.ui_tokens import resolve_ui_density


class MainWaveformPlotView(QtWidgets.QGraphicsView):
    channel_gutter_clicked = QtCore.pyqtSignal(str)
    measurement_point_selected = QtCore.pyqtSignal(str, float)
    wheel_time_scroll_requested = QtCore.pyqtSignal(float)
    wheel_channel_scroll_requested = QtCore.pyqtSignal(float)
    time_zoom_requested = QtCore.pyqtSignal(float, float)
    DEFAULT_TRACKPAD_SENSITIVITY = "default"
    TRACKPAD_SENSITIVITY_PROFILES = {
        "gentle": {
            "scroll_divisor": 128.0,
            "pinch_curve": 0.8,
            "pinch_sensitivity": 0.4,
        },
        "default": {
            "scroll_divisor": 96.0,
            "pinch_curve": 1.0,
            "pinch_sensitivity": 0.55,
        },
        "fast": {
            "scroll_divisor": 64.0,
            "pinch_curve": 1.3,
            "pinch_sensitivity": 0.8,
        },
    }
    MAX_GUTTER_TEXT_WIDTH = 220
    GUTTER_SIDE_PADDING = 8

    def __init__(self, plot_widget: pg.PlotWidget, channel_gutter_widget: pg.PlotWidget = None):
        super(MainWaveformPlotView, self).__init__()
        self.plot_widget = plot_widget
        self.channel_gutter_widget = channel_gutter_widget
        self._axis_width_cache = {}
        self._channel_label_items = []
        self._channel_display_labels = {}
        self._channel_tooltips = {}
        self._visible_channel_names = []
        self._visible_channel_locations = np.array([], dtype=float)
        self._visible_channel_offset = 0.0
        self._hovered_channel_name = None
        self.highlighted_channel_name = None
        self._main_row_highlight = None
        self._gutter_row_highlight = None
        self.cursor_enabled = False
        self.cursor_vline = None
        self.cursor_hline = None
        self.cursor_label = None
        self._cursor_proxy = None
        self._wheel_event_source = None
        self.trackpad_sensitivity = self.DEFAULT_TRACKPAD_SENSITIVITY
        self.touchpad_scroll_divisor = self.TRACKPAD_SENSITIVITY_PROFILES[self.DEFAULT_TRACKPAD_SENSITIVITY]["scroll_divisor"]
        self.native_pinch_curve = self.TRACKPAD_SENSITIVITY_PROFILES[self.DEFAULT_TRACKPAD_SENSITIVITY]["pinch_curve"]
        self.native_pinch_sensitivity = self.TRACKPAD_SENSITIVITY_PROFILES[self.DEFAULT_TRACKPAD_SENSITIVITY]["pinch_sensitivity"]
        self.measurement_enabled = False
        self.measurement_points = []
        self.measurement_summary_text = ""
        self.measurement_marker_item = None
        self.measurement_connector_item = None
        self.measurement_label = None
        self.measurement_vlines = []
        self._scale_bar_items = []
        self._scale_text_item = None
        self._init_plot_widget(plot_widget)
        if self.channel_gutter_widget is not None:
            self._init_channel_gutter(self.channel_gutter_widget)

    def _init_plot_widget(self, plot_widget: pg.PlotWidget):
        plot_widget.setMouseEnabled(x=False, y=False)
        plot_widget.getPlotItem().hideAxis('bottom')
        plot_widget.getPlotItem().hideAxis('left')
        plot_widget.setBackground('w')
        # Disable auto-ranging to use explicit ranges
        plot_widget.disableAutoRange()
        # Performance optimizations
        plot_widget.setAntialiasing(False)  # Faster rendering
        plot_widget.getPlotItem().setClipToView(True)  # Only render visible data
        plot_widget.getPlotItem().setDownsampling(auto=True, mode='peak')
        self._ensure_cursor_items()
        self._ensure_measurement_items()
        self._install_wheel_event_filter()
        if self._cursor_proxy is None:
            self._cursor_proxy = pg.SignalProxy(plot_widget.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved)
        plot_widget.scene().sigMouseClicked.connect(self._on_plot_scene_clicked)

    def _init_channel_gutter(self, plot_widget: pg.PlotWidget):
        plot_widget.setMouseEnabled(x=False, y=False)
        plot_item = plot_widget.getPlotItem()
        plot_item.showAxis('bottom')
        plot_item.hideAxis('left')
        plot_item.hideAxis('top')
        plot_item.hideAxis('right')
        plot_item.hideButtons()
        plot_item.setMenuEnabled(False)
        plot_widget.setBackground('w')
        plot_widget.disableAutoRange()
        plot_item.vb.setDefaultPadding(0.0)
        plot_item.vb.setMouseEnabled(x=False, y=False)
        bottom_axis = plot_widget.getAxis('bottom')
        bottom_axis.setStyle(showValues=False, tickLength=0)
        bottom_axis.setPen(pg.mkPen(0, 0, 0, 0))
        bottom_axis.setTextPen(pg.mkColor(0, 0, 0, 0))
        plot_widget.setRange(xRange=(0.0, 1.0), yRange=(-1.0, 1.0), padding=0, update=True)
        plot_widget.scene().sigMouseClicked.connect(self._on_gutter_scene_clicked)

    def _ensure_cursor_items(self):
        if self.cursor_vline is None:
            self.cursor_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#5f7385", width=1))
        if self.cursor_hline is None:
            self.cursor_hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#5f7385", width=1))
        if self.cursor_label is None:
            self.cursor_label = pg.TextItem("", color="#304657", anchor=(0, 1))

        for item in (self.cursor_vline, self.cursor_hline, self.cursor_label):
            if item is None:
                continue
            try:
                item.scene()
            except RuntimeError:
                continue
            if item.scene() is None:
                self.plot_widget.addItem(item, ignoreBounds=True)
        self._set_cursor_visibility(self.cursor_enabled)

    def _set_cursor_visibility(self, visible):
        for item in (self.cursor_vline, self.cursor_hline, self.cursor_label):
            if item is not None:
                item.setVisible(bool(visible))

    def _ensure_measurement_items(self):
        if len(self.measurement_vlines) != 2:
            self.measurement_vlines = [
                pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#5f7385", width=1, style=QtCore.Qt.DashLine)),
                pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#bf7b30", width=1, style=QtCore.Qt.DashLine)),
            ]
        if self.measurement_marker_item is None:
            self.measurement_marker_item = pg.ScatterPlotItem(size=8, pxMode=True)
        if self.measurement_connector_item is None:
            self.measurement_connector_item = pg.PlotDataItem(
                pen=pg.mkPen("#526b85", width=1.3),
                symbol=None,
            )
        if self.measurement_label is None:
            self.measurement_label = pg.TextItem("", color="#304657", anchor=(0, 1))

        for item in (*self.measurement_vlines, self.measurement_marker_item, self.measurement_connector_item, self.measurement_label):
            if item is None:
                continue
            try:
                item.scene()
            except RuntimeError:
                continue
            if item.scene() is None:
                self.plot_widget.addItem(item, ignoreBounds=True)
        self._render_measurement()

    def _set_measurement_visibility(self, visible):
        for item in (*self.measurement_vlines, self.measurement_marker_item, self.measurement_connector_item, self.measurement_label):
            if item is not None:
                item.setVisible(bool(visible))

    def _graphics_view_for_events(self):
        graphics_view = getattr(self.plot_widget, "graphics_widget", None)
        if graphics_view is None and hasattr(self.plot_widget, "viewport"):
            graphics_view = self.plot_widget
        return graphics_view

    def _install_wheel_event_filter(self):
        graphics_view = self._graphics_view_for_events()
        if graphics_view is None or not hasattr(graphics_view, "viewport"):
            return
        viewport = graphics_view.viewport()
        if viewport is None or viewport is self._wheel_event_source:
            return
        if self._wheel_event_source is not None:
            try:
                self._wheel_event_source.removeEventFilter(self)
            except Exception:
                pass
        viewport.installEventFilter(self)
        self._wheel_event_source = viewport

    def _wheel_step_deltas(self, event):
        pixel_delta = event.pixelDelta() if hasattr(event, "pixelDelta") else QtCore.QPoint(0, 0)
        if pixel_delta is not None and (pixel_delta.x() or pixel_delta.y()):
            return (
                float(pixel_delta.x()) / self.touchpad_scroll_divisor,
                float(pixel_delta.y()) / self.touchpad_scroll_divisor,
            )

        angle_delta = event.angleDelta() if hasattr(event, "angleDelta") else QtCore.QPoint(0, 0)
        return float(angle_delta.x()) / 120.0, float(angle_delta.y()) / 120.0

    def _is_touchpad_scroll_event(self, event):
        pixel_delta = event.pixelDelta() if hasattr(event, "pixelDelta") else QtCore.QPoint(0, 0)
        return bool(pixel_delta is not None and (pixel_delta.x() or pixel_delta.y()))

    def _zoom_anchor_time_for_scene_pos(self, scene_pos):
        x_range = self.plot_widget.getPlotItem().vb.viewRange()[0]
        anchor_time = float((x_range[0] + x_range[1]) / 2.0)
        if self._main_plot_contains_scene_pos(scene_pos):
            anchor_time = float(self.plot_widget.getPlotItem().vb.mapSceneToView(scene_pos).x())
        return anchor_time

    def _native_pinch_zoom_delta(self, gesture_value):
        gesture_value = float(gesture_value)
        if abs(gesture_value) < 1e-6:
            return 0.0
        scaled_delta = math.log1p(abs(gesture_value) * self.native_pinch_curve) * self.native_pinch_sensitivity
        return math.copysign(scaled_delta, gesture_value)

    def set_trackpad_sensitivity(self, profile_name):
        normalized_profile = str(profile_name or self.DEFAULT_TRACKPAD_SENSITIVITY).strip().lower()
        if normalized_profile not in self.TRACKPAD_SENSITIVITY_PROFILES:
            normalized_profile = self.DEFAULT_TRACKPAD_SENSITIVITY
        profile = self.TRACKPAD_SENSITIVITY_PROFILES[normalized_profile]
        self.trackpad_sensitivity = normalized_profile
        self.touchpad_scroll_divisor = float(profile["scroll_divisor"])
        self.native_pinch_curve = float(profile["pinch_curve"])
        self.native_pinch_sensitivity = float(profile["pinch_sensitivity"])

    def get_trackpad_sensitivity(self):
        return str(self.trackpad_sensitivity or self.DEFAULT_TRACKPAD_SENSITIVITY)

    def _main_plot_contains_scene_pos(self, scene_pos):
        return bool(scene_pos is not None and self.plot_widget.getPlotItem().vb.sceneBoundingRect().contains(scene_pos))

    def _gutter_contains_scene_pos(self, scene_pos):
        if scene_pos is None or self.channel_gutter_widget is None:
            return False
        return bool(self.channel_gutter_widget.getPlotItem().vb.sceneBoundingRect().contains(scene_pos))

    def eventFilter(self, obj, event):
        if self._wheel_event_source is not None and obj is not self._wheel_event_source:
            return super(MainWaveformPlotView, self).eventFilter(obj, event)

        graphics_view = self._graphics_view_for_events()
        if graphics_view is None or event is None:
            return super(MainWaveformPlotView, self).eventFilter(obj, event)

        event_type = event.type()
        if event_type == QtCore.QEvent.NativeGesture:
            if not hasattr(event, "pos") or event.gestureType() != QtCore.Qt.ZoomNativeGesture:
                return super(MainWaveformPlotView, self).eventFilter(obj, event)
            scene_pos = graphics_view.mapToScene(event.pos())
            over_main_plot = self._main_plot_contains_scene_pos(scene_pos)
            over_gutter = self._gutter_contains_scene_pos(scene_pos)
            if not over_main_plot and not over_gutter:
                return super(MainWaveformPlotView, self).eventFilter(obj, event)

            zoom_delta = self._native_pinch_zoom_delta(event.value())
            if abs(zoom_delta) < 1e-6:
                if hasattr(event, "accept"):
                    event.accept()
                return True
            self.time_zoom_requested.emit(zoom_delta, self._zoom_anchor_time_for_scene_pos(scene_pos))
            if hasattr(event, "accept"):
                event.accept()
            return True

        if event_type != QtCore.QEvent.Wheel:
            return super(MainWaveformPlotView, self).eventFilter(obj, event)
        if not hasattr(event, "pos"):
            return super(MainWaveformPlotView, self).eventFilter(obj, event)

        scene_pos = graphics_view.mapToScene(event.pos())
        over_main_plot = self._main_plot_contains_scene_pos(scene_pos)
        over_gutter = self._gutter_contains_scene_pos(scene_pos)
        if not over_main_plot and not over_gutter:
            return super(MainWaveformPlotView, self).eventFilter(obj, event)

        step_delta_x, step_delta_y = self._wheel_step_deltas(event)
        if abs(step_delta_x) < 1e-6 and abs(step_delta_y) < 1e-6:
            if hasattr(event, "accept"):
                event.accept()
            return True

        modifiers = event.modifiers() if hasattr(event, "modifiers") else QtCore.Qt.NoModifier
        if self._is_touchpad_scroll_event(event):
            if abs(step_delta_x) >= 1e-6:
                self.wheel_time_scroll_requested.emit(float(step_delta_x))
            if abs(step_delta_y) >= 1e-6:
                self.wheel_channel_scroll_requested.emit(float(step_delta_y))
        else:
            if modifiers & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier):
                if abs(step_delta_y) >= 1e-6:
                    self.time_zoom_requested.emit(float(step_delta_y) * math.log(1.25), self._zoom_anchor_time_for_scene_pos(scene_pos))
            elif modifiers & QtCore.Qt.ShiftModifier:
                if abs(step_delta_y) >= 1e-6:
                    self.wheel_time_scroll_requested.emit(float(step_delta_y))
            else:
                if abs(step_delta_y) >= 1e-6:
                    self.wheel_channel_scroll_requested.emit(float(step_delta_y))

        if hasattr(event, "accept"):
            event.accept()
        return True

    def _on_mouse_moved(self, event):
        scene_pos = event[0] if isinstance(event, (tuple, list)) else event
        self._update_hover_for_scene_pos(scene_pos)

    def set_cursor_enabled(self, enabled):
        self.cursor_enabled = bool(enabled)
        self._ensure_cursor_items()
        self._set_cursor_visibility(self.cursor_enabled)

    def is_cursor_enabled(self):
        return bool(self.cursor_enabled)

    def set_measurement_enabled(self, enabled):
        self.measurement_enabled = bool(enabled)
        self._ensure_measurement_items()
        if not self.measurement_enabled:
            self.clear_measurement()
        else:
            self._render_measurement()

    def is_measurement_enabled(self):
        return bool(self.measurement_enabled)

    def set_measurement(self, points, summary_text=""):
        self.measurement_points = [dict(point) for point in (points or [])[:2]]
        self.measurement_summary_text = str(summary_text or "")
        self._ensure_measurement_items()
        self._render_measurement()

    def clear_measurement(self):
        self.measurement_points = []
        self.measurement_summary_text = ""
        self._ensure_measurement_items()
        self._render_measurement()

    def get_highlighted_channel(self):
        return str(self.highlighted_channel_name) if self.highlighted_channel_name else None

    def clear(self):
        # Block updates during clear for better performance
        self.plot_widget.getPlotItem().vb.setMouseEnabled(x=False, y=False)
        self.plot_widget.clear()
        self._scale_bar_items = []
        self._scale_text_item = None
        self._ensure_cursor_items()
        self._ensure_measurement_items()
        self._main_row_highlight = None
        self._gutter_row_highlight = None
        self._hovered_channel_name = None
        self._visible_channel_names = []
        self._visible_channel_locations = np.array([], dtype=float)
        self._visible_channel_offset = 0.0
        if self.channel_gutter_widget is not None:
            self._clear_channel_gutter()

    def set_channel_presentation(self, presentation_by_channel):
        presentation_by_channel = presentation_by_channel or {}
        self._channel_display_labels = {
            str(channel_name): str(entry.get("display_label") or str(channel_name))
            for channel_name, entry in presentation_by_channel.items()
        }
        self._channel_tooltips = {
            str(channel_name): str(entry.get("tooltip") or str(channel_name))
            for channel_name, entry in presentation_by_channel.items()
        }
        self._axis_width_cache = {}

    def clear_channel_presentation(self):
        self._channel_display_labels = {}
        self._channel_tooltips = {}
        self._axis_width_cache = {}

    def _display_label_for_channel(self, channel_name):
        return self._channel_display_labels.get(str(channel_name), str(channel_name))

    def _tooltip_for_channel(self, channel_name):
        return self._channel_tooltips.get(str(channel_name), str(channel_name))

    def enable_axis_information(self):
        self.plot_widget.getPlotItem().showAxis('bottom')
        if self.channel_gutter_widget is None:
            self.plot_widget.getPlotItem().showAxis('left')
        else:
            self.plot_widget.getPlotItem().hideAxis('left')
        self.plot_widget.setTitle("")
        bottom_axis = self.plot_widget.getAxis('bottom')
        bottom_axis.setLabel("")
        bottom_axis.setStyle(tickTextOffset=4, tickLength=4)
        bottom_axis.setPen(pg.mkPen("#bcc7d1", width=1))
        bottom_axis.setTextPen(pg.mkColor("#7b8894"))
        if self.channel_gutter_widget is None:
            self.plot_widget.getAxis('left').setStyle(
                autoExpandTextSpace=False,
                autoReduceTextSpace=False,
            )
        else:
            self.channel_gutter_widget.setTitle("")
            gutter_bottom_axis = self.channel_gutter_widget.getAxis('bottom')
            desired_bottom_height = max(
                int(bottom_axis.height()),
                int(bottom_axis.style.get('tickTextHeight', 0) + bottom_axis.style.get('tickTextOffset', [0, 0])[1] + max(0, bottom_axis.style.get('tickLength', 0))),
            )
            gutter_bottom_axis.setStyle(showValues=False, tickLength=0)
            gutter_bottom_axis.setPen(pg.mkPen(0, 0, 0, 0))
            gutter_bottom_axis.setTextPen(pg.mkColor(0, 0, 0, 0))
            gutter_bottom_axis.setHeight(desired_bottom_height)

    def calculate_left_axis_width(self, channel_names):
        labels = tuple(self._display_label_for_channel(channel_name) for channel_name in channel_names)
        cached_width = self._axis_width_cache.get(labels)
        if cached_width is not None:
            return cached_width

        axis = self.plot_widget.getAxis('left')
        font = axis.style.get('tickFont') or self.plot_widget.font()
        metrics = QtGui.QFontMetrics(font)
        max_text_width = max((metrics.horizontalAdvance(label) for label in labels), default=0)
        max_text_width = min(max_text_width, self.MAX_GUTTER_TEXT_WIDTH)
        tick_offset = int(axis.style.get('tickTextOffset', [0, 0])[0])
        tick_length = max(0, int(axis.style.get('tickLength', 0)))
        label_padding = int(axis.label.boundingRect().height() * 0.8) if axis.label.isVisible() else 0
        reserved_width = max(int(axis.style.get('tickTextWidth', 0)), max_text_width) + tick_offset + tick_length + label_padding + 8

        if len(self._axis_width_cache) > 8:
            self._axis_width_cache.pop(next(iter(self._axis_width_cache)))
        self._axis_width_cache[labels] = reserved_width
        return reserved_width

    def _clear_channel_gutter(self):
        self.channel_gutter_widget.clear()
        self._channel_label_items = []

    def _refresh_channel_label_styles(self):
        if self.channel_gutter_widget is None or len(self._channel_label_items) != len(self._visible_channel_names):
            return
        base_font = self.plot_widget.getAxis('left').style.get('tickFont') or self.plot_widget.font()
        for text_item, channel_name in zip(self._channel_label_items, self._visible_channel_names):
            label_font = QtGui.QFont(base_font)
            is_selected = str(channel_name) == self.highlighted_channel_name
            is_hovered = str(channel_name) == self._hovered_channel_name
            label_font.setBold(is_selected or is_hovered)
            text_item.setFont(label_font)
            if is_selected:
                text_item.setColor("#304657")
            elif is_hovered:
                text_item.setColor("#526b85")
            else:
                text_item.setColor("#7b8894")

    def reserve_left_axis_width(self, channel_names):
        reserved_width = self.calculate_left_axis_width(channel_names)
        if self.channel_gutter_widget is not None:
            gutter_plot_item = self.channel_gutter_widget.getPlotItem()
            gutter_plot_item.setMinimumWidth(reserved_width)
            gutter_plot_item.setMaximumWidth(reserved_width)
            graphics_widget = getattr(self.channel_gutter_widget, "graphics_widget", None)
            if graphics_widget is not None:
                try:
                    graphics_widget.ci.layout.setColumnFixedWidth(0, reserved_width)
                except Exception:
                    pass
            return
        self.plot_widget.getAxis('left').setWidth(reserved_width)

    def _gutter_text_width_limit(self):
        return max(24, int(self.get_left_axis_width()) - self.GUTTER_SIDE_PADDING)

    def _build_row_highlight_item(self):
        region = pg.LinearRegionItem(
            values=[0.0, 0.0],
            orientation=pg.LinearRegionItem.Horizontal,
            movable=False,
            brush=pg.mkBrush(90, 122, 163, 34),
            pen=pg.mkPen(0, 0, 0, 0),
            hoverBrush=pg.mkBrush(90, 122, 163, 34),
            hoverPen=pg.mkPen(0, 0, 0, 0),
            span=(0.0, 1.0),
        )
        region.setZValue(-20)
        return region

    def _ensure_row_highlight_items(self):
        if self._main_row_highlight is None or self._main_row_highlight.scene() is None:
            self._main_row_highlight = self._build_row_highlight_item()
            self.plot_widget.addItem(self._main_row_highlight, ignoreBounds=True)
        if self.channel_gutter_widget is not None and (self._gutter_row_highlight is None or self._gutter_row_highlight.scene() is None):
            self._gutter_row_highlight = self._build_row_highlight_item()
            self.channel_gutter_widget.addItem(self._gutter_row_highlight, ignoreBounds=True)

    def _set_row_highlight_visible(self, visible):
        for item in (self._main_row_highlight, self._gutter_row_highlight):
            if item is not None:
                item.setVisible(bool(visible))

    def _active_channel_name(self):
        return self._hovered_channel_name or self.highlighted_channel_name

    def _set_hovered_channel(self, channel_name):
        hovered_channel_name = str(channel_name) if channel_name else None
        if hovered_channel_name == self._hovered_channel_name:
            return
        self._hovered_channel_name = hovered_channel_name
        self._refresh_channel_label_styles()
        self._render_row_highlight()

    def set_highlighted_channel(self, channel_name):
        self.highlighted_channel_name = str(channel_name) if channel_name else None
        self._hovered_channel_name = None
        self._refresh_channel_label_styles()
        self._render_row_highlight()

    def _render_row_highlight(self):
        active_channel_name = self._active_channel_name()
        if not active_channel_name or active_channel_name not in self._visible_channel_names:
            self._set_row_highlight_visible(False)
            return

        self._ensure_row_highlight_items()
        row_index = self._visible_channel_names.index(active_channel_name)
        y_center = float(self._visible_channel_locations[row_index])
        half_height = max(0.5, float(self._visible_channel_offset) * 0.5)
        region = [y_center - half_height, y_center + half_height]
        self._main_row_highlight.setRegion(region)
        self._main_row_highlight.setVisible(True)
        if self._gutter_row_highlight is not None:
            self._gutter_row_highlight.setRegion(region)
            self._gutter_row_highlight.setVisible(True)

    def _resolve_channel_name_for_y(self, y_value):
        if len(self._visible_channel_names) == 0 or self._visible_channel_locations.size == 0:
            return None
        distances = np.abs(self._visible_channel_locations - float(y_value))
        nearest_index = int(np.argmin(distances))
        threshold = max(0.6, float(self._visible_channel_offset) * 0.55) if self._visible_channel_offset else float("inf")
        if distances[nearest_index] > threshold:
            return None
        return self._visible_channel_names[nearest_index]

    def _activate_channel_for_y(self, y_value):
        channel_name = self._resolve_channel_name_for_y(y_value)
        if channel_name:
            self.set_highlighted_channel(channel_name)
            self.channel_gutter_clicked.emit(str(channel_name))
        return channel_name

    def _update_hover_for_scene_pos(self, scene_pos):
        if scene_pos is None:
            self._set_hovered_channel(None)
            self._set_cursor_visibility(False)
            return False

        main_view_box = self.plot_widget.getPlotItem().vb
        main_view_rect = main_view_box.sceneBoundingRect()
        if main_view_rect.contains(scene_pos):
            mouse_point = main_view_box.mapSceneToView(scene_pos)
            self._set_hovered_channel(self._resolve_channel_name_for_y(mouse_point.y()))
            if self.cursor_enabled:
                self._ensure_cursor_items()
                x_value = float(mouse_point.x())
                y_value = float(mouse_point.y())
                self.cursor_vline.setPos(x_value)
                self.cursor_hline.setPos(y_value)
                y_range = main_view_box.viewRange()[1]
                self.cursor_label.setHtml(
                    f"<div style='background-color:rgba(255,255,255,0.88); padding:2px 4px; color:#304657;'>"
                    f"{x_value:.2f} s</div>"
                )
                self.cursor_label.setPos(x_value, y_range[1])
                self._set_cursor_visibility(True)
            else:
                self._set_cursor_visibility(False)
            return True

        if self.channel_gutter_widget is not None:
            gutter_view_box = self.channel_gutter_widget.getPlotItem().vb
            gutter_view_rect = gutter_view_box.sceneBoundingRect()
            if gutter_view_rect.contains(scene_pos):
                mouse_point = gutter_view_box.mapSceneToView(scene_pos)
                self._set_hovered_channel(self._resolve_channel_name_for_y(mouse_point.y()))
                self._set_cursor_visibility(False)
                return True

        self._set_hovered_channel(None)
        self._set_cursor_visibility(False)
        return False

    def _on_gutter_scene_clicked(self, event):
        if self.channel_gutter_widget is None or event is None:
            return
        if hasattr(event, "button") and event.button() != QtCore.Qt.LeftButton:
            return
        scene_pos = event.scenePos() if hasattr(event, "scenePos") else None
        if scene_pos is None:
            return
        gutter_view_rect = self.channel_gutter_widget.getPlotItem().vb.sceneBoundingRect()
        if not gutter_view_rect.contains(scene_pos):
            return
        y_value = float(self.channel_gutter_widget.getPlotItem().vb.mapSceneToView(scene_pos).y())
        channel_name = self._activate_channel_for_y(y_value)
        if channel_name and hasattr(event, "accept"):
            event.accept()

    def _draw_channel_gutter_labels(self, channel_names_locs, visible_channel_names):
        self._clear_channel_gutter()
        font = self.plot_widget.getAxis('left').style.get('tickFont') or self.plot_widget.font()
        metrics = QtGui.QFontMetrics(font)
        text_width_limit = self._gutter_text_width_limit()
        for y_pos, channel_name in zip(channel_names_locs, visible_channel_names):
            full_label = str(channel_name)
            display_label = metrics.elidedText(self._display_label_for_channel(channel_name), QtCore.Qt.ElideRight, text_width_limit)
            is_selected = full_label == self.highlighted_channel_name
            label_font = QtGui.QFont(font)
            label_font.setBold(is_selected)
            text_item = pg.TextItem(display_label, color="#304657" if is_selected else "#7b8894", anchor=(1.0, 0.5))
            text_item.setFont(label_font)
            text_item.setPos(0.98, float(y_pos))
            text_item.setToolTip(self._tooltip_for_channel(channel_name))
            self.channel_gutter_widget.addItem(text_item, ignoreBounds=True)
            self._channel_label_items.append(text_item)
        self._refresh_channel_label_styles()

    def _measurement_display_y(self, point):
        channel_name = str(point.get("channel_name") or "")
        if channel_name not in self._visible_channel_names:
            return None
        row_index = self._visible_channel_names.index(channel_name)
        display_value = float(point.get("display_value", 0.0))
        return display_value - row_index * float(self._visible_channel_offset)

    def _render_measurement(self):
        if (
            not self.measurement_enabled
            or not self.measurement_points
            or not self._visible_channel_names
            or not self.measurement_vlines
            or self.measurement_marker_item is None
            or self.measurement_connector_item is None
            or self.measurement_label is None
        ):
            self._set_measurement_visibility(False)
            return

        render_points = []
        for point in self.measurement_points:
            display_y = self._measurement_display_y(point)
            if display_y is None:
                self._set_measurement_visibility(False)
                return
            render_points.append(
                {
                    "time_seconds": float(point.get("time_seconds", 0.0)),
                    "display_y": float(display_y),
                }
            )

        brushes = [pg.mkBrush("#5f7385"), pg.mkBrush("#bf7b30")]
        pens = [pg.mkPen("#5f7385", width=1), pg.mkPen("#bf7b30", width=1)]
        spots = []
        for index, point in enumerate(render_points):
            self.measurement_vlines[index].setPos(point["time_seconds"])
            self.measurement_vlines[index].setVisible(True)
            spots.append(
                {
                    "pos": (point["time_seconds"], point["display_y"]),
                    "data": index,
                    "brush": brushes[index],
                    "pen": pens[index],
                    "size": 8,
                }
            )
        for index in range(len(render_points), len(self.measurement_vlines)):
            self.measurement_vlines[index].setVisible(False)

        self.measurement_marker_item.setData(spots)
        self.measurement_marker_item.setVisible(True)

        if len(render_points) >= 2:
            self.measurement_connector_item.setData(
                [render_points[0]["time_seconds"], render_points[1]["time_seconds"]],
                [render_points[0]["display_y"], render_points[1]["display_y"]],
            )
            self.measurement_connector_item.setVisible(True)
        else:
            self.measurement_connector_item.setData([], [])
            self.measurement_connector_item.setVisible(False)

        y_range = self.plot_widget.getPlotItem().vb.viewRange()[1]
        label_x = min(point["time_seconds"] for point in render_points)
        label_text = self.measurement_summary_text or "Click second point"
        self.measurement_label.setHtml(
            f"<div style='background-color:rgba(255,255,255,0.9); padding:2px 4px; color:#304657;'>"
            f"{label_text}</div>"
        )
        self.measurement_label.setPos(label_x, y_range[1])
        self.measurement_label.setVisible(True)

    def begin_batch_update(self):
        """Begin batching updates for better performance"""
        self.plot_widget.getPlotItem().vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    
    def end_batch_update(self):
        """End batching updates and refresh the view"""
        self.plot_widget.getPlotItem().vb.update()

    def get_render_width(self):
        return max(self.plot_widget.width(), int(self.plot_widget.getPlotItem().vb.width()))

    def plot_waveform(self, x, y, color, width):
        # Optimize plotting with performance settings
        pen = pg.mkPen(color=color, width=width)
        curve = self.plot_widget.plot(x, y, pen=pen, 
                                      antialias=False,  # Faster rendering
                                      skipFiniteCheck=False,  # Handle NaN values
                                      connect='finite')  # Skip NaN values efficiently
        # Enable clipping and downsampling for better performance
        curve.setClipToView(True)
        if len(x) > 1000:  # Only downsample if there are many points
            curve.setDownsampling(auto=True, method='peak')

    def _clear_scale_bar(self):
        for item in (*self._scale_bar_items, self._scale_text_item):
            if item is None:
                continue
            try:
                if item.scene() is not None:
                    self.plot_widget.removeItem(item)
            except Exception:
                pass
        self._scale_bar_items = []
        self._scale_text_item = None

    def draw_scale_bar(self, x_pos, y_pos, y_100_length, y_scale_length):
        density = resolve_ui_density(self.screen())
        self._clear_scale_bar()

        view_box = self.plot_widget.getPlotItem().vb
        x_range = view_box.viewRange()[0]
        time_span = max(1e-6, float(x_range[1] - x_range[0]))
        inset = max(time_span * 0.028, 0.06)
        cap_half_width = max(time_span * 0.009, 0.028)
        label_gap = max(time_span * 0.018, 0.045)
        scale_x = min(float(x_pos), float(x_range[1]) - inset)
        scale_mid_y = float(y_pos) + (float(y_scale_length) / 2.0)

        scale_pen = pg.mkPen("#486277", width=2)
        scale_pen.setCapStyle(QtCore.Qt.RoundCap)
        scale_pen.setJoinStyle(QtCore.Qt.RoundJoin)

        line_segments = (
            ([scale_x, scale_x], [y_pos, y_pos + y_scale_length]),
            ([scale_x - cap_half_width, scale_x + cap_half_width], [y_pos, y_pos]),
            ([scale_x - cap_half_width, scale_x + cap_half_width], [y_pos + y_scale_length, y_pos + y_scale_length]),
        )

        for x_values, y_values in line_segments:
            line_item = pg.PlotDataItem(x_values, y_values, pen=scale_pen)
            line_item.setZValue(18)
            self.plot_widget.addItem(line_item, ignoreBounds=True)
            self._scale_bar_items.append(line_item)

        text_item = pg.TextItem(
            f" {y_100_length} μV ",
            color="#304657",
            anchor=(1, 0.5),
            border=pg.mkPen("#d5dde4", width=1),
            fill=pg.mkBrush(250, 252, 253, 236),
        )
        label_font = QtGui.QFont(self.plot_widget.font())
        label_font.setPointSize(max(8, density.compact_label_font))
        label_font.setWeight(QtGui.QFont.DemiBold)
        text_item.setFont(label_font)
        text_item.setZValue(19)
        text_item.setPos(scale_x - cap_half_width - label_gap, scale_mid_y)
        self.plot_widget.addItem(text_item, ignoreBounds=True)
        self._scale_text_item = text_item

    def draw_channel_names(self, offset_value, n_channels_to_plot, channels_to_plot, first_channel_to_plot, start_in_time, end_in_time):
        self.reserve_left_axis_width(channels_to_plot)
        channel_names_locs = -offset_value * np.arange(n_channels_to_plot)
        visible_channel_names = channels_to_plot[first_channel_to_plot:first_channel_to_plot + n_channels_to_plot]
        self._visible_channel_names = list(visible_channel_names)
        self._visible_channel_locations = np.asarray(channel_names_locs, dtype=float)
        self._visible_channel_offset = float(offset_value)

        if self.channel_gutter_widget is None:
            self.plot_widget.getAxis('left').setTicks([[(channel_names_locs[disp_i], self._display_label_for_channel(visible_channel_names[disp_i]))
                     for disp_i in range(len(visible_channel_names))]])
        
        # Set Y range to show all channels properly
        y_max = offset_value  # top margin
        y_min = -(n_channels_to_plot - 1) * offset_value - offset_value  # bottom margin
        
        # Batch range updates for better performance
        self.plot_widget.setRange(xRange=(start_in_time, end_in_time), 
                                  yRange=(y_min, y_max), 
                                  padding=0, 
                                  update=True)
        if self.channel_gutter_widget is not None:
            self.channel_gutter_widget.setRange(
                xRange=(0.0, 1.0),
                yRange=(y_min, y_max),
                padding=0,
                update=True,
            )
            self._draw_channel_gutter_labels(channel_names_locs, visible_channel_names)
        self._render_row_highlight()
        self._render_measurement()

    def get_left_axis_width(self):
        if self.channel_gutter_widget is not None:
            plot_item = self.channel_gutter_widget.getPlotItem()
            width_candidates = [0]
            if hasattr(plot_item, "minimumWidth"):
                try:
                    width_candidates.append(int(plot_item.minimumWidth()))
                except Exception:
                    pass
            if hasattr(plot_item, "maximumWidth"):
                try:
                    width_candidates.append(int(plot_item.maximumWidth()))
                except Exception:
                    pass
            return max(width_candidates)
        axis = self.plot_widget.getAxis('left')
        width_candidates = [0]
        if hasattr(axis, "width"):
            try:
                width_candidates.append(int(axis.width()))
            except Exception:
                pass
        if hasattr(axis, "size"):
            try:
                width_candidates.append(int(axis.size().width()))
            except Exception:
                pass
        return max(width_candidates)

    def _on_plot_scene_clicked(self, event):
        if event is None:
            return
        if hasattr(event, "button") and event.button() != QtCore.Qt.LeftButton:
            return
        scene_pos = event.scenePos() if hasattr(event, "scenePos") else None
        if scene_pos is None:
            return
        plot_rect = self.plot_widget.getPlotItem().vb.sceneBoundingRect()
        if not plot_rect.contains(scene_pos):
            return
        mouse_point = self.plot_widget.getPlotItem().vb.mapSceneToView(scene_pos)
        channel_name = self._activate_channel_for_y(mouse_point.y())
        if not channel_name:
            return
        if self.measurement_enabled:
            self.measurement_point_selected.emit(str(channel_name), float(mouse_point.x()))
        if hasattr(event, "accept"):
            event.accept()
