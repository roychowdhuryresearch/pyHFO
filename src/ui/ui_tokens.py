import sys
from dataclasses import dataclass, fields

from PyQt5 import QtWidgets


@dataclass(frozen=True)
class UiDensity:
    base_font: int = 10
    toolbar_font: int = 9
    button_font: int = 9
    compact_label_font: int = 9
    compact_unit_font: int = 8
    section_title_font: int = 10
    dock_title_font: int = 15
    body_font: int = 9
    small_font: int = 8
    tiny_font: int = 7
    status_value_font: int = 11

    compact_label_width: int = 56
    compact_unit_width: int = 14
    compact_input_height: int = 18
    compact_button_height: int = 22
    compact_tool_height: int = 16
    compact_radius: int = 5
    group_radius: int = 7

    waveform_header_title_font: int = 12
    waveform_header_meta_font: int = 9
    waveform_control_strip_height: int = 136

    inspector_dock_width: int = 280
    inspector_dock_min_width: int = 264
    inspector_dock_max_width: int = 308

    annotation_nav_button_width: int = 84
    annotation_jump_button_width: int = 88
    annotation_view_button_width: int = 64
    annotation_window_min_width: int = 1080
    annotation_window_min_height: int = 720
    annotation_window_default_width: int = 1240
    annotation_window_default_height: int = 820

    def scaled(self, scale: float) -> "UiDensity":
        values = {}
        for field in fields(self):
            base_value = getattr(self, field.name)
            if "font" in field.name:
                minimum = 7
            elif "radius" in field.name:
                minimum = 3
            elif "width" in field.name or "height" in field.name:
                minimum = 12
            else:
                minimum = 2
            values[field.name] = max(minimum, int(round(base_value * scale)))
        return UiDensity(**values)


UI_DENSITY = UiDensity()


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _platform_scale() -> float:
    if sys.platform.startswith("win"):
        return 0.96
    if sys.platform.startswith("linux"):
        return 0.98
    return 1.0


def resolve_ui_scale(screen=None) -> float:
    app = QtWidgets.QApplication.instance()
    screen = screen or (app.primaryScreen() if app is not None else None)
    if screen is None:
        return 1.0

    rect = screen.availableGeometry()
    logical_dpi = screen.logicalDotsPerInch() or 96.0

    resolution_scale = min(rect.width() / 1440.0, rect.height() / 900.0)
    dpi_scale = _clamp(logical_dpi / 96.0, 0.9, 1.2)
    blended_scale = (resolution_scale * 0.72) + (dpi_scale * 0.28)

    return _clamp(blended_scale * _platform_scale(), 0.9, 1.08)


def resolve_ui_density(screen=None) -> UiDensity:
    return UI_DENSITY.scaled(resolve_ui_scale(screen))


def resolve_window_size(
    *,
    default_width: int,
    default_height: int,
    min_width: int,
    min_height: int,
    screen=None,
    width_ratio: float = 0.88,
    height_ratio: float = 0.86,
):
    app = QtWidgets.QApplication.instance()
    screen = screen or (app.primaryScreen() if app is not None else None)
    if screen is None:
        return max(min_width, default_width), max(min_height, default_height)

    rect = screen.availableGeometry()
    width = max(min_width, min(default_width, int(rect.width() * width_ratio)))
    height = max(min_height, min(default_height, int(rect.height() * height_ratio)))
    return width, height
