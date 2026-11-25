from functools import lru_cache

from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt


_VIEWBOX = 24.0
_STROKE = 2.0
_OFF_COLOR = "#4b6476"
_ON_COLOR = "#203344"
_DISABLED_COLOR = "#a8b2bb"


def _pt(x, y):
    return QtCore.QPointF(float(x), float(y))


def _line(painter, start, end):
    painter.drawLine(_pt(*start), _pt(*end))


def _polyline(painter, points):
    path = QtGui.QPainterPath(_pt(*points[0]))
    for point in points[1:]:
        path.lineTo(_pt(*point))
    painter.drawPath(path)


def _circle(painter, x, y, radius, *, fill=False):
    if fill:
        brush = painter.brush()
        painter.setBrush(painter.pen().color())
        painter.drawEllipse(_pt(x, y), radius, radius)
        painter.setBrush(brush)
        return
    painter.drawEllipse(_pt(x, y), radius, radius)


def _rounded_rect(painter, x, y, width, height, radius=2.5):
    painter.drawRoundedRect(QtCore.QRectF(x, y, width, height), radius, radius)


def _draw_prev(painter):
    _line(painter, (18, 12), (7, 12))
    _line(painter, (11, 8), (7, 12))
    _line(painter, (11, 16), (7, 12))


def _draw_center(painter):
    _circle(painter, 12, 12, 4.5)
    _line(painter, (12, 3.5), (12, 7))
    _line(painter, (12, 17), (12, 20.5))
    _line(painter, (3.5, 12), (7, 12))
    _line(painter, (17, 12), (20.5, 12))


def _draw_next(painter):
    _line(painter, (6, 12), (17, 12))
    _line(painter, (13, 8), (17, 12))
    _line(painter, (13, 16), (17, 12))


def _draw_pending(painter):
    _circle(painter, 12, 12, 7)
    _line(painter, (12, 12), (12, 8))
    _line(painter, (12, 12), (15, 14))


def _draw_review(painter):
    _rounded_rect(painter, 4.5, 4.5, 10, 14, radius=2)
    _line(painter, (9, 8), (12, 8))
    _line(painter, (7.5, 11), (11.5, 11))
    _line(painter, (7.5, 14), (10.5, 14))
    _circle(painter, 15.5, 15.5, 3.2)
    _line(painter, (17.9, 17.9), (20, 20))


def _draw_snap(painter):
    _rounded_rect(painter, 4, 7, 16, 10, radius=2.5)
    _rounded_rect(painter, 8.5, 5, 5, 3, radius=1.5)
    _circle(painter, 12, 12, 3.2)


def _draw_raw(painter):
    _polyline(painter, [(3, 14), (6, 10), (8, 16), (10, 8), (13, 17), (16, 9), (21, 13)])


def _draw_filtered(painter):
    path = QtGui.QPainterPath(_pt(3, 13))
    path.cubicTo(_pt(6, 8), _pt(8.5, 8), _pt(11, 13))
    path.cubicTo(_pt(13.5, 18), _pt(16.5, 18), _pt(21, 11))
    painter.drawPath(path)


def _draw_notch(painter):
    _draw_filtered(painter)
    _line(painter, (18.5, 6), (5.5, 18))


def _draw_normalize(painter):
    _line(painter, (4.5, 18), (19.5, 18))
    for x in (6, 12, 18):
        _line(painter, (x, 17), (x, 8))
        _line(painter, (x - 2, 10), (x, 8))
        _line(painter, (x + 2, 10), (x, 8))
    _line(painter, (4.5, 6.5), (19.5, 6.5))


def _draw_channels(painter):
    for y in (7, 12, 17):
        _circle(painter, 5.5, y, 1.2, fill=True)
        _line(painter, (8.5, y), (19, y))


def _draw_overlap(painter):
    _circle(painter, 9.5, 12, 5)
    _circle(painter, 14.5, 12, 5)


def _draw_montage(painter):
    top = ((5.5, 7), (12, 7), (18.5, 7))
    bottom = ((5.5, 17), (12, 17), (18.5, 17))
    for point in (*top, *bottom):
        _circle(painter, point[0], point[1], 1.2, fill=True)
    _line(painter, top[0], bottom[1])
    _line(painter, top[1], bottom[2])
    _line(painter, bottom[0], top[1])
    _line(painter, bottom[1], top[2])


def _draw_ref(painter):
    for x in (6, 12, 18):
        _circle(painter, x, 6.5, 1.2, fill=True)
        _line(painter, (x, 8), (12, 15))
    _circle(painter, 12, 17.5, 1.4, fill=True)
    _line(painter, (7, 20), (17, 20))


def _draw_average(painter):
    for point in ((6, 7), (12, 7), (18, 7)):
        _circle(painter, point[0], point[1], 1.2, fill=True)
        _line(painter, (point[0], 8.5), (12, 13))
    _circle(painter, 12, 14.5, 3.2)
    _line(painter, (9.5, 14.5), (14.5, 14.5))


def _draw_auto_bipolar(painter):
    nodes = (4.5, 10, 15.5, 21)
    for x in nodes:
        _circle(painter, x, 12, 1.2, fill=True)
    _line(painter, (5.7, 12), (8.8, 12))
    _line(painter, (11.2, 12), (14.3, 12))
    _line(painter, (16.7, 12), (19.8, 12))
    _line(painter, (7.3, 8.2), (7.3, 11))
    _line(painter, (5.8, 9.6), (8.8, 9.6))
    _line(painter, (11.8, 9.6), (14.8, 9.6))
    _line(painter, (18, 8.2), (18, 11))
    _line(painter, (16.5, 9.6), (19.5, 9.6))


def _draw_focus(painter):
    _circle(painter, 12, 12, 6.5)
    _circle(painter, 12, 12, 2.2)
    _circle(painter, 12, 12, 0.9, fill=True)


def _draw_adjacent(painter):
    _line(painter, (5, 7), (16, 7))
    _line(painter, (4, 12), (20, 12))
    _line(painter, (5, 17), (16, 17))
    _line(painter, (18.5, 7), (18.5, 17))
    _line(painter, (18.5, 7), (16.5, 9))
    _line(painter, (18.5, 17), (16.5, 15))
    _circle(painter, 12, 12, 1.2, fill=True)


def _draw_clean(painter):
    path = QtGui.QPainterPath(_pt(9, 6))
    path.lineTo(_pt(18, 9))
    path.lineTo(_pt(13.5, 17))
    path.lineTo(_pt(4.5, 14))
    path.closeSubpath()
    painter.drawPath(path)
    _line(painter, (5, 19), (19, 19))
    _line(painter, (13.2, 8), (16.8, 9.2))


def _draw_events(painter):
    _polyline(painter, [(3, 15), (7, 15), (9, 9), (11, 17), (13, 12), (21, 12)])
    _line(painter, (8, 5), (8, 19))
    _line(painter, (16, 5), (16, 19))
    _circle(painter, 8, 8, 1.1, fill=True)
    _circle(painter, 16, 10, 1.1, fill=True)


def _draw_all(painter):
    for x in (5, 10.5, 16):
        for y in (5, 10.5, 16):
            painter.drawRect(QtCore.QRectF(x, y, 3, 3))


def _draw_go(painter):
    _line(painter, (5, 12), (16, 12))
    _line(painter, (12, 8), (16, 12))
    _line(painter, (12, 16), (16, 12))
    _line(painter, (19, 5), (19, 19))


def _draw_zoom_in(painter):
    _circle(painter, 10, 10, 5)
    _line(painter, (13.8, 13.8), (19, 19))
    _line(painter, (10, 7), (10, 13))
    _line(painter, (7, 10), (13, 10))


def _draw_zoom_out(painter):
    _circle(painter, 10, 10, 5)
    _line(painter, (13.8, 13.8), (19, 19))
    _line(painter, (7, 10), (13, 10))


def _draw_cursor(painter):
    _line(painter, (12, 4), (12, 20))
    _line(painter, (4, 12), (20, 12))
    _circle(painter, 12, 12, 1.4)


def _draw_measure(painter):
    _rounded_rect(painter, 4, 9, 16, 6, radius=1.5)
    for x in (7, 10, 13, 16):
        _line(painter, (x, 9), (x, 12 if x in (10, 16) else 11))


def _draw_hotspot(painter):
    path = QtGui.QPainterPath(_pt(12, 4.5))
    path.lineTo(_pt(13.8, 9))
    path.lineTo(_pt(18.5, 9.2))
    path.lineTo(_pt(14.8, 12.2))
    path.lineTo(_pt(16.2, 17))
    path.lineTo(_pt(12, 14.2))
    path.lineTo(_pt(7.8, 17))
    path.lineTo(_pt(9.2, 12.2))
    path.lineTo(_pt(5.5, 9.2))
    path.lineTo(_pt(10.2, 9))
    path.closeSubpath()
    painter.drawPath(path)


_ICON_DRAWERS = {
    "prev": _draw_prev,
    "center": _draw_center,
    "next": _draw_next,
    "pending": _draw_pending,
    "review": _draw_review,
    "snap": _draw_snap,
    "raw": _draw_raw,
    "filtered": _draw_filtered,
    "notch": _draw_notch,
    "normalize": _draw_normalize,
    "channels": _draw_channels,
    "overlap": _draw_overlap,
    "montage": _draw_montage,
    "referential": _draw_ref,
    "average": _draw_average,
    "auto_bipolar": _draw_auto_bipolar,
    "focus": _draw_focus,
    "adjacent": _draw_adjacent,
    "clean": _draw_clean,
    "events": _draw_events,
    "all": _draw_all,
    "go": _draw_go,
    "zoom_in": _draw_zoom_in,
    "zoom_out": _draw_zoom_out,
    "cursor": _draw_cursor,
    "measure": _draw_measure,
    "hotspot": _draw_hotspot,
}


def _render_icon(draw_fn, size, color, dpr=1.0):
    pixel_size = max(1, int(round(size * dpr)))
    pixmap = QtGui.QPixmap(pixel_size, pixel_size)
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
    painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
    if hasattr(QtGui.QPainter, "HighQualityAntialiasing"):
        painter.setRenderHint(QtGui.QPainter.HighQualityAntialiasing, True)
    painter.scale(size / _VIEWBOX, size / _VIEWBOX)
    painter.setPen(QtGui.QPen(QtGui.QColor(color), _STROKE, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.setBrush(Qt.NoBrush)
    draw_fn(painter)
    painter.end()
    return pixmap


@lru_cache(maxsize=None)
def get_waveform_toolbar_icon(name, size=16):
    if name not in _ICON_DRAWERS:
        raise KeyError(f"Unknown waveform toolbar icon: {name}")
    draw_fn = _ICON_DRAWERS[name]
    icon = QtGui.QIcon()
    for dpr in (1.0, 2.0, 3.0):
        icon.addPixmap(_render_icon(draw_fn, size, _OFF_COLOR, dpr), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon.addPixmap(_render_icon(draw_fn, size, _ON_COLOR, dpr), QtGui.QIcon.Normal, QtGui.QIcon.On)
        icon.addPixmap(_render_icon(draw_fn, size, _ON_COLOR, dpr), QtGui.QIcon.Active, QtGui.QIcon.Off)
        icon.addPixmap(_render_icon(draw_fn, size, _ON_COLOR, dpr), QtGui.QIcon.Active, QtGui.QIcon.On)
        icon.addPixmap(_render_icon(draw_fn, size, _DISABLED_COLOR, dpr), QtGui.QIcon.Disabled, QtGui.QIcon.Off)
        icon.addPixmap(_render_icon(draw_fn, size, _DISABLED_COLOR, dpr), QtGui.QIcon.Disabled, QtGui.QIcon.On)
    return icon
