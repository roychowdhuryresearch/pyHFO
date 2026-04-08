import os
import re
import sys
import queue
import traceback
from queue import Queue
from pathlib import Path
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from src.ui.ui_tokens import resolve_ui_density, resolve_window_size

COLOR_MAP = {'waveform': '#58748a',
             'HFO': '#6d8d74',
             'non_spike': '#6d8d74',
             'Real': '#6d8d74',
             'Accepted': '#6d8d74',
             'Unpredicted': '#6d8d74',
             'artifact': '#b46f55',
             'Artifact': '#b46f55',
             'spkHFO': '#7f6b93',
             'spike': '#7f6b93',
             'Spike': '#7f6b93',
             'eHFO': '#8d5f67',
             'ehfo': '#8d5f67',
             'spkHFO and eHFO': '#8d5f67',
             'eHFO and spkHFO': '#8d5f67'}

CHEVRON_DOWN_ICON = (Path(__file__).resolve().parents[1] / "ui" / "images" / "chevron_down.svg").as_posix()
CHECKBOX_CHECK_ICON = (Path(__file__).resolve().parents[1] / "ui" / "images" / "checkbox_check.svg").as_posix()


def build_checkbox_indeterminate_fill(fill_color, mark_color="#ffffff"):
    return (
        "qlineargradient("
        f"x1:0, y1:0, x2:0, y2:1, "
        f"stop:0 {fill_color}, "
        f"stop:0.36 {fill_color}, "
        f"stop:0.36 {mark_color}, "
        f"stop:0.64 {mark_color}, "
        f"stop:0.64 {fill_color}, "
        f"stop:1 {fill_color})"
    )


def build_checkbox_styles(
    prefix="",
    *,
    size=16,
    radius=5,
    spacing=8,
    label_color="#33495b",
    label_disabled_color="#94a2af",
    border_color="#b7c3ce",
    hover_border_color="#8da0b1",
    background="#ffffff",
    hover_background="#f5f8fb",
    accent="#355c72",
    accent_hover="#2d4e61",
    disabled_border="#d7dee5",
    disabled_background="#f5f7f9",
    disabled_accent="#b6c1cc",
):
    prefix = prefix or ""
    indeterminate_fill = build_checkbox_indeterminate_fill(accent)
    indeterminate_hover_fill = build_checkbox_indeterminate_fill(accent_hover)
    indeterminate_disabled_fill = build_checkbox_indeterminate_fill(disabled_accent)
    return f"""
    {prefix}QCheckBox {{
        color: {label_color};
        spacing: {spacing}px;
    }}
    {prefix}QCheckBox:disabled {{
        color: {label_disabled_color};
    }}
    {prefix}QCheckBox::indicator {{
        width: {size}px;
        height: {size}px;
        border: 1px solid {border_color};
        border-radius: {radius}px;
        background: {background};
    }}
    {prefix}QCheckBox::indicator:hover {{
        border-color: {hover_border_color};
        background: {hover_background};
    }}
    {prefix}QCheckBox::indicator:checked,
    {prefix}QCheckBox::indicator:indeterminate {{
        border-color: {accent};
        background: {accent};
    }}
    {prefix}QCheckBox::indicator:checked:hover,
    {prefix}QCheckBox::indicator:indeterminate:hover {{
        border-color: {accent_hover};
        background: {accent_hover};
    }}
    {prefix}QCheckBox::indicator:checked {{
        image: url("{CHECKBOX_CHECK_ICON}");
    }}
    {prefix}QCheckBox::indicator:indeterminate {{
        background: {indeterminate_fill};
        image: none;
    }}
    {prefix}QCheckBox::indicator:indeterminate:hover {{
        background: {indeterminate_hover_fill};
    }}
    {prefix}QCheckBox::indicator:disabled {{
        border-color: {disabled_border};
        background: {disabled_background};
    }}
    {prefix}QCheckBox::indicator:checked:disabled,
    {prefix}QCheckBox::indicator:indeterminate:disabled {{
        border-color: {disabled_accent};
        background: {disabled_accent};
    }}
    {prefix}QCheckBox::indicator:checked:disabled {{
        image: url("{CHECKBOX_CHECK_ICON}");
    }}
    {prefix}QCheckBox::indicator:indeterminate:disabled {{
        background: {indeterminate_disabled_fill};
        image: none;
    }}
    """


SUBWINDOW_STYLESHEET = """
    QDialog, QMainWindow, QMessageBox {
        background: #f3f4f6;
    }
    QLabel {
        color: #243746;
    }
    QLabel[dialogTitle="true"] {
        font-size: 18px;
        font-weight: 700;
        color: #243746;
    }
    QLabel[mutedText="true"] {
        color: #64717d;
    }
    QLabel[fieldLabel="true"] {
        color: #4b5563;
        font-weight: 600;
    }
    QLabel[fieldUnit="true"] {
        color: #7b8792;
    }
    QLabel[pageEyebrow="true"] {
        color: #73808b;
        font-weight: 700;
    }
    QLabel[pageTitle="true"] {
        color: #243746;
        font-weight: 700;
    }
    QLabel[pageSubtitle="true"] {
        color: #64717d;
    }
    QLabel[sectionTitle="true"] {
        color: #243746;
        font-weight: 700;
    }
    QLabel[sectionSubtitle="true"], QLabel[helperText="true"] {
        color: #64717d;
    }
    QLabel[metricLabel="true"] {
        color: #6a7784;
        font-weight: 600;
    }
    QLabel[metricValue="true"], QLabel[statusValue="true"] {
        color: #1f3448;
        font-weight: 700;
    }
    QLabel[stepBadge="true"] {
        background: #e7edf2;
        color: #435565;
        border-radius: 11px;
        font-weight: 700;
        padding: 1px 6px;
    }
    QLabel[stepTitle="true"] {
        color: #34485a;
        font-weight: 700;
    }
    QLabel[stepDescription="true"] {
        color: #7a8792;
    }
    QLabel[toolbarLabel="true"] {
        color: #60707d;
        font-weight: 700;
    }
    QLabel[dataBadge="true"] {
        background: #ffffff;
        border: 1px solid #d8dde3;
        border-radius: 8px;
        padding: 5px 8px;
        color: #243746;
        font-weight: 600;
    }
    QWidget#DialogScrollContent {
        background: #ffffff;
        border: 1px solid #d8dde3;
        border-radius: 12px;
    }
    QFrame[surfaceCard="true"], QFrame[metricCard="true"], QFrame[panelCard="true"] {
        background: #ffffff;
        border: 1px solid #d8dde3;
        border-radius: 12px;
    }
    QFrame[softCard="true"], QFrame[stepCard="true"] {
        background: #fafbfc;
        border: 1px solid #e5e9ee;
        border-radius: 10px;
    }
    QGroupBox {
        border: 1px solid #dde3ea;
        border-radius: 12px;
        margin-top: 12px;
        padding: 14px 12px 12px;
        background: #fbfcfd;
        font-weight: 700;
        color: #314657;
    }
    QGroupBox[tightGroup="true"] {
        padding-top: 12px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 4px;
        color: #314657;
    }
    QPushButton, QToolButton, QDialogButtonBox QPushButton {
        background: #ffffff;
        color: #243746;
        border: 1px solid #cfd6dd;
        border-radius: 8px;
        padding: 4px 10px;
        min-height: 16px;
    }
    QPushButton:hover, QToolButton:hover, QDialogButtonBox QPushButton:hover {
        background: #eef3f7;
        border-color: #c2ccd6;
    }
    QPushButton:pressed, QToolButton:pressed, QDialogButtonBox QPushButton:pressed {
        background: #e5ebf1;
        border-color: #b6c2cd;
    }
    QPushButton:disabled, QToolButton:disabled, QDialogButtonBox QPushButton:disabled {
        background: #f7f8fa;
        color: #a2aeb9;
        border-color: #dde2e8;
    }
    QPushButton[accentButton="true"], QToolButton[accentButton="true"], QDialogButtonBox QPushButton[accentButton="true"] {
        background: #304657;
        color: #ffffff;
        border-color: #304657;
        font-weight: 700;
    }
    QPushButton[accentButton="true"]:hover, QToolButton[accentButton="true"]:hover, QDialogButtonBox QPushButton[accentButton="true"]:hover {
        background: #3b5467;
        border-color: #3b5467;
    }
    QPushButton[accentButton="true"]:pressed, QToolButton[accentButton="true"]:pressed, QDialogButtonBox QPushButton[accentButton="true"]:pressed {
        background: #233746;
        border-color: #233746;
    }
    QPushButton[accentButton="true"]:disabled, QToolButton[accentButton="true"]:disabled, QDialogButtonBox QPushButton[accentButton="true"]:disabled {
        background: #f4f6f8;
        color: #96a3af;
        border-color: #d7dde4;
    }
    QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit, QComboBox {
        background: #ffffff;
        border: 1px solid #cfd6dd;
        border-radius: 8px;
        padding: 4px 8px;
        color: #243746;
        selection-background-color: #d9e5ef;
        selection-color: #243746;
    }
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus {
        border-color: #8aa0b2;
    }
    QLineEdit[readOnlyField="true"], QTextEdit[readOnlyField="true"], QPlainTextEdit[readOnlyField="true"] {
        background: #f5f7f9;
        color: #556472;
        border-color: #d7dde4;
    }
    QTextEdit[consolePanel="true"], QPlainTextEdit[consolePanel="true"] {
        background: #f7f9fb;
        color: #334455;
        border: 1px solid #d4dbe2;
        border-radius: 9px;
        font-family: Menlo, Monaco, monospace;
    }
""" + build_checkbox_styles(
    size=16,
    radius=5,
    spacing=8,
    label_color="#33495b",
    label_disabled_color="#94a2af",
    border_color="#b7c3ce",
    hover_border_color="#8da0b1",
    background="#ffffff",
    hover_background="#f5f8fb",
    accent="#355c72",
    accent_hover="#2d4e61",
    disabled_border="#d7dee5",
    disabled_background="#f5f7f9",
    disabled_accent="#b6c1cc",
) + """
    QRadioButton {
        color: #33495b;
        spacing: 8px;
    }
    QRadioButton::indicator {
        width: 16px;
        height: 16px;
    }
    QRadioButton::indicator {
        border: 1px solid #b7c3ce;
        border-radius: 8px;
        background: #ffffff;
    }
    QRadioButton::indicator:checked {
        border: 5px solid #304657;
        background: #ffffff;
    }
    QScrollArea {
        border: none;
        background: transparent;
    }
    QListWidget, QTreeWidget, QTableWidget {
        background: #ffffff;
        border: 1px solid #d8dde3;
        border-radius: 10px;
        gridline-color: #edf0f2;
        alternate-background-color: #f8fafc;
    }
    QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {
        background: #e7eef5;
        color: #243746;
    }
    QHeaderView::section {
        background: #f5f7f9;
        color: #556472;
        border: none;
        border-bottom: 1px solid #e5e9ee;
        padding: 4px 6px;
        font-weight: 600;
    }
    QTabWidget::pane {
        border: 1px solid #d8dde3;
        border-radius: 9px;
        background: #ffffff;
    }
    QTabBar::tab {
        background: #edf1f4;
        color: #5b6874;
        padding: 6px 10px;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        margin-right: 4px;
    }
    QTabBar::tab:selected {
        background: #ffffff;
        color: #243746;
        font-weight: 700;
    }
    QProgressBar {
        background: #e7edf2;
        border: 1px solid #d8dde3;
        border-radius: 8px;
        text-align: center;
        color: #243746;
    }
    QProgressBar::chunk {
        background: #5a758a;
        border-radius: 7px;
    }
    QStatusBar {
        background: #f7f8fa;
        color: #5a6b79;
        border-top: 1px solid #d8dde3;
    }
    QMenuBar {
        background: #f7f8fa;
        color: #243746;
        border-bottom: 1px solid #d8dde3;
    }
    QMenuBar::item:selected {
        background: #eef3f7;
    }
    QScrollBar:vertical {
        width: 12px;
        background: transparent;
        margin: 4px 0;
    }
    QScrollBar::handle:vertical {
        background: #c8d3dc;
        min-height: 36px;
        border-radius: 6px;
    }
    QScrollBar::handle:vertical:hover {
        background: #b7c5d0;
    }
    QScrollBar:horizontal {
        height: 12px;
        background: transparent;
        margin: 0 4px;
    }
    QScrollBar::handle:horizontal {
        background: #c8d3dc;
        min-width: 36px;
        border-radius: 6px;
    }
    QScrollBar::handle:horizontal:hover {
        background: #b7c5d0;
    }
    QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {
        border: none;
        background: transparent;
    }
"""


def build_popup_chrome_stylesheet(
    density,
    *,
    prefix="",
    control_radius=None,
    popup_radius=None,
    item_radius=None,
    drop_width=None,
    item_height=None,
    font_size=None,
    input_padding_right=None,
    drop_background="#f8fafc",
):
    control_radius = max(3, control_radius if control_radius is not None else density.compact_radius)
    popup_radius = max(control_radius, popup_radius if popup_radius is not None else density.group_radius + 1)
    item_radius = max(3, item_radius if item_radius is not None else control_radius)
    drop_width = max(18, drop_width if drop_width is not None else density.compact_input_height)
    item_height = max(16, item_height if item_height is not None else density.compact_input_height)
    font_size = max(8, font_size if font_size is not None else density.button_font)
    input_padding_right = max(drop_width + 8, input_padding_right if input_padding_right is not None else drop_width + 8)
    prefix = prefix or ""

    return f"""
        {prefix}QComboBox {{
            padding-right: {input_padding_right}px;
        }}
        {prefix}QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: {drop_width}px;
            border: none;
            border-left: 1px solid #dbe3ea;
            background: {drop_background};
            border-top-right-radius: {control_radius}px;
            border-bottom-right-radius: {control_radius}px;
        }}
        {prefix}QComboBox::down-arrow {{
            image: url("{CHEVRON_DOWN_ICON}");
            width: 10px;
            height: 10px;
        }}
        {prefix}QComboBox QAbstractItemView {{
            background: #ffffff;
            color: #243746;
            border: 1px solid #d8dde3;
            border-radius: {popup_radius}px;
            outline: 0px;
            padding: 4px;
            selection-background-color: #e7eef5;
            selection-color: #243746;
        }}
        {prefix}QComboBox QAbstractItemView::item {{
            min-height: {item_height}px;
            padding: 2px 8px;
            border-radius: {item_radius}px;
            margin: 1px 0px;
            font-size: {font_size}px;
        }}
        {prefix}QMenu {{
            background: #ffffff;
            color: #243746;
            border: 1px solid #d8dde3;
            border-radius: {popup_radius}px;
            padding: 4px;
        }}
        {prefix}QMenu::item {{
            padding: 5px 10px;
            border-radius: {item_radius}px;
            margin: 1px 0px;
            font-size: {font_size}px;
        }}
        {prefix}QMenu::item:selected {{
            background: #eef3f7;
            color: #243746;
        }}
        {prefix}QMenu::separator {{
            height: 1px;
            background: #e5e9ee;
            margin: 4px 6px;
        }}
        {prefix}QToolButton::menu-indicator {{
            image: url("{CHEVRON_DOWN_ICON}");
            width: 10px;
            height: 10px;
            subcontrol-origin: padding;
            subcontrol-position: center right;
        }}
    """


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    Supported signals are:
    finished
        No data
    error
        tuple (exctype, value, traceback.format_exc())
    result
        list object returned from thread
    progress
        int indicating % progress

    """
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    """
    Worker thread
    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function
    """

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            # Return the result of the function
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


class WriteStream(object):
    def __init__(self, q: Queue):
        self.queue = q

    def write(self, text):
        self.queue.put(text)

    def flush(self):
        pass


class STDOutReceiver(QThread):
    std_received_signal = pyqtSignal(str)

    def __init__(self, q: Queue, *args, **kwargs):
        QObject.__init__(self, *args, **kwargs)
        self.queue = q
        self._isRunning = True

    @pyqtSlot()
    def run(self):
        if not self._isRunning:
            self._isRunning = True

        while self._isRunning:
            try:
                text = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if text is None:
                break
            self.std_received_signal.emit(text)

    def stop(self):
        self._isRunning = False
        self.queue.put(None)


class STDErrReceiver(QThread):
    std_received_signal = pyqtSignal(str)

    def __init__(self, q: Queue, *args, **kwargs):
        QObject.__init__(self, *args, **kwargs)
        self.queue = q
        self._isRunning = True

    @pyqtSlot()
    def run(self):
        if not self._isRunning:
            self._isRunning = True

        while self._isRunning:
            try:
                text = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if text is None:
                break
            self.std_received_signal.emit(text)

    def stop(self):
        self._isRunning = False
        self.queue.put(None)


def clear_layout(layout):
    if layout is not None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
                continue

            child_layout = item.layout()
            if child_layout is not None:
                clear_layout(child_layout)
                child_layout.setParent(None)
                continue

            layout.removeItem(item)


def detach_layout(layout):
    if layout is None:
        return

    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.hide()
            continue

        child_layout = item.layout()
        if child_layout is not None:
            detach_layout(child_layout)
            continue

        layout.removeItem(item)


def clear_stacked_widget(stacked_widget):
    # Remove all pages from the QStackedWidget
    while stacked_widget.count() > 0:
        widget = stacked_widget.widget(0)  # Get the first page
        stacked_widget.removeWidget(widget)  # Remove the widget
        widget.deleteLater()  # Delete the widget


# def clear_frame(frame):
#     # Clear the frame by removing all child widgets
#     for child in frame.children():
#         # if isinstance(child, QLayout):
#         #     clear_layout(child)
#         if isinstance(child, QWidget):
#             child.deleteLater()
def safe_connect_signal_slot(signal, slot):
    """Ensure the signal is connected only once."""
    try:
        signal.disconnect(slot)
    except TypeError:
        pass
    signal.connect(slot)


def polish_widget_style(widget):
    if widget is None:
        return
    style = widget.style()
    if style is None:
        return
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def set_dynamic_property(widget, name, value):
    if widget is None:
        return
    widget.setProperty(name, value)
    polish_widget_style(widget)


def apply_compact_button_heights(root, density=None):
    if root is None:
        return

    if density is None:
        screen = root.screen() if hasattr(root, "screen") else None
        density = resolve_ui_density(screen)

    button_height = max(18, density.compact_button_height)
    tool_height = max(16, density.compact_tool_height)

    for button in root.findChildren(QPushButton):
        if button.property("preserveButtonHeight"):
            continue
        current_height = button.minimumHeight()
        if current_height <= 0 or current_height > button_height:
            button.setMinimumHeight(button_height)

    for button in root.findChildren(QToolButton):
        if button.property("preserveButtonHeight"):
            continue
        target_height = button_height
        if (
            button.property("waveformTool")
            or button.property("waveformPreset")
            or button.property("inspectorMenu")
            or (button.toolButtonStyle() == Qt.ToolButtonIconOnly and not button.text())
        ):
            target_height = tool_height
        current_height = button.minimumHeight()
        if current_height <= 0 or current_height > target_height:
            button.setMinimumHeight(target_height)


def apply_compact_input_heights(root, density=None):
    if root is None:
        return

    if density is None:
        screen = root.screen() if hasattr(root, "screen") else None
        density = resolve_ui_density(screen)

    input_height = max(16, density.compact_input_height)
    seen = set()

    for widget_class in (QLineEdit, QComboBox, QAbstractSpinBox):
        for widget in root.findChildren(widget_class):
            widget_id = id(widget)
            if widget_id in seen or widget.property("preserveInputHeight"):
                continue
            seen.add(widget_id)
            widget.setFixedHeight(input_height)


def apply_subwindow_theme(window, extra_stylesheet=""):
    if window is None:
        return
    window.setAttribute(Qt.WA_StyledBackground, True)
    stylesheet = SUBWINDOW_STYLESHEET
    screen = window.screen() if hasattr(window, "screen") else None
    density = resolve_ui_density(screen)
    button_min_height = max(16, density.compact_button_height - 4)
    popup_stylesheet = build_popup_chrome_stylesheet(
        density,
        control_radius=density.compact_radius + 1,
        popup_radius=density.group_radius + 2,
        item_radius=density.compact_radius + 1,
        drop_width=max(20, density.compact_input_height),
        item_height=density.compact_input_height,
        font_size=density.button_font,
    )
    density_overrides = f"""
        QLabel {{ font-size: {density.base_font}px; }}
        QLabel[dialogTitle="true"] {{ font-size: {density.dock_title_font}px; }}
        QLabel[fieldLabel="true"] {{ font-size: {density.compact_label_font}px; }}
        QLabel[fieldUnit="true"] {{ font-size: {density.compact_unit_font}px; }}
        QLabel[pageEyebrow="true"] {{ font-size: {max(9, density.base_font + 1)}px; }}
        QLabel[pageTitle="true"] {{ font-size: {max(18, density.dock_title_font + 3)}px; }}
        QLabel[pageSubtitle="true"] {{ font-size: {max(11, density.base_font + 2)}px; }}
        QLabel[sectionTitle="true"] {{ font-size: {max(12, density.section_title_font + 2)}px; }}
        QLabel[sectionSubtitle="true"], QLabel[helperText="true"] {{ font-size: {density.base_font}px; }}
        QLabel[metricLabel="true"] {{ font-size: {density.base_font}px; }}
        QLabel[metricValue="true"], QLabel[statusValue="true"] {{ font-size: {max(14, density.status_value_font + 3)}px; }}
        QLabel[stepBadge="true"] {{
            min-width: {max(18, density.compact_input_height + 2)}px;
            min-height: {max(18, density.compact_input_height + 2)}px;
            border-radius: {max(9, (density.compact_input_height + 2) // 2)}px;
            font-size: {density.base_font}px;
        }}
        QLabel[stepTitle="true"] {{ font-size: {max(11, density.base_font + 1)}px; }}
        QLabel[stepDescription="true"] {{ font-size: {density.base_font}px; }}
        QLabel[toolbarLabel="true"] {{ font-size: {max(8, density.small_font)}px; }}
        QLabel[dataBadge="true"] {{
            border-radius: {density.group_radius + 1}px;
            padding: 3px 6px;
            font-size: {density.base_font}px;
        }}
        QFrame[surfaceCard="true"], QFrame[metricCard="true"], QFrame[panelCard="true"] {{
            border-radius: {density.group_radius + 3}px;
        }}
        QFrame[softCard="true"], QFrame[stepCard="true"] {{
            border-radius: {density.group_radius + 1}px;
        }}
        QGroupBox {{
            border-radius: {density.group_radius + 3}px;
            margin-top: 10px;
            padding: 10px 10px 8px;
            font-size: {density.base_font}px;
        }}
        QGroupBox::title {{ left: 10px; }}
        QPushButton, QToolButton, QDialogButtonBox QPushButton {{
            border-radius: {density.compact_radius + 1}px;
            padding: 3px 8px;
            min-height: {button_min_height}px;
            font-size: {density.button_font}px;
        }}
        QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit, QComboBox {{
            border-radius: {density.compact_radius + 1}px;
            padding: 3px 6px;
            font-size: {density.button_font}px;
        }}
        QTextEdit[consolePanel="true"], QPlainTextEdit[consolePanel="true"] {{
            border-radius: {density.group_radius + 1}px;
            font-size: {density.button_font}px;
        }}
        QCheckBox, QRadioButton {{ spacing: 6px; font-size: {density.base_font}px; }}
        QCheckBox::indicator, QRadioButton::indicator {{ width: 14px; height: 14px; }}
        QListWidget, QTreeWidget, QTableWidget {{
            border-radius: {density.group_radius + 2}px;
            font-size: {density.button_font}px;
        }}
        QHeaderView::section {{ padding: 3px 5px; font-size: {density.small_font}px; }}
        QTabBar::tab {{
            padding: 4px 8px;
            border-top-left-radius: {density.group_radius + 1}px;
            border-top-right-radius: {density.group_radius + 1}px;
            font-size: {density.button_font}px;
        }}
        QProgressBar {{
            border-radius: {density.group_radius + 1}px;
            font-size: {density.button_font}px;
        }}
    """
    stylesheet = f"{stylesheet}\n{density_overrides}\n{popup_stylesheet}"
    if extra_stylesheet:
        stylesheet = f"{stylesheet}\n{extra_stylesheet}"
    window.setStyleSheet(stylesheet)
    apply_compact_button_heights(window, density)
    apply_compact_input_heights(window, density)
    polish_widget_style(window)


def fit_window_to_screen(
    window,
    *,
    default_width,
    default_height,
    min_width,
    min_height,
    width_ratio=0.88,
    height_ratio=0.86,
):
    if window is None:
        return
    screen = window.screen() if hasattr(window, "screen") else None
    width, height = resolve_window_size(
        default_width=default_width,
        default_height=default_height,
        min_width=min_width,
        min_height=min_height,
        screen=screen,
        width_ratio=width_ratio,
        height_ratio=height_ratio,
    )
    window.resize(width, height)
    window.setMinimumSize(min_width, min_height)


def set_accent_button(button, accent=True):
    if button is None:
        return
    button.setProperty("accentButton", bool(accent))
    polish_widget_style(button)


def style_value_badge(label, min_width=None, alignment=None, selectable=False):
    if label is None:
        return
    label.setProperty("dataBadge", True)
    if min_width is not None:
        label.setMinimumWidth(min_width)
    if alignment is None:
        alignment = Qt.AlignVCenter | Qt.AlignLeft
    label.setAlignment(alignment)
    if selectable:
        label.setTextInteractionFlags(label.textInteractionFlags() | Qt.TextSelectableByMouse)
    polish_widget_style(label)


def build_themed_message_box(
    parent=None,
    *,
    icon=QMessageBox.Information,
    title="",
    text="",
    informative_text="",
    detailed_text="",
    buttons=QMessageBox.Ok,
    default_button=QMessageBox.NoButton,
):
    message_box = QMessageBox(parent)
    message_box.setIcon(icon)
    message_box.setWindowTitle(title)
    message_box.setText(text)
    if informative_text:
        message_box.setInformativeText(informative_text)
    if detailed_text:
        message_box.setDetailedText(detailed_text)
    message_box.setStandardButtons(buttons)
    if default_button != QMessageBox.NoButton:
        message_box.setDefaultButton(default_button)
    apply_subwindow_theme(message_box)
    return message_box
