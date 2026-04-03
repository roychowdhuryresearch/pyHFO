import os
import re
import sys
import queue
import traceback
from queue import Queue
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

COLOR_MAP = {'waveform': '#58748a',
             'HFO': '#6d8d74',
             'non_spike': '#6d8d74',
             'Real': '#6d8d74',
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
    QGroupBox {
        border: 1px solid #dde3ea;
        border-radius: 12px;
        margin-top: 12px;
        padding: 14px 12px 12px;
        background: #fbfcfd;
        font-weight: 700;
        color: #314657;
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
        padding: 6px 12px;
        min-height: 18px;
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
    QComboBox::drop-down {
        border: none;
        width: 22px;
    }
    QCheckBox, QRadioButton {
        color: #33495b;
        spacing: 8px;
    }
    QCheckBox::indicator, QRadioButton::indicator {
        width: 16px;
        height: 16px;
    }
    QCheckBox::indicator {
        border: 1px solid #b7c3ce;
        border-radius: 4px;
        background: #ffffff;
    }
    QCheckBox::indicator:checked {
        border-color: #304657;
        background: #304657;
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
    QMenuBar::item:selected, QMenu::item:selected {
        background: #eef3f7;
    }
    QMenu {
        background: #ffffff;
        border: 1px solid #d8dde3;
        color: #243746;
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
                widget.deleteLater()  # Safely delete the widget
            else:
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


def apply_subwindow_theme(window, extra_stylesheet=""):
    if window is None:
        return
    window.setAttribute(Qt.WA_StyledBackground, True)
    stylesheet = SUBWINDOW_STYLESHEET
    if extra_stylesheet:
        stylesheet = f"{stylesheet}\n{extra_stylesheet}"
    window.setStyleSheet(stylesheet)
    polish_widget_style(window)


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
