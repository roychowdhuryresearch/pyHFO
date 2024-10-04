import os
import re
import sys
import traceback
from queue import Queue
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


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
    result = pyqtSignal(list)
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
            text = self.queue.get()
            self.std_received_signal.emit(text)

    def stop(self):
        self._isRunning = False


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
            text = self.queue.get()
            self.std_received_signal.emit(text)

    def stop(self):
        self._isRunning = False


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

