from PyQt5 import uic
from pathlib import Path
from src.utils.utils_gui import *
from src.ui.plot_waveform import *
from PyQt5.QtCore import QSize
from src.utils.utils_plotting import *
from src.ui.annotation_plot import AnnotationPlot, FFTPlot
from src.controllers import AnnotationController

ROOT_DIR = Path(__file__).parent


class Annotation(QtWidgets.QMainWindow):
    def __init__(self, backend=None, main_window=None, close_signal=None):
        super(Annotation, self).__init__(main_window)
        self.annotation_controller = AnnotationController(self, backend)

        print("initializing Annotation")
        self.backend = backend
        self.ui = uic.loadUi(os.path.join(ROOT_DIR, 'annotation.ui'), self)
        self.setWindowTitle("Annotator")
        self.setWindowIcon(QtGui.QIcon(os.path.join(ROOT_DIR, 'src/ui/images/icon.png')))
        self.threadpool = QThreadPool()
        self.close_signal = close_signal
        self.close_signal.connect(self.close)
        self.PreviousButton.clicked.connect(self.plot_prev)
        self.NextButton.clicked.connect(self.plot_next)
        self.Accept.clicked.connect(self.update_button_clicked)
        
        self.IntervalDropdownBox.currentIndexChanged.connect(self.update_interval)  # Connect the interval dropdown box

        # create the main waveform plot
        self.init_waveform_plot()

        # create fft plot
        self.init_fft_plot()

        channel, start, end = self.backend.hfo_features.get_current()
        self.annotation_controller.model.waveform_plot.plot(start, end, channel, interval=1.0)  # Default interval
        self.annotation_controller.model.fft_plot.plot(start, end, channel, interval=1.0)  # Default interval
        self.init_annotation_dropdown()
        self.update_infos()
        self.setInitialSize()
        
        self.setWindowModality(QtCore.Qt.ApplicationModal)  # Set as modal dialog

    def init_waveform_plot(self):
        self.annotation_controller.create_waveform_plot()

    def init_fft_plot(self):
        self.annotation_controller.create_fft_plot()

    def setInitialSize(self):
        # Getting screen resolution of your monitor
        screen = QApplication.primaryScreen()
        rect = screen.availableGeometry()

        # Calculating the window size as a fraction of the screen size
        width = rect.width() * 0.6  # 60% of the screen width
        height = rect.height() * 0.6  # 60% of the screen height
        width = int(width)
        height = int(height)
        # Setting the initial size and fixing it
        self.setGeometry(100, 100, width, height)
        self.setFixedSize(QSize(width, height))
    
    def get_current_interval(self):
        interval_text = self.IntervalDropdownBox.currentText()
        try:
            return float(interval_text.rstrip('s'))
        except (ValueError, AttributeError):
            return 1.0  # Default interval
        
    def plot_prev(self):
        # start, end: index of the prev hfo
        channel, start, end = self.backend.hfo_features.get_prev()
        interval = self.get_current_interval()
        # interval = float(self.IntervalDropdownBox.currentText().rstrip('s'))  # Get the current interval
        self.annotation_controller.model.waveform_plot.plot(start, end, channel, interval=interval)
        self.annotation_controller.model.plot(start, end, channel, interval=interval)
        self.update_infos()

    def plot_next(self):
        # start, end: index of the next hfo
        channel, start, end = self.backend.hfo_features.get_next()
        interval = self.get_current_interval()
        # interval = float(self.IntervalDropdownBox.currentText().rstrip('s'))  # Get the current interval
        self.annotation_controller.model.waveform_plot.plot(start, end, channel, interval=interval)
        self.annotation_controller.model.fft_plot.plot(start, end, channel, interval=interval)
        self.update_infos()

    def plot_jump(self):
        selected_index = self.AnotationDropdownBox.currentIndex()
        # start, end: index of the next hfo
        channel, start, end = self.backend.hfo_features.get_jump(selected_index)
        try:
            interval = float(self.IntervalDropdownBox.currentText().rstrip('s'))
        except (ValueError, AttributeError):
            interval = 1.0  # Default interval
        # interval = float(self.IntervalDropdownBox.currentText().rstrip('s'))  # Get the current interval
        self.annotation_controller.model.waveform_plot.plot(start, end, channel, interval=interval)
        self.annotation_controller.model.fft_plot.plot(start, end, channel, interval=interval)
        self.update_infos()

    def update_infos(self):
        info = self.backend.hfo_features.get_current_info()
        fs = self.backend.sample_freq
        self.channel_name_textbox.setText(info["channel_name"])
        self.start_textbox.setText(str(round(info["start_index"] / fs, 3)) + " s")
        self.end_textbox.setText(str(round(info["end_index"] / fs, 3)) + " s")
        self.length_textbox.setText(str(round((info["end_index"] - info["start_index"]) / fs, 3)) + " s")
        self.AnotationDropdownBox.setCurrentIndex(self.backend.hfo_features.index)
        print(info["prediction"])
        if info["prediction"] is not None:
            self.model_textbox.setText(info["prediction"])
            self.EventDropdown_Box.setCurrentText(info["prediction"])
        else:
            self.model_textbox.setText("Unpredicted")
            self.EventDropdown_Box.setCurrentText("--- Event Type ---")

    def update_button_clicked(self):
        # print("updating now...")
        selected_text = self.EventDropdown_Box.currentText()
        if selected_text in ["Artifact", "Spike", "Real"]:
            self.backend.hfo_features.doctor_annotation(selected_text)
            # Update the text of the selected item in the dropdown menu
            selected_index = self.backend.hfo_features.index
            item_text = self.backend.hfo_features.get_annotation_text(selected_index)
            self.AnotationDropdownBox.setItemText(selected_index, item_text)
            self.plot_next()

    def init_annotation_dropdown(self):
        # initialize the text in the dropdown menu
        for i in range(len(self.backend.hfo_features.annotated)):
            text = self.backend.hfo_features.get_annotation_text(i)
            self.AnotationDropdownBox.addItem(text)
        self.AnotationDropdownBox.activated.connect(self.plot_jump)
    
    def update_interval(self):
        interval = self.get_current_interval()
        
        # Update the plots to reflect the new interval
        channel, start, end = self.backend.hfo_features.get_current()
        self.annotation_controller.model.waveform_plot.plot(start, end, channel, interval=interval)
        self.annotation_controller.model.fft_plot.plot(start, end, channel, interval=interval)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = Annotation()
    mainWindow.show()
    sys.exit(app.exec_())