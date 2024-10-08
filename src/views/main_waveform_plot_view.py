from PyQt5 import QtWidgets
import pyqtgraph as pg
from PyQt5 import QtGui
import numpy as np


class MainWaveformPlotView:
    def __init__(self, plot_widget: pg.PlotWidget):
        self.plot_widget = plot_widget
        self._init_plot_widget(plot_widget)

    def _init_plot_widget(self, plot_widget: pg.PlotWidget):
        plot_widget.setMouseEnabled(x=False, y=False)
        plot_widget.getPlotItem().hideAxis('bottom')
        plot_widget.getPlotItem().hideAxis('left')
        plot_widget.setBackground('w')

    def clear(self):
        self.plot_widget.clear()

    def enable_axis_information(self):
        self.plot_widget.getPlotItem().showAxis('bottom')
        self.plot_widget.getPlotItem().showAxis('left')

    def plot_waveform(self, x, y, color, width):
        self.plot_widget.plot(x, y, pen=pg.mkPen(color=color, width=width))

    def draw_scale_bar(self, x_pos, y_pos, y_100_length, y_scale_length):
        scale_line = pg.PlotDataItem([x_pos, x_pos], [y_pos, y_pos + y_scale_length],
                             pen=pg.mkPen('black', width=10), fill=(0, 128, 255, 150)) 
        self.plot_widget.addItem(scale_line)

        text_item = pg.TextItem(f'Scale: {y_100_length} μV ', color='black', anchor=(1, 0.5))
        text_item.setFont(QtGui.QFont('Arial', 10, QtGui.QFont.Bold))
        text_item.setPos(x_pos, y_pos + y_scale_length / 2)
        self.plot_widget.addItem(text_item)

    def draw_channel_names(self, offset_value, n_channels_to_plot, channels_to_plot, first_channel_to_plot, start_in_time, end_in_time):
        #set y ticks to channel names
        channel_names_locs = -offset_value * np.arange(n_channels_to_plot)[:, None]  # + offset_value/2

        self.plot_widget.getAxis('left').setTicks([[(channel_names_locs[disp_i], channels_to_plot[chi_i])
                 for disp_i, chi_i in enumerate(range(first_channel_to_plot, first_channel_to_plot + n_channels_to_plot))]])
        #set the max and min of the x axis
        self.plot_widget.setXRange(start_in_time, end_in_time)
