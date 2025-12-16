from PyQt5 import QtWidgets
import pyqtgraph as pg
from PyQt5 import QtGui
import numpy as np


class MainWaveformPlotView(QtWidgets.QGraphicsView):
    def __init__(self, plot_widget: pg.PlotWidget):
        super(MainWaveformPlotView, self).__init__()
        self.plot_widget = plot_widget
        self._init_plot_widget(plot_widget)

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

    def clear(self):
        # Block updates during clear for better performance
        self.plot_widget.getPlotItem().vb.setMouseEnabled(x=False, y=False)
        self.plot_widget.clear()

    def enable_axis_information(self):
        self.plot_widget.getPlotItem().showAxis('bottom')
        self.plot_widget.getPlotItem().showAxis('left')
    
    def begin_batch_update(self):
        """Begin batching updates for better performance"""
        self.plot_widget.getPlotItem().vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    
    def end_batch_update(self):
        """End batching updates and refresh the view"""
        self.plot_widget.getPlotItem().vb.update()

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
        
        # Set Y range to show all channels properly
        y_max = offset_value  # top margin
        y_min = -(n_channels_to_plot - 1) * offset_value - offset_value  # bottom margin
        
        # Batch range updates for better performance
        self.plot_widget.setRange(xRange=(start_in_time, end_in_time), 
                                  yRange=(y_min, y_max), 
                                  padding=0, 
                                  update=True)
