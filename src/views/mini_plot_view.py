from PyQt5 import QtWidgets
import pyqtgraph as pg


class MiniPlotView(QtWidgets.QGraphicsView):
    def __init__(self, plot_widget: pg.PlotWidget):
        super(MiniPlotView, self).__init__()
        self.plot_widget = plot_widget
        self._init_plot_widget(plot_widget)

    def _init_plot_widget(self, plot_widget):
        plot_widget.setMouseEnabled(x=False, y=False)
        plot_widget.getPlotItem().hideAxis('bottom')
        plot_widget.getPlotItem().hideAxis('left')
        plot_widget.setBackground('w')
        
    def enable_axis_information(self):
        self.plot_widget.getPlotItem().showAxis('bottom')
        self.plot_widget.getPlotItem().showAxis('left')

    def add_linear_region(self):
        self.linear_region = pg.LinearRegionItem([0, 0], movable=False)
        self.linear_region.setZValue(-20)
        self.plot_widget.addItem(self.linear_region)

    def update_highlight_window(self, start, end, height):
        self.linear_region.setRegion([start,end])
        self.linear_region.setZValue(height)

    def plot_biomarker(self, start_time, end_time, top_value, color, width):
        self.plot_widget.plot([start_time, end_time], [top_value, top_value], pen=pg.mkPen(color=color, width=width))

    def set_miniplot_title(self, title, height):
        self.plot_widget.getAxis('left').setTicks([[(height, f'   {title}   ')]])

    def set_x_y_range(self, x_range, y_range):
        self.plot_widget.setXRange(x_range[0], x_range[1])
        self.plot_widget.setYRange(y_range[0], y_range[1])

    def clear(self):
        self.plot_widget.clear()
