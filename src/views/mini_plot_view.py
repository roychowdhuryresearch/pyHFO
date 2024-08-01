from PyQt5 import QtWidgets
import pyqtgraph as pg

class MiniPlotView(pg.PlotWidget):
    def __init__(self):
        super().__init__()
        self._init_mini_plot_display()

    def _init_mini_plot_display(self):
        self.setMouseEnabled(x=False, y=False)
        self.getPlotItem().hideAxis('bottom')
        self.getPlotItem().hideAxis('left')
        self.setBackground('w')
        self.clear()

    def enable_axis_information(self):
        self.getPlotItem().showAxis('bottom')
        self.getPlotItem().showAxis('left')

    def update_highlight_window(self, start, end):
        if hasattr(self, 'linear_region'):
            self.removeItem(self.linear_region)
        self.linear_region = pg.LinearRegionItem([start, end], movable=False)
        self.linear_region.setZValue(-20)
        self.addItem(self.linear_region)

    def update_plot(self, x, y):
        self.plot(x, y, clear=True)

    def set_miniplot_title(self, title):
        top_value = max(self.yData)
        self.getAxis('left').setTicks([[(top_value, f'   {title}   ')]])

    def set_x_y_range(self, x_range, y_range):
        self.setXRange(x_range[0], x_range[1])
        self.setYRange(y_range[0], y_range[1])
