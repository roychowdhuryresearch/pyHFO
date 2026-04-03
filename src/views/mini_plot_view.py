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
        plot_widget.getPlotItem().hideButtons()
        plot_widget.getPlotItem().setMenuEnabled(False)
        plot_widget.getPlotItem().vb.setDefaultPadding(0.0)
        plot_widget.setMaximumHeight(78)
        plot_widget.setMinimumHeight(58)
        self.linear_region = None
        self.timeline_baseline = None
        
    def enable_axis_information(self):
        self.plot_widget.getPlotItem().showAxis('bottom')
        self.plot_widget.getPlotItem().showAxis('left')
        self.plot_widget.setTitle("")
        bottom_axis = self.plot_widget.getAxis('bottom')
        bottom_axis.setLabel("")
        bottom_axis.setStyle(tickTextOffset=4, tickLength=4)
        bottom_axis.setHeight(26)
        bottom_axis.setPen(pg.mkPen("#bcc7d1", width=1))
        bottom_axis.setTextPen(pg.mkColor("#7b8894"))
        left_axis = self.plot_widget.getAxis('left')
        left_axis.setStyle(showValues=False)
        left_axis.setTicks([])
        left_axis.setPen(pg.mkPen("#ffffff", width=0))

    def add_linear_region(self):
        self.linear_region = pg.LinearRegionItem(
            [0, 0],
            movable=False,
            brush=pg.mkBrush(92, 121, 164, 45),
            pen=pg.mkPen("#5a7aa3", width=1),
            hoverBrush=pg.mkBrush(92, 121, 164, 55),
            hoverPen=pg.mkPen("#5a7aa3", width=1.2),
        )
        self.linear_region.setZValue(10)
        self.plot_widget.addItem(self.linear_region)

    def update_highlight_window(self, start, end, height):
        self.linear_region.setRegion([start,end])
        self.linear_region.setZValue(10)

    def plot_biomarker(self, start_time, end_time, top_value, color, width):
        self.plot_widget.plot([start_time, end_time], [top_value, top_value], pen=pg.mkPen(color=color, width=width))

    def set_miniplot_title(self, title, height):
        return

    def set_x_y_range(self, x_range, y_range):
        self.plot_widget.setXRange(x_range[0], x_range[1])
        self.plot_widget.setYRange(y_range[0], y_range[1])
        baseline_y = (y_range[0] + y_range[1]) / 2.0
        if self.timeline_baseline is None:
            self.timeline_baseline = pg.PlotDataItem(
                [x_range[0], x_range[1]],
                [baseline_y, baseline_y],
                pen=pg.mkPen("#d8dfe6", width=2),
            )
            self.timeline_baseline.setZValue(-30)
            self.plot_widget.addItem(self.timeline_baseline)
        else:
            self.timeline_baseline.setData([x_range[0], x_range[1]], [baseline_y, baseline_y])

    def clear(self):
        self.plot_widget.clear()
        self.linear_region = None
        self.timeline_baseline = None

    def sync_left_axis_width(self, axis_width):
        left_axis = self.plot_widget.getAxis('left')
        left_axis.show()
        left_axis.setStyle(showValues=False)
        left_axis.setTicks([])
        left_axis.setWidth(max(int(axis_width), 0))
