import sys
from src.models.mini_plot_model import MiniPlotModel
from src.views.mini_plot_view import MiniPlotView

class MiniPlotController:
    def __init__(self, mini_plot_widget, backend):
        self.model = MiniPlotModel(backend)
        self.view = MiniPlotView(mini_plot_widget)

    def clear(self):
        self.view.clear()

    def init_hfo_display(self):
        self.view.enable_axis_information()
        self.view.add_linear_region()

    def init_eeg_data(self):
        self.model.init_eeg_data()

    def plot_one_hfo(self, start_time, end_time, height, color, width=5):
        self.view.plot_hfo(start_time, end_time, height, color, width)

    def plot_all_current_hfos_for_channel(self, channel, plot_height):
        starts_in_time, ends_in_time, colors = self.model.get_all_hfos_for_channel_and_color(channel)
        
        for i in range(len(starts_in_time)):
            self.plot_one_hfo(starts_in_time[i], ends_in_time[i], plot_height, colors[i], 5)

    def set_miniplot_title(self, title):
        self.view.set_miniplot_title(title)

    def set_x_y_range(self, x_range, y_range):
        self.view.set_x_y_range(x_range, y_range)

    def update_highlight_window(self, start, end, height):
        self.view.update_highlight_window(start, end, height)

    def update_backend(self, new_backend):
        self.model.update_backend(new_backend)

    