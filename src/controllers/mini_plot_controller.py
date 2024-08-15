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

    def get_first_channel_to_plot(self):
        return self.model.first_channel_to_plot
    
    def set_channel_indices_to_plot(self, channel_indices_to_plot):
        self.model.set_channel_indices_to_plot(channel_indices_to_plot)

    def set_channels_to_plot(self, channels_to_plot):
        self.model.set_channels_to_plot(channels_to_plot)

    def update_channel_names(self, new_channel_names):
        self.model.update_channel_names(new_channel_names)

    def plot_one_hfo(self, start_time, end_time, height, color, width=5):
        self.view.plot_hfo(start_time, end_time, height, color, width)

    def plot_all_current_hfos_for_one_channel(self, channel, plot_height):
        starts_in_time, ends_in_time, colors = self.model.get_all_hfos_for_channel_and_color(channel)
        
        for i in range(len(starts_in_time)):
            self.plot_one_hfo(starts_in_time[i], ends_in_time[i], plot_height, colors[i], 5)

    def plot_all_current_hfos_for_all_channels(self, plot_height):
        first_channel_to_plot = self.get_first_channel_to_plot()
        n_channels_to_plot = self.model.n_channels_to_plot
        channels_to_plot = self.model.channels_to_plot
        for disp_i, ch_i in enumerate(range(first_channel_to_plot, first_channel_to_plot+n_channels_to_plot)):
            channel = channels_to_plot[ch_i]
            self.plot_all_current_hfos_for_one_channel(channel, plot_height)

    def set_miniplot_title(self, title):
        self.view.set_miniplot_title(title)

    def set_total_x_y_range(self, top_value):
        time_max = int(self.model.time.shape[0] / self.model.sample_freq)
        self.view.set_x_y_range([0, time_max], [top_value-0.25, top_value+0.25])

    def update_highlight_window(self, start, end, height):
        self.view.update_highlight_window(start, end, height)

    def update_backend(self, new_backend):
        self.model.update_backend(new_backend)

    