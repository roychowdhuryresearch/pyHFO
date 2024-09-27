from src.models.main_waveform_plot_model import MainWaveformPlotModel
from src.views.main_waveform_plot_view import MainWaveformPlotView
import pyqtgraph as pg
import numpy as np

class MainWaveformPlotController:
    def __init__(self, main_waveform_plot_widget, backend):
        self.model = MainWaveformPlotModel(backend)
        self.view = MainWaveformPlotView(main_waveform_plot_widget)
    
    def init_eeg_data(self):
        self.model.init_eeg_data()

    def clear(self):
        self.view.clear()

    def init_waveform_display(self):
        self.view.enable_axis_information()

    def plot_one_channel(self, x, y, color, width=5):
        self.view.plot_waveform(x, y, color, width)

    def update_backend(self, new_backend):
        self.model.update_backend(new_backend)

    def set_time_window(self, time_window:int):
        self.model.set_time_window(time_window)

    def set_n_channels_to_plot(self, n_channels_to_plot:int):
        self.model.set_n_channels_to_plot(n_channels_to_plot)

    def set_channel_indices_to_plot(self, channel_indices_to_plot:list):
        self.model.set_channel_indices_to_plot(channel_indices_to_plot)

    def set_channels_to_plot(self, channels_to_plot:list):
        self.model.set_channels_to_plot(channels_to_plot)

    def set_normalize_vertical(self, normalize_vertical:bool):
        self.model.set_normalize_vertical(normalize_vertical)
    
    def update_channel_names(self, new_channel_names):
        self.model.update_channel_names(new_channel_names)

    def set_first_channel_to_plot(self, first_channel_to_plot:int):
        self.model.set_first_channel_to_plot(first_channel_to_plot)

    def set_waveform_filter(self, filtered:bool):
        self.model.set_waveform_filter(filtered)
    
    def get_first_channel_to_plot(self):
        return self.model.first_channel_to_plot

    def get_current_eeg_data_to_display(self):
        return self.model.get_all_current_eeg_data_to_display()
    
    def set_current_time_window(self, start_in_time):
        self.model.set_current_time_window(start_in_time)

    def get_current_start_end(self):
        return self.model.get_current_start_end()

    def get_current_time_window(self):
        return self.model.get_current_time_window()
    
    def set_plot_HFOs(self, plot_HFOs:bool):
        self.model.set_plot_HFOs(plot_HFOs)

    def plot_all_current_channels_for_window(self):
        eeg_data_to_display, y_100_length, y_scale_length, offset_value = self.get_current_eeg_data_to_display()
        time_to_display = self.get_current_time_window()
        first_channel_to_plot = self.get_first_channel_to_plot()
        n_channels_to_plot = self.model.n_channels_to_plot
        waveform_color = self.model.get_waveform_color()

        for disp_i, ch_i in enumerate(range(first_channel_to_plot, first_channel_to_plot + n_channels_to_plot)):
            self.view.plot_waveform(time_to_display, eeg_data_to_display[ch_i] - disp_i*offset_value, waveform_color, 0.5)

        return eeg_data_to_display, y_100_length, y_scale_length, offset_value
        
    def plot_all_current_hfos_for_window(self, eeg_data_to_display, offset_value, top_value):
        first_channel_to_plot = self.get_first_channel_to_plot()
        n_channels_to_plot = self.model.n_channels_to_plot
        channels_to_plot = self.model.channels_to_plot
        start_in_time, end_in_time = self.get_current_start_end()

        for disp_i, ch_i in enumerate(range(first_channel_to_plot, first_channel_to_plot+n_channels_to_plot)):
            channel = channels_to_plot[ch_i]
            hfo_starts, hfo_ends, hfo_starts_in_time, hfo_ends_in_time, windows_in_time, colors = self.model.get_all_hfos_for_all_current_channels_and_color(channel)

            if self.model.plot_HFOs:
                for i in range(len(hfo_starts)):
                    event_start = int(hfo_starts[i]-start_in_time*self.model.sample_freq)
                    event_end = int(hfo_ends[i]-start_in_time*self.model.sample_freq)
                    self.view.plot_waveform(windows_in_time[i], eeg_data_to_display[ch_i, event_start:event_end]-disp_i*offset_value, colors[i], 2)
                    self.view.plot_waveform([hfo_starts_in_time[i], hfo_ends_in_time[i]], [top_value+0.2,top_value+0.2], colors[i], 10)


    def draw_scale_bar(self, eeg_data_to_display, offset_value, y_100_length, y_scale_length):
        start_in_time, end_in_time = self.get_current_start_end()
        n_channels_to_plot = self.model.n_channels_to_plot
        # Determine the position for the scale indicator (bottom right corner of the plot)
        x_pos = end_in_time #+ 0.15
        y_pos = np.min(eeg_data_to_display[-1]) - n_channels_to_plot * offset_value + 0.8 * offset_value

        # Use a dashed line for the scale
        self.view.draw_scale_bar(x_pos, y_pos, y_100_length, y_scale_length)

    def draw_channel_names(self, offset_value):
        n_channels_to_plot = self.model.n_channels_to_plot
        channels_to_plot = self.model.channels_to_plot
        first_channel_to_plot = self.get_first_channel_to_plot()
        start_in_time, end_in_time = self.get_current_start_end()
        self.view.draw_channel_names(offset_value, n_channels_to_plot, channels_to_plot, first_channel_to_plot, start_in_time, end_in_time)