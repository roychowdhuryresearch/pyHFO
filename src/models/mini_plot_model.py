import numpy as np
from src.hfo_app import HFO_App
import sys

class MiniPlotModel:
    def __init__(self, backend: HFO_App):
        self.backend = backend
        self.color_dict={"artifact":(245,130,48), #orange
                         "spike":(240,30,250), #pink
                         "non_spike":(60,180,75), #green
                         "HFO":(60,180,75), #green
                         }
        self.first_channel_to_plot = 0
        self.n_channels_to_plot = 10

    def update_backend(self, new_backend):
        self.backend = new_backend

    def init_eeg_data(self):
        eeg_data, self.channel_names = self.backend.get_eeg_data()
        self.edf_info = self.backend.get_edf_info()
        self.sample_freq = self.edf_info['sfreq']
        self.time = np.arange(0, eeg_data.shape[1]/self.sample_freq, 1/self.sample_freq)

        self.channel_names = list(self.channel_names)
        self.n_channels = len(self.channel_names)
        self.n_channels_to_plot = min(self.n_channels,self.n_channels_to_plot)
        self.channels_to_plot = self.channel_names.copy()
        self.channel_indices_to_plot = np.arange(self.n_channels)

    def set_first_channel_to_plot(self, first_channel_to_plot):
        self.first_channel_to_plot = first_channel_to_plot

    def set_n_channels_to_plot(self, n_channels_to_plot:int):
        self.n_channels_to_plot = n_channels_to_plot

    def set_channels_to_plot(self, channels_to_plot:list):
        self.channels_to_plot = channels_to_plot
        self.channel_indices_to_plot = [self.channel_names.index(channel) for channel in channels_to_plot]

    def set_channel_indices_to_plot(self,channel_indices_to_plot:list):
        self.channel_indices_to_plot = channel_indices_to_plot
        self.channels_to_plot = [self.channel_names[index] for index in channel_indices_to_plot]

    def update_channel_names(self, new_channel_names):
        self.channel_names = new_channel_names

    def get_all_hfos_for_channel(self, channel, t_start=0, t_end=sys.maxsize):
        return self.backend.event_features.get_HFOs_for_channel(channel, t_start, t_end)

    def get_all_hfos_for_channel_and_color(self, channel, t_start=0, t_end=sys.maxsize):
        starts, ends, artifacts, spikes = self.get_all_hfos_for_channel(channel, t_start, t_end)
        colors = []
        for j in range(len(starts)):
            try:
                if int(artifacts[j])<1:
                    color = self.color_dict["artifact"]
                elif spikes[j]:
                    color = self.color_dict["spike"]
                else:
                    color = self.color_dict["non_spike"]
            except:
                color = self.color_dict["non_spike"]
            colors.append(color)

        starts_in_time = [self.time[int(i)] for i in starts]
        ends_in_time = [self.time[min(int(i), len(self.time)-1)] for i in ends]
        return starts_in_time, ends_in_time, colors