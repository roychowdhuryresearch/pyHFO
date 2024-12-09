import numpy as np
from src.hfo_app import HFO_App
from src.spindle_app import SpindleApp
import sys


class MainWaveformPlotModel:
    def __init__(self, backend: HFO_App):
        self.backend = backend
        self.color_dict={"artifact":(245,130,48), #orange
                         "spike":(240,30,250), #pink
                         "non_spike":(60,180,75), #green
                         "HFO":(60,180,75), #green
                         "waveform": (0,0,255), 
                         }
        
        self.start_in_time = 0
        self.end_in_time = 20
        self.time_window = 20 #20 second time window
        self.first_channel_to_plot = 0
        self.n_channels_to_plot = 10
        self.filtered = False
        self.normalize_vertical = False
    
    def update_backend(self, new_backend):
        self.backend = new_backend

    def init_eeg_data(self):
        eeg_data, self.channel_names = self.backend.get_eeg_data()
        self.edf_info = self.backend.get_edf_info()
        self.sample_freq = self.edf_info['sfreq']
        self.time = np.arange(0, eeg_data.shape[1]/self.sample_freq, 1/self.sample_freq)

        self.filtered = False
        self.plot_biomarkers = False
        self.channel_names = list(self.channel_names)
        self.n_channels = len(self.channel_names)
        self.n_channels_to_plot = min(self.n_channels,self.n_channels_to_plot)
        self.channels_to_plot = self.channel_names.copy()
        self.channel_indices_to_plot = np.arange(self.n_channels)

    def set_time_window(self, time_window:int):
        self.time_window = time_window

    def set_plot_biomarkers(self, plot_biomarkers:bool):
        self.plot_biomarkers = plot_biomarkers

    def set_current_time_window(self, start_in_time):
        self.start_in_time = max(start_in_time, 0)
        self.end_in_time = min(start_in_time + self.time_window, self.time[-1])
        self.start_in_time, self.end_in_time = self.time[int(start_in_time * self.sample_freq)], self.time[int(self.end_in_time * self.sample_freq)]

    def get_current_start_end(self):
        return self.start_in_time, self.end_in_time
    
    def get_current_time_window(self):
        return self.time[int(self.start_in_time * self.sample_freq): int(self.end_in_time * self.sample_freq)]

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
        self.channel_names = list(new_channel_names)

    def set_waveform_filter(self, filtered):
        self.filtered = filtered
    
    def set_normalize_vertical(self, normalize_vertical:bool):
        self.normalize_vertical = normalize_vertical
        
    def get_waveform_color(self):
        return self.color_dict["waveform"]

    def get_all_current_eeg_data_to_display(self):
        eeg_data_to_display, _ = self.backend.get_eeg_data(int(self.start_in_time*self.sample_freq),int(self.end_in_time*self.sample_freq), self.filtered)
        eeg_data_to_display = eeg_data_to_display[self.channel_indices_to_plot,:]

        if self.normalize_vertical:
            eeg_data_to_display = (eeg_data_to_display-eeg_data_to_display.min(axis = 1,keepdims = True))
            eeg_data_to_display = eeg_data_to_display/np.max(eeg_data_to_display)
        else:
            if self.filtered:
                means = np.mean(eeg_data_to_display)
                self.stds = np.std(eeg_data_to_display) * 2
                eeg_data_to_display = (eeg_data_to_display - means) / self.stds
                eeg_data_to_display[np.isnan(eeg_data_to_display)] = 0
            else:
                # standardized signal globally
                means = np.mean(eeg_data_to_display)
                self.stds = np.std(eeg_data_to_display)
                eeg_data_to_display = (eeg_data_to_display - means) / self.stds
                #replace nans with 0
                eeg_data_to_display[np.isnan(eeg_data_to_display)] = 0
        #shift the ith channel by 1.1*i
        # eeg_data_to_display = eeg_data_to_display-1.1*np.arange(eeg_data_to_display.shape[0])[:,None]
        if self.filtered:
            # Add scale indicators
            # Set the length of the scale lines
            y_100_length = 50  # 100 microvolts
            offset_value = 6
            y_scale_length = y_100_length / self.stds
        else:
            y_100_length = 100  # 100 microvolts
            offset_value = 6
            y_scale_length = y_100_length / self.stds

        return eeg_data_to_display, y_100_length, y_scale_length, offset_value

    def get_all_biomarkers_for_all_current_channels_and_color(self, channel_in_name):
        starts, ends, artifacts, spikes = self.backend.event_features.get_biomarkers_for_channel(channel_in_name, int(self.start_in_time*self.sample_freq),int(self.end_in_time*self.sample_freq))
        colors = []
        windows_in_time = []

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
            # s_ind, e_ind = np.searchsorted(self.time, starts[j]), np.searchsorted(self.time, ends[j])
            windows_in_time.append(self.time[int(starts[j]):int(ends[j])])

        starts_in_time = [self.time[int(i)] for i in starts]
        ends_in_time = [self.time[min(int(i), len(self.time)-1)] for i in ends]
        
        return starts, ends, starts_in_time, ends_in_time, windows_in_time, colors

