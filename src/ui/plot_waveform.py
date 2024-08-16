import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg # We will try using pyqtgraph for plotting
import numpy as np
import time
import mne
# from superqt import QDoubleRangeSlider
from tqdm import tqdm
import os
from src.hfo_app import HFO_App
import random
from src.controllers import MiniPlotController, MainWaveformPlotController

curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(curr_dir))

class CenterWaveformAndMiniPlotController():
    def __init__(self, main_waveform_plot_widget: pg.PlotWidget, mini_plot_widget: pg.PlotWidget, backend: HFO_App):
        self.mini_plot_controller = MiniPlotController(mini_plot_widget, backend)
        self.main_waveform_plot_controller = MainWaveformPlotController(main_waveform_plot_widget, backend)

        self.time_window = 20 #20 second time window
        self.time_increment = 20
        # self.old_size = (self.waveform_display.x(),self.waveform_display.y(),self.waveform_display.width(),self.waveform_display.height())
        self.t_start = 0
        self.first_channel_to_plot = 0
        self.n_channels_to_plot = 10
        self.backend = backend
        self.filtered = False
        self.time_window_increment = 100 #in percent
        self.waveform_color = (0,0,255) #dark blue
        self.artifact_color=(245,130,48) #orange
        self.spike_color=(240,30,250) #pink
        self.non_spike_color=(60,180,75) #green
        self.HFO_color=self.non_spike_color
        self.color_dict={"artifact":self.artifact_color,"spike":self.spike_color,
                         "non_spike":self.non_spike_color,"HFO":self.HFO_color}
        self.plot_HFOs = False
        self.normalize_vertical = False
        self.stds = None
    
    def set_filtered(self,filtered:bool):
        self.main_waveform_plot_controller.set_waveform_filter(filtered)
        self.plot(self.t_start)

    def update_backend(self,new_backend:HFO_App,init_eeg_data:bool=True):
        self.backend = new_backend
        self.mini_plot_controller.update_backend(new_backend)
        self.main_waveform_plot_controller.update_backend(new_backend)
        if init_eeg_data:
            self.init_eeg_data()

    def init_eeg_data(self):
        self.mini_plot_controller.clear()
        self.main_waveform_plot_controller.clear()

        self.mini_plot_controller.init_eeg_data()
        self.main_waveform_plot_controller.init_eeg_data()

        self.mini_plot_controller.init_hfo_display()
        self.main_waveform_plot_controller.init_waveform_display()
      
    def get_n_channels(self):
        return self.main_waveform_plot_controller.model.n_channels
    
    def get_n_channels_to_plot(self):
        return self.n_channels_to_plot

    def get_total_time(self):
        return self.main_waveform_plot_controller.model.time[-1]

    def get_time_window(self):
        return self.time_window
    
    def get_time_increment(self):
        return self.time_increment
    
    def set_normalize_vertical(self,normalize_vertical:bool):
        self.normalize_vertical = normalize_vertical
        self.main_waveform_plot_controller.set_normalize_vertical(normalize_vertical)

    def set_time_window(self,time_window:float):
        self.time_window = time_window
        self.main_waveform_plot_controller.set_time_window(time_window)
        #replot
        # self.plot(self.t_start)

    def set_time_increment(self,time_increment:float):
        self.time_increment = time_increment
        
    def set_n_channels_to_plot(self,n_channels_to_plot:int):
        self.n_channels_to_plot = n_channels_to_plot
        self.main_waveform_plot_controller.set_n_channels_to_plot(n_channels_to_plot)
        self.mini_plot_controller.set_n_channels_to_plot(n_channels_to_plot)

    def set_plot_HFOs(self,plot_HFOs:bool):
        self.plot_HFOs = plot_HFOs
        self.main_waveform_plot_controller.set_plot_HFOs(plot_HFOs)
        self.plot(self.t_start, update_hfo=True)

    def get_channels_to_plot(self):
        return self.main_waveform_plot_controller.model.channels_to_plot
    
    def get_channel_indices_to_plot(self):
        return self.main_waveform_plot_controller.model.channel_indices_to_plot
    
    def update_channel_names(self,new_channel_names):
        self.mini_plot_controller.update_channel_names(new_channel_names)
        self.main_waveform_plot_controller.update_channel_names(new_channel_names)

    def set_channels_to_plot(self,channels_to_plot:list):
        self.main_waveform_plot_controller.set_channels_to_plot(channels_to_plot)
        self.mini_plot_controller.set_channels_to_plot(channels_to_plot)

    def set_channel_indices_to_plot(self,channel_indices_to_plot:list):
        self.main_waveform_plot_controller.set_channel_indices_to_plot(channel_indices_to_plot)
        self.mini_plot_controller.set_channel_indices_to_plot(channel_indices_to_plot)
    
    def plot(self, start_in_time:float = None, first_channel_to_plot:int = None, empty=False, update_hfo=False):

        if empty:
            self.main_waveform_plot_controller.clear()
            self.mini_plot_controller.clear()
            return
        
        if start_in_time is not None:
            self.main_waveform_plot_controller.set_current_time_window(start_in_time)
        start_in_time, end_in_time = self.main_waveform_plot_controller.get_current_start_end()

        if first_channel_to_plot is not None:
            self.main_waveform_plot_controller.set_first_channel_to_plot(first_channel_to_plot)
        first_channel_to_plot = self.main_waveform_plot_controller.get_first_channel_to_plot()

        self.main_waveform_plot_controller.clear()

        if update_hfo:
            self.mini_plot_controller.clear()
            self.mini_plot_controller.init_hfo_display()

        eeg_data_to_display, y_100_length, y_scale_length, offset_value = self.main_waveform_plot_controller.plot_all_current_channels_for_window()
        top_value = eeg_data_to_display[first_channel_to_plot].max()

        if self.plot_HFOs:
            self.main_waveform_plot_controller.plot_all_current_hfos_for_window(eeg_data_to_display, offset_value, top_value)

        if self.plot_HFOs and update_hfo:
            self.mini_plot_controller.plot_all_current_hfos_for_all_channels(top_value)

        self.main_waveform_plot_controller.draw_scale_bar(eeg_data_to_display, offset_value, y_100_length, y_scale_length)
        self.main_waveform_plot_controller.draw_channel_names(offset_value)

        self.mini_plot_controller.set_miniplot_title('HFO', top_value)
        self.mini_plot_controller.set_total_x_y_range(top_value)
        self.mini_plot_controller.update_highlight_window(start_in_time, end_in_time, top_value)
