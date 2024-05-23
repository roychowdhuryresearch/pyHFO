from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5 import uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import matplotlib.pyplot as plt

import pyqtgraph as pg
import matplotlib.ticker as ticker
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg # We will try using pyqtgraph for plotting
import time
import mne
# from superqt import QDoubleRangeSlider
from tqdm import tqdm
import os
from src.hfo_app import HFO_App
from src.hfo_feature import HFO_Feature

from src.utils.utils_annotation import *

import random
import scipy.fft as fft #FFT plot (5)
import scipy.signal as signal
import numpy as np

import re
from pathlib import Path
from src.hfo_app import HFO_App
from src.param.param_classifier import ParamClassifier
from src.param.param_detector import ParamDetector, ParamSTE, ParamMNI
from src.param.param_filter import ParamFilter
from src.utils.utils_gui import *
from src.ui.plot_waveform import *
# import FormatStrFormatter
from matplotlib.ticker import FormatStrFormatter


# from src.ui.plot_annotation_waveform import *
# from src.ui.a_channel_selection import AnnotationChannelSelection 

# from src.plot_time_frequency import PlotTimeFrequencyNoLabel
from src.utils.utils_plotting import *
# from src.plot_time_frequency import MainWindow

import multiprocessing as mp
import torch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
#import fft 
import scipy.fft as fft


def custom_formatter(x, pos):
    # if number >1000, then use scientific notation but still fix the width to 5
    max_width = 5
    if abs(x) > 1000:
        return f'{x:.0e}'
    # 4 digits + 1 for potential negative sign
    formatted_number = f' {x:.0f}' if x >= 0 else f'{x:.0f}'
    return f'{formatted_number:>{max_width}}'

class AnnotationPlot(FigureCanvasQTAgg):
      def __init__(self, parent=None, width=10, height=4, dpi=100, hfo_app=None):
        fig,self.axs = plt.subplots(3,1,figsize=(width, height), dpi=dpi)
        super(AnnotationPlot, self).__init__(fig)
        self.hfo_app = hfo_app
        FigureCanvasQTAgg.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)
        # self.setParent(parent)
        # self.plot()

      def plot(self,start_index: int = None, end_index: int = None, channel:str = None, interval=1.0):
        #first clear the plot
        for ax in self.axs:
            ax.cla()
        # check if the index are in correct range
        if None:
            return
        if start_index < 0:
            return
        
        channel_name = channel
        fs = self.hfo_app.sample_freq
        
        #both sets of data (filtered/unfiltered) for plots
        length = self.hfo_app.get_eeg_data_shape()[1]
        # window_start_index, window_end_index, relative_start_index, relative_end_end = calcuate_boundary(plot_start_index, plot_end_index, length, fs)
        window_start_index, window_end_index, relative_start_index, relative_end_end = calcuate_boundary(start_index, end_index, length,fs)
        unfiltered_eeg_data, self.channel_names = self.hfo_app.get_eeg_data(window_start_index, window_end_index)
        filtered_eeg_data,_ = self.hfo_app.get_eeg_data(window_start_index, window_end_index, filtered=True)

        unfiltered_eeg_data_to_display_one = unfiltered_eeg_data[self.channel_names == channel_name,:][0]
        filtered_eeg_data_to_display = filtered_eeg_data[self.channel_names == channel_name,:][0]
        # print("window_start_index: ", window_start_index)
        # print("window_end_index: ", window_end_index)
        # print("relative_start_index: ", relative_start_index)
        # print("relative_end_end: ", relative_end_end)
        time_to_display = np.arange(0, unfiltered_eeg_data_to_display_one.shape[0])/fs+window_start_index/fs
        # print("this is time to display: ", time_to_display.shape)
        # print("this is unfiltered_eeg_data_to_display_one: ", unfiltered_eeg_data_to_display_one.shape)
        # print("this is filtered_eeg_data_to_display: ", filtered_eeg_data_to_display.shape)
        self.axs[0].set_title("EEG Tracing")
        self.axs[0].plot(time_to_display, unfiltered_eeg_data_to_display_one, color='blue')
        # self.axs[0].plot(time_to_display[int(start_index - window_start_index):int(end_index - window_start_index)], 
        #                  unfiltered_eeg_data_to_display_one[int(start_index - window_start_index):int(end_index - window_start_index)], color='orange')
        self.axs[0].plot(time_to_display[relative_start_index:relative_end_end], unfiltered_eeg_data_to_display_one[relative_start_index:relative_end_end], color='orange')
        self.axs[0].set_xticks([])
        # keep the y axis label fixed (not moving when the plot is updated)
        
        self.axs[0].set_ylabel('Amplitude (uV)', rotation=90, labelpad=5)
        self.axs[0].yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
        #self.axs[0].yaxis.set_label_coords(-0.1, 0.5) 
        # set the y axis label to the right side
        self.axs[0].yaxis.set_label_position("right")
        self.axs[0].set_ylim([unfiltered_eeg_data_to_display_one.min(), unfiltered_eeg_data_to_display_one.max()])
        
        middle_index = (relative_start_index + relative_end_end) // 2
        half_interval_samples = int((interval * fs) // 2)
        plot_start_index = max(0, int(middle_index - half_interval_samples))
        plot_end_index = int(min(self.hfo_app.get_eeg_data_shape()[1], middle_index + half_interval_samples))
        plot_start_index = max(0, min(len(time_to_display) - 1, plot_start_index))
        plot_end_index = min(len(time_to_display) - 1, int(middle_index + half_interval_samples))
        
        # print(f"time_to_display range: {time_to_display[0]} to {time_to_display[-1]}")
        # print(f"plot_start_index: {plot_start_index}")
        # print(f"plot_end_index: {plot_end_index}")
        # print(f"relative_start_index: {relative_start_index}")
        # print(f"relative_end_index: {relative_end_end}")
        
        self.axs[0].set_xlim(time_to_display[plot_start_index], time_to_display[plot_end_index])
        
        #self.axs[0].grid()
        # print("this is time to display: ", time_to_display.shape)
        # print("this is filtered_eeg_data_to_display: ", filtered_eeg_data_to_display.shape)
        self.axs[1].set_title("Filtered Tracing")
        self.axs[1].plot(time_to_display, filtered_eeg_data_to_display,  color='blue')
        # self.axs[1].plot(time_to_display[int(start_index - window_start_index):int(end_index - window_start_index)], 
        #     filtered_eeg_data_to_display[int(start_index - window_start_index):int(end_index - window_start_index)], color='orange')
        self.axs[1].plot(time_to_display[relative_start_index:relative_end_end], filtered_eeg_data_to_display[relative_start_index:relative_end_end], color='orange')
        
        self.axs[1].set_ylabel('Amplitude (uV)', rotation=90, labelpad=6)
        self.axs[1].set_xticks([])
        self.axs[1].yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
        #self.axs[1].yaxis.set_label_coords(-0.1, 0.5) 
        # set the y axis label to the right side
        self.axs[1].yaxis.set_label_position("right")
        self.axs[1].set_ylim([filtered_eeg_data_to_display.min(), filtered_eeg_data_to_display.max()])  # Set y-axis limits
        self.axs[1].set_xlim(time_to_display[plot_start_index], time_to_display[plot_end_index])
        
        #self.axs[1].grid()

        time_frequency = calculate_time_frequency(unfiltered_eeg_data_to_display_one,fs)
        self.axs[2].set_title("Time Frequency")
        self.axs[2].imshow(time_frequency,extent=[time_to_display[0], time_to_display[-1], 10, 500], aspect='auto', cmap='jet')
        # set xticks as time
        self.axs[2].set_xticks(np.linspace(time_to_display[0], time_to_display[-1], 5))
        self.axs[2].set_xticklabels(np.round(np.linspace(time_to_display[0], time_to_display[-1], 5),1))
        # set yticks as frequency
        self.axs[2].set_yticks(np.linspace(10, 500, 5).astype(int))
        self.axs[2].yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
        self.axs[2].set_xlabel('Time (s)')
        self.axs[2].set_ylabel('Frequency (Hz)', rotation=90, labelpad=4)
        #self.axs[2].yaxis.set_label_coords(-0.1, 0.5) 
        # set the y axis label to the right side
        self.axs[2].yaxis.set_label_position("right")
        self.axs[2].set_xlim(time_to_display[plot_start_index], time_to_display[plot_end_index])
        
        #share x axis
        #self.axs[0].sharex(self.axs[1])
        # self.axs[0].sharex(self.axs[2])
        #self.axs[1].sharex(self.axs[2])
        #call the draw function
        plt.tight_layout()
        self.draw()
        
class FFTPlot(FigureCanvasQTAgg):
        def __init__(self, parent=None, width=5, height=4, dpi=100, hfo_app=None):
                fig,self.axs = plt.subplots(1,1,figsize=(width, height), dpi=dpi)
                super(FFTPlot, self).__init__(fig)
                self.hfo_app = hfo_app
                
                FigureCanvasQTAgg.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
                FigureCanvasQTAgg.updateGeometry(self)
        

        def plot(self, start_index: int = None, end_index: int = None, channel: str = None, interval=1.0):
            self.axs.cla()
            start_index = int(start_index)
            fs = self.hfo_app.sample_freq
            middle_index = (start_index + end_index) // 2
            half_interval_samples = int((interval * fs) // 2)
            plot_start_index = int(max(0, middle_index - half_interval_samples))
            plot_end_index = int(min(self.hfo_app.get_eeg_data_shape()[1], middle_index + half_interval_samples))
            
            unfiltered_eeg_data, channel_names = self.hfo_app.get_eeg_data(plot_start_index, plot_end_index)
            unfiltered_eeg_data = unfiltered_eeg_data[channel_names == channel, :][0]
            # Compute the FFT
            f, Pxx_den = signal.periodogram(unfiltered_eeg_data, fs)

            # Plotting the FFT
            self.axs.semilogy(f, Pxx_den)
            self.axs.set_xlabel('Frequency (Hz)')
            self.axs.set_ylabel(r"PSD (V$^2$/Hz)")
            
            self.axs.set_ylim([1e-7, 1e3])
            # self.axs.set_ylim([0, Pxx_den.max()])
            self.axs.set_xlim([min(f), max(f)])  # Ensure the x-axis covers the full frequency range
            self.axs.grid()
            plt.tight_layout()
            self.draw()