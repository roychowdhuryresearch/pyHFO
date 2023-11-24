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
    max_width = 5  # 4 digits + 1 for potential negative sign
    formatted_number = f' {x:,.0f}' if x >= 0 else f'{x:,.0f}'
    return f'{formatted_number:>{max_width}}'

class AnnotationPlot(FigureCanvasQTAgg):
      def __init__(self, parent=None, width=5, height=4, dpi=100, hfo_app=None):
        fig,self.axs = plt.subplots(3,1,figsize=(width, height), dpi=dpi)
        super(AnnotationPlot, self).__init__(fig)
        self.hfo_app = hfo_app
        FigureCanvasQTAgg.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)
        # self.setParent(parent)
        # self.plot()

      def plot(self,start_index: int = None, end_index: int = None, channel:str = None):
        #first clear the plot
        for ax in self.axs:
                ax.cla()
        # check if the index are in correct range
        if None:
                return
        if start_index < 0:
                return
        if start_index < 0:
                start_index = self.index

        channel_name = channel
        # print("this is channel: ", channel)
        # print("this is channel_name: ", channel_name)
        
        #both sets of data (filtered/unfiltered) for plots
        length = self.hfo_app.get_eeg_data_shape()[1]
        fs = self.hfo_app.sample_freq
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
        self.axs[0].plot(time_to_display[relative_start_index:relative_end_end], unfiltered_eeg_data_to_display_one[relative_start_index:relative_end_end], color='orange')
        self.axs[0].set_xticks([])
        # keep the y axis label fixed (not moving when the plot is updated)
        
        self.axs[0].set_ylabel('Amplitude (uV)', rotation=90, labelpad=5)
        self.axs[0].yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
        self.axs[0].yaxis.set_label_coords(-0.1, 0.5) 

        # print("this is time to display: ", time_to_display.shape)
        # print("this is filtered_eeg_data_to_display: ", filtered_eeg_data_to_display.shape)
        self.axs[1].set_title("Filtered Tracing")
        self.axs[1].plot(time_to_display, filtered_eeg_data_to_display,  color='blue')
        self.axs[1].plot(time_to_display[relative_start_index:relative_end_end], filtered_eeg_data_to_display[relative_start_index:relative_end_end], color='orange')
        self.axs[1].set_ylabel('Amplitude (uV)', rotation=90, labelpad=6)
        self.axs[1].set_xticks([])
        self.axs[1].yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
        self.axs[1].yaxis.set_label_coords(-0.1, 0.5) 

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
        self.axs[2].yaxis.set_label_coords(-0.1, 0.5) 
        

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
        
        def plot(self,start_index: int = None, end_index: int = None, channel:str = None):
                self.axs.cla()
                start_index = int(start_index)
                end_index = int(end_index)
                unfiltered_eeg_data, channel_names = self.hfo_app.get_eeg_data(start_index, end_index)
                unfiltered_eeg_data = unfiltered_eeg_data[channel_names == channel,:][0]
                # filtered_eeg_data,_ = self.hfo_app.get_eeg_data(start_index, end_index, filtered=True)
                #compute the fft
                fs = self.hfo_app.sample_freq

                f, Pxx_den = signal.periodogram(unfiltered_eeg_data, fs)
                self.axs.semilogy(f, Pxx_den)
                self.axs.set_xlabel('frequency [Hz]')
                self.axs.set_ylabel('PSD [V**2/Hz]')
                self.axs.set_ylim([1e-7, 1e2])
                self.axs.set_xlim([0, 1000])
                self.axs.grid()
                plt.tight_layout()

                # fft_magnitudes = np.abs(fft.fft(unfiltered_eeg_data))
                # fft_magnitudes = (fft_magnitudes-np.min(fft_magnitudes))/np.ptp(fft_magnitudes)
                # # print(np.max(data))
                # # print(np.min(data))
                # max_frequencies = 500
                # fft_freqs = fft.fftfreq(len(unfiltered_eeg_data), 1/fs)
                # # fft_magnitudes = fft_freqs[np.abs(fft_freqs)<max_frequencies] #change this later to be able to be edited
                # # fft_freqs = fft_freqs[np.abs(fft_freqs)<max_frequencies]
                # #only keep the frequencies above 0
                # fft_magnitudes = fft_magnitudes[fft_freqs>0]
                # fft_freqs = fft_freqs[fft_freqs>0]
                # #sort the frequencies
                # # print("this is fft_magnitudes before: ", fft_magnitudes)
                # # print("this is fft_freqs before: ", fft_freqs)
                # fft_magnitudes = fft_magnitudes[np.argsort(fft_freqs)]
                # fft_freqs = fft_freqs[np.argsort(fft_freqs)]
                # # print("this is fft_magnitudes after: ", fft_magnitudes)
                # # print("this is fft_freqs after: ", fft_freqs)
                # #plot as a stem plot

                # #self.axs.stem(fft_freqs, fft_magnitudes, use_line_collection=True)
                # # only plot 10~500 Hz
                # index = np.where((fft_freqs>10) & (fft_freqs<500))

                # self.axs.plot(fft_freqs[index], np.abs(fft_magnitudes)[index], color='blue')
                # self.axs.set_xlabel('Frequency (HZ)')
                # self.axs.set_ylabel('Magnitude')
                self.draw()
                
                
                
                
                
        

              
