import matplotlib.pyplot as plt
import numpy as np
import os
def plot_feature(folder, feature, start, end, data, data_filtered, channel_name, hfo_start, hfo_end):
    channel_data = data
    channel_data_f = data_filtered
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    channel_data = np.squeeze(channel_data)
    channel_data_f = np.squeeze(channel_data_f)
    ax1.imshow(feature[0])
    ax2.plot(channel_data, color='blue')
    ax2.plot(np.arange(hfo_start, hfo_end), channel_data[hfo_start:hfo_end], color='red')
    ax3.plot(channel_data_f, color='blue')
    ax3.plot(np.arange(hfo_start, hfo_end), channel_data_f[hfo_start:hfo_end], color='red')
    plt.suptitle(f"{channel_name}_{start}_{end} with length: {(end - start)*0.5} ms")
    fn = f'{channel_name}_{start}_{end}.jpg'
    plt.savefig(os.path.join(folder,fn))
    plt.close()

from PyQt5.QtWidgets import QDialog, QApplication, QGridLayout
import matplotlib
import pyqtgraph as pg
import sys
# from src.hfo_app import HFO_App
from src.utils.utils_feature import *
from scipy.signal import butter, sosfiltfilt

from PyQt5.QtWidgets import QDialog, QApplication, QGridLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def calculate_time_frequency(data,fs):
    filter_range = [58., 62.]
    sos = butter(20, filter_range, btype='bandstop', output='sos', fs=fs)
    filtered_data = sosfiltfilt(sos, data)
    # sample_freq = hfo_app.get_sample_freq()
    # print("filtered_data shape:", filtered_data.shape)
    # print("1data shape:", data.shape)
    # print("2data shape:", data.shape)
    #calculate the time frequency
    #why data matrix vector
    time_frequency = compute_spectrum(data, ps_SampleRate = fs)
    return time_frequency