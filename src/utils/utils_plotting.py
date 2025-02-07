import matplotlib.pyplot as plt
import numpy as np
import os


def plot_feature(folder, feature, start, end, data, data_filtered, channel_name, biomarker_start, biomarker_end):
    channel_data = data
    channel_data_f = data_filtered
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    channel_data = np.squeeze(channel_data)
    channel_data_f = np.squeeze(channel_data_f)
    ax1.imshow(feature[0])
    ax2.plot(channel_data, color='blue')
    ax2.plot(np.arange(biomarker_start, biomarker_end), channel_data[biomarker_start:biomarker_end], color='red')
    ax3.plot(channel_data_f, color='blue')
    ax3.plot(np.arange(biomarker_start, biomarker_end), channel_data_f[biomarker_start:biomarker_end], color='red')
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


def calculate_time_frequency(data, fs, freq_min=10, freq_max=500):
    # filter_range = [58., 62.]
    # sos = butter(5, filter_range, btype='bandstop', output='sos', fs=fs)
    # filtered_data = sosfiltfilt(sos, data)
    time_frequency = compute_spectrum(data, ps_SampleRate=fs, ps_MinFreqHz=freq_min, ps_MaxFreqHz=freq_max)
    return time_frequency