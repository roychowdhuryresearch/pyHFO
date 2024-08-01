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

curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(curr_dir))


class CenterWaveformAndMiniPlotController():
    def __init__(self, plot_loc:pg.PlotWidget, hfo_loc:pg.PlotWidget, backend: HFO_App):
        super().__init__()
        self.hfo_display = hfo_loc
        self.hfo_display.setMouseEnabled(x=False, y=False)
        self.hfo_display.getPlotItem().hideAxis('bottom')
        self.hfo_display.getPlotItem().hideAxis('left')
        self.hfo_display.setBackground('w')

        self.waveform_display = plot_loc #pg.PlotWidget(plot_loc)
        # self.waveform_display.getPlotItem().getAxis('bottom').setHeight(10)
        self.waveform_display.setMouseEnabled(x=False, y=False)
        self.waveform_display.getPlotItem().hideAxis('bottom')
        self.waveform_display.getPlotItem().hideAxis('left')
        self.plot_loc = plot_loc
        self.waveform_display.setBackground('w')

        self.time_window = 20 #20 second time window
        self.time_increment =20
        self.old_size = (self.plot_loc.x(),self.plot_loc.y(),self.plot_loc.width(),self.plot_loc.height())
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
        self.filtered = filtered
        self.plot(self.t_start)

    def update_backend(self,new_backend:HFO_App,init_eeg_data:bool=True):
        self.backend = new_backend
        if init_eeg_data:
            self.init_eeg_data()

    def init_eeg_data(self):
        # print("reinit eeg data")
        #reinitalize self
        # self = PlotWaveform(self.plot_loc,self.hfo_loc,self.backend)
        # self.eeg_data = eeg_data
        # #normalize each to 0-1
        # self.eeg_data = (self.eeg_data-self.eeg_data.min(axis = 1,keepdims = True))/(np.ptp(self.eeg_data,axis = 1,keepdims = True))
        # #shift the ith channel by 1.1*i
        # self.eeg_data = self.eeg_data-1.1*np.arange(self.eeg_data.shape[0])[:,None]
        self.filtered = False
        self.plot_HFOs = False
        self.hfo_display.clear()
        self.waveform_display.clear()
        eeg_data,self.channel_names=self.backend.get_eeg_data()
        ## print("eeg_data.shape",eeg_data.shape)
        # print("self.channel_names",self.channel_names)
        self.channel_names = list(self.channel_names)
        self.edf_info=self.backend.get_edf_info()
        # print(self.edf_info)
        # self.channel_names_locs=np.mean(self.eeg_data,axis = 1)
        self.sample_freq = self.edf_info['sfreq']
        self.time = np.arange(0,eeg_data.shape[1]/self.sample_freq,1/self.sample_freq) # time in seconds
        self.n_channels = len(self.channel_names)
        # print("here")
        self.n_channels_to_plot = min(self.n_channels,self.n_channels_to_plot)
        self.channels_to_plot = self.channel_names.copy()
        self.channel_indices_to_plot = np.arange(self.n_channels)
        self.init_hfo_display()
        self.waveform_display.getPlotItem().showAxis('bottom')
        self.waveform_display.getPlotItem().showAxis('left')
        # print(self.plot_loc.x(),self.plot_loc.y(),self.plot_loc.width(),self.plot_loc.height())
      
    def init_hfo_display(self):
        # print("init hfo display")
        self.hfo_display.getPlotItem().showAxis('bottom')
        self.hfo_display.getPlotItem().showAxis('left')
        self.lr = pg.LinearRegionItem([0,0], movable = False)
        self.lr.setZValue(-20)
        self.hfo_display.addItem(self.lr)

    def get_n_channels(self):
        return self.n_channels
    
    def get_n_channels_to_plot(self):
        return self.n_channels_to_plot

    def get_total_time(self):
        return self.time[-1]

    def get_time_window(self):
        return self.time_window
    
    def get_time_increment(self):
        return self.time_increment
    
    def set_normalize_vertical(self,normalize_vertical:bool):
        self.normalize_vertical = normalize_vertical

    def set_time_window(self,time_window:float):
        self.time_window = time_window
        #replot
        # self.plot(self.t_start)

    def set_time_increment(self,time_increment:float):
        self.time_increment = time_increment
        
    def set_n_channels_to_plot(self,n_channels_to_plot:int):
        self.n_channels_to_plot = n_channels_to_plot

    def set_plot_HFOs(self,plot_HFOs:bool):
        self.plot_HFOs = plot_HFOs
        self.plot(self.t_start, update_hfo=True)

    def get_channels_to_plot(self):
        return self.channels_to_plot
    
    def get_channel_indices_to_plot(self):
        return self.channel_indices_to_plot
    
    def update_channel_names(self,new_channel_names):
        self.channel_names = list(new_channel_names)

    def set_channels_to_plot(self,channels_to_plot:list):
        self.channels_to_plot = channels_to_plot
        self.channel_indices_to_plot = [self.channel_names.index(channel) for channel in channels_to_plot]
        # self.n_channels_to_plot = len(self.channels_to_plot)
        # self.plot(self.t_start)

    def set_channel_indices_to_plot(self,channel_indices_to_plot:list):
        self.channel_indices_to_plot = channel_indices_to_plot
        self.channels_to_plot = [self.channel_names[index] for index in channel_indices_to_plot]
        # self.n_channels_to_plot = len(self.channels_to_plot)
        # self.plot(self.t_start)
    
    def plot(self,t_start:float = None,first_channel_to_plot:int = None, empty=False, update_hfo=False):
        # print("plot HFOs",self.plot_HFOs)
        if empty:
            self.waveform_display.clear()
            self.hfo_display.clear()
            return
        if t_start is None:
            t_start = self.t_start
        else:
            self.t_start = t_start #this allows us to keep track of the start time of the plot and thus replot when the time window changes or when the number of channels
        if first_channel_to_plot is None:
            first_channel_to_plot = self.first_channel_to_plot
        else:
            self.first_channel_to_plot = first_channel_to_plot
        self.waveform_display.clear()
        if update_hfo:
            self.hfo_display.clear()
            self.init_hfo_display()
        #to show changes
        t_end = min(t_start+self.time_window,self.time[-1])
        # print(t_start,t_end,self.time[-1])
        # start_time = time.time()
        eeg_data_to_display,_=self.backend.get_eeg_data(int(t_start*self.sample_freq),int(t_end*self.sample_freq), self.filtered)
        #normalize each to 0-1
        eeg_data_to_display = eeg_data_to_display[self.channel_indices_to_plot,:]
        if self.normalize_vertical:
            eeg_data_to_display = (eeg_data_to_display-eeg_data_to_display.min(axis = 1,keepdims = True))
            eeg_data_to_display = eeg_data_to_display/np.max(eeg_data_to_display)
        else:
            # eeg_data_to_display = (eeg_data_to_display-eeg_data_to_display.min(axis = 1,keepdims = True))/(np.ptp(eeg_data_to_display,axis = 1,keepdims = True))

            # standardized signal by channel
            # means = np.mean(eeg_data_to_display, axis=1, keepdims=True)
            # stds = np.std(eeg_data_to_display, axis=1, keepdims=True)
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
        time_to_display = self.time[int(t_start*self.sample_freq):int(t_end*self.sample_freq)]
        top_value=eeg_data_to_display[first_channel_to_plot].max()
        # print("top value:",top_value)
        # bottom_value=eeg_data_to_display[-1].min()
        # print("bottom value:",bottom_value)
        # print("channel means",np.mean(eeg_data_to_display,axis = 1))
        for disp_i, ch_i in enumerate(range(first_channel_to_plot,first_channel_to_plot+self.n_channels_to_plot)):
            channel = self.channels_to_plot[ch_i]

            self.waveform_display.plot(time_to_display, eeg_data_to_display[ch_i] - disp_i*offset_value, pen=pg.mkPen(color=self.waveform_color, width=0.5))
            if self.plot_HFOs:
                starts, ends, artifacts, spikes = self.backend.hfo_features.get_HFOs_for_channel(channel,int(t_start*self.sample_freq),int(t_end*self.sample_freq))
                # print("channel:", channel, starts,ends, artifacts, spikes)
                for j in range(len(starts)):
                    try:
                        if int(artifacts[j])<1:
                            color = self.artifact_color
                            name="artifact"
                        elif spikes[j]:
                            color = self.spike_color
                            name="spike"
                        else:
                            color = self.non_spike_color
                            name="non-spike"
                    except:
                        color = self.HFO_color
                        name="HFO"
                    # print(time_to_display[starts[j]:ends[j]])
                    self.waveform_display.plot(self.time[int(starts[j]):int(ends[j])],
                                               eeg_data_to_display[ch_i, int(starts[j])-int(t_start*self.sample_freq):int(ends[j])-int(t_start*self.sample_freq)]-disp_i*offset_value,
                                               pen=pg.mkPen(color=color, width=2))
                    # print("plotting",self.time[int(starts[j])],self.time[int(ends[j])],"name:",name,"channel:",channel)
                    # print(starts[j],ends[j])
                    # print(eeg_data_to_display[i,int(starts[j]):int(ends[j])])
                    self.waveform_display.plot([self.time[int(starts[j])],self.time[int(ends[j])]],[
                        top_value+0.2,top_value+0.2
                    ],pen = pg.mkPen(color = color,width=10))

            # mini plot
            if self.plot_HFOs and update_hfo:
                starts, ends, artifacts, spikes = self.backend.hfo_features.get_HFOs_for_channel(channel,
                                                                                                 0,
                                                                                                 sys.maxsize)
                for j in range(len(starts)):
                    try:
                        if int(artifacts[j])<1:
                            color = self.artifact_color
                            name="artifact"
                        elif spikes[j]:
                            color = self.spike_color
                            name="spike"
                        else:
                            color = self.non_spike_color
                            name="non-spike"
                    except:
                        color = self.HFO_color
                        name="HFO"
                    # x = self.time[int(starts[j]):int(ends[j])]
                    # y = eeg_data_to_display[i, int(starts[j]):int(ends[j])]
                    # # self.waveform_mini_item.setData(x, y, pen=pg.mkPen(color=color, width=2))
                    # self.hfo_display.plot(x, y, pen=pg.mkPen(color=color, width=2))
                    end = min(int(ends[j]), len(self.time)-1)
                    self.hfo_display.plot([self.time[int(starts[j])], self.time[end]], [
                        top_value, top_value
                    ], pen=pg.mkPen(color=color, width=5))

        # Determine the position for the scale indicator (bottom right corner of the plot)
        x_pos = t_end #+ 0.15
        # y_pos = top_value - 0.1 * (top_value - np.min(eeg_data_to_display))  # Adjust as needed
        y_pos = np.min(eeg_data_to_display[-1]) - self.n_channels_to_plot * offset_value + 0.8 * offset_value

        # # Draw the x and y scale lines
        # self.waveform_display.plot([x_pos, x_pos], [y_pos, y_pos + y_scale_length], pen=pg.mkPen('black', width=2))

        # # Add text annotations for the scale lines
        # text_item = pg.TextItem(f'{y_100_length} μV', color='black', anchor=(1, 0.5))
        # text_item.setPos(x_pos, y_pos + y_scale_length / 2)
        # self.waveform_display.addItem(text_item)

        # Use a dashed line for the scale
        scale_line = pg.PlotDataItem([x_pos, x_pos], [y_pos, y_pos + y_scale_length],
                             pen=pg.mkPen('black', width=10), fill=(0, 128, 255, 150)) 
        self.waveform_display.addItem(scale_line)
        
        text_item = pg.TextItem(f'Scale: {y_100_length} μV ', color='black', anchor=(1, 0.5))
        text_item.setFont(QtGui.QFont('Arial', 10, QtGui.QFont.Bold))
        text_item.setPos(x_pos, y_pos + y_scale_length / 2)
        self.waveform_display.addItem(text_item)

        # print("time to plot:",time.time()-start_time)
        #set y ticks to channel names
        # channel_names_locs = -offset_value * np.arange(eeg_data_to_display.shape[0])[:, None] # + offset_value/2
        channel_names_locs = -offset_value * np.arange(self.n_channels_to_plot)[:, None]  # + offset_value/2

        self.waveform_display.getAxis('left').setTicks([[(channel_names_locs[disp_i], self.channels_to_plot[chi_i])
                 for disp_i, chi_i in enumerate(range(first_channel_to_plot,first_channel_to_plot+self.n_channels_to_plot))]])
        #set the max and min of the x axis
        self.waveform_display.setXRange(t_start,t_end)

        self.hfo_display.getAxis('left').setTicks([[(top_value, '   HFO   ')]])
        self.hfo_display.setXRange(0, int(self.time.shape[0] / self.sample_freq))
        self.hfo_display.setYRange(top_value-0.25, top_value+0.25)
        
        self.lr.setRegion([t_start,t_end])
        self.lr.setZValue(top_value)
        #set background to white
        # self.waveform_display.setBackground('w')

        # plot out on top bars of where the HFOs are
        # all_channels_starts = np.array(all_channels_starts)
        # all_channels_ends = np.array(all_channels_ends)
        # all_channels_names = np.array(all_channels_names)
        # for name in np.unique(all_channels_names):
        #     self.waveform_display.plot(self.time[all_channels_starts[all_channels_names==name]],
        #                                [0]*
        #                                ,pen = pg.mkPen(color = self.color_dict[name],width=5))
