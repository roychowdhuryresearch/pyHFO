 
import torch
import torch.nn as nn
# Data utils and dataloader
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import numpy as np
import pandas as pd


class NeuralCNN_ehfo(torch.nn.Module):
    def __init__(self, num_classes=2, num_extra_features=0, dropout_p=0, freeze_layers=False):
        super(NeuralCNN_ehfo, self).__init__()
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        self.cnn = models.resnet18(pretrained=True)
        if freeze_layers:
            for param in self.cnn.parameters():
                param.requires_grad = False

        self.cnn.fc = nn.Sequential(nn.Linear(512, 32))
        for param in self.cnn.fc.parameters():
            param.requires_grad = True
        self.bn0 = nn.BatchNorm1d(32)
        self.relu0 = nn.LeakyReLU()

        self.fc = nn.Linear(32 + num_extra_features, 32)
        self.bn = nn.BatchNorm1d(32)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(32, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.LeakyReLU()

        self.dropout = nn.Dropout(dropout_p)
        if num_classes < 3:
            self.fc_out = nn.Linear(16, 1)
            self.final_ac = nn.Sigmoid()
        else:
            self.fc_out = nn.Linear(16, num_classes)
            self.final_ac = nn.Softmax(dim=-1)

    def forward(self, x, additional_feature=None):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        assert x.shape[1] == 2, "Should have 2 channel"
        x = x[:, 0:1, :, :]
        x = x.repeat(1, 3, 1, 1)
        batch = self.cnn(x)
        # batch = self.bn0(batch)
        # print(batch.shape)
        if additional_feature != None:
            batch = torch.cat((batch, additional_feature), 1)
        batch = self.dropout(self.bn(self.relu(self.fc(batch))))
        batch = self.dropout(self.bn1(self.relu1(self.fc1(batch))))
        batch = self.final_ac(self.fc_out(batch))
        return batch

class NeuralCNN(torch.nn.Module):
    def __init__(self, in_channels, outputs, freeze = False, channel_selection = True):
        super(NeuralCNN, self).__init__()
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        self.in_channels = in_channels
        self.outputs = outputs
        self.channel_selection = channel_selection
        self.cnn = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.conv1= nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.cnn.fc = nn.Sequential(nn.Linear(512, 32))
        for param in self.cnn.fc.parameters():
            param.requires_grad = not freeze
        self.bn0 = nn.BatchNorm1d(32)
        self.relu0 = nn.LeakyReLU()
        self.fc = nn.Linear(32,32)
        self.bn = nn.BatchNorm1d(32)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(32, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.LeakyReLU()
        
        self.fc_out = nn.Linear(16, self.outputs)
        self.final_ac = nn.Sigmoid()
        
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        if self.in_channels == 1:
            x = x[:,0:1,:,:]
        batch = self.cnn(x)
        batch = self.bn(self.relu(self.fc(batch)))
        batch = self.bn1(self.relu1(self.fc1(batch)))
        batch = self.final_ac(self.fc_out(batch))
        return batch


class PreProcessing():
    def __init__(self, image_size, fs, freq_range_hz, event_length, selected_window_size_ms, selected_freq_range_hz,
                    random_shift_ms):

        # original data parameter
        self.image_size = image_size
        self.freq_range = freq_range_hz # in HZ
        self.fs = fs # in HZ
        self.event_length = event_length # in ms

        # cropped data parameter
        self.crop_time = selected_window_size_ms # in ms
        self.crop_freq = selected_freq_range_hz # in HZ
        self.random_shift_time = random_shift_ms # in ms

        self.initialize()

    def initialize(self):
        self.freq_range_low = self.freq_range[0]   # in HZ
        self.freq_range_high = self.freq_range[1]  # in HZ
        self.time_range = [0, self.event_length] # in ms
        self.crop_range_index = self.crop_time / self.event_length * self.image_size # in index
        self.crop_freq_low = self.crop_freq[0] # in HZ
        self.crop_freq_high = self.crop_freq[1] # in HZ
        self.crop = self.freq_range_low == self.crop_freq_low and self.freq_range_high == self.crop_freq_high and self.crop_time*2 == self.event_length
        self.calculate_crop_index()
        self.random_shift_index = int(self.random_shift_time*(self.image_size/self.event_length)) # in index
        self.random_shift = self.random_shift_time != 0

    @staticmethod
    def from_dict(d):
        data_meta = pd.DataFrame({
            'image_size': d['image_size'],
            'freq_min_hz': d['freq_range_hz'][0],
            'freq_max_hz': d['freq_range_hz'][1],
            'resample': d['fs'],
            'time_window_ms': d['time_range_ms'][1],
        }, index=[0])
        return PreProcessing.from_df_args(data_meta, d)

    @staticmethod
    def from_df_args(data_meta, args):
        if len(data_meta) != 1:
            AssertionError("Data meta should be a single row")
        freq_range_hz = [data_meta["freq_min_hz"].values[0], data_meta["freq_max_hz"].values[0]]
        fs = data_meta["resample"].values[0] 
        event_length = data_meta["time_window_ms"].values[0]
        image_size = data_meta["image_size"].values[0]
        selected_window_size_ms = args['selected_window_size_ms']
        selected_freq_range_hz = args['selected_freq_range_hz']
        random_shift_ms = args['random_shift_ms']
        preProcessing = PreProcessing(image_size, fs, freq_range_hz, event_length, selected_window_size_ms, selected_freq_range_hz, random_shift_ms)
        return preProcessing
    @staticmethod
    def from_param(param):
        freq_range_hz = param.freq_range
        fs = param.fs
        event_length = param.time_range[1]
        image_size = param.image_size
        selected_window_size_ms =  param.crop_time
        selected_freq_range_hz = param.crop_freq
        random_shift_ms = 0
        preProcessing = PreProcessing(image_size, fs, freq_range_hz, event_length, selected_window_size_ms, selected_freq_range_hz, random_shift_ms)
        return preProcessing
    
    def check_bound(self, x, text):
        if x < 0 or x > self.image_size:
            raise AssertionError(f"Index out of bound on {text}")
        return True

    def calculate_crop_index(self):
        # calculate the index of the crop, high_freq is low index
        self.crop_freq_index_low = self.image_size - self.image_size / (self.freq_range_high - self.freq_range_low) * (self.crop_freq_low - self.freq_range_low)
        self.crop_freq_index_high = self.image_size - self.image_size / (self.freq_range_high - self.freq_range_low) * (self.crop_freq_high - self.freq_range_low)  
        self.crop_freq_index = np.array([self.crop_freq_index_high, self.crop_freq_index_low]).astype(int) # in index
        self.crop_time_index = np.array([-self.crop_range_index, self.crop_range_index]).astype(int) # in index
        self.crop_time_index_r = self.image_size//2 + self.crop_time_index # in index
        #print("crop freq: ", self.crop_freq, "crop time: ", self.crop_time, "crop freq index: ", self.crop_freq_index, "crop time index: ", self.crop_time_index_r)
        self.check_bound(self.crop_freq_index_low, "selected_freq_range_hz_low")
        self.check_bound(self.crop_freq_index_high, "selected_freq_range_hz_high")
        self.check_bound(self.crop_time_index_r[0], "crop_time")
        self.check_bound(self.crop_time_index_r[1], "crop_time")
        self.crop_index_w = np.abs(self.crop_time_index_r[0]- self.crop_time_index_r[1])
        self.crop_index_h = np.abs(self.crop_freq_index[0]- self.crop_freq_index[1])
    
    def enable_random_shift(self):
        self.random_shift = self.random_shift_time != 0
    
    def disable_random_shift(self):
        self.random_shift = False


    def to_dict(self):
        return {
            'image_size': self.image_size,
            'freq_range_hz': self.freq_range,
            'time_range_ms': self.time_range,
            'fs': self.fs,
            'random_shift_ms': self.random_shift_time,
            'selected_freq_range_hz': self.crop_freq,
            'selected_window_size_ms': self.crop_time,
        }

    def _cropping(self, data):
        time_crop_index = self.crop_time_index_r.copy()
        if self.random_shift:
            shift = np.random.randint(-self.random_shift_index, self.random_shift_index)
            time_crop_index += shift
        data = data[:,:,self.crop_freq_index[0]:self.crop_freq_index[1] , time_crop_index[0]:time_crop_index[1]]
        return data

    def __call__(self, data):
        data = self._cropping(data)
        return data

    def process_biomarker_feature(self, feature):
        data = feature.get_features()
        self.freq_range = feature.freq_range
        self.event_length = max(feature.time_range)
        self.fs = feature.sample_freq
        self.initialize()
        self.disable_random_shift()
        data = self(data)
        return data