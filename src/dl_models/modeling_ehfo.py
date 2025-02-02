import numpy as np
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import math
import pandas as pd
import os
import copy
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
class NeuralCNN_ehfo(torch.nn.Module):
    def __init__(self,num_classes =2, num_extra_features =0, dropout_p = 0, freeze_layers = False):
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

        self.fc = nn.Linear(32 + num_extra_features,32)
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
            self.final_ac = nn.Softmax(dim =-1)

    def forward(self, x, additional_feature = None):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        assert x.shape[1] == 2, "Should have 2 channel"
        x = x[:,0,:,:]
        x = x.repeat(1, 3, 1, 1)
        batch = self.cnn(x)
        #batch = self.bn0(batch)
        #print(batch.shape)
        if additional_feature != None:
            batch = torch.cat((batch ,additional_feature),1)
        batch = self.dropout(self.bn(self.relu(self.fc(batch))))
        batch = self.dropout(self.bn1(self.relu1(self.fc1(batch))))
        batch = self.final_ac(self.fc_out(batch))
        return batch

class PreProcessing_ehfo():
    def __init__(self, image_size, fs, freq_range, event_length, crop_time, crop_freq,
                    random_shift_time, feature_param, device):

        # original data parameter
        self.image_size = image_size
        self.fs = fs # in HZ
        self.freq_range = freq_range # in HZ
        self.event_length = event_length # in ms

        # cropped data parameter
        self.crop_time = crop_time # in ms
        self.crop_freq = crop_freq # in HZ
        self.random_shift_time = random_shift_time # in ms
        self.feature_param = feature_param
        self.device = device
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
        # print("random shift time:", self.random_shift_time, "image size:", self.image_size, "event length:", self.event_length)
        self.random_shift_index = int(self.random_shift_time*(self.image_size/self.event_length)) # in index
        self.random_shift = self.random_shift_time != 0

    # @staticmethod
    # def from_df_args(data_meta, args, feature_param, device):
    #     if len(data_meta) != 1:
    #         AssertionError("Data meta should be a single row")
    #     freq_range_hz = [data_meta["freq_min_hz"].values[0], data_meta["freq_max_hz"].values[0]]
    #     fs = data_meta["resample"].values[0] 
    #     event_length = data_meta["time_window_ms"].values[0]
    #     image_size = data_meta["image_size"].values[0]
    #     selected_window_size_ms = args['selected_window_size_ms']
    #     selected_freq_range_hz = args['selected_freq_range_hz']
    #     random_shift_ms = args['random_shift_ms']
    #     preProcessing = PreProcessing(image_size, fs, freq_range_hz, event_length, selected_window_size_ms, selected_freq_range_hz, random_shift_ms, feature_param, device)
        # return preProcessing
    @staticmethod
    def from_dict(d, device):
        return PreProcessing_ehfo(d["image_size"], d["fs"], d["freq_range"], d["event_length"], d["crop_time"], d["crop_freq"], d["random_shift_time"], d["feature_param"], device)
    def to_dict(self):
        return {
        'image_size': self.image_size,
        "fs": self.fs,
        'freq_range': self.freq_range,
        'event_length': self.event_length,
        'random_shift_time': self.random_shift_time,
        'crop_time': self.crop_time,
        'crop_freq': self.crop_freq,
        'feature_param': self.feature_param,
    }
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
        #print("crop freq: ", self.crop_freq, "crop time: ", self.crop_time, "crop freq index: ", self.crop_freq_index, "crop time index: ", self.crop_time_index_r, self.crop_time_index)
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




    def _cropping(self, data):
        time_crop_index = self.crop_time_index_r.copy()
        if self.random_shift:
            shift = np.random.randint(-self.random_shift_index, self.random_shift_index)
            time_crop_index += shift
        self.crop_freq_index[0] = min(max(0, self.crop_freq_index[0]), self.image_size)
        self.crop_freq_index[1] = min(max(0, self.crop_freq_index[1]), self.image_size)
        time_crop_index[0] = min(max(0, time_crop_index[0]), self.image_size)
        time_crop_index[1] = min(max(0, time_crop_index[1]), self.image_size)
        data = data[:,:,self.crop_freq_index[0]:self.crop_freq_index[1] , time_crop_index[0]:time_crop_index[1]]
        return data
    def __call__(self, data):
        # data is shape (batch_size, channel, freq, time)
        # we only want first channel
        # data = data[:,0,:,:]
        # # normalized_data = normalize_img(data)
        # if normalized_data.shape[1] == 1:
        #     normalized_data = normalized_data.repeat(1, 3, 1, 1)
        return data
    # def __call__(self, pt_names, data, label, channel_name, start, end, debug=False):
    #     """Combined preprocessing pipeline"""
    #     # 1. Compute spectrum (from pack_batch)
    #     spectrum_data = compute_spectrum_batch_with_resize(
    #         data, 
    #         sample_rate=self.feature_param["resample"],
    #         win_size=self.feature_param["image_size"],
    #         ps_MinFreqHz=self.feature_param["freq_min_hz"],
    #         ps_MaxFreqHz=self.feature_param["freq_max_hz"],
    #         resize_img=True,
    #         device=self.device
    #     )
    #     # 2. Normalize the data
    #     normalized_data = normalize_img(spectrum_data)
    #     # if only 1 channel, duplicate to make 3 channels
    #     if normalized_data.shape[1] == 1:
    #         normalized_data = normalized_data.repeat(1, 3, 1, 1)
    #     # 4. Create batch dictionary
    #     batch = {
    #         "inputs": normalized_data,
    #         "label": label.squeeze().float().to(self.device),
    #         "channel_name": np.array(channel_name),
    #         "start": start,
    #         "end": end,
    #         "pt_name": np.array(pt_names)
    #     }
        
    #     return batch
    def process_biomarker_feature(self, feature):
        data = feature.get_features()
        self.freq_range = feature.freq_range
        self.event_length = max(feature.time_range)
        self.fs = feature.sample_freq
        self.initialize()
        self.disable_random_shift()
        data = self(data)
        return data

def normalize_img(a):
    batch_num = a.shape[0]
    c = a.shape[1]
    h = a.shape[2]
    w = a.shape[3]
    a_reshape = a.reshape(batch_num * c, -1)
    a_min = torch.min(a_reshape, -1)[0].unsqueeze(1)
    a_max = torch.max(a_reshape, -1)[0].unsqueeze(1)
    normalized = 255.0 * (a_reshape - a_min)/(a_max - a_min)
    normalized = normalized.reshape(batch_num,c, h, w)
    return normalized
def compute_spectrum_batch_with_resize(org_sigs, sample_rate, win_size, ps_MinFreqHz, ps_MaxFreqHz, resize_img=True, device='cpu'):
        # Get original spectrum
    spectrum_imgs = compute_spectrum_batch(org_sigs, ps_SampleRate=sample_rate, ps_FreqSeg=win_size, ps_MinFreqHz=ps_MinFreqHz, ps_MaxFreqHz=ps_MaxFreqHz, device=device)
    # print("spectrum_imgs shape:", spectrum_imgs.shape)
    if resize_img:
        if torch.is_tensor(spectrum_imgs):
                # Use torch interpolate if already a tensor
            return torch.nn.functional.interpolate(
                spectrum_imgs.unsqueeze(1),  # Add channel dim [batch, 1, freq, time]
                size=(win_size, win_size),
                mode='bilinear',  # or 'bicubic' for potentially better quality
                align_corners=False
            )
        else:
            # If it's numpy, convert to tensor first
            spectrum_imgs = torch.from_numpy(spectrum_imgs).to(device)
            resized = torch.nn.functional.interpolate(
                spectrum_imgs.unsqueeze(1),
                size=(win_size, win_size),
                mode='bilinear',
                align_corners=False
            )
            return resized
        
    raise ValueError("We must reshape the spectrum image to the desired size")
def compute_spectrum_batch(org_sigs, ps_SampleRate=2000, ps_FreqSeg=512, ps_MinFreqHz=10, ps_MaxFreqHz=500, device='cpu'):
    # Ensure device compatibility
    org_sigs_device = org_sigs.device if org_sigs.is_cuda else torch.device(device)
    
    # Adjust device for computations
    org_sigs = org_sigs.to(org_sigs_device)
    
    batch_size, sig_len = org_sigs.shape
    ii, jj = int(sig_len // 2), int(sig_len // 2 + sig_len)

    # Create extended signals directly on the correct device
    extend_sigs = create_extended_sig_batch(org_sigs)
    ps_StDevCycles = 3
    s_Len = extend_sigs.shape[1]
    s_HalfLen = math.floor(s_Len / 2) + 1

    # Define frequency and window axes on the correct device
    v_WAxis = (torch.linspace(0, 2 * np.pi, s_Len, device=org_sigs_device)[:-1] * ps_SampleRate).float()
    v_WAxisHalf = v_WAxis[:s_HalfLen].repeat(ps_FreqSeg, 1)
    v_FreqAxis = torch.linspace(ps_MaxFreqHz, ps_MinFreqHz, steps=ps_FreqSeg, device=org_sigs_device).float()

    # Initialize FFT window matrix on the correct device
    v_WinFFT = torch.zeros(ps_FreqSeg, s_Len, device=org_sigs_device).float()
    s_StDevSec = (1 / v_FreqAxis) * ps_StDevCycles
    v_WinFFT[:, :s_HalfLen] = torch.exp(-0.5 * (v_WAxisHalf - (2 * torch.pi * v_FreqAxis.view(-1, 1)))**2 * (s_StDevSec**2).view(-1, 1))
    
    # Normalize the FFT windows
    v_WinFFT = v_WinFFT * math.sqrt(s_Len) / torch.norm(v_WinFFT, dim=-1).view(-1, 1)

    # Perform FFT on the extended signals and apply windowed filters
    v_InputSignalFFT = torch.fft.fft(extend_sigs, dim=1)
    
    # Corrected reshaping to align dimensions
    res = torch.fft.ifft(v_InputSignalFFT.unsqueeze(1) * v_WinFFT.unsqueeze(0), dim=2)[:, :, ii:jj]
    res = res / torch.sqrt(s_StDevSec).view(1, -1, 1)

    # Return the magnitude, ensuring the result is moved to CPU if required
    res = res.abs()
    return res.cpu().numpy() if not org_sigs.is_cuda else res
def create_extended_sig_batch(sigs):
    batch_size, s_len = sigs.shape
    s_halflen = int(np.ceil(s_len / 2)) + 1

    # Compute start and end windows for each signal in the batch
    start_win = sigs[:, :s_halflen] - sigs[:, [0]]
    end_win = sigs[:, s_len - s_halflen - 1:] - sigs[:, [-1]]

    start_win = -start_win.flip(dims=[1]) + sigs[:, [0]]
    end_win = -end_win.flip(dims=[1]) + sigs[:, [-1]]

    # Concatenate to form the final extended signals
    final_sigs = torch.cat((start_win[:, :-1], sigs, end_win[:, 1:]), dim=1)

    # Ensure the final signals have an odd length
    if final_sigs.shape[1] % 2 == 0:
        final_sigs = final_sigs[:, :-1]

    return final_sigs
def pick_best_model(model, best_model ,epoch, v_loss, best_loss, checkpoint_folder, model_name="a", preprocessing=None, save = True, verbose = True):
    if v_loss < best_loss:
        best_loss = v_loss 
        best_model = copy.deepcopy(model)
        pre_processing = preprocessing.to_dict()
        if verbose:
            print(f"best_model of {model_name} in epoch ", epoch +1)
        if save:
            save_checkpoint({'epoch': epoch + 1,
                'model': best_model,
                # "model_param":{"in_channels": model.in_channels, "outputs": model.outputs,"model_name": model_name},
                "preprocessing": pre_processing,
                },
                filename= os.path.join(checkpoint_folder, f'model_{model_name}.tar'))
    return best_loss, best_model
def save_checkpoint(state, filename='models/checkpoint.pth.tar'):
    torch.save(state, filename)