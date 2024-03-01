import numpy as np
import math
import scipy.linalg as LA
import mne
from skimage.transform import resize
import pandas as pd
import os, shutil
from scipy.signal import cheb2ord, cheby2, zpk2sos, sosfilt, sosfiltfilt, ellip


class FilterCheby2(object):
    def __init__(self, fp, fs, rp, rs, space, sample_freq):
        filter_freq = [fp, fs]
        space = space

        nyq = sample_freq / 2
        # MGNM
        if fs >= .99 * nyq:
            fs = nyq * .99
        low, high = fp, fs

        scale = 0
        while 0 < low < 1:
            low *= 10
            scale += 1
        low = filter_freq[0] - (space * 10 ** (-1 * scale))

        scale = 0
        while high < 1:
            high *= 10
            scale += 1
        high = filter_freq[1] + (space * 10 ** (-1 * scale))
        stop_freq = [low, high]
        ps_order, wst = cheb2ord([filter_freq[0] / nyq, filter_freq[1] / nyq],
                                 [stop_freq[0] / nyq, stop_freq[1] / nyq], rp, rs)
        # z, p, k = cheby2(ps_order, rs, wst, btype='bandpass', analog=0, output='zpk')
        # sos = zpk2sos(z, p, k)
        self.sos = cheby2(ps_order, rs, wst, btype='bandpass', analog=False, output='sos')

    def filter_data(self, data):
        filtered = sosfilt(self.sos, data)
        filtered = sosfilt(self.sos, np.flipud(filtered))
        filtered = np.flipud(filtered)
        return filtered


class FilterEllip(object):
    def __init__(self, order=4, rp=0.1, rs=40, wn=(20, 80), btype='bandpass', fs=2000):
        filter_freq = [wn[0], wn[1]]
        self.sos = ellip(order, rp, rs, filter_freq, btype=btype, output='sos', fs=fs)

    def filter_data(self, data):
        filtered = sosfiltfilt(self.sos, data)
        return filtered


def clean_folder(saved_fn):
    if not os.path.exists(saved_fn):
        #os.mkdir(saved_fn)
        os.makedirs(saved_fn)
    else:
        shutil.rmtree(saved_fn)
        os.mkdir(saved_fn)


def read_raw(raw_path, resample=2000):
    raw = mne.io.read_raw_edf(raw_path, verbose= False)
    if raw.info['sfreq'] != resample:
        raw = raw.resample(resample, npad='auto')
    raw_channels = raw.info['ch_names']
    channels = np.array([ch for ch in raw_channels])
    data = []

    for raw_ch in raw_channels:
        ch_data = raw.get_data(raw_ch) * 1E6
        data.append(ch_data)
    
    data = np.squeeze(np.array(data))
    return data, channels


def compute_spectrum(org_sig, ps_SampleRate=2000, ps_FreqSeg=512, ps_MinFreqHz=10, ps_MaxFreqHz=500):

    def create_extended_sig(sig):
        s_len = len(sig)
        s_halflen = int(np.ceil(s_len/2)) + 1
        start_win = sig[:s_halflen] - sig[0]
        end_win = sig[s_len - s_halflen - 1:] - sig[-1]
        start_win = -start_win[::-1] + sig[0]
        end_win = -end_win[::-1] + sig[-1]
        final_sig = np.concatenate((start_win[:-1],sig, end_win[1:]))
        if len(final_sig)%2 == 0:
            final_sig = final_sig[:-1]
        return final_sig
    
    extend_sig = create_extended_sig(org_sig)
    # extend_sig = org_sig
    s_Len = len(extend_sig)
    s_HalfLen = math.floor(s_Len/2)+1
    v_WAxis = np.linspace(0, 2*np.pi, s_Len, endpoint=False)
    v_WAxis = v_WAxis* ps_SampleRate
    v_WAxisHalf = v_WAxis[:s_HalfLen]
    v_FreqAxis = np.linspace(ps_MinFreqHz, ps_MaxFreqHz, num=ps_FreqSeg)#ps_MinFreqHz:s_FreqStep:ps_MaxFreqHz
    v_FreqAxis = v_FreqAxis[::-1]
    
    v_InputSignalFFT = np.fft.fft(extend_sig)
    ps_StDevCycles = 3
    m_GaborWT = np.zeros((ps_FreqSeg, s_Len), dtype=complex)
    for i, s_FreqCounter in enumerate(v_FreqAxis):
        v_WinFFT = np.zeros(s_Len)
        s_StDevSec = (1 / s_FreqCounter) * ps_StDevCycles
        v_WinFFT[:s_HalfLen] = np.exp(-0.5*np.power(v_WAxisHalf - (2 * np.pi * s_FreqCounter), 2)*
            (s_StDevSec**2))
        v_WinFFT = v_WinFFT * np.sqrt(s_Len)/ LA.norm(v_WinFFT, 2)
        m_GaborWT[i, :] = np.fft.ifft(v_InputSignalFFT* v_WinFFT)/np.sqrt(s_StDevSec)
    ii, jj = int(len(org_sig)//2), int(len(org_sig)//2 + len(org_sig))
    return np.abs(m_GaborWT[:, ii:jj])


def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s)
        # pre-sorting of locals min based on relative position with respect to s_mid
        lmin = lmin[s[lmin] < s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid
        lmax = lmax[s[lmax] > s_mid]

    # global min of dmin-chunks of locals min
    lmin = lmin[[i + np.argmin(s[lmin[i:i + dmin]]) for i in range(0, len(lmin), dmin)]]
    # global max of dmax-chunks of locals max
    lmax = lmax[[i + np.argmax(s[lmax[i:i + dmax]]) for i in range(0, len(lmax), dmax)]]

    return lmin, lmax
