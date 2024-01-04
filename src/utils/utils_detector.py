from HFODetector import ste, mni
from yasa import spindles_detect
import pandas as pd
import numpy as np

def set_STE_detector(args):
    detector = ste.STEDetector(sample_freq=args.sample_freq, filter_freq=[args.pass_band, args.stop_band], 
                rms_window=args.rms_window, min_window=args.min_window, min_gap=args.min_gap, 
                epoch_len=args.epoch_len, min_osc=args.min_osc, rms_thres=args.rms_thres, peak_thres=args.peak_thres,
                n_jobs=args.n_jobs, front_num=1)
    return detector
def set_MNI_detector(args):
    detector = mni.MNIDetector(sample_freq=args.sample_freq, filter_freq=[args.pass_band, args.stop_band],epoch_time=args.epoch_time,
            epo_CHF=args.epo_CHF,per_CHF=args.per_CHF,min_win=args.min_win,min_gap=args.min_gap,
            thrd_perc=args.thrd_perc,base_seg=args.base_seg,base_shift=args.base_shift,
            base_thrd=args.base_thrd,base_min=args.base_min,n_jobs=args.n_jobs,front_num=1)
    return detector
def set_Spindle_detector(args):
    detector = SpindleDetector(sf=args.sample_freq, hypno=None, include=(1, 2, 3),
                #  freq_sp=(args.pass_band, args.stop_band),
                #  freq_broad=(0, 500),
                 freq_sp=(12, 15),
                 freq_broad=(1, 30),
                 duration=(0.5, 2), min_distance=500,
                 thresh={'corr': 0.65, 'rel_pow': 0.2, 'rms': 1.5},
                 multi_only=False, remove_outliers=False, verbose=False)
    return detector

class SpindleDetector:
    def __init__(self, sf=None, hypno=None, include=(1, 2, 3),
                 freq_sp=(12, 15), freq_broad=(1, 500), duration=(0.5, 2), min_distance=500,
                 thresh={'corr': 0.65, 'rel_pow': 0.2, 'rms': 1.5},
                 multi_only=False, remove_outliers=False, verbose=False):
        self.sf = sf
        self.hypno = hypno
        self.include = include
        self.freq_sp = freq_sp
        self.freq_broad = freq_broad
        self.duration = duration
        self.min_distance = min_distance
        self.thresh = thresh
        self.multi_only = multi_only
        self.remove_outliers = remove_outliers
        self.verbose = verbose

    def detect_multi_channels(self, filter_data, channel_names, filtered=True):
        detection = spindles_detect(filter_data, sf=self.sf, ch_names=channel_names, hypno=self.hypno, include=self.include,
                                   freq_sp=self.freq_sp, freq_broad=self.freq_broad, duration=self.duration, min_distance=self.min_distance,
                                   thresh=self.thresh,
                                   multi_only=self.multi_only, remove_outliers=self.remove_outliers, verbose=self.verbose)
        result = detection.summary()
        if result is None:
            print("No spindles detected")
            return [], np.array()
        # each spindles are represented by a list, which is an interval [start, end]
        spindles = pd.concat([result.Start * self.sf, result.End * self.sf], axis=1).values
        channel_names_result = result.Channel.values
        return channel_names_result, spindles