from HFODetector import ste, mni
from src.utils.detectors.spike_detector import SpikeDetector
from src.utils.detectors.spindle_detector import SpindleDetector

def set_STE_detector(args):
    detector = ste.STEDetector(
        sample_freq=args.sample_freq,
        filter_freq=[args.pass_band, args.stop_band],
        rms_window=args.rms_window,
        min_window=args.min_window,
        min_gap=args.min_gap,
        epoch_len=args.epoch_len,
        min_osc=args.min_osc,
        rms_thres=args.rms_thres,
        peak_thres=args.peak_thres,
        n_jobs=args.n_jobs,
        front_num=1,
    )
    return detector


def set_MNI_detector(args):
    detector = mni.MNIDetector(
        sample_freq=args.sample_freq,
        filter_freq=[args.pass_band, args.stop_band],
        epoch_time=args.epoch_time,
        epo_CHF=args.epo_CHF,
        per_CHF=args.per_CHF,
        min_win=args.min_win,
        min_gap=args.min_gap,
        thrd_perc=args.thrd_perc,
        base_seg=args.base_seg,
        base_shift=args.base_shift,
        base_thrd=args.base_thrd,
        base_min=args.base_min,
        n_jobs=args.n_jobs,
        front_num=1,
    )
    return detector


def set_Spindle_detector(args):
    detector = SpindleDetector(
        sf=args.sample_freq,
        hypno=args.hypno,
        include=args.include,
        freq_sp=args.freq_sp,
        freq_broad=args.freq_broad,
        duration=args.duration,
        min_distance=args.min_distance,
        thresh=args.thresh,
        multi_only=args.multi_only,
        remove_outliers=args.remove_outliers,
        verbose=args.verbose,
    )
    return detector


def set_Spike_detector(args):
    detector = SpikeDetector(
        resample_rate=args.resample_rate,  # hz
        window_size=args.window_size,  # seconds
        ps_FreqSeg=args.ps_FreqSeg,
        ps_MinFreqHz=args.ps_MinFreqHz,
        ps_MaxFreqHz=args.ps_MaxFreqHz,
        n_jobs=args.n_jobs,
        n_pre_spike=args.n_pre_spike,
        n_post_spike=args.n_post_spike,
        threshold_factor=args.threshold_factor,
        threshold_window=args.threshold_window,  # seconds
        filter_type=args.filter_type,
        detect_mode=args.detect_mode,  # pos, neg, all
    )
    return detector