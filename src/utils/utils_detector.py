from HFODetector import ste, mni, hil
import yasa


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


def set_HIL_detector(args):
    detector = hil.HILDetector(sample_freq=args.sample_freq, filter_freq=[args.pass_band, args.stop_band],
                               sd_thres=args.sd_threshold, min_window=args.min_window,
                               epoch_len=args.epoch_time, n_jobs=args.n_jobs, front_num=1)
    return detector


def set_YASA_detector(args):
    detector = yasa
    return {'yasa': detector, 'args': args}
