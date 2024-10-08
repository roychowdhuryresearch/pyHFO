class ParamDetector:
    def __init__(self, detector_param, detector_type='ste'):
        self.detector_param = detector_param
        self.detector_type = detector_type
    def to_dict(self):
        return {'detector_param': self.detector_param.to_dict(), 'detector_type': self.detector_type}
    @staticmethod
    def from_dict(param_dict):
        param = ParamDetector(None)
        detector_type = param_dict['detector_type']
        param.detector_type = detector_type
        if detector_type.lower() == 'ste':
            param.detector_param = ParamSTE.from_dict(param_dict['detector_param'])
        elif detector_type.lower() == 'mni':
            param.detector_param = ParamMNI.from_dict(param_dict['detector_param'])
        elif detector_type.lower() == 'hil':
            param.detector_param = ParamHIL.from_dict(param_dict['detector_param'])
        return param
    
class ParamSTE:
    def __init__(self, sample_freq, pass_band=1, stop_band=50, rms_window=3*1e-3, min_window=6*1e-3, min_gap=10 * 1e-3, 
                epoch_len=600, min_osc=6, rms_thres=5, peak_thres=3,
                n_jobs=32):
        self.sample_freq = sample_freq
        self.pass_band = pass_band
        self.stop_band = stop_band
        self.rms_window = rms_window
        self.min_window = min_window
        self.min_gap = min_gap
        self.epoch_len = epoch_len
        self.min_osc = min_osc
        self.rms_thres = rms_thres
        self.peak_thres = peak_thres
        self.n_jobs = n_jobs
    def to_dict(self):
        d = {
            "sample_freq": self.sample_freq,
            "pass_band": self.pass_band,
            "stop_band": self.stop_band,
            "rms_window": self.rms_window,
            "min_window": self.min_window,
            "min_gap": self.min_gap,
            "epoch_len": self.epoch_len,
            "min_osc": self.min_osc,
            "rms_thres": self.rms_thres,
            "peak_thres": self.peak_thres,
            "n_jobs": self.n_jobs
        }
        return d

    @staticmethod
    def from_dict(d):
        return ParamSTE(
            sample_freq = d["sample_freq"],
            pass_band = d["pass_band"],
            stop_band = d["stop_band"],
            rms_window = d["rms_window"],
            min_window = d["min_window"],
            min_gap = d["min_gap"],
            epoch_len = d["epoch_len"],
            min_osc = d["min_osc"],
            rms_thres = d["rms_thres"],
            peak_thres = d["peak_thres"],
            n_jobs = d["n_jobs"]
        )
class ParamMNI:
    def __init__(self,sample_freq, pass_band = 80, stop_band = 500, 
                epoch_time=10, epo_CHF=60, per_CHF=95/100, 
                min_win=10*1e-3, min_gap=10*1e-3, thrd_perc=99.9999/100, 
                base_seg=125*1e-3, base_shift=0.5, base_thrd=0.67, base_min=5,
                n_jobs=32):
        self.sample_freq = sample_freq
        self.pass_band = pass_band
        self.stop_band = stop_band
        self.epoch_time = epoch_time
        self.epo_CHF = epo_CHF
        self.per_CHF = per_CHF
        self.min_win = min_win
        self.min_gap = min_gap
        self.thrd_perc = thrd_perc
        self.base_seg = base_seg
        self.base_shift = base_shift
        self.base_thrd = base_thrd
        self.base_min = base_min
        self.n_jobs = n_jobs

    def to_dict(self):
        return {
            "sample_freq": self.sample_freq,
            "pass_band": self.pass_band,
            "stop_band": self.stop_band,
            "epoch_time": self.epoch_time,
            "epo_CHF": self.epo_CHF,
            "per_CHF": self.per_CHF,
            "min_win": self.min_win,
            "min_gap": self.min_gap,
            "thrd_perc": self.thrd_perc,
            "base_seg": self.base_seg,
            "base_shift": self.base_shift,
            "base_thrd": self.base_thrd,
            "base_min": self.base_min,
            "n_jobs": self.n_jobs,
        }
    @staticmethod
    def from_dict(d):
        return ParamMNI(
            d["sample_freq"],
            d["pass_band"],
            d["stop_band"],
            d["epoch_time"],
            d["epo_CHF"],
            d["per_CHF"],
            d["min_win"],
            d["min_gap"],
            d["thrd_perc"],
            d["base_seg"],
            d["base_shift"],
            d["base_thrd"],
            d["base_min"],
            d["n_jobs"]
        )

class ParamHIL:
    def __init__(self, sample_freq=2000, pass_band=80, stop_band=500, epoch_time=10, sliding_window=5, min_window=0.006, n_jobs=8):
        self.sample_freq = sample_freq
        self.pass_band = pass_band
        self.stop_band = stop_band
        self.epoch_time = epoch_time
        self.sliding_window = sliding_window
        self.min_window = min_window
        self.n_jobs = n_jobs

    def to_dict(self):
        return {
            "sample_freq": self.sample_freq,
            "pass_band": self.pass_band,
            "stop_band": self.stop_band,
            "epoch_time": self.epoch_time,
            "sliding_window": self.sliding_window,
            "min_window": self.min_window,
            "n_jobs": self.n_jobs
        }

    @staticmethod
    def from_dict(d):
        return ParamHIL(
            d["sample_freq"],
            d["pass_band"],
            d["stop_band"],
            d["epoch_time"],
            d["sliding_window"],
            d["min_window"],
            d["n_jobs"]
        )