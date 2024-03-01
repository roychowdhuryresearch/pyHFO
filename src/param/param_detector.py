class ParamDetector:
    def __init__(self, detector_param, detector_type="ste"):
        self.detector_param = detector_param
        self.detector_type = detector_type

    def to_dict(self):
        return {
            "detector_param": self.detector_param.to_dict(),
            "detector_type": self.detector_type,
        }

    @staticmethod
    def from_dict(param_dict):
        param = ParamDetector(None)
        detector_type = param_dict["detector_type"]
        param.detector_type = detector_type
        if detector_type.lower() == "ste":
            param.detector_param = ParamSTE.from_dict(param_dict["detector_param"])
        elif detector_type.lower() == "mni":
            param.detector_param = ParamMNI.from_dict(param_dict["detector_param"])
        elif detector_type.lower() == "spindle":
            param.detector_param = ParamSpindle.from_dict(param_dict["detector_param"])
        elif detector_type.lower() == "spike":
            param.detector_param = ParamSpike.from_dict(param_dict["detector_param"])
        return param


class ParamSTE:
    def __init__(
        self,
        sample_freq,
        pass_band=1,
        stop_band=50,
        rms_window=3 * 1e-3,
        min_window=6 * 1e-3,
        min_gap=10 * 1e-3,
        epoch_len=600,
        min_osc=6,
        rms_thres=5,
        peak_thres=3,
        n_jobs=32,
    ):
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
            "n_jobs": self.n_jobs,
        }
        return d

    @staticmethod
    def from_dict(d):
        return ParamSTE(
            sample_freq=d["sample_freq"],
            pass_band=d["pass_band"],
            stop_band=d["stop_band"],
            rms_window=d["rms_window"],
            min_window=d["min_window"],
            min_gap=d["min_gap"],
            epoch_len=d["epoch_len"],
            min_osc=d["min_osc"],
            rms_thres=d["rms_thres"],
            peak_thres=d["peak_thres"],
            n_jobs=d["n_jobs"],
        )


class ParamMNI:
    def __init__(
        self,
        sample_freq,
        pass_band=80,
        stop_band=500,
        epoch_time=10,
        epo_CHF=60,
        per_CHF=95 / 100,
        min_win=10 * 1e-3,
        min_gap=10 * 1e-3,
        thrd_perc=99.9999 / 100,
        base_seg=125 * 1e-3,
        base_shift=0.5,
        base_thrd=0.67,
        base_min=5,
        n_jobs=32,
    ):
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
            d["n_jobs"],
        )


class ParamSpindle:
    def __init__(
        self,
        sample_freq,
        freq_sp=(12, 15),
        hypno=None,
        include=(1, 2, 3),
        freq_broad=(1, 30),
        duration=(0.5, 2),
        min_distance=500,
        thresh={"corr": 0.65, "rel_pow": 0.2, "rms": 1.5},
        multi_only=False,
        remove_outliers=False,
        verbose=False,
    ):
        self.freq_sp = freq_sp
        self.sample_freq = sample_freq
        self.pass_band = freq_sp[0]
        self.stop_band = freq_sp[1]
        self.hypno = hypno
        self.include = include
        self.freq_broad = freq_broad
        self.duration = duration
        self.min_distance = min_distance
        self.thresh = thresh
        self.multi_only = multi_only
        self.remove_outliers = remove_outliers
        self.verbose = verbose

    @staticmethod
    def from_dict(d):
        return ParamSpindle(
            sample_freq=d["sample_freq"],
            freq_sp=(d["freq_sp"]),
            hypno=d["hypno"],
            include=d["include"],
            freq_broad=d["freq_broad"],
            duration=d["duration"],
            min_distance=d["min_distance"],
            thresh=d["thresh"],
            multi_only=d["multi_only"],
            remove_outliers=d["remove_outliers"],
            verbose=d["verbose"],
        )


class ParamSpike:
    def __init__(
        self,
        resample_rate=2000,  # hz
        window_size=5,  # seconds
        save_folder="results",
        ps_FreqSeg=128,
        ps_MinFreqHz=3,
        ps_MaxFreqHz=50,
        n_jobs=16,
        n_pre_spike=23,
        n_post_spike=51,
        threshold_factor=5,
        threshold_window=30,  # seconds
        filter_type="ellip",
        detect_mode="all",  # pos, neg, all
    ):
        self.resample_rate = resample_rate
        self.window_size = window_size
        self.save_folder = save_folder
        self.ps_FreqSeg = ps_FreqSeg
        self.ps_MinFreqHz = ps_MinFreqHz
        self.ps_MaxFreqHz = ps_MaxFreqHz
        self.n_jobs = n_jobs
        self.n_pre_spike = n_pre_spike
        self.n_post_spike = n_post_spike
        self.threshold_factor = threshold_factor
        self.threshold_window = threshold_window
        self.filter_type = filter_type
        self.detect_mode = detect_mode

    @staticmethod
    def from_dict(d):
        return ParamSpike(
            resample_rate=d["resample_rate"],  # hz
            window_size=d["window_size"],  # seconds
            ps_FreqSeg=d["ps_FreqSeg"],
            ps_MinFreqHz=d["ps_MinFreqHz"],
            ps_MaxFreqHz=d["ps_MaxFreqHz"],
            n_jobs=d["n_jobs"],
            n_pre_spike=d["n_pre_spike"],
            n_post_spike=d["n_post_spike"],
            threshold_factor=d["threshold_factor"],
            threshold_window=d["threshold_window"],  # seconds
            filter_type=d["filter_type"],
            detect_mode=d["detect_mode"],  # pos, neg, all
        )
