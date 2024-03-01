from yasa import spindles_detect
import numpy as np

class SpindleDetector:
    def __init__(
        self,
        sf=None,
        hypno=None,
        include=(1, 2, 3),
        freq_sp=(12, 15),
        freq_broad=(1, 500),
        duration=(0.5, 2),
        min_distance=500,
        thresh={"corr": 0.65, "rel_pow": 0.2, "rms": 1.5},
        multi_only=False,
        remove_outliers=False,
        verbose=False,
    ):
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
        detection = spindles_detect(
            filter_data,
            sf=self.sf,
            ch_names=channel_names,
            hypno=self.hypno,
            include=self.include,
            freq_sp=self.freq_sp,
            freq_broad=self.freq_broad,
            duration=self.duration,
            min_distance=self.min_distance,
            thresh=self.thresh,
            multi_only=self.multi_only,
            remove_outliers=self.remove_outliers,
            verbose=self.verbose,
        )
        if detection is None:
            print("No spindles detected")
            return [], np.array()
        result = detection.summary()
        # each spindles are represented by a list, which is an interval [start, end]
        spindles = np.concatenate(
            (result.Start * self.sf, result.End * self.sf), axis=1
        ).values
        channel_names_result = result.Channel.values
        return channel_names_result, spindles
