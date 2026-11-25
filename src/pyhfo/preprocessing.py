from __future__ import annotations

import numpy as np


class PreProcessing:
    def __init__(
        self,
        image_size,
        fs,
        freq_range_hz,
        event_length,
        selected_window_size_ms,
        selected_freq_range_hz,
        random_shift_ms,
    ):
        self.image_size = image_size
        self.freq_range = freq_range_hz
        self.fs = fs
        self.event_length = event_length

        self.crop_time = selected_window_size_ms
        self.crop_freq = selected_freq_range_hz
        self.random_shift_time = random_shift_ms

        self.initialize()

    def initialize(self):
        self.freq_range_low = self.freq_range[0]
        self.freq_range_high = self.freq_range[1]
        self.time_range = [0, self.event_length]
        self.crop_range_index = self.crop_time / self.event_length * self.image_size
        self.crop_freq_low = self.crop_freq[0]
        self.crop_freq_high = self.crop_freq[1]
        self.crop = (
            self.freq_range_low == self.crop_freq_low
            and self.freq_range_high == self.crop_freq_high
            and self.crop_time * 2 == self.event_length
        )
        self.calculate_crop_index()
        self.random_shift_index = int(self.random_shift_time * (self.image_size / self.event_length))
        self.random_shift = self.random_shift_time != 0

    @staticmethod
    def from_dict(d):
        return PreProcessing(
            d["image_size"],
            d["fs"],
            d["freq_range_hz"],
            d["time_range_ms"][1],
            d["selected_window_size_ms"],
            d["selected_freq_range_hz"],
            d["random_shift_ms"],
        )

    @staticmethod
    def from_df_args(data_meta, args):
        if len(data_meta) != 1:
            AssertionError("Data meta should be a single row")
        freq_range_hz = [data_meta["freq_min_hz"].values[0], data_meta["freq_max_hz"].values[0]]
        fs = data_meta["resample"].values[0]
        event_length = data_meta["time_window_ms"].values[0]
        image_size = data_meta["image_size"].values[0]
        selected_window_size_ms = args["selected_window_size_ms"]
        selected_freq_range_hz = args["selected_freq_range_hz"]
        random_shift_ms = args["random_shift_ms"]
        return PreProcessing(
            image_size,
            fs,
            freq_range_hz,
            event_length,
            selected_window_size_ms,
            selected_freq_range_hz,
            random_shift_ms,
        )

    @staticmethod
    def from_param(param):
        return PreProcessing(
            param.image_size,
            param.fs,
            param.freq_range,
            param.time_range[1],
            param.crop_time,
            param.crop_freq,
            0,
        )

    def check_bound(self, x, text):
        if x < 0 or x > self.image_size:
            raise AssertionError(
                f"Index out of bound on {text}: index {x:.1f} is outside [0, {self.image_size}]. "
                f"selected_freq_range_hz={self.crop_freq} must lie within freq_range_hz={self.freq_range} "
                "so this model's crop band does not exceed its own spectrogram range."
            )
        return True

    def calculate_crop_index(self):
        self.crop_freq_index_low = self.image_size - self.image_size / (
            self.freq_range_high - self.freq_range_low
        ) * (self.crop_freq_low - self.freq_range_low)
        self.crop_freq_index_high = self.image_size - self.image_size / (
            self.freq_range_high - self.freq_range_low
        ) * (self.crop_freq_high - self.freq_range_low)
        self.crop_freq_index = np.array([self.crop_freq_index_high, self.crop_freq_index_low]).astype(int)
        self.crop_time_index = np.array([-self.crop_range_index, self.crop_range_index]).astype(int)
        self.crop_time_index_r = self.image_size // 2 + self.crop_time_index
        self.check_bound(self.crop_freq_index_low, "selected_freq_range_hz_low")
        self.check_bound(self.crop_freq_index_high, "selected_freq_range_hz_high")
        self.check_bound(self.crop_time_index_r[0], "crop_time")
        self.check_bound(self.crop_time_index_r[1], "crop_time")
        self.crop_index_w = np.abs(self.crop_time_index_r[0] - self.crop_time_index_r[1])
        self.crop_index_h = np.abs(self.crop_freq_index[0] - self.crop_freq_index[1])

    def enable_random_shift(self):
        self.random_shift = self.random_shift_time != 0

    def disable_random_shift(self):
        self.random_shift = False

    def to_dict(self):
        return {
            "image_size": self.image_size,
            "freq_range_hz": self.freq_range,
            "time_range_ms": self.time_range,
            "fs": self.fs,
            "random_shift_ms": self.random_shift_time,
            "selected_freq_range_hz": self.crop_freq,
            "selected_window_size_ms": self.crop_time,
        }

    def _cropping(self, data):
        time_crop_index = self.crop_time_index_r.copy()
        if self.random_shift:
            shift = np.random.randint(-self.random_shift_index, self.random_shift_index)
            time_crop_index += shift
        return data[
            :,
            :,
            self.crop_freq_index[0] : self.crop_freq_index[1],
            time_crop_index[0] : time_crop_index[1],
        ]

    def __call__(self, data):
        return self._cropping(data)

    def process_biomarker_feature(self, feature):
        data = feature.get_features()
        self.freq_range = feature.freq_range
        self.event_length = max(feature.time_range)
        self.fs = feature.sample_freq
        self.initialize()
        self.disable_random_shift()
        return self(data)
