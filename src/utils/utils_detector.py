from HFODetector import ste, mni
import numpy as np
from scipy import signal
try:
    from HFODetector import hil
except Exception:  # pragma: no cover - optional runtime dependency
    hil = None
try:
    import yasa
except Exception:  # pragma: no cover - optional runtime dependency
    yasa = None


def has_hil():
    return hil is not None


def has_yasa():
    return yasa is not None


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
    if hil is None:
        raise ImportError("HIL detection is not available in the installed HFODetector package.")
    detector = hil.HILDetector(sample_freq=args.sample_freq, filter_freq=[args.pass_band, args.stop_band],
                               sd_thres=args.sd_threshold, min_window=args.min_window,
                               epoch_len=args.epoch_time, n_jobs=args.n_jobs, front_num=1)
    return detector


def set_YASA_detector(args):
    if yasa is None:
        raise ImportError("YASA is required for spindle detection but is not installed.")
    detector = yasa
    return {'yasa': detector, 'args': args}


class SpikeRMSLLDetector:
    def __init__(
        self,
        *,
        sample_freq,
        filter_freq,
        rms_window,
        ll_window,
        rms_thres,
        ll_thres,
        peak_thres,
        min_window,
        max_window,
        min_gap,
        n_jobs=1,
        front_num=1,
    ):
        self.sample_freq = float(sample_freq)
        self.filter_freq = list(filter_freq)
        self.rms_window = float(rms_window)
        self.ll_window = float(ll_window)
        self.rms_thres = float(rms_thres)
        self.ll_thres = float(ll_thres)
        self.peak_thres = float(peak_thres)
        self.min_window = float(min_window)
        self.max_window = float(max_window)
        self.min_gap = float(min_gap)
        self.n_jobs = int(n_jobs)
        self.front_num = int(front_num)

    def _window_samples(self, seconds, *, minimum=1):
        return max(int(round(float(seconds) * self.sample_freq)), int(minimum))

    def _robust_center_scale(self, values):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return 0.0, 1.0
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 0.0, 1.0
        center = float(np.median(finite))
        mad = float(np.median(np.abs(finite - center)))
        scale = mad * 1.4826
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(np.std(finite))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        return center, scale

    def _moving_rms(self, values, window_samples):
        squared = np.square(np.asarray(values, dtype=float))
        kernel = np.ones(int(window_samples), dtype=float) / float(window_samples)
        return np.sqrt(np.convolve(squared, kernel, mode="same"))

    def _moving_line_length(self, values, window_samples):
        diffs = np.abs(np.diff(np.asarray(values, dtype=float), prepend=float(values[0]) if len(values) else 0.0))
        kernel = np.ones(int(window_samples), dtype=float) / float(window_samples)
        return np.convolve(diffs, kernel, mode="same")

    def _event_bounds_from_peak(self, abs_signal, peak_index, baseline_level, max_half_window):
        peak_index = int(peak_index)
        left_limit = max(0, peak_index - max_half_window)
        right_limit = min(len(abs_signal) - 1, peak_index + max_half_window)

        boundary_level = max(float(baseline_level), float(abs_signal[peak_index]) * 0.35)
        if boundary_level >= float(abs_signal[peak_index]):
            boundary_level = float(abs_signal[peak_index]) * 0.8

        start = peak_index
        while start > left_limit and abs_signal[start] > boundary_level:
            start -= 1
        end = peak_index
        while end < right_limit and abs_signal[end] > boundary_level:
            end += 1

        if start == peak_index:
            start = max(0, peak_index - 1)
        if end == peak_index:
            end = min(len(abs_signal) - 1, peak_index + 1)
        return int(start), int(end + 1)

    def detect_channel(self, channel_data):
        data = np.asarray(channel_data, dtype=float)
        if data.size == 0 or not np.any(np.isfinite(data)):
            return np.empty((0, 2), dtype=int)

        rms_window_samples = self._window_samples(self.rms_window)
        ll_window_samples = self._window_samples(self.ll_window)
        min_window_samples = self._window_samples(self.min_window, minimum=2)
        max_window_samples = max(min_window_samples, self._window_samples(self.max_window, minimum=2))
        min_gap_samples = self._window_samples(self.min_gap, minimum=1)

        rms = self._moving_rms(data, rms_window_samples)
        ll = self._moving_line_length(data, ll_window_samples)
        rms_center, rms_scale = self._robust_center_scale(rms)
        ll_center, ll_scale = self._robust_center_scale(ll)

        rms_z = (rms - rms_center) / rms_scale
        ll_z = (ll - ll_center) / ll_scale

        abs_data = np.abs(data)
        amplitude_center, amplitude_scale = self._robust_center_scale(abs_data)
        peak_height = amplitude_center + (self.peak_thres * amplitude_scale)
        baseline_level = amplitude_center + amplitude_scale

        peak_indexes, _ = signal.find_peaks(
            abs_data,
            height=peak_height,
            distance=min_gap_samples,
        )

        events = []
        for peak_index in peak_indexes.tolist():
            if rms_z[peak_index] < self.rms_thres and ll_z[peak_index] < self.ll_thres:
                continue
            start, end = self._event_bounds_from_peak(abs_data, peak_index, baseline_level, max_window_samples // 2)
            duration_samples = max(1, int(end - start))
            if duration_samples < min_window_samples or duration_samples > max_window_samples:
                continue
            events.append((int(start), int(end)))

        if not events:
            return np.empty((0, 2), dtype=int)

        merged = []
        for start, end in sorted(events):
            if not merged:
                merged.append([start, end])
                continue
            previous_start, previous_end = merged[-1]
            if start - previous_end <= min_gap_samples:
                merged[-1][1] = max(previous_end, end)
                continue
            merged.append([start, end])
        return np.asarray(merged, dtype=int)

    def detect_multi_channels(self, data, channel_names, filtered=True):
        data = np.asarray(data, dtype=float)
        channel_names = np.asarray(channel_names)
        event_channel_names = []
        event_intervals = []
        for channel_index, channel_name in enumerate(channel_names):
            intervals = self.detect_channel(data[channel_index])
            if len(intervals) == 0:
                continue
            event_channel_names.append(channel_name)
            event_intervals.append(intervals)
        return np.asarray(event_channel_names), event_intervals


def set_spike_rms_ll_detector(args):
    return SpikeRMSLLDetector(
        sample_freq=args.sample_freq,
        filter_freq=[args.pass_band, args.stop_band],
        rms_window=args.rms_window,
        ll_window=args.ll_window,
        rms_thres=args.rms_thres,
        ll_thres=args.ll_thres,
        peak_thres=args.peak_thres,
        min_window=args.min_window,
        max_window=args.max_window,
        min_gap=args.min_gap,
        n_jobs=args.n_jobs,
        front_num=1,
    )
