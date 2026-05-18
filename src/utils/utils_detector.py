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
from src.utils.kramer_lsm_spindle import KramerLSMSpindleDetector


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


class LocalThresholdDetectorMixin:
    def _window_samples(self, seconds, *, minimum=1):
        return max(int(round(float(seconds) * self.sample_freq)), int(minimum))

    def _coerce_multi_channel_input(self, data, channel_names):
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        channel_names = np.asarray(channel_names if np.ndim(channel_names) > 0 else [channel_names], dtype=object)
        if channel_names.size != data.shape[0]:
            channel_names = np.asarray([f"Ch{index + 1}" for index in range(data.shape[0])], dtype=object)
        return data, channel_names

    def _robust_center_scale(self, values):
        values = np.asarray(values, dtype=float)
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

    def _moving_average(self, values, window_samples):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return np.asarray([], dtype=float)
        window_samples = max(1, int(window_samples))
        kernel = np.ones(window_samples, dtype=float) / float(window_samples)
        return np.convolve(values, kernel, mode="same")

    def _moving_rms(self, values, window_samples):
        return np.sqrt(np.maximum(self._moving_average(np.square(np.asarray(values, dtype=float)), window_samples), 0.0))

    def _moving_line_length(self, values, window_samples):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return np.asarray([], dtype=float)
        diffs = np.abs(np.diff(values, prepend=float(values[0])))
        return self._moving_average(diffs, window_samples)

    def _mask_to_intervals(self, mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.size == 0 or not np.any(mask):
            return np.empty((0, 2), dtype=int)
        padded = np.concatenate([[False], mask, [False]])
        changes = np.diff(padded.astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        return np.column_stack([starts, ends]).astype(int)

    def _merge_close_intervals(self, intervals, gap_samples):
        intervals = np.asarray(intervals, dtype=int)
        if intervals.size == 0:
            return np.empty((0, 2), dtype=int)
        merged = []
        for start, end in sorted(intervals.tolist()):
            if not merged or start - merged[-1][1] > gap_samples:
                merged.append([int(start), int(end)])
            else:
                merged[-1][1] = max(int(end), merged[-1][1])
        return np.asarray(merged, dtype=int)

    def _filter_by_duration(self, intervals, min_samples, max_samples=None):
        intervals = np.asarray(intervals, dtype=int)
        if intervals.size == 0:
            return np.empty((0, 2), dtype=int)
        durations = intervals[:, 1] - intervals[:, 0]
        keep = durations >= int(min_samples)
        if max_samples is not None:
            keep &= durations <= int(max_samples)
        return intervals[keep]

    def _bandpass_filter(self, values, freq_range):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return values
        finite = np.isfinite(values)
        if not finite.all():
            fill_value = float(np.median(values[finite])) if finite.any() else 0.0
            values = np.where(finite, values, fill_value)
        low, high = [float(value) for value in freq_range]
        nyquist = self.sample_freq / 2.0
        high = min(high, nyquist * 0.99)
        low = max(low, 0.01)
        if low >= high or nyquist <= 0:
            return values
        sos = signal.butter(4, [low, high], btype="bandpass", fs=self.sample_freq, output="sos")
        try:
            return signal.sosfiltfilt(sos, values)
        except ValueError:
            return signal.sosfilt(sos, values)


class HFOThresholdDetector(LocalThresholdDetectorMixin):
    def __init__(
        self,
        *,
        sample_freq,
        filter_freq,
        metric,
        metric_window,
        threshold,
        peak_threshold,
        min_window,
        max_window,
        min_gap,
        n_jobs=1,
        front_num=1,
    ):
        self.sample_freq = float(sample_freq)
        self.filter_freq = list(filter_freq)
        self.metric = str(metric)
        self.metric_window = float(metric_window)
        self.threshold = float(threshold)
        self.peak_threshold = float(peak_threshold)
        self.min_window = float(min_window)
        self.max_window = float(max_window)
        self.min_gap = float(min_gap)
        self.n_jobs = int(n_jobs)
        self.front_num = int(front_num)

    def _metric_values(self, data):
        window_samples = self._window_samples(self.metric_window)
        if self.metric == "line_length":
            return self._moving_line_length(data, window_samples)
        return self._moving_rms(data, window_samples)

    def detect_channel(self, channel_data, *, filtered=True):
        data = np.asarray(channel_data, dtype=float)
        if data.size == 0 or not np.any(np.isfinite(data)):
            return np.empty((0, 2), dtype=int)
        work_data = data if filtered else self._bandpass_filter(data, self.filter_freq)
        metric_values = self._metric_values(work_data)
        metric_center, metric_scale = self._robust_center_scale(metric_values)
        metric_z = (metric_values - metric_center) / metric_scale
        mask = metric_z >= self.threshold

        intervals = self._mask_to_intervals(mask)
        min_samples = self._window_samples(self.min_window, minimum=2)
        max_samples = max(min_samples, self._window_samples(self.max_window, minimum=2))
        gap_samples = self._window_samples(self.min_gap, minimum=1)
        intervals = self._merge_close_intervals(intervals, gap_samples)
        intervals = self._filter_by_duration(intervals, min_samples, max_samples)
        if intervals.size == 0:
            return intervals

        abs_data = np.abs(work_data)
        peak_center, peak_scale = self._robust_center_scale(abs_data)
        peak_level = peak_center + (self.peak_threshold * peak_scale)
        peak_filtered = []
        for start, end in intervals.tolist():
            if np.max(abs_data[start:end]) >= peak_level:
                peak_filtered.append([start, end])
        if not peak_filtered:
            return np.empty((0, 2), dtype=int)
        return np.asarray(peak_filtered, dtype=int)

    def detect_multi_channels(self, data, channel_names, filtered=True):
        data, channel_names = self._coerce_multi_channel_input(data, channel_names)
        event_channel_names = []
        event_intervals = []
        for channel_index, channel_name in enumerate(channel_names):
            intervals = self.detect_channel(data[channel_index], filtered=filtered)
            if len(intervals) == 0:
                continue
            event_channel_names.append(channel_name)
            event_intervals.append(intervals)
        return np.asarray(event_channel_names, dtype=object), event_intervals


def set_HFO_RMS_detector(args):
    return HFOThresholdDetector(
        sample_freq=args.sample_freq,
        filter_freq=[args.pass_band, args.stop_band],
        metric="rms",
        metric_window=args.rms_window,
        threshold=args.threshold,
        peak_threshold=args.peak_threshold,
        min_window=args.min_window,
        max_window=args.max_window,
        min_gap=args.min_gap,
        n_jobs=args.n_jobs,
        front_num=1,
    )


def set_HFO_line_length_detector(args):
    return HFOThresholdDetector(
        sample_freq=args.sample_freq,
        filter_freq=[args.pass_band, args.stop_band],
        metric="line_length",
        metric_window=args.ll_window,
        threshold=args.threshold,
        peak_threshold=args.peak_threshold,
        min_window=args.min_window,
        max_window=args.max_window,
        min_gap=args.min_gap,
        n_jobs=args.n_jobs,
        front_num=1,
    )


def set_YASA_detector(args):
    if yasa is None:
        raise ImportError("YASA is required for spindle detection but is not installed.")
    detector = yasa
    return {'yasa': detector, 'args': args}


def set_LSM_spindle_detector(args):
    return KramerLSMSpindleDetector(
        sample_freq=args.sample_freq,
        parameter_file=args.parameter_file,
        model_parameters=getattr(args, "model_parameters", {}),
        prob_threshold=args.prob_threshold,
        min_spindle_duration=args.min_spindle_duration,
        spindle_separation_threshold=args.spindle_separation_threshold,
        min_peak_prominence=args.min_peak_prominence,
        start_frequency=args.start_frequency,
        stop_frequency=args.stop_frequency,
        n_jobs=args.n_jobs,
    )


class SpindleThresholdDetector(LocalThresholdDetectorMixin):
    def __init__(
        self,
        *,
        sample_freq,
        method,
        freq_sp,
        duration,
        min_distance,
        smooth_window,
        rms_threshold,
        freq_broad=None,
        relative_power_threshold=0.0,
        correlation_threshold=0.0,
        n_jobs=1,
    ):
        self.sample_freq = float(sample_freq)
        self.method = str(method).upper()
        self.freq_sp = tuple(float(value) for value in freq_sp)
        self.freq_broad = tuple(float(value) for value in (freq_broad or (1, 30)))
        self.duration = tuple(float(value) for value in duration)
        self.min_distance = float(min_distance)
        self.smooth_window = float(smooth_window)
        self.rms_threshold = float(rms_threshold)
        self.relative_power_threshold = float(relative_power_threshold)
        self.correlation_threshold = float(correlation_threshold)
        self.n_jobs = int(n_jobs)

    def _rolling_correlation(self, x, y, window_samples):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mean_x = self._moving_average(x, window_samples)
        mean_y = self._moving_average(y, window_samples)
        mean_xy = self._moving_average(x * y, window_samples)
        mean_x2 = self._moving_average(x * x, window_samples)
        mean_y2 = self._moving_average(y * y, window_samples)
        numerator = mean_xy - (mean_x * mean_y)
        denominator = np.sqrt(np.maximum(mean_x2 - mean_x * mean_x, 0.0) * np.maximum(mean_y2 - mean_y * mean_y, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = numerator / denominator
        return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    def detect_channel(self, channel_data):
        data = np.asarray(channel_data, dtype=float)
        if data.size == 0 or not np.any(np.isfinite(data)):
            return np.empty((0, 2), dtype=int)

        window_samples = self._window_samples(self.smooth_window, minimum=2)
        sigma_data = self._bandpass_filter(data, self.freq_sp)
        sigma_rms = self._moving_rms(sigma_data, window_samples)
        center, scale = self._robust_center_scale(sigma_rms)
        rms_z = (sigma_rms - center) / scale
        mask = rms_z >= self.rms_threshold

        if self.method == "A7":
            broad_data = self._bandpass_filter(data, self.freq_broad)
            sigma_power = self._moving_average(np.square(sigma_data), window_samples)
            broad_power = self._moving_average(np.square(broad_data), window_samples)
            relative_power = sigma_power / np.maximum(broad_power, 1e-20)
            correlation = self._rolling_correlation(broad_data, sigma_data, window_samples)
            mask &= relative_power >= self.relative_power_threshold
            mask &= correlation >= self.correlation_threshold

        intervals = self._mask_to_intervals(mask)
        min_samples = self._window_samples(self.duration[0], minimum=2)
        max_samples = max(min_samples, self._window_samples(self.duration[1], minimum=2))
        gap_samples = self._window_samples(self.min_distance, minimum=1)
        intervals = self._merge_close_intervals(intervals, gap_samples)
        return self._filter_by_duration(intervals, min_samples, max_samples)

    def detect_multi_channels(self, data, channel_names, filtered=False):
        data, channel_names = self._coerce_multi_channel_input(data, channel_names)
        event_channel_names = []
        event_intervals = []
        for channel_index, channel_name in enumerate(channel_names):
            intervals = self.detect_channel(data[channel_index])
            if len(intervals) == 0:
                continue
            event_channel_names.append(channel_name)
            event_intervals.append(intervals)
        return np.asarray(event_channel_names, dtype=object), event_intervals


def set_A7_spindle_detector(args):
    return SpindleThresholdDetector(
        sample_freq=args.sample_freq,
        method="A7",
        freq_sp=args.freq_sp,
        freq_broad=args.freq_broad,
        duration=args.duration,
        min_distance=args.min_distance,
        smooth_window=args.smooth_window,
        rms_threshold=args.rms_threshold,
        relative_power_threshold=args.relative_power_threshold,
        correlation_threshold=args.correlation_threshold,
        n_jobs=args.n_jobs,
    )


def set_spindle_rms_detector(args):
    return SpindleThresholdDetector(
        sample_freq=args.sample_freq,
        method=getattr(args, "method", "MOLLE"),
        freq_sp=args.freq_sp,
        duration=args.duration,
        min_distance=args.min_distance,
        smooth_window=args.smooth_window,
        rms_threshold=args.rms_threshold,
        n_jobs=args.n_jobs,
    )


class SpikeRMSLLDetector(LocalThresholdDetectorMixin):
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
