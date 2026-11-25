from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import loadmat


@dataclass
class KramerLSMParameters:
    window_duration: float
    step_duration: float
    theta_index: np.ndarray
    nine_15_index: np.ndarray
    mu: dict
    sigma: dict
    transition_matrix: np.ndarray

    def to_dict(self, *, name="", source=""):
        return {
            "schema_version": 1,
            "format": "pyhfo_kramer_lsm_preset",
            "name": name,
            "source": source,
            "index_base": 0,
            "params": {
                "window_duration": self.window_duration,
                "step_duration": self.step_duration,
                "theta_index": self.theta_index.astype(int).tolist(),
                "nine_15_index": self.nine_15_index.astype(int).tolist(),
            },
            "mu": dict(self.mu),
            "sigma": dict(self.sigma),
            "transition_matrix": np.asarray(self.transition_matrix, dtype=float).tolist(),
        }


def _mat_struct_to_dict(value):
    field_names = getattr(value, "_fieldnames", None)
    if not field_names:
        return {}
    return {field: getattr(value, field) for field in field_names}


def _matlab_indexes_to_python(indexes):
    values = np.atleast_1d(indexes).astype(int)
    return np.maximum(values - 1, 0)


def _build_parameters_from_payload(payload, *, index_base=1):
    missing = [key for key in ("params", "mu", "sigma", "transition_matrix") if key not in payload]
    if missing:
        raise ValueError(f"Kramer LSM parameter file is missing fields: {', '.join(missing)}")

    params = payload["params"] if isinstance(payload["params"], dict) else _mat_struct_to_dict(payload["params"])
    mu = _mat_struct_to_dict(payload["mu"])
    sigma = _mat_struct_to_dict(payload["sigma"])
    if isinstance(payload["mu"], dict):
        mu = payload["mu"]
    if isinstance(payload["sigma"], dict):
        sigma = payload["sigma"]
    transition_matrix = np.asarray(payload["transition_matrix"], dtype=float)
    if transition_matrix.shape != (2, 2):
        raise ValueError("Kramer LSM transition_matrix must be 2x2.")

    theta_index = np.atleast_1d(params.get("theta_index", 7)).astype(int)
    nine_15_index = np.atleast_1d(params.get("nine_15_index", [12, 14])).astype(int)
    if int(index_base) == 1:
        theta_index = np.maximum(theta_index - 1, 0)
        nine_15_index = np.maximum(nine_15_index - 1, 0)

    return KramerLSMParameters(
        window_duration=float(params.get("window_duration", 0.5)),
        step_duration=float(params.get("step_duration", 0.1)),
        theta_index=theta_index,
        nine_15_index=nine_15_index,
        mu={key: float(value) for key, value in mu.items()},
        sigma={key: float(value) for key, value in sigma.items()},
        transition_matrix=transition_matrix,
    )


def load_kramer_lsm_parameters(parameter_file):
    path = Path(str(parameter_file or "")).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Kramer LSM parameter file not found: {path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "pyhfo_kramer_lsm_preset":
            raise ValueError("Kramer LSM JSON preset has an unsupported format.")
        return _build_parameters_from_payload(payload, index_base=int(payload.get("index_base", 0)))

    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    return _build_parameters_from_payload(payload, index_base=1)


def load_kramer_lsm_parameters_from_dict(parameter_dict):
    payload = dict(parameter_dict or {})
    if payload.get("format") not in (None, "pyhfo_kramer_lsm_preset"):
        raise ValueError("Kramer LSM parameter dictionary has an unsupported format.")
    return _build_parameters_from_payload(payload, index_base=int(payload.get("index_base", 0)))


def export_kramer_lsm_json_preset(parameter_file, output_file, *, name="", source=""):
    params = load_kramer_lsm_parameters(parameter_file)
    output_path = Path(output_file).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(params.to_dict(name=name, source=source), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


class KramerLSMSpindleDetector:
    """Python implementation of the Kramer latent-state-model spindle detector."""

    def __init__(
        self,
        *,
        sample_freq,
        parameter_file,
        model_parameters=None,
        prob_threshold=0.95,
        min_spindle_duration=0.5,
        spindle_separation_threshold=1.0,
        min_peak_prominence=2e-6,
        start_frequency=None,
        stop_frequency=None,
        n_jobs=1,
    ):
        self.sample_freq = float(sample_freq)
        self.parameter_file = str(parameter_file or "")
        self.model_parameters = dict(model_parameters or {})
        self.prob_threshold = float(prob_threshold)
        self.min_spindle_duration = float(min_spindle_duration)
        self.spindle_separation_threshold = float(spindle_separation_threshold)
        self.min_peak_prominence = float(min_peak_prominence)
        self.start_frequency = None if start_frequency in (None, "") else float(start_frequency)
        self.stop_frequency = None if stop_frequency in (None, "") else float(stop_frequency)
        self.n_jobs = int(n_jobs)
        if self.model_parameters:
            self.params = load_kramer_lsm_parameters_from_dict(self.model_parameters)
        else:
            self.params = load_kramer_lsm_parameters(self.parameter_file)
        self.last_probabilities = []

        if self.sample_freq <= 0:
            raise ValueError("sample_freq must be positive.")
        if not 0 < self.prob_threshold <= 1:
            raise ValueError("prob_threshold must be in the interval (0, 1].")
        if self.start_frequency is None and self.stop_frequency is not None:
            raise ValueError("start_frequency must be set when stop_frequency is set.")
        if self.start_frequency is not None and self.stop_frequency is None:
            raise ValueError("stop_frequency must be set when start_frequency is set.")
        if self.start_frequency is not None and self.start_frequency >= self.stop_frequency:
            raise ValueError("start_frequency must be lower than stop_frequency.")

    def _normal_likelihood(self, value, mu, sigma):
        value = float(value)
        mu = float(mu)
        sigma = float(sigma)
        if not np.isfinite(value) or not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
            return 0.0
        z = (value - mu) / sigma
        return float(np.exp(-0.5 * z * z) / (np.sqrt(2 * np.pi) * sigma))

    def _feature_likelihood(self, name, value, state):
        suffix = "1" if state == 1 else "0"
        return self._normal_likelihood(
            value,
            self.params.mu[f"{name}{suffix}"],
            self.params.sigma[f"{name}{suffix}"],
        )

    def _narrowband_likelihood(self, frequency):
        if self.start_frequency is None or self.stop_frequency is None:
            return 1.0
        if not np.isfinite(frequency):
            return 0.0
        return 1.0 if self.start_frequency <= float(frequency) <= self.stop_frequency else 0.0

    def _bandpass_for_fano(self, data):
        nyquist = self.sample_freq / 2.0
        high = min(25.0, nyquist * 0.95)
        low = min(3.0, high * 0.5)
        if low <= 0 or high <= low:
            return np.asarray(data, dtype=float)
        sos = signal.butter(4, [low, high], btype="bandpass", fs=self.sample_freq, output="sos")
        cleaned = np.asarray(data, dtype=float)
        finite = np.isfinite(cleaned)
        if not np.all(finite):
            cleaned = cleaned.copy()
            cleaned[~finite] = 0.0
        try:
            return signal.sosfiltfilt(sos, cleaned)
        except ValueError:
            return signal.sosfilt(sos, cleaned)

    def _mean_power_at_indexes(self, power, indexes):
        indexes = np.asarray(indexes, dtype=int)
        indexes = indexes[(indexes >= 0) & (indexes < len(power))]
        if indexes.size == 0:
            return np.nan
        return float(np.mean(power[indexes]))

    def _window_features(self, raw_window, fano_window):
        raw_window = signal.detrend(np.asarray(raw_window, dtype=float))
        fft_length = max(int(round(self.sample_freq)), len(raw_window))
        windowed = signal.windows.hann(len(raw_window), sym=False) * raw_window
        power = np.abs(np.fft.fft(windowed, n=fft_length))
        scale = float(np.sum(power[1:min(51, len(power))]))
        if np.isfinite(scale) and scale > 0:
            power = power / scale

        pow_theta = self._mean_power_at_indexes(power, self.params.theta_index)
        pow_9_15 = self._mean_power_at_indexes(power, self.params.nine_15_index)

        fano_window = signal.detrend(np.asarray(fano_window, dtype=float))
        min_peak_distance = max(1, int(np.ceil((1.0 / 35.0) * self.sample_freq)))
        pos_locs, _ = signal.find_peaks(
            fano_window,
            distance=min_peak_distance,
            prominence=self.min_peak_prominence,
        )
        neg_locs, _ = signal.find_peaks(
            -fano_window,
            distance=min_peak_distance,
            prominence=self.min_peak_prominence,
        )
        if len(pos_locs) > 1 and len(neg_locs) > 1:
            ipi = np.concatenate([np.diff(pos_locs), np.diff(neg_locs)]) / self.sample_freq * 1000.0
        else:
            ipi = np.asarray([np.nan])

        if ipi.size > 1 and np.isfinite(np.nanmean(ipi)) and np.nanmean(ipi) > 0:
            mean_ipi = float(np.nanmean(ipi))
            fano = float(np.nanvar(ipi) / mean_ipi)
            instant_freq = float(1.0 / (mean_ipi / 1000.0))
        else:
            fano = np.nan
            instant_freq = np.nan
        return pow_theta, pow_9_15, fano, instant_freq

    def compute_channel_probabilities(self, channel_data):
        data = np.asarray(channel_data, dtype=float)
        if data.size == 0 or not np.any(np.isfinite(data)):
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        filtered = self._bandpass_for_fano(data)
        window_size = max(1, int(round(self.params.window_duration * self.sample_freq)))
        step_size = max(1, int(round(self.params.step_duration * self.sample_freq)))
        if data.size < window_size:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        probabilities = []
        times = []
        posterior_state = np.asarray([0.5, 0.5], dtype=float)
        for start in range(0, data.size - window_size + 1, step_size):
            stop = start + window_size
            pow_theta, pow_9_15, fano, instant_freq = self._window_features(
                data[start:stop],
                filtered[start:stop],
            )

            prior = self.params.transition_matrix @ posterior_state
            prior_sum = float(np.sum(prior))
            if not np.isfinite(prior_sum) or prior_sum <= 0:
                prior = np.asarray([0.5, 0.5], dtype=float)
            else:
                prior = prior / prior_sum

            log_pow_theta = np.log(pow_theta) if pow_theta > 0 else np.nan
            log_pow_9_15 = np.log(pow_9_15) if pow_9_15 > 0 else np.nan
            spindle_like = (
                self._feature_likelihood("log_P_9_15_", log_pow_9_15, 1)
                * self._feature_likelihood("log_P_theta", log_pow_theta, 1)
                * self._narrowband_likelihood(instant_freq)
            )
            non_spindle_like = (
                self._feature_likelihood("log_P_9_15_", log_pow_9_15, 0)
                * self._feature_likelihood("log_P_theta", log_pow_theta, 0)
            )
            if np.isfinite(fano) and fano > 0:
                log_fano = np.log(fano)
                spindle_like *= self._feature_likelihood("F", log_fano, 1)
                non_spindle_like *= self._feature_likelihood("F", log_fano, 0)

            posterior = np.asarray([spindle_like * prior[0], non_spindle_like * prior[1]], dtype=float)
            posterior_sum = float(np.sum(posterior))
            if not np.isfinite(posterior_sum) or posterior_sum <= 0:
                posterior_state = np.asarray([0.5, 0.5], dtype=float)
            else:
                posterior_state = posterior / posterior_sum
            probabilities.append(float(posterior_state[0]))
            times.append(float(start / self.sample_freq))

        return np.asarray(probabilities, dtype=float), np.asarray(times, dtype=float)

    def compute_probabilities(self, data, channel_names):
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        channel_names = np.atleast_1d(np.asarray(channel_names, dtype=object))
        probabilities = []
        for channel_index, channel_name in enumerate(channel_names):
            if channel_index >= data.shape[0]:
                break
            prob, times = self.compute_channel_probabilities(data[channel_index])
            probabilities.append(
                {
                    "label": str(channel_name),
                    "prob": prob,
                    "t": times,
                    "Fs": self.sample_freq,
                    "params": {"step_duration": self.params.step_duration},
                }
            )
        self.last_probabilities = probabilities
        return probabilities

    def detections_from_probabilities(self, spindle_probabilities):
        detections_by_channel = []
        for row in spindle_probabilities:
            prob = np.asarray(row["prob"], dtype=float)
            times = np.asarray(row["t"], dtype=float)
            if prob.size == 0 or times.size == 0:
                detections_by_channel.append(
                    {"label": row["label"], "intervals": np.empty((0, 2), dtype=int), "Fs": row["Fs"]}
                )
                continue

            detections = (prob >= self.prob_threshold).astype(int)
            if not np.any(detections):
                start_times = np.asarray([], dtype=float)
                end_times = np.asarray([], dtype=float)
            elif np.all(detections):
                start_times = np.asarray([times[0]], dtype=float)
                end_times = np.asarray([times[-1]], dtype=float)
            else:
                detections[0] = 0
                detections[-1] = 0
                starts = np.where(np.diff(detections) == 1)[0] + 1
                ends = np.where(np.diff(detections) == -1)[0] + 1
                start_times = times[starts]
                end_times = times[ends]

            if start_times.size:
                durations = end_times - start_times
                keep = durations >= self.min_spindle_duration
                start_times = start_times[keep]
                end_times = end_times[keep]

            if start_times.size > 1:
                merged_starts = [float(start_times[0])]
                merged_ends = [float(end_times[0])]
                for start, end in zip(start_times[1:], end_times[1:]):
                    if start - merged_ends[-1] < self.spindle_separation_threshold:
                        merged_ends[-1] = float(end)
                    else:
                        merged_starts.append(float(start))
                        merged_ends.append(float(end))
                start_times = np.asarray(merged_starts, dtype=float)
                end_times = np.asarray(merged_ends, dtype=float)

            if end_times.size:
                end_times = end_times + self.params.step_duration
            intervals = np.column_stack(
                [
                    np.rint(start_times * self.sample_freq).astype(int),
                    np.rint(end_times * self.sample_freq).astype(int),
                ]
            ) if start_times.size else np.empty((0, 2), dtype=int)
            detections_by_channel.append({"label": row["label"], "intervals": intervals, "Fs": row["Fs"]})
        return detections_by_channel

    def detect_multi_channels(self, data, channel_names, filtered=False):
        spindle_probabilities = self.compute_probabilities(data, channel_names)
        detections = self.detections_from_probabilities(spindle_probabilities)
        event_channel_names = []
        event_intervals = []
        for row in detections:
            intervals = np.asarray(row["intervals"], dtype=int)
            if intervals.size == 0:
                continue
            event_channel_names.append(row["label"])
            event_intervals.append(intervals.reshape(-1, 2))
        return np.asarray(event_channel_names, dtype=object), event_intervals
