import numpy as np

from src.hfo_app import HFO_App
from src.param.param_detector import (
    ParamDetector,
    ParamHFOLineLength,
    ParamHFORMS,
    ParamSpindleA7,
    ParamSpindleRMS,
)
from src.param.param_filter import ParamFilter
from src.spindle_app import SpindleApp
from src.utils.utils_detector import HFOThresholdDetector, SpindleThresholdDetector


def _synthetic_hfo(sample_freq=1000, duration_seconds=2.0):
    times = np.arange(int(sample_freq * duration_seconds)) / sample_freq
    data = np.random.default_rng(1).normal(scale=0.03, size=times.shape)
    burst = (times >= 0.8) & (times <= 0.86)
    data[burst] += 1.6 * np.sin(2 * np.pi * 120 * times[burst])
    return data, burst


def _synthetic_spindle(sample_freq=200, duration_seconds=4.0):
    times = np.arange(int(sample_freq * duration_seconds)) / sample_freq
    data = np.random.default_rng(2).normal(scale=0.02, size=times.shape)
    envelope = np.exp(-((times - 2.0) / 0.35) ** 2)
    data += 0.8 * envelope * np.sin(2 * np.pi * 13 * times)
    return data


def test_local_hfo_rms_detector_finds_synthetic_burst():
    sample_freq = 1000
    data, _burst = _synthetic_hfo(sample_freq=sample_freq)
    detector = HFOThresholdDetector(
        sample_freq=sample_freq,
        filter_freq=[80, 250],
        metric="rms",
        metric_window=0.006,
        threshold=2.0,
        peak_threshold=2.0,
        min_window=0.01,
        max_window=0.2,
        min_gap=0.01,
    )

    events = detector.detect_channel(data, filtered=False)

    assert len(events) >= 1
    assert any(start <= 820 <= end for start, end in events.tolist())


def test_local_hfo_line_length_detector_finds_synthetic_burst():
    sample_freq = 1000
    data, _burst = _synthetic_hfo(sample_freq=sample_freq)
    detector = HFOThresholdDetector(
        sample_freq=sample_freq,
        filter_freq=[80, 250],
        metric="line_length",
        metric_window=0.01,
        threshold=2.0,
        peak_threshold=2.0,
        min_window=0.01,
        max_window=0.2,
        min_gap=0.01,
    )

    events = detector.detect_channel(data, filtered=False)

    assert len(events) >= 1
    assert any(start <= 820 <= end for start, end in events.tolist())


def test_hfo_app_runs_local_threshold_detector_without_new_dependency():
    sample_freq = 1000
    data, _burst = _synthetic_hfo(sample_freq=sample_freq)
    app = HFO_App()
    app.sample_freq = sample_freq
    app.eeg_data = data.reshape(1, -1)
    app.eeg_data_un60 = app.eeg_data.copy()
    app.filter_data = app.eeg_data.copy()
    app.filter_data_un60 = app.filter_data.copy()
    app.filtered = True
    app.channel_names = np.array(["A1"], dtype=object)
    app.recording_channel_names = app.channel_names.copy()
    app.edf_param = {"lowpass": 500, "edf_fn": "synthetic"}
    app.param_filter = ParamFilter(fp=80, fs=250, sample_freq=sample_freq)
    app.set_detector(
        ParamDetector(
            ParamHFORMS(
                sample_freq=sample_freq,
                pass_band=80,
                stop_band=250,
                rms_window=0.006,
                threshold=2.0,
                peak_threshold=2.0,
                min_window=0.01,
                max_window=0.2,
                min_gap=0.01,
                n_jobs=1,
            ),
            detector_type="RMS",
        )
    )

    app.detect_biomarker()

    assert app.detected is True
    assert app.event_features.HFO_type == "RMS"
    assert app.event_features.get_num_biomarker() >= 1


def test_spindle_a7_detector_finds_synthetic_spindle():
    sample_freq = 200
    data = _synthetic_spindle(sample_freq=sample_freq)
    detector = SpindleThresholdDetector(
        sample_freq=sample_freq,
        method="A7",
        freq_sp=(11, 16),
        freq_broad=(1, 30),
        duration=(0.3, 2.0),
        min_distance=0.2,
        smooth_window=0.2,
        rms_threshold=0.8,
        relative_power_threshold=0.05,
        correlation_threshold=0.1,
    )

    events = detector.detect_channel(data)

    assert len(events) >= 1
    assert any(start <= 400 <= end for start, end in events.tolist())


def test_spindle_app_runs_molle_style_threshold_detector():
    sample_freq = 200
    data = _synthetic_spindle(sample_freq=sample_freq)
    app = SpindleApp()
    app.sample_freq = sample_freq
    app.eeg_data = data.reshape(1, -1)
    app.eeg_data_un60 = app.eeg_data.copy()
    app.channel_names = np.array(["C3"], dtype=object)
    app.recording_channel_names = app.channel_names.copy()
    app.edf_param = {"lowpass": 100, "edf_fn": "synthetic"}
    app.set_detector(
        ParamDetector(
            ParamSpindleRMS(
                sample_freq=sample_freq,
                method="MOLLE",
                freq_sp=(12, 15),
                duration=(0.3, 2.0),
                min_distance=0.2,
                smooth_window=0.2,
                rms_threshold=0.8,
                n_jobs=1,
            ),
            detector_type="MOLLE",
        )
    )

    app.detect_biomarker()

    assert app.detected is True
    assert app.event_features.detector_type == "MOLLE"
    assert app.event_features.get_num_biomarker() >= 1


def test_param_detector_restores_new_threshold_detector_params():
    restored_rms = ParamDetector.from_dict(
        {
            "detector_type": "RMS",
            "detector_param": ParamHFORMS(sample_freq=1000, threshold=4.0).to_dict(),
        }
    )
    restored_ll = ParamDetector.from_dict(
        {
            "detector_type": "LineLength",
            "detector_param": ParamHFOLineLength(sample_freq=1000, threshold=4.0).to_dict(),
        }
    )
    restored_a7 = ParamDetector.from_dict(
        {
            "detector_type": "A7",
            "detector_param": ParamSpindleA7(sample_freq=200, rms_threshold=1.2).to_dict(),
        }
    )

    assert isinstance(restored_rms.detector_param, ParamHFORMS)
    assert isinstance(restored_ll.detector_param, ParamHFOLineLength)
    assert isinstance(restored_a7.detector_param, ParamSpindleA7)
    assert restored_a7.detector_param.rms_threshold == 1.2
