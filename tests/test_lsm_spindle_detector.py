import numpy as np
from scipy.io import savemat

from src.param.param_detector import ParamDetector, ParamSpindleLSM
from src.spindle_app import SpindleApp
from src.utils.kramer_lsm_spindle import (
    KramerLSMSpindleDetector,
    export_kramer_lsm_json_preset,
    load_kramer_lsm_parameters,
)


def _write_lsm_parameter_file(path):
    feature_names = (
        "log_P_theta1",
        "log_P_theta0",
        "log_P_9_15_1",
        "log_P_9_15_0",
        "F1",
        "F0",
    )
    mu = {name: 0.0 for name in feature_names}
    sigma = {
        "log_P_theta1": 1e6,
        "log_P_theta0": 1e18,
        "log_P_9_15_1": 1e6,
        "log_P_9_15_0": 1e18,
        "F1": 1e6,
        "F0": 1e18,
    }
    savemat(
        path,
        {
            "params": {
                "window_duration": 0.5,
                "step_duration": 0.1,
                "theta_index": 7,
                "nine_15_index": np.array([12, 14], dtype=np.uint8),
            },
            "mu": mu,
            "sigma": sigma,
            "transition_matrix": np.array([[0.99, 0.01], [0.01, 0.99]], dtype=float),
        },
    )
    return path


def test_lsm_parameter_roundtrip_loads_kramer_mat_shape(tmp_path):
    parameter_file = _write_lsm_parameter_file(tmp_path / "lsm_params.mat")

    params = load_kramer_lsm_parameters(parameter_file)

    assert params.window_duration == 0.5
    assert params.step_duration == 0.1
    assert params.theta_index.tolist() == [6]
    assert params.nine_15_index.tolist() == [11, 13]
    assert params.transition_matrix.shape == (2, 2)


def test_lsm_parameter_export_loads_expanded_json(tmp_path):
    parameter_file = _write_lsm_parameter_file(tmp_path / "lsm_params.mat")
    json_file = export_kramer_lsm_json_preset(parameter_file, tmp_path / "lsm_params.json", name="demo")

    params = load_kramer_lsm_parameters(json_file)

    assert params.theta_index.tolist() == [6]
    assert params.nine_15_index.tolist() == [11, 13]


def test_lsm_detector_converts_probabilities_to_merged_intervals(tmp_path):
    parameter_file = _write_lsm_parameter_file(tmp_path / "lsm_params.mat")
    detector = KramerLSMSpindleDetector(
        sample_freq=200,
        parameter_file=str(parameter_file),
        prob_threshold=0.8,
        min_spindle_duration=0.15,
        spindle_separation_threshold=0.5,
        min_peak_prominence=0.01,
    )

    probabilities = [
        {
            "label": "C3",
            "prob": np.array([0.1, 0.9, 0.91, 0.1, 0.92, 0.93, 0.1]),
            "t": np.arange(7) * 0.1,
            "Fs": 200,
            "params": {"step_duration": 0.1},
        }
    ]

    detections = detector.detections_from_probabilities(probabilities)

    assert detections[0]["label"] == "C3"
    assert detections[0]["intervals"].tolist() == [[20, 140]]


def test_spindle_app_runs_python_lsm_detector(tmp_path):
    sample_freq = 200
    duration_seconds = 4.0
    times = np.arange(int(sample_freq * duration_seconds)) / sample_freq
    data = 0.02 * np.random.default_rng(2).normal(size=(2, times.size))
    data[0] += 0.2 * np.sin(2 * np.pi * 13 * times)
    parameter_file = _write_lsm_parameter_file(tmp_path / "lsm_params.mat")

    app = SpindleApp()
    app.sample_freq = sample_freq
    app.eeg_data = data
    app.eeg_data_un60 = data.copy()
    app.channel_names = np.array(["C3", "C4"], dtype=object)
    app.recording_channel_names = app.channel_names.copy()
    app.edf_param = {"lowpass": 100, "edf_fn": "synthetic"}
    app.set_detector(
        ParamDetector(
            ParamSpindleLSM(
                sample_freq=sample_freq,
                parameter_file=str(parameter_file),
                prob_threshold=0.6,
                min_spindle_duration=0.3,
                spindle_separation_threshold=0.5,
                min_peak_prominence=0.01,
                n_jobs=1,
            ),
            detector_type="LSM",
        )
    )

    app.detect_biomarker()

    assert app.detected is True
    assert app.event_features.detector_type == "LSM"
    assert app.event_features.get_num_biomarker() >= 1
    assert "C3" in app.event_features.channel_names.tolist()


def test_lsm_detector_accepts_single_channel_arrays_and_scalar_labels(tmp_path):
    sample_freq = 200
    duration_seconds = 3.0
    times = np.arange(int(sample_freq * duration_seconds)) / sample_freq
    data = 0.2 * np.sin(2 * np.pi * 13 * times)
    parameter_file = _write_lsm_parameter_file(tmp_path / "lsm_params.mat")
    detector = KramerLSMSpindleDetector(
        sample_freq=sample_freq,
        parameter_file=str(parameter_file),
        prob_threshold=0.6,
        min_spindle_duration=0.3,
        spindle_separation_threshold=0.5,
        min_peak_prominence=0.01,
    )

    channels, intervals = detector.detect_multi_channels(data, "C3")

    assert channels.tolist() == ["C3"]
    assert len(intervals) == 1
    assert len(intervals[0]) >= 1


def test_lsm_detector_accepts_manual_model_parameters_without_file(tmp_path):
    sample_freq = 200
    duration_seconds = 3.0
    times = np.arange(int(sample_freq * duration_seconds)) / sample_freq
    data = 0.2 * np.sin(2 * np.pi * 13 * times)
    parameter_file = _write_lsm_parameter_file(tmp_path / "lsm_params.mat")
    model_parameters = load_kramer_lsm_parameters(parameter_file).to_dict()
    detector = KramerLSMSpindleDetector(
        sample_freq=sample_freq,
        parameter_file="",
        model_parameters=model_parameters,
        prob_threshold=0.6,
        min_spindle_duration=0.3,
        spindle_separation_threshold=0.5,
        min_peak_prominence=0.01,
    )

    channels, intervals = detector.detect_multi_channels(data, "C3")

    assert channels.tolist() == ["C3"]
    assert len(intervals[0]) >= 1


def test_param_detector_restores_lsm_parameters():
    restored = ParamDetector.from_dict(
        {
            "detector_type": "LSM",
            "detector_param": {
                "sample_freq": 200,
                "parameter_file": "/tmp/params.mat",
                "prob_threshold": 0.9,
                "min_spindle_duration": 0.4,
                "spindle_separation_threshold": 0.8,
                "min_peak_prominence": 1e-6,
                "start_frequency": 11,
                "stop_frequency": 16,
                "n_jobs": 2,
            },
        }
    )

    assert isinstance(restored.detector_param, ParamSpindleLSM)
    assert restored.detector_param.prob_threshold == 0.9
    assert restored.detector_param.start_frequency == 11
