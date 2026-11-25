import numpy as np

from src.hfo_app import HFO_App
from src.hfo_feature import HFO_Feature
from src.param.param_classifier import ParamClassifier
from src.param.param_detector import ParamDetector, ParamSTE
from src.param.param_filter import ParamFilter
from src.utils.analysis_session import DetectionRun
from src.utils.protocol_presets import (
    load_protocol_preset,
    protocol_from_run,
    restore_protocol_params,
    save_protocol_preset,
)


def _build_run():
    param_filter = ParamFilter(fp=80, fs=500, sample_freq=2000)
    param_detector = ParamDetector(
        ParamSTE(2000, rms_thres=4.0, n_jobs=2),
        detector_type="STE",
    )
    param_classifier = ParamClassifier(
        artifact_path="/tmp/model_a.tar",
        artifact_card="roychowdhuryresearch/HFO-artifact",
        use_spike=False,
        use_ehfo=False,
        source_preference="huggingface",
    )
    return DetectionRun.create(
        biomarker_type="HFO",
        detector_name="STE",
        selected_channels=np.array(["A1"]),
        param_filter=param_filter,
        param_detector=param_detector,
        param_classifier=param_classifier,
        event_features=HFO_Feature(np.array(["A1"]), np.array([[10, 20]]), sample_freq=2000),
    )


def test_protocol_preset_round_trips_params(tmp_path):
    run = _build_run()
    preset = protocol_from_run(run, name="HFO STE Lab")
    path = save_protocol_preset(tmp_path / "hfo_ste.json", preset)

    restored = load_protocol_preset(path)
    params = restore_protocol_params(restored)

    assert restored["name"] == "HFO STE Lab"
    assert params["biomarker_type"] == "HFO"
    assert params["param_filter"].fp == 80
    assert params["param_detector"].detector_type == "STE"
    assert params["param_detector"].detector_param.rms_thres == 4.0
    assert params["param_classifier"].source_preference == "huggingface"


def test_backend_exports_and_applies_protocol_preset(tmp_path):
    run = _build_run()
    app = HFO_App()
    app.param_filter = run.param_filter
    app.param_detector = run.param_detector
    app.param_classifier = run.param_classifier
    app.analysis_session.add_run(run)

    preset_path = app.export_protocol_preset(tmp_path / "preset.json", name="Current HFO")
    target = HFO_App()
    params = target.apply_protocol_preset(preset_path)

    assert params["param_detector"].detector_type == "STE"
    assert target.param_filter.fp == 80
    assert target.param_classifier.artifact_card == "roychowdhuryresearch/HFO-artifact"
