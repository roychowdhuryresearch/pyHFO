import json

import numpy as np

from src.hfo_app import HFO_App
from src.hfo_feature import HFO_Feature
from src.param.param_detector import ParamDetector, ParamMNI, ParamSTE
from src.param.param_filter import ParamFilter
from src.utils.analysis_session import DetectionRun
from src.utils.reporting import export_analysis_report


def _build_hfo_run(detector_name, intervals, channel_names):
    detector_param = ParamSTE(2000).to_dict() if detector_name == "STE" else ParamMNI(2000).to_dict()
    return DetectionRun.create(
        biomarker_type="HFO",
        detector_name=detector_name,
        selected_channels=np.array(channel_names),
        param_filter=ParamFilter(),
        param_detector=ParamDetector.from_dict({"detector_type": detector_name, "detector_param": detector_param}),
        event_features=HFO_Feature(np.array(channel_names), np.array(intervals), np.array([]), detector_name, 2000),
        detector_output=np.array([intervals], dtype=object),
        classified=False,
    )


def _build_backend():
    backend = HFO_App()
    backend.edf_param = {
        "edf_fn": "/tmp/demo_case.edf",
        "sfreq": 2000,
        "nchan": 2,
        "highpass": 1,
        "lowpass": 500,
        "channels": ["A1Ref", "A2Ref"],
    }
    backend.sample_freq = 2000
    backend.channel_names = np.array(["A1Ref", "A2Ref"])
    backend.eeg_data = np.zeros((2, 4000))

    first_run = _build_hfo_run("STE", [[10, 20], [30, 40]], ["A1Ref", "A2Ref"])
    second_run = _build_hfo_run("MNI", [[10, 20], [80, 100]], ["A1Ref", "A1Ref"])
    backend.analysis_session.add_run(first_run)
    backend.analysis_session.add_run(second_run)
    backend.analysis_session.accept_run(first_run.run_id)
    return backend


def test_export_analysis_report_creates_shareable_bundle(tmp_path):
    backend = _build_backend()
    snapshot_path = tmp_path / "snapshot.png"
    snapshot_path.write_bytes(b"fake-png")

    report_path = export_analysis_report(
        tmp_path / "demo_report.html",
        backend,
        biomarker_label="HFO",
        snapshot_source_path=snapshot_path,
    )

    assets_dir = tmp_path / "demo_report_files"
    assert report_path.exists()
    assert (assets_dir / "clinical_summary.xlsx").exists()
    assert (assets_dir / "events.csv").exists()
    assert (assets_dir / "metadata.json").exists()
    assert (assets_dir / "waveform_snapshot.png").exists()

    report_html = report_path.read_text(encoding="utf-8")
    assert "Detector Agreement" in report_html
    assert "demo_case.edf" in report_html
    assert "Channel Ranking" in report_html

    metadata = json.loads((assets_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["selected_run"]["detector_name"] == "STE"
    assert metadata["summary"]["run_count"] == 2


def test_export_analysis_report_adds_html_suffix_when_missing(tmp_path):
    backend = _build_backend()

    report_path = export_analysis_report(tmp_path / "summary_report", backend, biomarker_label="HFO")

    assert report_path.suffix == ".html"
    assert report_path.exists()
