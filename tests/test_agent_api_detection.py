import json
from pathlib import Path

import numpy as np

from src.agent_api import run_detection


class _FakeFeature:
    channel_names = np.array(["A1", "A2", "A1"])
    starts = np.array([1, 2, 3])
    ends = np.array([4, 5, 6])

    def get_num_biomarker(self):
        return 3

    def export_excel(self, path):
        Path(path).write_text("excel", encoding="utf-8")

    def export_csv(self, path):
        Path(path).write_text("csv", encoding="utf-8")


class _FakeSession:
    active_run_id = "run-1"


class _FakeBackend:
    def __init__(self):
        self.sample_freq = 2000
        self.event_features = _FakeFeature()
        self.analysis_session = _FakeSession()
        self.calls = []
        self.edf_param = {"sfreq": 2000, "nchan": 2}

    def set_n_jobs(self, n_jobs):
        self.calls.append(("set_n_jobs", n_jobs))

    def load_edf(self, path):
        self.calls.append(("load_edf", path))

    def set_filter_parameter(self, param):
        self.calls.append(("set_filter_parameter", param.fp, param.fs))

    def filter_eeg_data(self):
        self.calls.append(("filter_eeg_data",))

    def set_detector(self, param):
        self.calls.append(("set_detector", param.detector_type))

    def detect_biomarker(self):
        self.calls.append(("detect_biomarker",))

    def export_excel(self, path):
        self.event_features.export_excel(path)

    def export_report(self, path):
        self.event_features.export_csv(path)

    def export_app(self, path):
        Path(path).write_text("session", encoding="utf-8")

    def export_clinical_summary(self, path):
        Path(path).write_text("clinical", encoding="utf-8")

    def get_run_summaries(self):
        return [{"run_id": "run-1", "num_events": 3}]

    def get_decision_summary(self):
        return {"active_run_id": "run-1"}


def test_run_detection_returns_structured_result_and_artifacts(monkeypatch, tmp_path):
    input_path = tmp_path / "input.edf"
    input_path.write_text("fake", encoding="utf-8")
    backend = _FakeBackend()

    monkeypatch.setattr("src.agent_api.pipeline._backend_for_biomarker", lambda _biomarker: backend)

    result = run_detection(
        {
            "input_path": str(input_path),
            "output_dir": str(tmp_path / "out"),
            "biomarker": "HFO",
            "detector": "STE",
            "filter": {"fp": 80, "fs": 500},
            "exports": ["excel", "csv", "session", "clinical-summary", "summary-json"],
            "n_jobs": 2,
        }
    )

    assert result["status"] == "complete"
    assert result["num_events"] == 3
    assert result["num_channels"] == 2
    assert result["active_run_id"] == "run-1"
    assert set(result["artifacts"]) == {"excel", "csv", "session", "clinical_summary", "summary_json"}
    for artifact_path in result["artifacts"].values():
        assert Path(artifact_path).exists()
    assert [call[0] for call in backend.calls] == [
        "set_n_jobs",
        "load_edf",
        "set_filter_parameter",
        "filter_eeg_data",
        "set_detector",
        "detect_biomarker",
    ]
    json.dumps(result)


def test_run_detection_can_return_failed_result(monkeypatch, tmp_path):
    input_path = tmp_path / "input.edf"
    input_path.write_text("fake", encoding="utf-8")

    class BrokenBackend(_FakeBackend):
        def detect_biomarker(self):
            raise RuntimeError("boom")

    monkeypatch.setattr("src.agent_api.pipeline._backend_for_biomarker", lambda _biomarker: BrokenBackend())

    result = run_detection(
        {
            "input_path": str(input_path),
            "output_dir": str(tmp_path / "out"),
            "biomarker": "HFO",
            "detector": "STE",
            "fail_on_error": False,
        }
    )

    assert result["status"] == "failed"
    assert result["error"]["code"] == "RuntimeError"
    assert Path(result["output_dir"], "summary.json").exists()
