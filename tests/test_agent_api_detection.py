import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pyhfo.agent_api import classify_existing_run, run_detection
from pyhfo.hfo_app import HFO_App
from pyhfo.utils.default_onnx_models import default_hf_onnx_sources


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

    def set_classifier(self, param):
        self.calls.append(("set_classifier", param.device, param.batch_size, param.model_type))

    def classify_artifacts(self, ignore_region=None, threshold=0.5):
        self.calls.append(("classify_artifacts", ignore_region, threshold))

    def classify_spikes(self):
        self.calls.append(("classify_spikes",))

    def classify_ehfos(self):
        self.calls.append(("classify_ehfos",))

    def export_excel(self, path):
        self.event_features.export_excel(path)

    def export_report(self, path):
        self.event_features.export_csv(path)

    def export_app(self, path):
        Path(path).write_text("session", encoding="utf-8")

    def export_clinical_summary(self, path):
        Path(path).write_text("clinical", encoding="utf-8")

    def activate_run(self, run_id):
        self.calls.append(("activate_run", run_id))
        self.analysis_session.active_run_id = run_id

    def get_run_summaries(self):
        return [{"run_id": "run-1", "num_events": 3}]

    def get_decision_summary(self):
        return {"active_run_id": "run-1"}


class _FakeRaw:
    def __init__(self):
        self.ch_names = ["A1", "A2", "EKG1", "X1"]
        self.picked = None

    def pick_channels(self, selected, ordered=True):
        self.picked = list(selected)
        self.ch_names = list(selected)


class _ChannelSelectionBackend(_FakeBackend):
    def __init__(self):
        super().__init__()
        self.raw = _FakeRaw()


def test_run_detection_returns_structured_result_and_artifacts(monkeypatch, tmp_path):
    input_path = tmp_path / "input.edf"
    input_path.write_text("fake", encoding="utf-8")
    backend = _FakeBackend()

    monkeypatch.setattr("pyhfo.agent_api.pipeline._backend_for_biomarker", lambda _biomarker: backend)

    result = run_detection(
        {
            "input_path": str(input_path),
            "output_dir": str(tmp_path / "out"),
            "biomarker": "HFO",
            "detector": "STE",
            "filter": {"fp": 80, "fs": 500},
            "classifier": {"enabled": False},
            "exports": ["excel", "csv", "session", "clinical-summary", "summary-json"],
            "n_jobs": 2,
        }
    )

    assert result["status"] == "complete"
    assert result["num_events"] == 3
    assert result["num_channels"] == 2
    assert result["active_run_id"] == "run-1"
    assert result["classifier_metadata"] == {"enabled": False}
    assert result["recording_fingerprint"] == hashlib.sha256(input_path.read_bytes()).hexdigest()
    assert len(result["parameters_sha256"]) == 64
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


def test_run_detection_quick_detection_can_request_legacy_npz_session(monkeypatch, tmp_path):
    input_path = tmp_path / "input.edf"
    input_path.write_text("fake", encoding="utf-8")
    backend = _FakeBackend()

    monkeypatch.setattr("pyhfo.agent_api.pipeline._backend_for_biomarker", lambda _biomarker: backend)

    result = run_detection(
        {
            "input_path": str(input_path),
            "output_dir": str(tmp_path / "out"),
            "biomarker": "HFO",
            "detector": "STE",
            "exports": ["session", "summary-json"],
            "session_format": "npz",
        }
    )

    assert result["status"] == "complete"
    assert result["artifacts"]["session"].endswith("session.npz")
    assert Path(result["artifacts"]["session"]).exists()
    summary = json.loads(Path(result["artifacts"]["summary_json"]).read_text(encoding="utf-8"))
    assert summary["request"]["session_format"] == "npz"


def test_run_detection_honors_include_and_exclude_channels(monkeypatch, tmp_path):
    input_path = tmp_path / "input.edf"
    input_path.write_text("fake", encoding="utf-8")
    backend = _ChannelSelectionBackend()

    monkeypatch.setattr("pyhfo.agent_api.pipeline._backend_for_biomarker", lambda _biomarker: backend)
    monkeypatch.setattr("pyhfo.agent_api.pipeline.get_edf_info", lambda raw: {"sfreq": 2000, "nchan": len(raw.ch_names)})
    monkeypatch.setattr(
        "pyhfo.agent_api.pipeline.read_eeg_data",
        lambda raw: (np.zeros((len(raw.ch_names), 8)), np.array(raw.ch_names)),
    )

    result = run_detection(
        {
            "input_path": str(input_path),
            "output_dir": str(tmp_path / "out"),
            "biomarker": "HFO",
            "detector": "STE",
            "filter": {"fp": 80, "fs": 500},
            "include_channels": ["A1", "A2", "EKG1", "MISSING"],
            "exclude_channels": ["EKG1", "X1", "ALSO_MISSING"],
            "exports": ["summary-json"],
        }
    )

    assert result["status"] == "complete"
    assert backend.raw.picked == ["A1", "A2"]
    assert backend.channel_names.tolist() == ["A1", "A2"]
    assert backend.edf_param["nchan"] == 2
    assert "Detection restricted to 2 selected channels." in result["warnings"]
    assert any("Ignored missing include_channels" in warning and "MISSING" in warning for warning in result["warnings"])
    assert any("Ignored missing exclude_channels" in warning and "ALSO_MISSING" in warning for warning in result["warnings"])
    summary = json.loads(Path(result["artifacts"]["summary_json"]).read_text(encoding="utf-8"))
    assert summary["request"]["include_channels"] == ["A1", "A2", "EKG1", "MISSING"]
    assert summary["request"]["exclude_channels"] == ["EKG1", "X1", "ALSO_MISSING"]


def test_run_detection_supports_spindle_molle_smoke_on_synthetic_fif(tmp_path):
    import mne

    sample_freq = 200.0
    duration_seconds = 4.0
    times = np.arange(int(sample_freq * duration_seconds)) / sample_freq
    rng = np.random.default_rng(42)
    data = rng.normal(scale=0.02, size=(2, times.size))
    envelope = np.exp(-((times - 2.0) / 0.35) ** 2)
    data[0] += 0.8 * envelope * np.sin(2 * np.pi * 13 * times)

    recording_path = tmp_path / "synthetic_spindle_raw_eeg.fif"
    raw = mne.io.RawArray(
        data,
        mne.create_info(["C3", "C4"], sfreq=sample_freq, ch_types=["eeg", "eeg"]),
        verbose=False,
    )
    raw.save(recording_path, overwrite=True, verbose=False)

    result = run_detection(
        {
            "input_path": str(recording_path),
            "output_dir": str(tmp_path / "spindle-out"),
            "biomarker": "Spindle",
            "detector": "MOLLE",
            "filter": {"fp": 1, "fs": 30},
            "detector_params": {
                "freq_sp": [12, 15],
                "duration": [0.3, 2.0],
                "min_distance": 0.2,
                "smooth_window": 0.2,
                "rms_threshold": 0.8,
            },
            "n_jobs": 1,
            "exports": ["csv", "summary-json"],
        }
    )

    assert result["status"] == "complete"
    assert result["biomarker"] == "Spindle"
    assert result["detector"] == "MOLLE"
    assert result["num_events"] >= 1
    assert Path(result["artifacts"]["csv"]).exists()
    assert Path(result["artifacts"]["summary_json"]).exists()
    summary = json.loads(Path(result["artifacts"]["summary_json"]).read_text(encoding="utf-8"))
    assert summary["request"]["biomarker"] == "Spindle"
    assert summary["request"]["detector"] == "MOLLE"


def test_run_detection_can_classify_during_pipeline(monkeypatch, tmp_path):
    input_path = tmp_path / "input.edf"
    input_path.write_text("fake", encoding="utf-8")
    backend = _FakeBackend()
    monkeypatch.setattr("pyhfo.agent_api.pipeline._backend_for_biomarker", lambda _biomarker: backend)

    result = run_detection(
        {
            "input_path": str(input_path),
            "output_dir": str(tmp_path / "out"),
            "biomarker": "HFO",
            "detector": "STE",
            "classifier": {
                "enabled": True,
                "device": "cpu",
                "batch_size": 16,
                "model_type": "default_cpu",
                "ignore_region": [2, 3],
                "steps": ["artifact", "spike", "ehfo"],
            },
        }
    )

    assert result["status"] == "complete"
    assert ("set_classifier", "cpu", 16, "default_cpu") in backend.calls
    assert ("classify_artifacts", [2, 3], 0.5) in backend.calls
    assert ("classify_spikes",) in backend.calls
    assert ("classify_ehfos",) in backend.calls
    assert result["classifier_metadata"]["enabled"] is True
    assert result["classifier_metadata"]["model_type"] == "default_cpu"
    assert result["classifier_metadata"]["source_preference"] is None
    assert result["classifier_metadata"]["ignore_region"] == [2, 3]
    assert result["classifier_metadata"]["executed_steps"] == ["artifact", "spike", "ehfo"]


def test_enabled_classifier_loads_pinned_cpu_models_and_records_provenance(monkeypatch, tmp_path):
    class OfflineBackend(_FakeBackend, HFO_App):
        """Use the real HFO preset with the fake backend's set_classifier and inference methods."""

    backend = OfflineBackend()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("PYHFO_ONNX_DIR", raising=False)
    monkeypatch.setattr("pyhfo.agent_api.pipeline._backend_for_biomarker", lambda _: backend)
    recording = tmp_path / "input.edf"
    recording.touch()
    result = run_detection({
        "input_path": str(recording),
        "output_dir": str(tmp_path / "out"),
        "classifier": {"enabled": True},
        "exports": ["summary-json"],
    })

    assert result["status"] == "complete"
    assert ("set_classifier", "cpu", 32, "onnx_cpu") in backend.calls
    assert not any(warning.startswith("Classifier step") for warning in result["warnings"])
    metadata = result["classifier_metadata"]
    assert metadata["model_preset"] == "default_cpu"
    assert metadata["model_type"] == "onnx_cpu"
    assert metadata["executed_steps"] == ["artifact", "spike", "ehfo"]
    sources = default_hf_onnx_sources()
    assert [metadata[f"{step}_onnx"] for step in metadata["executed_steps"]] == [
        sources["a"], sources["s"], sources["e"],
    ]
    summary = json.loads(Path(result["artifacts"]["summary_json"]).read_text(encoding="utf-8"))
    assert summary["request"]["classifier"]["model_type"] == "default_cpu"
    assert summary["classifier_metadata"] == metadata


@pytest.mark.parametrize(
    ("key", "source"),
    [
        ("artifact_path", "/models/custom.tar"),
        ("artifact_onnx", "/models/custom.onnx"),
        ("artifact_card", "example/custom-classifier"),
    ],
)
def test_explicit_classifier_source_without_model_type_bypasses_cpu_preset(monkeypatch, tmp_path, key, source):
    backend = _FakeBackend()
    monkeypatch.setattr(
        backend, "set_default_cpu_classifier", lambda: pytest.fail("unexpected CPU preset"), raising=False,
    )
    monkeypatch.setattr("pyhfo.agent_api.pipeline._backend_for_biomarker", lambda _: backend)
    loaded = []
    monkeypatch.setattr(backend, "set_classifier", loaded.append)
    recording = tmp_path / "input.edf"
    recording.touch()
    result = run_detection({
        "input_path": str(recording),
        "output_dir": str(tmp_path / "out"),
        "classifier": {"enabled": True, key: source, "steps": ["artifact"]},
    })
    assert result["status"] == "complete"
    assert getattr(loaded[0], key) == source
    assert result["classifier_metadata"][key] == source


def test_classify_existing_run_imports_session_without_detector_rerun(monkeypatch, tmp_path):
    session_path = tmp_path / "session.pybrain"
    session_path.write_text("session", encoding="utf-8")
    backend = _FakeBackend()
    monkeypatch.setattr("pyhfo.agent_api.pipeline._import_app_from_session", lambda biomarker, path: backend)

    result = classify_existing_run(
        {
            "session_path": str(session_path),
            "output_dir": str(tmp_path / "classified"),
            "biomarker": "HFO",
            "run_id": "run-1",
            "classifier": {"enabled": True, "device": "cpu", "batch_size": 8, "model_type": "default_cpu", "steps": ["artifact", "spike"]},
            "exports": ["csv", "session", "summary-json"],
        }
    )

    assert result["status"] == "complete"
    assert result["active_run_id"] == "run-1"
    assert result["num_events"] == 3
    assert result["artifacts"]["session"].endswith("classified_session.pybrain")
    assert Path(result["artifacts"]["csv"]).exists()
    assert Path(result["artifacts"]["session"]).exists()
    assert Path(result["artifacts"]["summary_json"]).exists()
    assert ("activate_run", "run-1") in backend.calls
    assert ("set_classifier", "cpu", 8, "default_cpu") in backend.calls
    assert ("classify_artifacts", [1, 1], 0.5) in backend.calls
    assert ("classify_spikes",) in backend.calls
    assert "detect_biomarker" not in [call[0] for call in backend.calls]
    assert result["classifier_metadata"]["classification_mode"] == "active_run_session"
    assert result["classifier_metadata"]["source_run_id"] == "run-1"
    summary = json.loads(Path(result["artifacts"]["summary_json"]).read_text(encoding="utf-8"))
    assert summary["request"]["session_path"] == str(session_path)


def test_run_detection_can_return_failed_result(monkeypatch, tmp_path):
    input_path = tmp_path / "input.edf"
    input_path.write_text("fake", encoding="utf-8")

    class BrokenBackend(_FakeBackend):
        def detect_biomarker(self):
            raise RuntimeError("boom")

    monkeypatch.setattr("pyhfo.agent_api.pipeline._backend_for_biomarker", lambda _biomarker: BrokenBackend())

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


def test_run_detection_rejects_nyquist_invalid_filter(monkeypatch, tmp_path):
    input_path = tmp_path / "input.edf"
    input_path.write_text("fake", encoding="utf-8")

    class LowSamplingBackend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.sample_freq = 100

    monkeypatch.setattr("pyhfo.agent_api.pipeline._backend_for_biomarker", lambda _biomarker: LowSamplingBackend())

    result = run_detection(
        {
            "input_path": str(input_path),
            "output_dir": str(tmp_path / "out"),
            "biomarker": "HFO",
            "detector": "STE",
            "filter": {"fp": 80, "fs": 500},
            "fail_on_error": False,
        }
    )

    assert result["status"] == "failed"
    assert result["error"]["code"] == "invalid_filter"
