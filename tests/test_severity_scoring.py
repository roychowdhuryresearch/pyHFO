"""Severity scoring: scalp-EEG detection, the scorer itself, and the agent_api surface.

The model checkpoint is large and CPU inference is slow, so the tests that need real
weights are opt-in via PYHFO_RUN_SEVERITY_MODEL=1. Everything that can be checked
without the checkpoint (eeg_type inference, schema plumbing, dispatch/route wiring,
bad-input handling) runs unconditionally.
"""
import json
import os
import sys
import threading
import urllib.request
from types import SimpleNamespace

import mne
import numpy as np
import pytest

from pyhfo.agent_api.errors import PyHFOAgentError
import pyhfo.agent_api.pipeline as severity_pipeline
from pyhfo.agent_api.pipeline import _severity_artifact_flags, get_recording_info, run_severity_scoring
from pyhfo.agent_api.schemas import SeverityScoringResult
from pyhfo.agent_api.server import build_server
from pyhfo.agent_api.sidecar import PyHFOSidecarSession, get_capability_manifest
from pyhfo.agent_api.subprocess_runner import _VALID_FNS
from pyhfo.hfo_app import HFO_App
from pyhfo.spindle_app import SpindleApp
from pyhfo.utils.utils_montage import infer_eeg_type


TUEG_19 = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
    "FZ", "CZ", "PZ",
]

_MODEL_TESTS_ENABLED = os.environ.get("PYHFO_RUN_SEVERITY_MODEL") == "1"
_requires_model = pytest.mark.skipif(
    not _MODEL_TESTS_ENABLED,
    reason="Set PYHFO_RUN_SEVERITY_MODEL=1 to run tests that load the BASED checkpoint.",
)


def _write_raw(tmp_path, channel_names, duration_seconds=30.0, sfreq=200.0, seed=0):
    samples = int(sfreq * duration_seconds)
    data = np.random.default_rng(seed).normal(scale=20e-6, size=(len(channel_names), samples))
    info = mne.create_info(list(channel_names), sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    path = tmp_path / "severity_raw.fif"
    raw.save(path, overwrite=True, verbose=False)
    return path


# ── scalp detection ───────────────────────────────────────────────────────────

def test_infer_eeg_type_recognises_conventional_scalp_layout():
    assert infer_eeg_type(np.array(TUEG_19)) == "scalp"


def test_infer_eeg_type_reports_ieeg_for_numbered_contacts():
    assert infer_eeg_type(np.array(["A1", "A2", "A3", "A4", "B1", "B2"])) == "ieeg"


def test_recording_info_exposes_eeg_type(tmp_path):
    info = get_recording_info(_write_raw(tmp_path, TUEG_19, duration_seconds=2.0))
    assert info["eeg_type"] == "scalp"


def test_backends_record_eeg_type_on_load(tmp_path):
    path = str(_write_raw(tmp_path, TUEG_19, duration_seconds=2.0))
    for app in (HFO_App(), SpindleApp()):
        app.load_edf(path)
        assert app.eeg_type == "scalp"


# ── backend severity API surface (no checkpoint needed) ───────────────────────

def test_backends_start_without_a_severity_result():
    for app in (HFO_App(), SpindleApp()):
        assert app.has_severity_result() is False
        assert app.get_severity_summary() is None


def test_export_severity_csv_requires_a_run_first():
    app = HFO_App()
    with pytest.raises(ValueError):
        app.export_severity_csv("unused.csv")


def test_run_severity_scoring_requires_a_loaded_recording():
    app = HFO_App()
    with pytest.raises(ValueError):
        app.run_severity_scoring(scorer=None)


def test_unpublished_severity_model_gives_local_setup_instructions(monkeypatch):
    from pyhfo.utils import torch_models

    monkeypatch.setattr(torch_models, "SEVERITY_REVISION", "")
    monkeypatch.delenv("PYHFO_SEVERITY_MODEL_DIR", raising=False)
    for app_cls in (HFO_App, SpindleApp):
        with pytest.raises(PyHFOAgentError, match="PYHFO_SEVERITY_MODEL_DIR") as exc:
            app_cls.default_severity_model_dir()
        assert exc.value.code == "severity_model_unpublished"


def test_agent_exposes_unpublished_severity_without_loading_torch(tmp_path, monkeypatch):
    from pyhfo.utils import torch_models

    monkeypatch.setattr(torch_models, "SEVERITY_REVISION", "")
    monkeypatch.delenv("PYHFO_SEVERITY_MODEL_DIR", raising=False)
    monkeypatch.setitem(sys.modules, "pyhfo.severity_app", None)
    path = _write_raw(tmp_path, TUEG_19, duration_seconds=15.0)
    with pytest.raises(PyHFOAgentError) as exc:
        run_severity_scoring({"input_path": str(path)})
    assert exc.value.code == "severity_model_unpublished"


def test_severity_local_override_precedence_and_lfs_pointer_error(tmp_path, monkeypatch):
    from pyhfo.utils.torch_models import severity_model_dir

    model_dir = tmp_path / "standalone"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    weights = model_dir / "model.safetensors"
    weights.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    monkeypatch.setenv("PYHFO_SEVERITY_MODEL_DIR", str(tmp_path / "missing"))
    with pytest.raises(PyHFOAgentError, match="Git LFS pointer"):
        severity_model_dir(model_dir)
    weights.write_bytes(b"local test weights")
    assert severity_model_dir(model_dir) == str(model_dir)
    monkeypatch.setenv("PYHFO_SEVERITY_MODEL_DIR", str(model_dir))
    assert HFO_App.default_severity_model_dir() == str(model_dir)


def test_published_severity_uses_verified_snapshot(tmp_path, monkeypatch):
    import hashlib
    from pyhfo.utils import torch_models

    contents = {"config.json": b"{}", "model.safetensors": b"weights"}
    monkeypatch.delenv("PYHFO_SEVERITY_MODEL_DIR", raising=False)
    monkeypatch.setenv("PYHFO_SEVERITY_REPO", "org/mirror")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(torch_models, "SEVERITY_REVISION", "a" * 40)
    monkeypatch.setattr(torch_models, "SEVERITY_CONFIG_SHA256", hashlib.sha256(contents["config.json"]).hexdigest())
    monkeypatch.setattr(torch_models, "SEVERITY_MODEL_SHA256", hashlib.sha256(contents["model.safetensors"]).hexdigest())

    def fetch(repo, filename, *, revision, local_files_only, cache_dir):
        assert (repo, revision, local_files_only, cache_dir) == ("org/mirror", "a" * 40, True, tmp_path / "cache")
        path = tmp_path / filename
        path.write_bytes(contents[filename])
        return str(path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=fetch))
    assert torch_models.severity_model_dir() == str(tmp_path)
    contents["model.safetensors"] = b"wrong"
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        torch_models.severity_model_dir()


# ── agent_api plumbing ────────────────────────────────────────────────────────

def test_severity_scoring_result_serialises_all_reported_fields():
    result = SeverityScoringResult(status="complete", input_path="x.fif", eeg_type="scalp")
    payload = result.to_dict()
    for key in ("status", "input_path", "eeg_type", "recording_fingerprint", "model_name",
                "model_version", "model_sha256", "parameters_sha256", "num_segments", "mean_score",
                "peak_score", "peak_time_sec", "num_clean_segments", "clean_mean_score",
                "clean_peak_score", "clean_peak_time_sec", "artifact_quality", "scores",
                "artifacts", "warnings", "timing"):
        assert key in payload


def test_severity_model_digest_binds_names_and_bytes(tmp_path):
    from pyhfo.agent_api.pipeline import _model_directory_sha256

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"version":1}', encoding="utf-8")
    original = _model_directory_sha256(model_dir)
    (model_dir / "config.json").write_text('{"version":2}', encoding="utf-8")
    changed_contents = _model_directory_sha256(model_dir)
    assert changed_contents != original
    (model_dir / "config.json").rename(model_dir / "renamed.json")
    assert _model_directory_sha256(model_dir) != changed_contents


def test_severity_scoring_is_registered_for_subprocess_isolation():
    # A full-recording transformer pass must run out-of-process so it can't starve
    # latency-sensitive endpoints (notably /waveform/window) of the GIL.
    assert "run_severity_scoring" in _VALID_FNS

    from pyhfo.agent_api.pipeline_worker import _FUNCTIONS

    assert "run_severity_scoring" in _FUNCTIONS


def test_capability_manifest_advertises_severity_scoring():
    actions = {action["name"]: action for action in get_capability_manifest()["actions"]}
    assert "run_severity_scoring" in actions
    assert actions["run_severity_scoring"]["status"] == "callable"


def test_run_severity_scoring_rejects_a_missing_file():
    with pytest.raises(PyHFOAgentError) as exc:
        run_severity_scoring({"input_path": "definitely-not-here.fif"})
    assert exc.value.code == "file_not_found"


def test_run_severity_scoring_rejects_a_recording_with_no_scalp_channels(tmp_path):
    # Zero overlap with TUEG-19 is a bad-input condition (HTTP 400), not a server error.
    path = _write_raw(tmp_path, ["A1", "A2", "A3", "A4"], duration_seconds=20.0)
    with pytest.raises(PyHFOAgentError) as exc:
        run_severity_scoring({"input_path": str(path)})
    assert exc.value.code == "invalid_recording"


def test_severity_state_starts_empty_and_is_reported_in_session_state():
    session = PyHFOSidecarSession()
    assert session.last_severity is None
    assert session.get_state()["last_severity_summary"] is None


def _severity_scores(count=2):
    return [
        {"segment_idx": index, "start_sec": float(index * 15), "end_sec": float((index + 1) * 15), "score": float(index + 1)}
        for index in range(count)
    ]


def test_severity_artifact_screening_preserves_normal_clean_and_artifact_flags():
    eeg = np.zeros((1, 30 * 200), dtype=float)
    eeg[0, 15 * 200:] = 1001.0
    app = SimpleNamespace(channel_names=np.array(["EEG Fp1-Ref"]), sample_freq=200, eeg_data=eeg)

    flags, quality, warnings = _severity_artifact_flags(app, _severity_scores())

    assert flags == [False, True]
    assert quality == {
        "status": "assessed",
        "method": "tueg19_peak_amplitude",
        "threshold_uv": 1000.0,
        "assessed_segments": 2,
        "total_segments": 2,
        "channel_count": 1,
    }
    assert warnings == []


def test_severity_artifact_screening_marks_missing_tueg_evidence_unknown():
    app = SimpleNamespace(
        channel_names=np.array(["A1", "A2"]),
        sample_freq=200,
        eeg_data=np.zeros((2, 30 * 200), dtype=float),
    )

    flags, quality, warnings = _severity_artifact_flags(app, _severity_scores())

    assert flags == [None, None]
    assert quality["status"] == "unavailable"
    assert quality["assessed_segments"] == 0
    assert "TUEG-19" in quality["reason"]
    assert any("not classified as clean" in warning for warning in warnings)


def test_severity_artifact_screening_surfaces_eeg_read_failures():
    class BrokenEegApp:
        channel_names = np.array(["FP1"])
        sample_freq = 200

        @property
        def eeg_data(self):
            raise RuntimeError("unreadable EEG buffer")

    flags, quality, warnings = _severity_artifact_flags(BrokenEegApp(), _severity_scores())

    assert flags == [None, None]
    assert quality["status"] == "unavailable"
    assert quality["channel_count"] == 1
    assert "RuntimeError" in quality["reason"]
    assert any("unreadable EEG buffer" in warning for warning in warnings)


class _FakeScoresFrame:
    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def to_dict(self, orient):
        assert orient == "records"
        return [dict(record) for record in self.records]


def test_empty_scorer_output_is_no_data_instead_of_complete_zero_segments(tmp_path, monkeypatch):
    input_path = tmp_path / "scalp.fif"
    input_path.write_bytes(b"fake recording")

    class FakeApp:
        eeg_type = "scalp"
        channel_names = np.array(["FP1"])
        sample_freq = 200
        eeg_data = np.zeros((1, 30 * 200), dtype=float)

        def load_edf(self, _path):
            return None

        def default_severity_model_dir(self):
            return "unused-model"

        def run_severity_scoring(self, _scorer):
            return SimpleNamespace(
                scores_df=_FakeScoresFrame([]),
                mean_score=float("nan"),
                peak_score=float("nan"),
                peak_time_sec=float("nan"),
            )

    monkeypatch.setattr(severity_pipeline, "HFO_App", FakeApp)
    monkeypatch.setattr(severity_pipeline, "_model_directory_sha256", lambda _model_dir: "a" * 64)
    monkeypatch.setitem(sys.modules, "pyhfo.severity_app", SimpleNamespace(SeverityScorer=lambda _model_dir: object()))

    with pytest.raises(PyHFOAgentError) as exc:
        run_severity_scoring({"input_path": str(input_path)})
    assert exc.value.code == "no_data"
    assert exc.value.details == {"minimum_segment_seconds": 15.0, "num_segments": 0}


def test_unavailable_artifact_evidence_never_promotes_raw_scores_to_clean(tmp_path, monkeypatch):
    input_path = tmp_path / "scalp.fif"
    input_path.write_bytes(b"fake recording")
    scores = _severity_scores()

    class FakeApp:
        eeg_type = "scalp"
        channel_names = np.array(["A1"])
        sample_freq = 200
        eeg_data = np.zeros((1, 30 * 200), dtype=float)

        def load_edf(self, _path):
            return None

        def default_severity_model_dir(self):
            return "unused-model"

        def run_severity_scoring(self, _scorer):
            return SimpleNamespace(
                scores_df=_FakeScoresFrame(scores),
                mean_score=1.5,
                peak_score=2.0,
                peak_time_sec=15.0,
            )

    monkeypatch.setattr(severity_pipeline, "HFO_App", FakeApp)
    monkeypatch.setattr(severity_pipeline, "_model_directory_sha256", lambda _model_dir: "a" * 64)
    monkeypatch.setattr(severity_pipeline, "file_fingerprint", lambda _path: {"sha256": "b" * 64})
    monkeypatch.setitem(sys.modules, "pyhfo.severity_app", SimpleNamespace(SeverityScorer=lambda _model_dir: object()))

    result = run_severity_scoring({"input_path": str(input_path)})

    assert result["status"] == "complete"
    assert result["artifact_quality"]["status"] == "unavailable"
    assert [record["artifact"] for record in result["scores"]] == [None, None]
    assert result["num_clean_segments"] == 0
    assert result["clean_mean_score"] is None
    assert result["clean_peak_score"] is None
    assert result["clean_peak_time_sec"] is None
    assert any("not classified as clean" in warning for warning in result["warnings"])


# ── end-to-end with the real checkpoint (opt-in) ──────────────────────────────

@_requires_model
def test_severity_scoring_scores_a_scalp_recording(tmp_path):
    path = _write_raw(tmp_path, TUEG_19, duration_seconds=30.0)
    csv_path = tmp_path / "scores.csv"
    result = run_severity_scoring({"input_path": str(path), "export_csv": str(csv_path)})

    assert result["status"] == "complete"
    assert result["eeg_type"] == "scalp"
    assert len(result["recording_fingerprint"]) == 64
    assert result["model_name"] == "BASED"
    assert result["model_version"].startswith("based-severity-")
    assert len(result["model_sha256"]) == 64
    assert len(result["parameters_sha256"]) == 64
    # 30s of signal at a 15s segment length.
    assert result["num_segments"] == 2
    assert len(result["scores"]) == 2
    assert 0.0 <= result["mean_score"] <= 5.0
    assert 0.0 <= result["peak_score"] <= 5.0
    assert result["artifacts"]["csv"] == str(csv_path)
    assert csv_path.exists()


def test_severity_scoring_rejects_an_intracranial_recording_before_loading_the_model(tmp_path):
    # Partial TUEG overlap must not turn an intracranial recording into a scalp severity result.
    path = _write_raw(tmp_path, ["A1", "A2", "C3", "C4"], duration_seconds=20.0)
    with pytest.raises(PyHFOAgentError) as exc:
        run_severity_scoring({"input_path": str(path)})
    assert exc.value.code == "invalid_recording"
    assert "scalp EEG" in str(exc.value)


def test_severity_scoring_rejects_a_recording_without_one_complete_segment(tmp_path, monkeypatch):
    path = _write_raw(tmp_path, TUEG_19, duration_seconds=5.0)
    monkeypatch.setattr(
        severity_pipeline,
        "_model_directory_sha256",
        lambda _model_dir: pytest.fail("a too-short recording must fail before loading or hashing the model"),
    )
    with pytest.raises(PyHFOAgentError) as exc:
        run_severity_scoring({"input_path": str(path)})
    assert exc.value.code == "no_data"
    assert exc.value.details["num_segments"] == 0
    assert exc.value.details["minimum_segment_seconds"] == 15.0
    assert "complete 15-second segment" in str(exc.value)


@_requires_model
def test_severity_routes_score_and_expose_the_result(tmp_path):
    path = _write_raw(tmp_path, TUEG_19, duration_seconds=30.0)
    session = PyHFOSidecarSession()
    server = build_server("127.0.0.1", 0, session)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def request(method, route, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{route}", data=data, method=method,
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.loads(response.read().decode())

    try:
        assert request("GET", "/severity")["status"] == "not_run"

        loaded = request("POST", "/recordings/load", {"input_path": str(path)})
        assert loaded["result"]["eeg_type"] == "scalp"

        scored = request("POST", "/severity/run", {})["result"]
        assert scored["status"] == "complete"
        assert scored["num_segments"] == 2

        assert request("GET", "/severity")["result"]["mean_score"] == scored["mean_score"]
        assert request("GET", "/state")["last_severity_summary"]["num_segments"] == 2
    finally:
        server.shutdown()
