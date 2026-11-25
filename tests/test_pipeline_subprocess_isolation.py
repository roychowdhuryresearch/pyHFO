"""Guards for running heavy pipeline calls out-of-process (the GIL-contention fix).

Prepare/detection/classify/batch are CPU/GIL-bound or memory-heavy; running them
inline in the single ThreadingHTTPServer process starved /waveform/window past
its 20s abort and made very large EDF preparation take down the backend process.
They now run in a fresh interpreter via subprocess_runner.run_pipeline_isolated
when PYHFO_PIPELINE_SUBPROCESS is set (the server's main() opts in; tests/CLI stay
inline by default).

These assert deterministic MECHANISMS (separate process, behavioural parity,
crash isolation, inline default) rather than timing, which would be flaky in CI.
The real responsiveness win (scroll the waveform while a detection runs) is
verified manually via `npm run qa:electron-real-recording`.
"""
from __future__ import annotations

import os

import pytest

from pyhfo.agent_api import subprocess_runner
from pyhfo.agent_api.errors import PyHFOAgentError
from pyhfo.agent_api.subprocess_runner import run_pipeline_isolated

CHANNELS = ["B10Ref", "A2Ref"]

# The app supports THREE biomarkers, each backed by a different app
# (HFO_App/SpikeApp/SpindleApp) selected by run_detection via _backend_for_biomarker.
# They all flow through the same run_detection chokepoint we decouple, so the fix
# must hold for every one — not just HFO. One dependency-free detector each
# (Spindle's default YASA needs the optional `yasa` package, so use MOLLE).
BIOMARKERS = [("HFO", "STE"), ("Spike", "RMS/LL"), ("Spindle", "MOLLE")]


def _detect_payload(tiny_fif_path, output_dir, biomarker="HFO", detector="STE"):
    # fail_on_error=False so a failure returns a dict (vs raising) -> clean
    # inline/subprocess comparison. No `classifier` key -> no NN model load, so
    # this exercises the real classical pipeline without the ckpt LFS dependency.
    return {
        "input_path": str(tiny_fif_path),
        "output_dir": str(output_dir),
        "biomarker": biomarker,
        "detector": detector,
        "include_channels": CHANNELS,
        "exports": ["csv", "summary-json"],
        "n_jobs": 1,
        "fail_on_error": False,
    }


def _prepare_payload(tiny_fif_path, output_path):
    return {
        "input_path": str(tiny_fif_path),
        "output_path": str(output_path),
        "drop_bad_channels": True,
        "drop_flat_channels": True,
        "mark_outlier_channels": True,
        "drop_outlier_channels": False,
        "n_jobs": 1,
    }


@pytest.fixture(autouse=True)
def _inline_by_default(monkeypatch):
    # Each test opts into subprocess mode explicitly; start from a known-inline env.
    monkeypatch.delenv("PYHFO_PIPELINE_SUBPROCESS", raising=False)
    yield


def test_inline_when_env_disabled(tiny_fif_path, tmp_path):
    assert subprocess_runner.subprocess_enabled() is False
    result = run_pipeline_isolated("run_detection", _detect_payload(tiny_fif_path, tmp_path / "out"))
    assert isinstance(result, dict)
    assert result.get("status") in {"complete", "failed"}
    iso = subprocess_runner.last_isolation()
    assert iso.get("mode") == "inline"
    assert iso.get("pid") == os.getpid()


def test_prepare_runs_in_a_separate_process(monkeypatch, tiny_fif_path, tmp_path):
    monkeypatch.setenv("PYHFO_PIPELINE_SUBPROCESS", "1")
    output_path = tmp_path / "prepared.fif"
    result = run_pipeline_isolated("prepare_recording", _prepare_payload(tiny_fif_path, output_path))
    assert result["status"] == "complete"
    assert result["output_path"] == str(output_path.resolve())
    assert output_path.exists()
    iso = subprocess_runner.last_isolation()
    assert iso.get("mode") == "subprocess"
    assert isinstance(iso.get("pid"), int) and iso["pid"] > 0
    assert iso["pid"] != os.getpid()


@pytest.mark.parametrize("biomarker,detector", BIOMARKERS)
def test_detection_runs_in_a_separate_process(monkeypatch, tiny_fif_path, tmp_path, biomarker, detector):
    monkeypatch.setenv("PYHFO_PIPELINE_SUBPROCESS", "1")
    result = run_pipeline_isolated("run_detection", _detect_payload(tiny_fif_path, tmp_path / "out", biomarker, detector))
    assert isinstance(result, dict)
    assert result.get("status") in {"complete", "failed"}
    if result.get("status") == "complete":
        assert result.get("biomarker") == biomarker  # the worker selected the right app for this biomarker
    iso = subprocess_runner.last_isolation()
    assert iso.get("mode") == "subprocess"
    assert isinstance(iso.get("pid"), int) and iso["pid"] > 0
    assert iso["pid"] != os.getpid()  # the heavy work ran off the server's interpreter


@pytest.mark.parametrize("biomarker,detector", BIOMARKERS)
def test_subprocess_result_matches_inline(monkeypatch, tiny_fif_path, tmp_path, biomarker, detector):
    inline = run_pipeline_isolated("run_detection", _detect_payload(tiny_fif_path, tmp_path / "inline", biomarker, detector))
    monkeypatch.setenv("PYHFO_PIPELINE_SUBPROCESS", "1")
    sub = run_pipeline_isolated("run_detection", _detect_payload(tiny_fif_path, tmp_path / "sub", biomarker, detector))
    assert inline.get("status") == sub.get("status")
    assert inline.get("num_events") == sub.get("num_events")
    assert inline.get("biomarker") == sub.get("biomarker")
    assert inline.get("detector") == sub.get("detector")
    if sub.get("status") == "complete":
        assert (tmp_path / "sub" / "summary.json").exists()


def test_child_crash_raises_clean_error_and_parent_survives(monkeypatch, tiny_fif_path, tmp_path):
    monkeypatch.setenv("PYHFO_PIPELINE_SUBPROCESS", "1")
    # Point the worker at a missing module so the child exits non-zero with no
    # result and no summary.json: the runner must surface a clean PyHFOAgentError
    # rather than hang or crash THIS process (which stands in for the HTTP server).
    monkeypatch.setattr(subprocess_runner, "_WORKER_MODULE", "pyhfo.agent_api._missing_worker_xyz")
    payload = _detect_payload(tiny_fif_path, tmp_path / "no-summary-here")
    with pytest.raises(PyHFOAgentError):
        run_pipeline_isolated("run_detection", payload)
    # Parent is alive and the runner still works after a child failure.
    assert subprocess_runner.subprocess_enabled() is True


def test_subprocess_preserves_pipeline_error_without_summary(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    missing_session = "missing.pybrain"

    with pytest.raises(PyHFOAgentError) as inline_exc:
        run_pipeline_isolated("classify_existing_run", {"session_path": str(missing_session), "biomarker": "HFO"})
    assert inline_exc.value.code == "file_not_found"

    monkeypatch.setenv("PYHFO_PIPELINE_SUBPROCESS", "1")
    with pytest.raises(PyHFOAgentError) as subprocess_exc:
        run_pipeline_isolated("classify_existing_run", {"session_path": str(missing_session), "biomarker": "HFO"})
    assert subprocess_exc.value.code == inline_exc.value.code
    assert subprocess_exc.value.message == inline_exc.value.message


def test_unknown_fn_rejected_before_spawn(monkeypatch):
    monkeypatch.setenv("PYHFO_PIPELINE_SUBPROCESS", "1")
    with pytest.raises(PyHFOAgentError):
        run_pipeline_isolated("definitely_not_a_pipeline_fn", {})
