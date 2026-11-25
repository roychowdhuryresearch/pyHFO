"""Regression tests for sidecar bug-hunt fixes (#5 stale ROI, #6 cross-file race)."""

from pyhfo.agent_api import sidecar
from pyhfo.agent_api.sidecar import PyHFOSidecarSession


def _patch_common(monkeypatch):
    monkeypatch.setattr(
        sidecar, "get_recording_info",
        lambda path: {"input_path": path, "sample_freq": 2000.0, "channel_names": ["A1", "A2", "B1", "B2"]},
    )
    monkeypatch.setattr(
        sidecar, "infer_brain_signal_roi_from_channels",
        lambda chans: {"brain_signal_channels": list(chans), "default_detection_exclusions": []},
    )
    monkeypatch.setattr(
        "pyhfo.agent_api.preprocessing.prepare_recording",
        lambda args: {"status": "complete", "input_path": args["input_path"], "output_path": args["output_path"]},
    )


# --- Bug #5: a bipolar montage renames channels, so a monopolar ROI is stale --

def test_bipolar_prepare_invalidates_stale_brain_signal_roi(monkeypatch):
    _patch_common(monkeypatch)
    session = PyHFOSidecarSession(session_id="roi")
    session.dispatch("load_recording", {"input_path": "a.fif"})
    session.dispatch("infer_brain_signal_roi", {"channel_names": ["A1", "A2"]})
    assert session.brain_signal_roi is not None

    session.dispatch("prepare_recording", {
        "input_path": "a.fif", "output_path": "clean.fif", "bipolar_montage": "auto_adjacent",
    })
    # The monopolar ROI no longer matches the renamed bipolar channels -> cleared, so the
    # next run_detection won't auto-inject all-missing channels and fail.
    assert session.brain_signal_roi is None


def test_non_bipolar_prepare_keeps_brain_signal_roi(monkeypatch):
    _patch_common(monkeypatch)
    session = PyHFOSidecarSession(session_id="roi2")
    session.dispatch("load_recording", {"input_path": "a.fif"})
    session.dispatch("infer_brain_signal_roi", {"channel_names": ["A1", "A2"]})
    # A notch-only prepare (no montage) preserves channel names -> ROI stays valid.
    session.dispatch("prepare_recording", {"input_path": "a.fif", "output_path": "clean.fif"})
    assert session.brain_signal_roi is not None


# --- Bug #6: don't store a detection result for a file the session moved off of --

def test_run_detection_skips_writeback_when_recording_switched_concurrently(monkeypatch):
    _patch_common(monkeypatch)
    session = PyHFOSidecarSession(session_id="race")
    session.dispatch("load_recording", {"input_path": "A.fif"})

    def fake_run_detection(args):
        # Simulate a concurrent load_recording for a DIFFERENT file landing while the
        # detection subprocess runs.
        session.recording_info = {"input_path": "B.fif"}
        return {"status": "complete", "output_dir": args["output_dir"], "num_events": 7,
                "artifacts": {"summary_json": "s.json"}}

    monkeypatch.setattr("pyhfo.agent_api.pipeline.run_detection", fake_run_detection)
    session.dispatch("run_detection", {"input_path": "A.fif", "output_dir": "out"})
    # The session moved to B mid-run, so the A-detection must NOT become last_detection.
    assert session.last_detection is None


def test_run_detection_stores_result_when_recording_unchanged(monkeypatch):
    _patch_common(monkeypatch)
    session = PyHFOSidecarSession(session_id="normal")
    session.dispatch("load_recording", {"input_path": "A.fif"})
    monkeypatch.setattr(
        "pyhfo.agent_api.pipeline.run_detection",
        lambda args: {"status": "complete", "output_dir": args["output_dir"], "num_events": 3,
                      "artifacts": {"summary_json": "s.json"}},
    )
    # Normal flow: detect on the prepared file (path differs from recording_info) but no
    # concurrent load -> result IS stored.
    session.dispatch("run_detection", {"input_path": "clean.fif", "output_dir": "out"})
    assert session.last_detection is not None
    assert session.last_detection["num_events"] == 3
