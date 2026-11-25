from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from pyhfo.agent_api.server import build_server
from pyhfo.agent_api.sidecar import PyHFOSidecarSession


def _json(url: str, *, payload: dict | None = None) -> dict:
    if payload is None:
        with urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    req = Request(url, data=json.dumps(payload).encode("utf-8"), headers={"content-type": "application/json"}, method="POST")
    with urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _fixture_path() -> Path:
    raw = os.getenv("PYHFO_PARITY_FIXTURE_RECORDING", "").strip()
    if not raw:
        pytest.skip("Set PYHFO_PARITY_FIXTURE_RECORDING to run the real recording Electron/PyQt parity smoke.")
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        pytest.fail(f"PYHFO_PARITY_FIXTURE_RECORDING does not exist: {path}")
    return path


def _output_dir(tmp_path: Path) -> Path:
    raw = os.getenv("PYHFO_PARITY_FIXTURE_OUTPUT_DIR", "").strip()
    path = Path(raw).expanduser().resolve() if raw else tmp_path / "real-fixture-smoke"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _unwrap_action(result: dict) -> dict:
    if "result" in result and isinstance(result["result"], dict):
        return result["result"]
    return result


def test_real_recording_load_waveform_session_and_optional_detection(tmp_path):
    recording = _fixture_path()
    output_dir = _output_dir(tmp_path)
    session = PyHFOSidecarSession(session_id="real-recording-parity")
    server = build_server("127.0.0.1", 0, session)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = str(server.server_address[0]), int(server.server_address[1])
    base = f"http://{host}:{port}"
    try:
        health = _json(base + "/health")
        assert health["status"] == "ok"

        loaded = _unwrap_action(_json(base + "/recordings/load", payload={"input_path": str(recording)}))
        assert loaded["input_path"] == str(recording)
        assert float(loaded["sample_freq"]) > 0
        assert float(loaded["duration_seconds"]) > 0
        channels = [str(channel) for channel in loaded.get("channel_names") or loaded.get("channels") or []]
        assert channels, "real fixture must expose channel names"
        expected_channel = os.getenv("PYHFO_PARITY_FIXTURE_EXPECTED_CHANNEL", "").strip()
        channel = expected_channel or channels[0]
        assert channel in channels, f"Expected channel {channel!r} not present in loaded recording"

        waveform = _json(base + "/waveform/window?" + urlencode({
            "input_path": str(recording),
            "channels": channel,
            "start_seconds": 0,
            "duration_seconds": min(2.0, max(0.1, float(loaded["duration_seconds"]))),
            "target_samples_per_channel": 256,
            "units": "uV",
        }))
        assert waveform["returned_samples_per_channel"] <= 256
        assert waveform["channels"][0] == channel
        assert waveform["data"], "waveform response must contain bounded samples"

        roi = _unwrap_action(_json(base + "/recordings/roi", payload={"input_path": str(recording)}))
        assert "brain_signal_channel_count" in roi
        qc = _unwrap_action(_json(base + "/recordings/qc", payload={"input_path": str(recording), "max_duration_seconds": 2.0}))
        assert qc.get("status") == "complete"
        spectral = _unwrap_action(_json(base + "/recordings/spectral", payload={"input_path": str(recording), "max_channels": 4, "max_duration_seconds": 2.0}))
        assert spectral.get("status") == "complete"

        pybrain_path = output_dir / "real-fixture-session.pybrain"
        session_state = {
            "format": "electron-real-recording-parity-smoke",
            "recording_path": str(recording),
            "recording_info": loaded,
            "prepare_controls": {"fp": "80", "fs": "250", "rp": "0.5", "rs": "40", "nJobs": 1},
            "waveform_controls": {"visibleChannels": 8, "timeWindowSeconds": 5, "startSeconds": 0, "selectedChannel": channel},
            "run_records": [],
            "review_events": [],
            "runtime_log": [{"level": "success", "message": "Real fixture load/waveform/QC smoke completed"}],
        }
        exported = _json(base + "/exports/session", payload={"output_path": str(pybrain_path), "session_state": session_state})
        assert exported["artifact_type"] == "pybrain_session"
        reopened = _json(base + "/sessions/open-saved", payload={"session_path": exported["artifact_path"]})
        assert reopened["recording_info"]["input_path"] == str(recording)

        if os.getenv("PYHFO_PARITY_RUN_DETECTION", "").strip().lower() in {"1", "true", "yes", "on"}:
            biomarker = os.getenv("PYHFO_PARITY_BIOMARKER", "HFO")
            detector = os.getenv("PYHFO_PARITY_DETECTOR", "STE")
            detection = _unwrap_action(_json(base + "/detections/run", payload={
                "input_path": str(recording),
                "output_dir": str(output_dir / "detection"),
                "biomarker": biomarker,
                "detector": detector,
                "filter": {"fp": float(os.getenv("PYHFO_PARITY_FILTER_FP", "80")), "fs": float(os.getenv("PYHFO_PARITY_FILTER_FS", "250"))},
                "n_jobs": int(os.getenv("PYHFO_PARITY_N_JOBS", "1")),
                "exports": ["summary-json", "session"],
            }))
            assert detection["status"] == "complete"
            assert detection.get("artifacts", {}).get("summary_json")
    finally:
        server.shutdown(); server.server_close()
