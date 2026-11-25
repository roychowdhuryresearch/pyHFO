"""End-to-end smoke for release builds, run from outside the source checkout."""

import hashlib
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest


@pytest.mark.skipif(os.getenv("PYHFO_TEST_WHEEL") != "1", reason="requires a wheel installation")
def test_installed_wheel(tmp_path):
    import edfio
    import numpy as np

    scripts = Path(sysconfig.get_path("scripts"))
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYHFO_AGENT_SESSION_DIR", None)
    home = tmp_path / "home"
    env["PYHFO_HOME"] = str(home)
    env["HF_HUB_OFFLINE"] = "1"

    def run(*args, stdin=None):
        process = subprocess.run(args, cwd=tmp_path, env=env, input=stdin,
                                 capture_output=True, text=True, timeout=90)
        assert process.returncode == 0, process.stderr or process.stdout
        return process.stdout

    version = run(sys.executable, "-c", """
import importlib.util, pathlib, pyhfo, sys
from importlib.metadata import version
assert pathlib.Path(pyhfo.__file__).is_relative_to(sys.prefix)
assert importlib.util.find_spec("src") is None
assert pyhfo.__version__ == version("pyhfo")
print(pyhfo.__version__)
""").strip()
    assert run(scripts / "pyhfo", "--help")
    assert run(scripts / "pyhfo-server", "--help")
    request = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    response = json.loads(run(scripts / "pyhfo-mcp", stdin=json.dumps(request) + "\n"))
    assert response["result"]["serverInfo"]["version"] == version

    # A 150 Hz burst in a 2 kHz EDF gives the RMS detector a known candidate.
    sfreq = 2000
    time = np.arange(8 * sfreq) / sfreq
    signal = np.random.default_rng(42).normal(0, 2, len(time))
    burst = (time >= 3) & (time < 3.15)
    signal[burst] += 80 * np.sin(2 * np.pi * 150 * time[burst])
    recording = tmp_path / "burst.edf"
    edfio.Edf([edfio.EdfSignal(signal, sfreq, label="A1", physical_dimension="uV")]).write(recording)
    result = json.loads(run(scripts / "pyhfo", "detect", "--input", recording,
                            "--output", tmp_path / "run", "--biomarker", "HFO",
                            "--detector", "RMS", "--n-jobs", "1", "--export", "csv",
                            "--export", "summary-json"))
    assert result["status"] == "complete", result
    assert result["num_events"] > 0
    assert result["recording_fingerprint"] == hashlib.sha256(recording.read_bytes()).hexdigest()
    assert Path(result["artifacts"]["csv"]).is_file()
    assert Path(result["artifacts"]["summary_json"]).is_file()

    run(scripts / "pyhfo", "action", "--name", "load_recording", "--session-id", "wheel",
        "--args-json", json.dumps({"input_path": str(recording)}))
    assert (home / "sessions" / "wheel.json").is_file()
    state = json.loads(run(scripts / "pyhfo", "state", "--session-id", "wheel"))
    assert state["loaded_recording_path"] == str(recording)
