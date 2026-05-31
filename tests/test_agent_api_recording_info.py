import json

import pytest

from src.agent_api import PyHFOAgentError, get_recording_info


def test_get_recording_info_reads_tiny_fif(tiny_fif_path):
    info = get_recording_info(tiny_fif_path)

    assert info["input_path"].endswith("tiny_raw.fif")
    assert info["format"] == ".fif"
    assert info["sample_freq"] == 200.0
    assert info["n_channels"] == 3
    assert info["duration_seconds"] > 0
    assert info["channel_names"] == ["B10Ref", "A2Ref", "A1Ref"]
    json.dumps(info)


def test_get_recording_info_rejects_missing_file(tmp_path):
    with pytest.raises(PyHFOAgentError) as exc_info:
        get_recording_info(tmp_path / "missing.edf")

    assert exc_info.value.code == "file_not_found"
