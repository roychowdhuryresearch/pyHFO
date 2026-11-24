import hashlib
import json

import mne
import numpy as np
import pytest

from pyhfo.agent_api import PyHFOAgentError, detect_line_noise, get_recording_info


def test_get_recording_info_reads_tiny_fif(tiny_fif_path):
    info = get_recording_info(tiny_fif_path)

    assert info["input_path"].endswith("tiny_raw.fif")
    assert info["format"] == ".fif"
    assert info["sample_freq"] == 200.0
    assert info["n_channels"] == 3
    assert info["duration_seconds"] > 0
    assert info["channel_names"] == ["B10Ref", "A2Ref", "A1Ref"]
    assert info["recording_fingerprint"] == hashlib.sha256(tiny_fif_path.read_bytes()).hexdigest()
    json.dumps(info)


def test_get_recording_info_rejects_missing_file(tmp_path):
    with pytest.raises(PyHFOAgentError) as exc_info:
        get_recording_info(tmp_path / "missing.edf")

    assert exc_info.value.code == "file_not_found"


def test_detect_line_noise_uses_explicit_header_frequency_for_notch_decision(tmp_path):
    sfreq = 1000.0
    samples = int(sfreq * 2.0)
    times = np.arange(samples) / sfreq
    data = np.vstack(
        [
            20e-6 * np.sin(2 * np.pi * 10 * times) + 120e-6 * np.sin(2 * np.pi * 60 * times),
            15e-6 * np.sin(2 * np.pi * 8 * times) + 90e-6 * np.sin(2 * np.pi * 60 * times),
            200e-6 * np.sin(2 * np.pi * 1.0 * times),
        ]
    )
    raw = mne.io.RawArray(data, mne.create_info(["A1", "A2", "EKG1"], sfreq, ch_types="eeg"), verbose=False)
    raw.info["line_freq"] = 60
    path = tmp_path / "line_noise_raw.fif"
    raw.save(path, overwrite=True, verbose=False)

    result = detect_line_noise(
        {
            "input_path": str(path),
            "exclude_channels": ["EKG1"],
            "candidate_frequencies_hz": [50, 60],
            "max_duration_seconds": 2.0,
            "ratio_threshold": 3.0,
        }
    )

    assert result["status"] == "complete"
    assert result["metadata_line_freq_hz"] == 60.0
    assert result["recommended_notch_hz"] == 60.0
    assert result["decision"]["should_notch"] is True
    assert result["decision"]["requires_user_confirmation"] is False
    assert result["excluded_channels"] == ["EKG1"]
    assert any(item["base_frequency_hz"] == 60.0 for item in result["detected_frequencies"])


def test_detect_line_noise_requires_confirmation_without_explicit_frequency(tmp_path):
    sfreq = 1000.0
    samples = int(sfreq * 2.0)
    times = np.arange(samples) / sfreq
    data = np.vstack([20e-6 * np.sin(2 * np.pi * 10 * times) + 120e-6 * np.sin(2 * np.pi * 60 * times)])
    raw = mne.io.RawArray(data, mne.create_info(["A1"], sfreq, ch_types="eeg"), verbose=False)
    path = tmp_path / "line_noise_no_header_raw.fif"
    raw.save(path, overwrite=True, verbose=False)

    result = detect_line_noise(
        {
            "input_path": str(path),
            "candidate_frequencies_hz": [50, 60],
            "max_duration_seconds": 2.0,
            "ratio_threshold": 3.0,
        }
    )

    assert result["metadata_line_freq_hz"] is None
    assert result["decision"]["should_notch"] is False
    assert result["decision"]["requires_user_confirmation"] is True
    assert result["recommended_notch_hz"] is None
