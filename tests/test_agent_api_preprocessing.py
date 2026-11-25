import hashlib

import mne
import numpy as np

from pyhfo.agent_api.preprocessing import auto_adjacent_bipolar_pairs, prepare_recording


def test_auto_adjacent_bipolar_pairs_skips_already_bipolar_labels():
    assert auto_adjacent_bipolar_pairs(["A1", "A2", "A3", "B1-B2", "Aux"]) == [
        {"anode": "A1", "cathode": "A2", "name": "A1-A2"},
        {"anode": "A2", "cathode": "A3", "name": "A2-A3"},
    ]


def test_prepare_recording_writes_cleaned_fif_with_bipolar_and_notch(tmp_path):
    sfreq = 1000.0
    samples = int(sfreq * 2)
    times = np.arange(samples) / sfreq
    data = np.vstack(
        [
            np.sin(2 * np.pi * 10 * times),
            np.sin(2 * np.pi * 10 * times + 0.2),
            np.zeros_like(times),
        ]
    ) * 1e-6
    raw = mne.io.RawArray(data, mne.create_info(["A1", "A2", "Flat"], sfreq, ch_types="eeg"), verbose=False)
    src = tmp_path / "raw.fif"
    dst = tmp_path / "clean.fif"
    raw.save(src, overwrite=True, verbose=False)

    result = prepare_recording(
        {
            "input_path": str(src),
            "output_path": str(dst),
            "bipolar_montage": "auto_adjacent",
            "line_noise_notch_hz": 60,
        }
    )

    assert result["status"] == "complete"
    assert result["output_path"] == str(dst.resolve())
    assert "Flat" in result["flat_channels"]
    assert "Flat" in result["dropped_channels"]
    assert result["bipolar_montage"]["added_channels"] == ["A1-A2"]
    assert result["line_noise_notch_hz"] == 60
    assert result["input_recording_fingerprint"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert result["output_recording_fingerprint"] == hashlib.sha256(dst.read_bytes()).hexdigest()
    assert len(result["parameters_sha256"]) == 64
    assert dst.exists()
