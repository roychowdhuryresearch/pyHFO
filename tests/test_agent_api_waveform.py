from __future__ import annotations

import pytest

from pyhfo.agent_api.errors import PyHFOAgentError
from pyhfo.agent_api.waveform import waveform_channels, waveform_overview, waveform_window


def test_waveform_window_returns_bounded_downsampled_data(tiny_fif_path):
    result = waveform_window(
        tiny_fif_path,
        start_seconds=0.0,
        duration_seconds=1.0,
        channels=["B10Ref", "A2Ref"],
        target_samples_per_channel=25,
    )

    assert result["status"] == "complete"
    assert result["channels"] == ["B10Ref", "A2Ref"]
    assert result["returned_samples_per_channel"] <= 25
    assert len(result["data"]) == 2
    assert len(result["data"][0]) == result["returned_samples_per_channel"]
    assert result["units"] == "uV"


def test_waveform_window_rejects_bad_channel_and_time(tiny_fif_path):
    with pytest.raises(PyHFOAgentError) as bad_channel:
        waveform_window(tiny_fif_path, channels=["missing"])
    assert bad_channel.value.code == "channel_not_found"

    with pytest.raises(PyHFOAgentError) as bad_time:
        waveform_window(tiny_fif_path, start_seconds=999, duration_seconds=1)
    assert bad_time.value.code == "invalid_time_window"


def test_waveform_overview_and_channels_are_json_safe(tiny_fif_path):
    channels = waveform_channels(tiny_fif_path)
    overview = waveform_overview(tiny_fif_path, channels=["B10Ref"], bins=16)

    assert channels["channels"][0]["name"] == "B10Ref"
    assert overview["bins"] == 16
    assert len(overview["min"][0]) == 16


def test_waveform_window_enforces_hard_payload_caps(tiny_fif_path):
    with pytest.raises(PyHFOAgentError) as too_long:
        waveform_window(tiny_fif_path, duration_seconds=120)
    assert too_long.value.code == "request_too_large"

    with pytest.raises(PyHFOAgentError) as too_many_samples:
        waveform_window(tiny_fif_path, target_samples_per_channel=10_000)
    assert too_many_samples.value.code == "request_too_large"

    with pytest.raises(PyHFOAgentError) as too_many_bins:
        waveform_overview(tiny_fif_path, bins=3000)
    assert too_many_bins.value.code == "request_too_large"


def test_waveform_requires_explicit_channels_when_recording_exceeds_channel_cap(monkeypatch, tmp_path):
    from pyhfo.agent_api import waveform as waveform_module
    import numpy as np

    class ManyChannelRaw:
        info = {"sfreq": 200.0, "ch_names": [f"CH{i}" for i in range(40)]}
        n_times = 200

        def get_data(self, picks=None, start=0, stop=None):
            picks = picks if picks is not None else list(range(40))
            return np.zeros((len(picks), (stop or self.n_times) - start))

        def get_channel_types(self, picks=None):
            return ["eeg"]

    path = tmp_path / "many.fif"
    path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(waveform_module, "load_mne_raw", lambda _path: ManyChannelRaw())

    with pytest.raises(PyHFOAgentError) as exc_info:
        waveform_window(path)
    assert exc_info.value.code == "request_too_large"

    selected = waveform_window(path, channels=["CH1"], target_samples_per_channel=10)
    assert selected["channels"] == ["CH1"]


def test_waveform_rejects_excessive_source_samples_before_downsampling(monkeypatch, tmp_path):
    from pyhfo.agent_api import waveform as waveform_module
    import numpy as np

    class HugeRaw:
        info = {"sfreq": 10_000.0, "ch_names": ["A"]}
        n_times = 10_000_000
        def get_data(self, picks=None, start=0, stop=None):  # pragma: no cover - should not be reached
            raise AssertionError("get_data should not run for oversized source windows")
        def get_channel_types(self, picks=None):
            return ["eeg"]

    path = tmp_path / "huge.fif"
    path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(waveform_module, "load_mne_raw", lambda _path: HugeRaw())

    with pytest.raises(PyHFOAgentError) as exc_info:
        waveform_window(path, channels=["A"], duration_seconds=60)
    assert exc_info.value.code == "request_too_large"

def test_waveform_window_applies_notch_filter_and_reports_filter_metadata(monkeypatch, tmp_path):
    from pyhfo.agent_api import waveform as waveform_module
    import numpy as np

    sfreq = 1000.0
    times = np.arange(0, 2.0, 1.0 / sfreq)
    ten_hz = np.sin(2 * np.pi * 10 * times)
    sixty_hz = 0.8 * np.sin(2 * np.pi * 60 * times)
    raw_trace = ten_hz + sixty_hz

    class FilterRaw:
        info = {"sfreq": sfreq, "ch_names": ["A"]}
        n_times = len(times)

        def get_data(self, picks=None, start=0, stop=None):
            return raw_trace[start:stop][None, :]

        def get_channel_types(self, picks=None):
            return ["eeg"]

    path = tmp_path / "filter.fif"
    path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(waveform_module, "load_mne_raw", lambda _path: FilterRaw())

    raw = waveform_window(path, channels=["A"], duration_seconds=2.0, target_samples_per_channel=2000)
    filtered = waveform_window(path, channels=["A"], duration_seconds=2.0, target_samples_per_channel=2000, filter_60hz=True)

    raw_data = np.asarray(raw["data"][0], dtype=float)
    filtered_data = np.asarray(filtered["data"][0], dtype=float)
    freqs = np.fft.rfftfreq(raw_data.size, d=1.0 / sfreq)
    raw_fft = np.abs(np.fft.rfft(raw_data))
    filtered_fft = np.abs(np.fft.rfft(filtered_data))
    idx_60 = int(np.argmin(np.abs(freqs - 60.0)))

    assert filtered["source"] == "filtered"
    assert filtered["filter_request"] == {"show_filtered": False, "filter_60hz": True}
    assert "notch_60hz" in filtered["filters_applied"]
    assert filtered_fft[idx_60] < raw_fft[idx_60] * 0.35


def test_waveform_window_display_band_overrides_default_clinical_band(monkeypatch, tmp_path):
    """An explicit filter_low/filter_high band reveals a high-frequency biomarker
    (e.g. an HFO) that the default 1-80 Hz filtered view would suppress."""
    from pyhfo.agent_api import waveform as waveform_module
    import numpy as np

    sfreq = 1000.0
    times = np.arange(0, 2.0, 1.0 / sfreq)
    low_freq = np.sin(2 * np.pi * 10 * times)          # 10 Hz — in the default 1-80 band
    high_freq = np.sin(2 * np.pi * 150 * times)        # 150 Hz — an HFO-like component
    raw_trace = low_freq + high_freq

    class BandRaw:
        info = {"sfreq": sfreq, "ch_names": ["A"]}
        n_times = len(times)

        def get_data(self, picks=None, start=0, stop=None):
            return raw_trace[start:stop][None, :]

        def get_channel_types(self, picks=None):
            return ["eeg"]

    path = tmp_path / "band.fif"
    path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(waveform_module, "load_mne_raw", lambda _path: BandRaw())

    default_band = waveform_window(path, channels=["A"], duration_seconds=2.0, target_samples_per_channel=2000, show_filtered=True)
    hfo_band = waveform_window(path, channels=["A"], duration_seconds=2.0, target_samples_per_channel=2000, show_filtered=True, filter_low=80, filter_high=250)

    # The label reports the actual band so the renderer/agent can see which view it got.
    assert "bandpass_1_80hz" in default_band["filters_applied"]
    assert "bandpass_80_250hz" in hfo_band["filters_applied"]
    # filter_request stays the {show_filtered, filter_60hz} contract — band lives in filters_applied.
    assert hfo_band["filter_request"] == {"show_filtered": True, "filter_60hz": False}

    freqs = np.fft.rfftfreq(2000, d=1.0 / sfreq)
    idx_150 = int(np.argmin(np.abs(freqs - 150.0)))
    default_fft = np.abs(np.fft.rfft(np.asarray(default_band["data"][0], dtype=float)))
    hfo_fft = np.abs(np.fft.rfft(np.asarray(hfo_band["data"][0], dtype=float)))
    # The 150 Hz HFO is killed by the 1-80 default but preserved by the 80-250 band.
    assert hfo_fft[idx_150] > default_fft[idx_150] * 5
