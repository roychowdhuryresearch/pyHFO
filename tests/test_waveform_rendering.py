import numpy as np

from src.models.main_waveform_plot_model import MainWaveformPlotModel
from src.models.mini_plot_model import MiniPlotModel


class FakeEventFeatures:
    def get_biomarkers_for_channel(self, channel_name, min_start=None, max_end=None):
        starts = np.array([50, 2000, 4200], dtype=int)
        ends = np.array([70, 2040, 4260], dtype=int)
        artifacts = np.ones_like(starts)
        spikes = np.zeros_like(starts)
        ehfos = np.zeros_like(starts)
        if min_start is not None and max_end is not None:
            mask = (starts >= min_start) & (ends <= max_end)
            starts = starts[mask]
            ends = ends[mask]
            artifacts = artifacts[mask]
            spikes = spikes[mask]
            ehfos = ehfos[mask]
        return starts, ends, artifacts, spikes, ehfos


class FakeBackend:
    def __init__(self, n_channels=4, n_samples=200_000, sample_freq=1000.0):
        self.sample_freq = sample_freq
        self._channel_names = np.array([f"A{i}Ref" for i in range(1, n_channels + 1)])
        times = np.arange(n_samples) / sample_freq
        self._eeg = np.vstack(
            [
                np.sin(2 * np.pi * 8 * times),
                np.cos(2 * np.pi * 12 * times) * 0.8,
                np.sin(2 * np.pi * 16 * times) * 0.5,
                np.cos(2 * np.pi * 3 * times) * 0.25,
            ]
        )[:n_channels]
        self.event_features = FakeEventFeatures()

    def get_eeg_data(self, start=None, end=None, filtered=False):
        if start is None and end is None:
            return self._eeg, self._channel_names
        return self._eeg[:, start:end], self._channel_names

    def get_edf_info(self):
        return {"sfreq": self.sample_freq}


def test_main_waveform_plot_model_downsamples_long_windows():
    backend = FakeBackend()
    model = MainWaveformPlotModel(backend)

    model.init_eeg_data()
    model.set_render_width_pixels(600)
    model.set_time_window(50)
    model.set_current_time_window(0)

    eeg_data, y_100_length, y_scale_length, offset_value = model.get_all_current_eeg_data_to_display()
    time_axis = model.get_current_time_window()

    assert not hasattr(model, "time")
    assert model.get_total_time() > 0
    assert eeg_data.shape[1] == len(time_axis)
    assert eeg_data.shape[1] <= model._target_render_points() + 1
    assert y_100_length in (50, 100)
    assert y_scale_length > 0
    assert offset_value == 6


def test_mini_plot_model_tracks_total_time_without_full_time_array():
    backend = FakeBackend()
    model = MiniPlotModel(backend)

    model.init_eeg_data()
    starts_in_time, ends_in_time, _ = model.get_all_biomarkers_for_channel_and_color("A1Ref")

    assert not hasattr(model, "time")
    assert model.total_time > 0
    assert np.all(ends_in_time >= starts_in_time)
