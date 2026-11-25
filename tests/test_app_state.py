import numpy as np

from src.utils.app_state import APP_STATE_VERSION, build_base_checkpoint, checkpoint_get, checkpoint_version


class DummyApp:
    def __init__(self):
        self.biomarker_type = "HFO"
        self.n_jobs = 2
        self.eeg_data = np.arange(12).reshape(2, 6)
        self.eeg_data_un60 = self.eeg_data
        self.eeg_data_60 = self.eeg_data + 1
        self.edf_param = {"edf_fn": "demo.edf", "sfreq": 2000}
        self.sample_freq = 2000
        self.channel_names = np.array(["A1Ref", "A2Ref"])
        self.param_filter = None
        self.classified = False
        self.filtered = False
        self.detected = False
        self.filter_data = None
        self.filter_data_un60 = None
        self.filter_data_60 = None


def test_build_base_checkpoint_sets_versioned_contract():
    checkpoint = build_base_checkpoint(DummyApp(), "HFO")

    assert checkpoint_version(checkpoint) == APP_STATE_VERSION
    assert checkpoint_get(checkpoint, "biomarker_type") == "HFO"
    np.testing.assert_array_equal(checkpoint["eeg_data_60"], np.arange(12).reshape(2, 6) + 1)


def test_legacy_checkpoint_defaults_to_version_one():
    checkpoint = {"sample_freq": np.array(2000)}
    assert checkpoint_version(checkpoint) == 1
