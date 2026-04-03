import numpy as np

from src.utils.utils_io import get_edf_info, load_mne_raw, read_eeg_data, sort_channel


class FakeRaw:
    def __init__(self):
        self.info = {"ch_names": ["B10Ref", "A2Ref", "A1Ref"]}
        self._data = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )

    def get_data(self):
        return self._data


def test_sort_channel_orders_numeric_suffixes():
    _, ordered = sort_channel(np.array(["B10Ref", "A2Ref", "A1Ref"]))
    assert ordered.tolist() == ["A1Ref", "A2Ref", "B10Ref"]


def test_read_eeg_data_reads_in_bulk_and_scales_to_microvolts():
    eeg_data, channel_names = read_eeg_data(FakeRaw())
    assert channel_names.tolist() == ["A1Ref", "A2Ref", "B10Ref"]
    np.testing.assert_array_equal(
        eeg_data,
        np.array(
            [
                [7.0e6, 8.0e6, 9.0e6],
                [4.0e6, 5.0e6, 6.0e6],
                [1.0e6, 2.0e6, 3.0e6],
            ]
        ),
    )


def test_load_mne_raw_supports_small_fif_dataset(tiny_fif_path):
    raw = load_mne_raw(str(tiny_fif_path))

    assert raw.info["sfreq"] == 200.0
    assert raw.n_times == 400
    assert raw.info["ch_names"] == ["B10Ref", "A2Ref", "A1Ref"]


def test_get_edf_info_reads_metadata_from_small_mne_dataset(tiny_mne_raw):
    info = get_edf_info(tiny_mne_raw)

    assert info["sfreq"] == 200.0
    assert info["nchan"] == 3
    assert info["channels"] == ["B10Ref", "A2Ref", "A1Ref"]
