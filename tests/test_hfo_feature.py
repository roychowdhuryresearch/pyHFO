import numpy as np

from src.hfo_feature import HFO_Feature


def test_channel_lookup_uses_cached_channel_indexing():
    feature = HFO_Feature(
        np.array(["A1Ref", "A1Ref", "B2Ref"]),
        np.array([[10, 12], [20, 22], [30, 35]]),
        sample_freq=2000,
    )
    feature.update_pred(np.array([1, -1, 1]), np.array([0, 1, 0]), np.array([0, 0, 1]))

    starts, ends, artifacts, spikes, ehfos = feature.get_biomarkers_for_channel("A1Ref", 0, 25)

    assert starts.tolist() == [10, 20]
    assert ends.tolist() == [12, 22]
    assert artifacts.tolist() == [1, -1]
    assert spikes.tolist() == [False, True]
    assert ehfos.tolist() == [False, False]


def test_channel_lookup_keeps_events_that_overlap_window_edges():
    feature = HFO_Feature(
        np.array(["A1Ref"]),
        np.array([[1950, 2050]]),
        sample_freq=1000,
    )
    feature.update_pred(np.array([1]), np.array([0]), np.array([0]))

    starts, ends, artifacts, spikes, ehfos = feature.get_biomarkers_for_channel("A1Ref", 2000, 2100)

    assert starts.tolist() == [1950]
    assert ends.tolist() == [2050]
    assert artifacts.tolist() == [1]
    assert spikes.tolist() == [False]
    assert ehfos.tolist() == [False]
