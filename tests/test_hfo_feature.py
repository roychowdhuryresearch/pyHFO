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
