"""Regression tests for round-4 bug-hunt fixes (pyhfo compute core)."""

import numpy as np

from pyhfo.utils.utils_io import sort_channel
from pyhfo.spindle_feature import SpindleFeature


# --- Bug #1: sort_channel must not crash on mixed digit-/letter-leading labels ---

def test_sort_channel_handles_mixed_numeric_and_alpha_labels():
    # '1'/'2'/'10' produce int sort tokens; 'ECG' produces a str token. The old code
    # compared int vs str inside sorted() and raised TypeError, aborting recording load.
    idx, names = sort_channel(["1", "2", "10", "ECG", "POL A1", "B2Ref"])
    names = list(names)
    assert set(names) == {"1", "2", "10", "ECG", "POL A1", "B2Ref"}
    # numeric labels still sort numerically (10 after 2, not after 1)
    assert names.index("2") < names.index("10")


def test_sort_channel_pure_numeric_labels():
    idx, names = sort_channel(["3", "1", "20", "2"])
    assert list(names) == ["1", "2", "3", "20"]


# --- Bug #3: a zero-event spindle run must not crash on feature access ---

def test_zero_event_spindle_feature_has_attributes():
    sf = SpindleFeature.construct_from_channel_intervals(
        ["ch1", "ch2"], [np.array([]), np.array([])]
    )
    # Previously raised AttributeError: 'SpindleFeature' has no attribute 'features'.
    assert sf.has_feature() is False
    assert list(sf.spike_predictions) == []
    assert len(sf.features) == 0
