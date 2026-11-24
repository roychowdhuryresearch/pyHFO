"""P1-ER backend parity: review events must surface the classifier-prediction
columns PyQt uses for review scoping (artifact/spike/ehfo/overlap), index-aligned
with the visible event rows, so the Electron Event Review prediction scopes
(Artifact / Non-artifact / spkHFO / eHFO / Overlap tagged) work on real recordings."""

import numpy as np

from pyhfo.agent_api.sessions import _review_events_for_run


class _FakeFeatures:
    def __init__(self, n=3, artifact=None, spike=None, ehfo=None, overlap=None,
                 artifact_predicted=True, spike_predicted=True, ehfo_predicted=True,
                 visible_mask=None):
        self.sample_freq = 1000.0
        self.channel_names = np.array([f"CH{i}" for i in range(n)])
        self.starts = np.arange(n) * 100
        self.ends = self.starts + 50
        self.artifact_predictions = np.array(artifact if artifact is not None else [1] * n, dtype=float)
        self.spike_predictions = np.array(spike if spike is not None else [0] * n, dtype=float)
        self.ehfo_predictions = np.array(ehfo if ehfo is not None else [0] * n, dtype=float)
        self.overlap_tagged = np.array(overlap if overlap is not None else [0] * n, dtype=int)
        self.artifact_predicted = artifact_predicted
        self.spike_predicted = spike_predicted
        self.ehfo_predicted = ehfo_predicted
        self._visible_mask = visible_mask

    def get_visible_array(self, attr_name):
        values = np.array(getattr(self, attr_name, np.array([])))
        if self._visible_mask is None or len(values) == 0 or len(values) != len(self.starts):
            return values
        return values[self._visible_mask]


class _FakeRun:
    def __init__(self, features, run_id="run-1", biomarker_type="HFO"):
        self.event_features = features
        self.run_id = run_id
        self.biomarker_type = biomarker_type


def test_prediction_columns_surface_in_features():
    f = _FakeFeatures(n=3, artifact=[1, 0, 1], spike=[1, 0, 0], ehfo=[0, 0, 1], overlap=[0, 0, 1])
    events = _review_events_for_run(_FakeRun(f))
    assert len(events) == 3
    assert events[0]["features"]["artifact_prediction"] == 1.0  # non-artifact (real)
    assert events[0]["features"]["spike_prediction"] == 1.0     # spkHFO
    assert events[0]["features"]["ehfo_prediction"] == 0.0
    assert events[1]["features"]["artifact_prediction"] == 0.0  # artifact
    assert events[2]["features"]["ehfo_prediction"] == 1.0      # eHFO
    assert events[2]["features"]["overlap_tagged"] == 1
    # original index payload preserved
    assert events[0]["features"]["start_index"] == 0


def test_zero_overlap_is_omitted_to_avoid_scope_false_positive():
    f = _FakeFeatures(n=2, overlap=[0, 0])
    events = _review_events_for_run(_FakeRun(f))
    for event in events:
        assert "overlap_tagged" not in event["features"]


def test_unpredicted_columns_are_omitted():
    f = _FakeFeatures(n=2, artifact_predicted=False, spike_predicted=False, ehfo_predicted=False)
    events = _review_events_for_run(_FakeRun(f))
    for event in events:
        assert "artifact_prediction" not in event["features"]
        assert "spike_prediction" not in event["features"]
        assert "ehfo_prediction" not in event["features"]


def test_predictions_stay_index_aligned_when_overlap_events_are_hidden():
    # Hide the middle event (visibility mask drops index 1). The visible rows are
    # events 0 and 2, and their predictions must follow the same mask.
    mask = np.array([True, False, True])
    f = _FakeFeatures(n=3, artifact=[1, 0, 0], spike=[1, 0, 0], ehfo=[0, 0, 1], visible_mask=mask)
    events = _review_events_for_run(_FakeRun(f))
    assert len(events) == 2
    assert events[0]["channel"] == "CH0"
    assert events[0]["features"]["artifact_prediction"] == 1.0
    # second visible event is the original index 2 -> ehfo prediction 1.0, artifact 0.0
    assert events[1]["channel"] == "CH2"
    assert events[1]["features"]["artifact_prediction"] == 0.0
    assert events[1]["features"]["ehfo_prediction"] == 1.0
