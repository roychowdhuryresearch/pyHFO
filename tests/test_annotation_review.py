import numpy as np

from pyhfo.hfo_feature import HFO_Feature
from pyhfo.spike_feature import SpikeFeature
from pyhfo.spindle_feature import SpindleFeature


def test_hfo_relabel_clears_previous_choice():
    feature = HFO_Feature(np.array(["A1"]), np.array([[10, 20]]), sample_freq=2000)

    feature.doctor_annotation("Artifact")
    feature.doctor_annotation("Pathological")
    feature.doctor_annotation("Physiological")

    assert feature.artifact_annotations.tolist() == [0.0]
    assert feature.pathological_annotations.tolist() == [0.0]
    assert feature.physiological_annotations.tolist() == [1.0]
    assert feature.get_annotation() == "Physiological"


def test_spindle_relabel_to_artifact_clears_spike_flag():
    feature = SpindleFeature(np.array(["A1"]), np.array([10]), np.array([20]), sample_freq=2000)

    feature.doctor_annotation("Spike")
    feature.doctor_annotation("Artifact")

    assert feature.spike_annotations.tolist() == [0.0]
    assert feature.get_annotation() == "Artifact"
    assert feature.get_annotation_counts()["Spike"] == 0


def test_unannotated_navigation_skips_reviewed_events():
    feature = HFO_Feature(
        np.array(["A1", "A1", "A2"]),
        np.array([[10, 20], [30, 40], [50, 60]]),
        sample_freq=2000,
    )
    feature.doctor_annotation("Artifact")

    assert feature.get_prediction_scope_options() == ["All", "Unreviewed"]
    assert feature.get_matching_indexes("Unreviewed").tolist() == [1, 2]

    next_event = feature.get_next_unannotated()
    assert next_event == ("A1", 30, 40)
    assert feature.index == 1

    feature.doctor_annotation("Pathological")
    next_event = feature.get_next_unannotated()
    assert next_event == ("A2", 50, 60)
    assert feature.index == 2


def test_hfo_prediction_scope_navigation_tracks_match_groups():
    feature = HFO_Feature(
        np.array(["A1", "A1", "A2", "A2"]),
        np.array([[10, 20], [30, 40], [50, 60], [70, 80]]),
        sample_freq=2000,
    )
    feature.update_pred(
        np.array([0, 1, 1, 1]),
        np.array([0, 1, 0, 1]),
        np.array([0, 0, 1, 1]),
    )

    assert feature.get_prediction_scope_options() == ["All", "Unreviewed", "Artifact", "Non-artifact", "spkHFO", "eHFO"]
    assert feature.get_matching_indexes("Artifact").tolist() == [0]
    assert feature.get_matching_indexes("spkHFO").tolist() == [1, 3]
    assert feature.get_matching_indexes("eHFO").tolist() == [2, 3]

    next_event = feature.get_next_matching("eHFO")
    assert next_event == ("A2", 50, 60)
    assert feature.index == 2

    prev_event = feature.get_prev_matching("spkHFO")
    assert prev_event == ("A1", 30, 40)
    assert feature.index == 1


def test_hfo_prediction_scope_can_intersect_with_unannotated_only():
    feature = HFO_Feature(
        np.array(["A1", "A1", "A2", "A2"]),
        np.array([[10, 20], [30, 40], [50, 60], [70, 80]]),
        sample_freq=2000,
    )
    feature.update_pred(
        np.array([1, 1, 1, 1]),
        np.array([0, 1, 0, 1]),
        np.array([0, 1, 1, 1]),
    )
    feature.get_jump(1)
    feature.doctor_annotation("Pathological")

    assert feature.get_matching_indexes("eHFO").tolist() == [1, 2, 3]
    assert feature.get_matching_indexes("eHFO", unannotated_only=True).tolist() == [2, 3]

    next_event = feature.get_next_matching("eHFO", unannotated_only=True)
    assert next_event == ("A2", 50, 60)
    assert feature.index == 2


def test_spindle_prediction_scope_navigation_tracks_spike_matches():
    feature = SpindleFeature(
        np.array(["A1", "A2", "A3"]),
        np.array([10, 30, 50]),
        np.array([20, 40, 60]),
        sample_freq=2000,
    )
    feature.update_pred(np.array([0, 1, 1]), np.array([0, 1, 0]))

    assert feature.get_prediction_scope_options() == ["All", "Unreviewed", "Artifact", "Non-artifact", "Spike"]
    assert feature.get_matching_indexes("Unreviewed").tolist() == [0, 1, 2]
    assert feature.get_matching_indexes("Artifact").tolist() == [0]
    assert feature.get_matching_indexes("Spike").tolist() == [1]

    next_event = feature.get_next_matching("Spike")
    assert next_event == ("A2", 30, 40)
    assert feature.index == 1


def test_spike_review_scope_exposes_unreviewed_queue():
    feature = SpikeFeature(
        np.array(["A1", "A2", "A3"]),
        np.array([10, 30, 50]),
        np.array([20, 40, 60]),
        sample_freq=2000,
    )
    feature.doctor_annotation("Accepted")

    assert feature.get_prediction_scope_options() == ["All", "Unreviewed"]
    assert feature.get_matching_indexes("Unreviewed").tolist() == [1, 2]
