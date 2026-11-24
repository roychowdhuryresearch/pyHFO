"""Unit tests for the short-recording detection warning (pipeline.short_recording_warning).

A 1.0 s recording (e.g. the anomalous sub-Multicenterumf005 ictal ECoG file) makes STE/MNI return
0 events legitimately; run_detection should explain that rather than report a bare empty result.
"""

from pyhfo.agent_api.pipeline import MIN_DETECTION_DURATION_SECONDS, short_recording_warning


def test_floor_is_one_minute():
    assert MIN_DETECTION_DURATION_SECONDS == 60.0


def test_warns_for_short_zero_event_run():
    warning = short_recording_warning(1.0, 0)
    assert warning is not None
    assert "1.0s" in warning
    assert "60s" in warning
    # A 20 s recording is still below the floor.
    assert short_recording_warning(20.0, 0) is not None


def test_no_warning_when_events_found():
    # Events present -> the run worked; never explain it as "too short".
    assert short_recording_warning(1.0, 5) is None
    assert short_recording_warning(40.0, 9) is None


def test_no_warning_for_adequate_or_unknown_duration():
    # Boundary: exactly the floor is adequate.
    assert short_recording_warning(60.0, 0) is None
    assert short_recording_warning(71.0, 0) is None
    # Unknown / zero / negative duration -> cannot conclude "too short".
    assert short_recording_warning(None, 0) is None
    assert short_recording_warning(0.0, 0) is None
    assert short_recording_warning(-5.0, 0) is None
