from __future__ import annotations

import numpy as np

from pyhfo.agent_api.review import annotate_event, compare_detector_runs, event_cursor, event_feature_summary, event_snippet, events_from_csv, export_review_state, list_events, overlap_review, review_time_frequency
from pyhfo.utils.session_store import save_session_checkpoint
from pyhfo.agent_api.sidecar import PyHFOSidecarSession


def _events():
    return [
        {"event_id": "evt-1", "run_id": "run", "channel": "B10Ref", "start_seconds": 0.2, "end_seconds": 0.4, "duration_seconds": 0.2, "features": {"score": "0.9"}},
        {"event_id": "evt-2", "run_id": "run", "channel": "A2Ref", "start_seconds": 0.3, "end_seconds": 0.5, "duration_seconds": 0.2, "features": {"score": "0.8"}},
    ]


def test_event_cursor_annotation_history_and_export_state():
    session = PyHFOSidecarSession()
    session.review_events = _events()

    selected = event_cursor(session, action="current")
    assert selected["event"]["event_id"] == "evt-1"
    assert list_events(session)["cursor_event_id"] == "evt-1"
    assert event_cursor(session, action="next")["event"]["event_id"] == "evt-2"
    assert event_cursor(session, action="next")["event"]["event_id"] == "evt-1"
    assert event_cursor(session, action="select", event_id="evt-2")["cursor"]["index"] == 1
    assert list_events(session)["cursor"]["event_id"] == "evt-2"

    staged = annotate_event(session, "evt-1", label="artifact", notes="motion", events=_events())
    assert staged["status"] == "staged"
    assert list_events(session)["events"][0]["annotation"] is None

    annotated = annotate_event(session, "evt-1", label="artifact", notes="motion", confirmed=True, events=_events())
    assert annotated["annotation"]["notes"] == "motion"
    assert list_events(session)["events"][0]["annotation"]["label"] == "artifact"
    assert export_review_state(session)["annotation_history"][0]["event_id"] == "evt-1"


def test_run_scoped_review_events_cursor_and_annotations():
    session = PyHFOSidecarSession()
    session.review_events = [{"event_id": "fallback", "run_id": "fallback", "channel": "Z", "start_seconds": 9.0, "end_seconds": 9.1}]
    session.review_events_by_run = {
        "run-a": [
            {"event_id": "a-1", "run_id": "run-a", "channel": "A1", "start_seconds": 0.0, "end_seconds": 0.1},
            {"event_id": "a-2", "run_id": "run-a", "channel": "A2", "start_seconds": 0.2, "end_seconds": 0.3},
        ],
        "run-b": [
            {"event_id": "b-1", "run_id": "run-b", "channel": "B1", "start_seconds": 1.0, "end_seconds": 1.1},
        ],
    }
    session.review_active_run_id = "run-b"

    active = list_events(session)
    run_a = list_events(session, run_id="run-a")
    assert active["run_id"] == "run-b"
    assert [event["event_id"] for event in active["events"]] == ["b-1"]
    assert [event["event_id"] for event in run_a["events"]] == ["a-1", "a-2"]

    moved = event_cursor(session, action="next", run_id="run-a")
    assert moved["run_id"] == "run-a"
    assert moved["event"]["event_id"] == "a-2"
    assert session.review_active_run_id == "run-a"

    annotated = annotate_event(session, "a-2", label="accepted", confirmed=True, run_id="run-a")
    assert annotated["annotation"]["run_id"] == "run-a"
    assert list_events(session, run_id="run-a")["events"][1]["annotation"]["label"] == "accepted"
    assert list_events(session, run_id="run-b")["events"][0]["annotation"] is None
    exported = export_review_state(session)
    assert exported["active_run_id"] == "run-a"
    assert set(exported["events_by_run"]) == {"run-a", "run-b"}


def test_event_feature_summary_and_overlap_review():
    session = PyHFOSidecarSession()
    session.review_events = _events()

    summary = event_feature_summary(session, "evt-1")
    overlap = overlap_review(session)

    assert summary["summary"]["features"]["score"] == "0.9"
    assert summary["summary"]["spectrum"]["status"] == "not_computed"
    assert overlap["overlap_count"] == 1


def test_events_from_pyhfo_csv_converts_sample_indices_using_summary(tmp_path):
    events_csv = tmp_path / "events.csv"
    events_csv.write_text(
        "channel_names,starts,ends,artifact\nA1,3991,4131,0.0\n",
        encoding="utf-8",
    )
    (tmp_path / "summary.json").write_text(
        '{"recording_info":{"sfreq":1000.0}}',
        encoding="utf-8",
    )

    events = events_from_csv(events_csv, run_id="run-1", biomarker="HFO")

    assert events[0]["channel"] == "A1"
    assert events[0]["start_seconds"] == 3.991
    assert events[0]["end_seconds"] == 4.131
    assert np.isclose(events[0]["duration_seconds"], 0.14)
    assert events[0]["features"]["starts"] == "3991"


def test_event_feature_summary_uses_run_scope_when_event_ids_repeat():
    session = PyHFOSidecarSession()
    session.review_events_by_run = {
        "run-a": [{"event_id": "evt-1", "run_id": "run-a", "channel": "A1", "duration_seconds": 0.1, "features": {"score": "0.1"}}],
        "run-b": [{"event_id": "evt-1", "run_id": "run-b", "channel": "B1", "duration_seconds": 0.2, "features": {"score": "0.9"}}],
    }

    summary = event_feature_summary(session, "evt-1", run_id="run-b")

    assert summary["summary"]["channel"] == "B1"
    assert summary["summary"]["features"]["score"] == "0.9"


def test_annotation_preserves_pyqt_display_label_with_normalized_backend_label():
    session = PyHFOSidecarSession()
    session.review_events = _events()

    annotated = annotate_event(session, "evt-1", label="accepted", display_label="Pathological", notes="agent confirmed", confirmed=True, events=_events())

    assert annotated["annotation"]["label"] == "accepted"
    assert annotated["annotation"]["display_label"] == "Pathological"
    listed = list_events(session)["events"][0]["annotation"]
    assert listed["label"] == "accepted"
    assert listed["display_label"] == "Pathological"
    assert export_review_state(session)["annotation_history"][0]["display_label"] == "Pathological"


def test_annotation_confirmation_requires_json_boolean_true():
    session = PyHFOSidecarSession()
    session.review_events = _events()

    for value in ["false", "0", "true", 1, None]:
        result = annotate_event(session, "evt-1", label="accepted", confirmed=(value is True), events=_events())
        assert result["status"] == "staged"
        assert session.review_annotations == {}

    applied = annotate_event(session, "evt-1", label="accepted", confirmed=True, events=_events())
    assert applied["status"] == "complete"
    assert session.review_annotations["evt-1"]["label"] == "accepted"


def test_spindle_previous_cursor_clamps_at_first_event():
    session = PyHFOSidecarSession()
    session.review_events = [{**event, "biomarker": "Spindle"} for event in _events()]

    assert event_cursor(session, action="previous")["event"]["event_id"] == "evt-1"
    assert event_cursor(session, action="next")["event"]["event_id"] == "evt-2"

def test_event_snippet_returns_pyqt_review_plot_payloads(tiny_fif_path):
    session = PyHFOSidecarSession()
    session.review_events = [{"event_id": "evt-1", "run_id": "run", "channel": "B10Ref", "start_seconds": 0.2, "end_seconds": 0.4, "duration_seconds": 0.2, "biomarker": "Spike"}]
    session.last_detection = {"biomarker": "Spike", "request": {"filter": {"fp": 4, "fs": 80, "rp": 0.5, "rs": 93}}}

    snippet = event_snippet(session, "evt-1", input_path=tiny_fif_path, padding_seconds=0.1, target_samples_per_channel=160)

    assert snippet["status"] == "complete"
    assert snippet["waveform"]["returned_samples_per_channel"] <= 160
    assert snippet["filtered_trace"]["status"] in {"complete", "fallback"}
    assert snippet["filtered_trace"]["data"]
    assert len(snippet["filtered_trace"]["data"]) == snippet["waveform"]["returned_samples_per_channel"]
    assert snippet["time_frequency"]["status"] == "complete"
    assert snippet["time_frequency"]["source"] == "pyhfo.utils.utils_plotting.calculate_time_frequency"
    assert snippet["time_frequency"]["high_frequency_at_top"] is True
    assert snippet["time_frequency"]["display_transform"] == "pyqt_log10_percentile"
    assert snippet["time_frequency"]["display_percentiles"] == [8.0, 99.7]
    assert snippet["time_frequency"]["colormap"] == "magma"
    assert len(snippet["time_frequency"]["display_limits"]) == 2
    assert snippet["time_frequency"]["intensity"]
    assert len(snippet["time_frequency"]["intensity"]) <= 48
    assert len(snippet["time_frequency"]["intensity"][0]) <= 160


def test_review_time_frequency_keeps_high_frequency_energy_at_top():
    sample_freq = 2000.0
    samples = 4096
    times = np.arange(samples) / sample_freq
    session = PyHFOSidecarSession()
    session.last_detection = {"biomarker": "HFO", "request": {"filter": {"fp": 1, "fs": 1000}}}
    event = {"event_id": "evt-1", "run_id": "run", "channel": "A1", "biomarker": "HFO"}
    waveform = {
        "sample_freq": sample_freq,
        "times_seconds": times.tolist(),
        "data": [np.zeros(samples).tolist()],
    }

    def peak_frequency_for(tone_hz: float) -> float:
        signal = np.sin(2 * np.pi * tone_hz * times)
        tf = review_time_frequency(session, event, waveform, {"data": signal.tolist(), "filter": {"fp": 1, "fs": 1000}})
        assert tf["status"] == "complete"
        frequencies = np.asarray(tf["frequencies_hz"], dtype=float)
        intensity = np.asarray(tf["intensity"], dtype=float)
        assert frequencies[0] > frequencies[-1]
        row_power = intensity.mean(axis=1)
        return float(frequencies[int(row_power.argmax())])

    low_peak = peak_frequency_for(20.0)
    high_peak = peak_frequency_for(400.0)

    assert low_peak < 80.0
    assert high_peak > 250.0
    assert high_peak > low_peak



def test_backend_consensus_run_uses_run_scoped_events_and_updates_session_state():
    from pyhfo.agent_api.review import create_consensus_run

    session = PyHFOSidecarSession()
    session.review_events_by_run = {
        "run-a": [
            {"event_id": "a-1", "run_id": "run-a", "channel": "A1", "start_seconds": 0.10, "end_seconds": 0.20, "biomarker": "HFO"},
            {"event_id": "a-2", "run_id": "run-a", "channel": "A2", "start_seconds": 1.00, "end_seconds": 1.10, "biomarker": "HFO"},
        ],
        "run-b": [
            {"event_id": "b-1", "run_id": "run-b", "channel": "A1", "start_seconds": 0.15, "end_seconds": 0.25, "biomarker": "HFO"},
        ],
        "run-c": [
            {"event_id": "c-1", "run_id": "run-c", "channel": "A3", "start_seconds": 3.00, "end_seconds": 3.10, "biomarker": "HFO"},
        ],
    }

    majority = create_consensus_run(session, run_ids=["run-a", "run-b", "run-c"], strategy="majority", biomarker="HFO")

    assert majority["status"] == "complete"
    assert majority["support_threshold"] == 2
    assert majority["run"]["detector"] == "Consensus Majority"
    assert majority["run"]["parameters"]["backend_consensus"] is True
    assert len(majority["events"]) == 1
    event = majority["events"][0]
    assert event["channel"] == "A1"
    assert event["features"]["support_count"] == 2
    assert event["features"]["source_run_ids"] == ["run-a", "run-b"]
    assert session.review_active_run_id == majority["run"]["runId"]
    assert session.review_events == majority["events"]
    assert session.review_events_by_run[majority["run"]["runId"]] == majority["events"]
    assert session.review_visible_run_ids[0] == majority["run"]["runId"]


def test_compare_detector_runs_reports_stable_and_run_specific_channels():
    session = PyHFOSidecarSession()
    session.review_events_by_run = {
        "run-a": [
            {"event_id": "a-1", "run_id": "run-a", "channel": "A1", "start_seconds": 0.10, "end_seconds": 0.20, "biomarker": "HFO"},
            {"event_id": "a-2", "run_id": "run-a", "channel": "A2", "start_seconds": 1.00, "end_seconds": 1.10, "biomarker": "HFO"},
        ],
        "run-b": [
            {"event_id": "b-1", "run_id": "run-b", "channel": "A1", "start_seconds": 0.15, "end_seconds": 0.25, "biomarker": "HFO"},
            {"event_id": "b-2", "run_id": "run-b", "channel": "B1", "start_seconds": 2.00, "end_seconds": 2.10, "biomarker": "HFO"},
        ],
    }

    comparison = compare_detector_runs(session, run_ids=["run-a", "run-b"])

    assert comparison["status"] == "complete"
    assert comparison["source_run_ids"] == ["run-a", "run-b"]
    stable = {row["channel"]: row for row in comparison["stable_channels"]}
    assert stable["A1"]["support_count"] == 2
    assert stable["A1"]["stability"] == "present_all_runs"
    run_specific = {row["channel"] for row in comparison["run_specific_channels"]}
    assert run_specific == {"A2", "B1"}
    assert comparison["temporal_overlap_count"] == 1
    assert comparison["temporal_overlaps"][0]["left_event_id"] == "a-1"
    assert "not clinical labels" in comparison["limitations"][1]

def test_event_snippet_falls_back_to_saved_session_eeg_data(tmp_path):
    sample_freq = 200.0
    samples = 300
    times = np.arange(samples) / sample_freq
    eeg_data = np.vstack([np.sin(2 * np.pi * 12 * times) * 25.0, np.cos(2 * np.pi * 8 * times) * 10.0])
    session_path = tmp_path / "saved.pybrain"
    save_session_checkpoint(session_path, {"biomarker_type": "Spike", "sample_freq": sample_freq, "channel_names": np.array(["A1", "A2"]), "recording_channel_names": np.array(["A1", "A2"]), "eeg_data": eeg_data})
    session = PyHFOSidecarSession(saved_session_path=str(session_path))
    session.review_events = [{"event_id": "evt-saved", "run_id": "run", "channel": "A1", "start_seconds": 0.2, "end_seconds": 0.4, "duration_seconds": 0.2, "biomarker": "Spike"}]

    snippet = event_snippet(session, "evt-saved", input_path="", padding_seconds=0.1, target_samples_per_channel=80)

    assert snippet["waveform"]["source"] == "saved_session_eeg_data"
    assert snippet["waveform"]["channels"] == ["A1"]
    assert snippet["waveform"]["returned_samples_per_channel"] <= 80
    assert snippet["filtered_trace"]["data"]
    assert snippet["time_frequency"]["intensity"]
