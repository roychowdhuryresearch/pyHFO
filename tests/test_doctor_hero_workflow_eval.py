from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pytest

from pyhfo.agent_api.sidecar import PyHFOSidecarSession


@pytest.fixture(scope="module")
def doctor_eval_case(tmp_path_factory):
    work_dir = tmp_path_factory.mktemp("doctor_hero_eval")
    input_path = work_dir / "doctor_hero_synthetic_raw.fif"

    sfreq = 1000.0
    duration_seconds = 12.0
    samples = int(sfreq * duration_seconds)
    times = np.arange(samples) / sfreq
    rng = np.random.default_rng(7)
    data = 3e-6 * rng.normal(size=(5, samples))
    data[0] += 30e-6 * np.sin(2 * np.pi * 8 * times)
    data[1] += 20e-6 * np.sin(2 * np.pi * 10 * times)
    hfo_window = (times >= 4.0) & (times <= 4.12)
    data[0, hfo_window] += 80e-6 * np.sin(2 * np.pi * 120 * times[hfo_window])
    spike_start = int(7.0 * sfreq)
    data[1, spike_start : spike_start + 8] += np.hanning(8) * 180e-6
    data[2] += 200e-6 * np.sin(2 * np.pi * 1.2 * times)
    data[4] = 0.0

    info = mne.create_info(["A1", "A2", "EKG1", "X1", "A3"], sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.info["line_freq"] = 60
    raw.save(input_path, overwrite=True, verbose=False)

    session = PyHFOSidecarSession(session_id="doctor-hero-eval")
    loaded = session.dispatch("load_recording", {"input_path": str(input_path)})
    roi = session.dispatch("infer_brain_signal_roi", {"input_path": str(input_path)})
    exclusions = roi["result"]["default_detection_exclusions"]
    qc = session.dispatch("compute_channel_qc", {"input_path": str(input_path), "exclude_channels": exclusions})
    spectral = session.dispatch(
        "compute_spectral_profile",
        {"input_path": str(input_path), "exclude_channels": exclusions, "max_channels": 4},
    )
    detector_plan = session.dispatch("recommend_detector_plan", {"input_path": str(input_path)})

    prepared_path = work_dir / "doctor_hero_synthetic_clean.fif"
    prepared = session.dispatch(
        "prepare_recording",
        {
            "input_path": str(input_path),
            "output_path": str(prepared_path),
            "bipolar_montage": "auto_adjacent",
            "line_noise_notch_hz": 60,
        },
    )

    detection = session.dispatch(
        "run_detection",
        {
            "input_path": str(input_path),
            "output_dir": str(work_dir / "hfo-smoke"),
            "biomarker": "HFO",
            "detector": "RMS",
            "filter": {"fp": 80, "fs": 300},
            "n_jobs": 1,
            "exports": ["summary-json", "csv"],
        },
    )
    ranking = session.dispatch(
        "rank_candidate_channels",
        {
            "runs": [{"biomarker": "HFO", "events_csv": detection["result"]["artifacts"]["csv"]}],
            "duration_seconds": duration_seconds,
            "exclude_channels": exclusions,
            "top_n": 10,
        },
    )
    events = session.dispatch("list_events", {})["result"]
    first_event_id = events["events"][0]["event_id"]
    snippet = session.dispatch(
        "event_snippet",
        {
            "event_id": first_event_id,
            "input_path": str(input_path),
            "padding_seconds": 0.25,
            "target_samples_per_channel": 160,
        },
    )
    overlap = session.dispatch("cross_channel_overlap_review", {})["result"]
    staged = session.dispatch(
        "annotate_event",
        {"event_id": first_event_id, "label": "uncertain", "notes": "Agent suggestion only"},
    )
    after_staged = session.dispatch("list_events", {})["result"]
    confirmed = session.dispatch(
        "annotate_event",
        {"event_id": first_event_id, "label": "uncertain", "notes": "Confirmed by evaluator", "confirmed": True},
    )
    report = session.dispatch(
        "export_report",
        {"output_path": str(work_dir / "doctor_hero_report.html"), "allow_absolute_path": True},
    )

    return {
        "input_path": input_path,
        "work_dir": work_dir,
        "session": session,
        "loaded": loaded,
        "roi": roi,
        "qc": qc,
        "spectral": spectral,
        "detector_plan": detector_plan,
        "prepared": prepared,
        "detection": detection,
        "ranking": ranking,
        "events": events,
        "snippet": snippet,
        "overlap": overlap,
        "staged": staged,
        "after_staged": after_staged,
        "confirmed": confirmed,
        "report": report,
    }


def test_doctor_eval_recording_overview_keeps_clinical_labels_disabled(doctor_eval_case):
    state = doctor_eval_case["loaded"]["state"]
    info = state["recording_info"]

    assert info["sample_freq"] == 1000.0
    assert info["duration_seconds"] == 12.0
    assert info["n_channels"] == 5
    assert info["channel_names"] == ["A1", "A2", "EKG1", "X1", "A3"]
    assert state["bids_channel_annotations_loaded"] is False
    assert state["bids_channel_annotations"] is None


def test_doctor_eval_roi_excludes_non_brain_and_uncertain_channels(doctor_eval_case):
    roi = doctor_eval_case["roi"]["result"]

    assert roi["brain_signal_channels"] == ["A1", "A2", "A3"]
    assert roi["default_detection_exclusions"] == ["EKG1", "X1"]
    assert [item["channel"] for item in roi["excluded_non_brain_channels"]] == ["EKG1"]
    assert [item["channel"] for item in roi["uncertain_channels"]] == ["X1"]
    assert "clinical pathology localization" in " ".join(roi["warnings"])


def test_doctor_eval_qc_flags_flat_channel_without_calling_it_pathology(doctor_eval_case):
    qc = doctor_eval_case["qc"]["result"]

    assert qc["roi_channel_count"] == 3
    assert qc["flat_channels"] == ["A3"]
    assert qc["excluded_channels"] == ["EKG1", "X1"]
    assert any("heuristic" in warning for warning in qc["warnings"])


def test_doctor_eval_hfo_feasibility_uses_nyquist_safe_detector_plan(doctor_eval_case):
    spectral = doctor_eval_case["spectral"]["result"]
    plan = doctor_eval_case["detector_plan"]["result"]
    hfo_run = next(run for run in plan["recommended_runs"] if run["biomarker"] == "HFO")

    assert spectral["nyquist_hz"] == 500.0
    assert spectral["recommended_hfo_band"] == [80.0, 475.0]
    assert hfo_run["filter"]["fs"] <= spectral["nyquist_hz"] * 0.95
    assert hfo_run["n_jobs"] == 1
    assert any("short" in warning.lower() for warning in plan["warnings"])


def test_doctor_eval_preprocessing_writes_clean_copy_without_overwriting_source(doctor_eval_case):
    prepared = doctor_eval_case["prepared"]["result"]
    input_path = doctor_eval_case["input_path"]

    assert prepared["status"] == "complete"
    assert Path(prepared["output_path"]).exists()
    assert Path(prepared["output_path"]) != input_path
    assert prepared["dropped_channels"] == ["A3"]
    assert prepared["bipolar_montage"]["added_channels"] == ["A1-A2"]


def test_doctor_eval_smoke_detection_and_candidate_ranking_are_reviewable(doctor_eval_case):
    detection = doctor_eval_case["detection"]["result"]
    ranking = doctor_eval_case["ranking"]["result"]

    assert detection["status"] == "complete"
    assert detection["num_events"] >= 1
    assert Path(detection["artifacts"]["summary_json"]).exists()
    assert Path(detection["artifacts"]["csv"]).exists()
    assert ranking["status"] == "complete"
    assert ranking["candidate_channel_count"] >= 1
    assert all(item["channel"] not in {"EKG1", "X1"} for item in ranking["top_channels"])
    assert "not clinical labels" in ranking["interpretation_policy"]


def test_doctor_eval_review_queue_event_snippet_and_overlap_are_time_correct(doctor_eval_case):
    events = doctor_eval_case["events"]
    first_event = events["events"][0]
    snippet = doctor_eval_case["snippet"]["result"]
    overlap = doctor_eval_case["overlap"]

    assert events["count"] >= 1
    assert 3.9 <= first_event["start_seconds"] <= 4.1
    assert first_event["end_seconds"] > first_event["start_seconds"]
    assert snippet["status"] == "complete"
    assert snippet["event"]["event_id"] == first_event["event_id"]
    assert snippet["waveform"]["returned_samples_per_channel"] <= 160
    assert snippet["filtered_trace"]["status"] in {"complete", "fallback"}
    assert snippet["time_frequency"]["status"] == "complete"
    assert overlap["status"] == "complete"


def test_doctor_eval_staged_annotation_and_report_export_keep_human_review_boundary(doctor_eval_case):
    staged = doctor_eval_case["staged"]["result"]
    after_staged = doctor_eval_case["after_staged"]
    confirmed = doctor_eval_case["confirmed"]["result"]
    report = doctor_eval_case["report"]["result"]

    assert staged["status"] == "staged"
    assert staged["requires_confirmation"] is True
    assert all(event["annotation"] is None for event in after_staged["events"])
    assert confirmed["status"] == "complete"
    assert confirmed["annotation"]["confirmed"] is True
    assert report["status"] == "complete"
    assert Path(report["artifact_path"]).exists()
    assert "Signal-derived candidates are not clinical diagnoses." in report["report"]["limits"]
