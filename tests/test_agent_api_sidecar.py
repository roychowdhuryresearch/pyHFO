import json

import pytest

from pyhfo.agent_api import sidecar
from pyhfo.agent_api.sidecar import PyHFOSidecarSession, get_capability_manifest, infer_brain_signal_roi_from_channels, load_bids_channel_annotations, load_sidecar_session, save_sidecar_session


def test_manifest_exposes_decoupled_actions_and_missing_review_wrappers():
    manifest = get_capability_manifest()
    names = {action["name"] for action in manifest["actions"]}

    assert manifest["schema_version"] == "pyhfo-sidecar.v1"
    assert "generic_agent_role" in manifest["architecture"]
    assert "optional_dependencies" in manifest
    assert "detector_availability" in manifest
    assert "get_detector_availability" in names
    assert manifest["optional_dependencies"]["yasa"]["detectors"][0] == {"biomarker": "Spindle", "detector": "YASA"}
    if not manifest["optional_dependencies"]["yasa"]["available"]:
        assert "optional 'yasa' package" in manifest["optional_dependencies"]["yasa"]["install_hint"]
    spindle_detectors = {
        detector["detector"]: detector
        for detector in manifest["detector_availability"]["detectors_by_biomarker"]["Spindle"]
    }
    assert manifest["detector_availability"]["recommended_smoke_detectors"]["Spindle"] == "MOLLE"
    assert spindle_detectors["MOLLE"]["status"] == "callable"
    assert spindle_detectors["MOLLE"]["can_run_without_extra_config"] is True
    assert spindle_detectors["A7"]["status"] == "callable"
    assert spindle_detectors["LSM"]["status"] == "callable_requires_configuration"
    assert "parameter_file" in spindle_detectors["LSM"]["required_configuration"]
    assert spindle_detectors["YASA"]["status"] in {"callable", "unavailable_optional_dependency"}
    assert manifest["classifier"]["cpu"]["available"] is True
    assert manifest["classifier"]["gpu"]["available"] in {True, False}
    assert manifest["classifier"]["gpu"]["ui_behavior"]
    assert "GPU classifier" in manifest["classifier"]["gpu"]["ui_behavior"]
    assert {
        "load_recording",
        "infer_brain_signal_roi",
        "prepare_recording",
        "run_detection",
        "get_state",
        "detector_defaults",
        "recommend_parameters",
        "validate_parameters",
        "sweep_parameters",
    }.issubset(names)
    assert {
        "list_events",
        "detect_line_noise",
        "event_snippet",
        "annotate_event",
        "cross_channel_overlap_review",
        "compare_detector_runs",
        "create_consensus_run",
        "export_review_state",
        "export_report",
        "save_project",
        "compare_candidate_channels_to_bids_labels",
    }.issubset(names)
    prepare = next(action for action in manifest["actions"] if action["name"] == "prepare_recording")
    assert prepare["input_schema"]["properties"]["bipolar_montage"]["enum"] == ["none", "auto_adjacent", "pairs"]
    assert prepare["input_schema"]["properties"]["line_noise_notch_hz"]["type"] == "number"
    line_noise = next(action for action in manifest["actions"] if action["name"] == "detect_line_noise")
    assert line_noise["input_schema"]["required"] == ["input_path"]
    assert line_noise["input_schema"]["properties"]["candidate_frequencies_hz"]["items"]["type"] == "number"


def test_manifest_callable_only_hides_missing_wrappers():
    manifest = get_capability_manifest(include_missing=False)
    statuses = {action["name"]: action["status"] for action in manifest["actions"]}
    assert statuses["load_bids_channel_annotations"] == "disabled_by_default"
    assert statuses["compare_candidate_channels_to_bids_labels"] == "disabled_by_default"
    assert all(
        action["status"] == "callable"
        for action in manifest["actions"]
        if action["name"] not in {"load_bids_channel_annotations", "compare_candidate_channels_to_bids_labels"}
    )


def test_manifest_label_actions_become_callable_when_env_enabled(monkeypatch):
    monkeypatch.setenv("PYHFO_AGENT_ENABLE_LABEL_LOADING", "1")

    manifest = get_capability_manifest(include_missing=False)
    statuses = {action["name"]: action["status"] for action in manifest["actions"]}

    assert statuses["load_bids_channel_annotations"] == "callable"
    assert statuses["compare_candidate_channels_to_bids_labels"] == "callable"


def test_sidecar_dispatch_returns_detector_availability():
    result = PyHFOSidecarSession().dispatch("get_detector_availability", {})

    assert result["action"] == "get_detector_availability"
    assert result["result"]["schema_version"] == "pyhfo-detectors.v1"
    assert result["result"]["recommended_smoke_detectors"]["Spindle"] == "MOLLE"
    spindle_detectors = {
        detector["detector"]: detector
        for detector in result["result"]["detectors_by_biomarker"]["Spindle"]
    }
    assert spindle_detectors["MOLLE"]["status"] == "callable"


def test_sidecar_dispatch_returns_detector_defaults():
    result = PyHFOSidecarSession().dispatch(
        "detector_defaults",
        {"biomarker": "HFO", "detector": "STE", "sample_freq": 1000},
    )

    assert result["action"] == "detector_defaults"
    assert result["result"]["status"] == "complete"
    assert result["result"]["source"] == "pyhfo_param_detector_defaults"
    assert result["result"]["detector"] == "STE"
    assert result["result"]["detector_params"]["sample_freq"] == 1000.0


def test_sidecar_dispatch_updates_session_state(monkeypatch):
    monkeypatch.setattr(
        sidecar,
        "get_recording_info",
        lambda path: {"input_path": path, "sample_freq": 2000.0, "channel_names": ["A1", "A2"]},
    )
    # dispatch("prepare_recording"/"run_detection") routes through
    # run_pipeline_isolated; with subprocess mode disabled, it resolves the real
    # target modules inline — fake those chokepoints there.
    monkeypatch.setattr(
        "pyhfo.agent_api.preprocessing.prepare_recording",
        lambda args: {"status": "complete", "input_path": args["input_path"], "output_path": args["output_path"]},
    )
    monkeypatch.setattr(
        "pyhfo.agent_api.pipeline.run_detection",
        lambda args: {"status": "complete", "output_dir": args["output_dir"], "num_events": 3, "artifacts": {"summary_json": "summary.json"}},
    )
    session = PyHFOSidecarSession(session_id="test")

    loaded = session.dispatch("load_recording", {"input_path": "sample.fif"})
    prepared = session.dispatch("prepare_recording", {"input_path": "sample.fif", "output_path": "clean.fif"})
    detected = session.dispatch("run_detection", {"input_path": "clean.fif", "output_dir": "out"})

    assert loaded["state"]["loaded_recording_path"] == "sample.fif"
    assert prepared["state"]["prepared_recording_path"] == "clean.fif"
    assert detected["state"]["last_detection_summary"]["num_events"] == 3
    assert [item["action"] for item in session.action_log] == ["load_recording", "prepare_recording", "run_detection"]


def test_sidecar_run_detection_uses_loaded_roi(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "pyhfo.agent_api.pipeline.run_detection",
        lambda args: captured.update(args) or {"status": "complete", "output_dir": args["output_dir"], "num_events": 0},
    )
    session = PyHFOSidecarSession(session_id="roi-detection")
    session.brain_signal_roi = {
        "brain_signal_channels": ["A1-A2"],
        "default_detection_exclusions": ["EKG1-EKG2", "X1-X2"],
    }

    session.dispatch("run_detection", {"input_path": "sample.fif", "output_dir": "out"})

    assert captured["include_channels"] == ["A1-A2"]
    assert captured["exclude_channels"] == ["EKG1-EKG2", "X1-X2"]


def test_sidecar_read_summary_and_unknown_action(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"status": "complete", "num_events": 2}), encoding="utf-8")
    session = PyHFOSidecarSession()

    result = session.dispatch("read_summary", {"summary_path": str(summary)})

    assert result["result"]["num_events"] == 2
    with pytest.raises(Exception):
        session.dispatch("does_not_exist", {})


def test_infer_brain_signal_roi_excludes_auxiliary_and_uncertain_channels():
    roi = infer_brain_signal_roi_from_channels(["LG01-LG02", "A1", "EKG1-EKG2", "X1-X2", "TRIG1"] )

    assert roi["brain_signal_channels"] == ["LG01-LG02", "A1"]
    assert roi["brain_signal_channel_count"] == 2
    assert [item["channel"] for item in roi["excluded_non_brain_channels"]] == ["EKG1-EKG2", "TRIG1"]
    assert [item["channel"] for item in roi["uncertain_channels"]] == ["X1-X2"]
    assert roi["default_detection_exclusions"] == ["EKG1-EKG2", "TRIG1", "X1-X2"]


def test_load_bids_channel_annotations_counts_soz_resection_and_bad(tmp_path):
    edf = tmp_path / "sub-1_ses-1_task-test_run-01_ieeg.edf"
    edf.write_text("placeholder", encoding="utf-8")
    channels = tmp_path / "sub-1_ses-1_task-test_run-01_channels.tsv"
    channels.write_text(
        "\t".join(["name", "good", "soz", "resection"]) + "\n"
        + "A1-A2\t1\t1\t0\n"
        + "A2-A3\t1\t0\t1\n"
        + "A3-A4\t0\t1\t1\n",
        encoding="utf-8",
    )

    annotations = load_bids_channel_annotations(edf)
    assert annotations is not None

    assert annotations["channels_tsv"] == str(channels)
    assert annotations["soz_channel_count"] == 2
    assert annotations["soz_channels"] == ["A1-A2", "A3-A4"]
    assert annotations["resection_channel_count"] == 2
    assert annotations["soz_or_resection_channel_count"] == 3
    assert annotations["soz_or_resection_channels"] == ["A1-A2", "A2-A3", "A3-A4"]
    assert annotations["bad_channels"] == ["A3-A4"]


def test_sidecar_load_recording_does_not_load_bids_labels_by_default(monkeypatch, tmp_path):
    edf = tmp_path / "sub-1_ses-1_task-test_run-01_ieeg.edf"
    edf.write_text("placeholder", encoding="utf-8")
    (tmp_path / "sub-1_ses-1_task-test_run-01_channels.tsv").write_text(
        "name\tgood\tsoz\tresection\nA1-A2\t1\t1\t0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sidecar,
        "get_recording_info",
        lambda path: {"input_path": str(edf), "sample_freq": 512.0, "channel_names": ["A1-A2"]},
    )

    loaded = PyHFOSidecarSession().dispatch("load_recording", {"input_path": str(edf)})

    assert "bids_channel_annotations" not in loaded["state"]["recording_info"]
    assert loaded["state"]["bids_channel_annotations_loaded"] is False
    assert loaded["state"]["bids_channel_annotations"] is None


def test_sidecar_explicitly_infers_brain_signal_roi(monkeypatch):
    monkeypatch.setattr(
        sidecar,
        "get_recording_info",
        lambda path: {"input_path": path, "sample_freq": 512.0, "channel_names": ["LG01-LG02", "EKG1-EKG2", "X1-X2"]},
    )
    session = PyHFOSidecarSession()
    loaded = session.dispatch("infer_brain_signal_roi", {"channel_names": ["LG01-LG02", "EKG1-EKG2", "X1-X2"]})

    roi = loaded["state"]["brain_signal_roi"]
    assert loaded["state"]["brain_signal_roi_loaded"] is True
    assert roi["brain_signal_channels"] == ["LG01-LG02"]
    assert roi["default_detection_exclusions"] == ["EKG1-EKG2", "X1-X2"]


def test_sidecar_dispatch_blocks_bids_annotations_by_default(monkeypatch, tmp_path):
    edf = tmp_path / "sub-1_ses-1_task-test_run-01_ieeg.edf"
    edf.write_text("placeholder", encoding="utf-8")
    (tmp_path / "sub-1_ses-1_task-test_run-01_channels.tsv").write_text(
        "name\tgood\tsoz\tresection\nA1-A2\t1\t1\t0\n",
        encoding="utf-8",
    )
    session = PyHFOSidecarSession()

    with pytest.raises(Exception, match="label loading is disabled"):
        session.dispatch("load_bids_channel_annotations", {"input_path": str(edf)})


def test_sidecar_dispatch_blocks_bids_label_comparison_by_default(tmp_path):
    edf = tmp_path / "sub-1_ses-1_task-test_run-01_ieeg.edf"
    edf.write_text("placeholder", encoding="utf-8")
    (tmp_path / "sub-1_ses-1_task-test_run-01_channels.tsv").write_text(
        "name\tgood\tsoz\tresection\nA1-A2\t1\t1\t0\n",
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="label comparison is disabled"):
        PyHFOSidecarSession().dispatch(
            "compare_candidate_channels_to_bids_labels",
            {"input_path": str(edf), "candidate_channels": ["A1-A2"]},
        )


def test_sidecar_explicitly_loads_bids_annotations_when_env_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("PYHFO_AGENT_ENABLE_LABEL_LOADING", "1")
    edf = tmp_path / "sub-1_ses-1_task-test_run-01_ieeg.edf"
    edf.write_text("placeholder", encoding="utf-8")
    (tmp_path / "sub-1_ses-1_task-test_run-01_channels.tsv").write_text(
        "name\tgood\tsoz\tresection\nA1-A2\t1\t1\t0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sidecar,
        "get_recording_info",
        lambda path: {"input_path": str(edf), "sample_freq": 512.0, "channel_names": ["A1-A2"]},
    )
    session = PyHFOSidecarSession()
    session.dispatch("load_recording", {"input_path": str(edf)})

    loaded = session.dispatch("load_bids_channel_annotations", {"input_path": str(edf)})

    annotations = loaded["state"]["bids_channel_annotations"]
    assert loaded["state"]["bids_channel_annotations_loaded"] is True
    assert annotations is not None
    assert annotations["soz_channel_count"] == 1
    assert annotations["definitions"]["soz"].startswith("Seizure onset zone")


def test_sidecar_compares_candidates_to_bids_labels_when_env_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("PYHFO_AGENT_ENABLE_LABEL_LOADING", "1")
    edf = tmp_path / "sub-1_ses-1_task-test_run-01_ieeg.edf"
    edf.write_text("placeholder", encoding="utf-8")
    (tmp_path / "sub-1_ses-1_task-test_run-01_channels.tsv").write_text(
        "name\tgood\tsoz\tresection\n"
        "A1-A2\t1\t1\t0\n"
        "A2-A3\t1\t0\t1\n"
        "A3-A4\t0\t0\t0\n",
        encoding="utf-8",
    )
    session = PyHFOSidecarSession()

    compared = session.dispatch(
        "compare_candidate_channels_to_bids_labels",
        {
            "input_path": str(edf),
            "candidate_analysis": {
                "top_channels": [
                    {"channel": "A1-A2", "hfo_count": 3},
                    {"channel": "A3-A4", "hfo_count": 1},
                    {"channel": "A9-A10", "hfo_count": 1},
                ]
            },
        },
    )

    result = compared["result"]
    assert result["schema_version"] == "pyhfo-label-comparison.v1"
    assert result["matches"]["soz"] == ["A1-A2"]
    assert result["matches"]["resection"] == []
    assert result["matches"]["bad_channel"] == ["A3-A4"]
    assert result["missing_label_channels"]["resection"] == ["A2-A3"]
    assert result["candidates_without_label_overlap"] == ["A9-A10"]
    assert result["summary"]["candidate_matches_soz_or_resection_count"] == 1
    assert "Do not use BIDS labels to create or alter" in result["interpretation_policy"][1]
    assert session.bids_channel_annotations is not None
    assert session.candidate_analysis["label_comparison"]["label_source"].endswith("_channels.tsv")


def test_signal_analysis_qc_spectral_plan_and_rank(monkeypatch, tmp_path):
    from pyhfo.agent_api import signal_analysis
    import numpy as np

    class FakeRaw:
        ch_names = ["A1-A2", "A2-A3", "A3-A4", "EKG1-EKG2", "X1-X2"]
        info = {"sfreq": 512.0}
        n_times = 1024

        def get_data(self, picks=None):
            times = np.arange(self.n_times) / self.info["sfreq"]
            data = np.vstack(
                [
                    np.sin(2 * np.pi * 10 * times) * 1e-6,
                    np.sin(2 * np.pi * 60 * times) * 1e-6,
                    np.zeros_like(times),
                    np.sin(2 * np.pi * 1 * times) * 1e-3,
                    np.sin(2 * np.pi * 20 * times) * 1e-6,
                ]
            )
            return data[picks]

    monkeypatch.setattr(signal_analysis, "_read_raw", lambda path: (tmp_path / "case.edf", FakeRaw()))

    qc = signal_analysis.compute_channel_qc({"input_path": "case.edf"})
    assert qc["roi_channel_count"] == 3
    assert "EKG1-EKG2" in qc["excluded_channels"]
    assert "X1-X2" in qc["excluded_channels"]
    assert "A3-A4" in qc["flat_channels"]

    spectral = signal_analysis.compute_spectral_profile({"input_path": "case.edf", "max_channels": 2, "max_psd_points": 32})
    assert spectral["nyquist_hz"] == 256.0
    assert spectral["recommended_hfo_band"] == [80.0, 243.2]
    assert spectral["channels_analyzed"] == ["A1-A2", "A2-A3"]
    assert 8 <= len(spectral["psd_curve"]) <= 32
    assert {"frequency_hz", "db", "relative_power"}.issubset(spectral["psd_curve"][0])
    assert all(0.0 <= point["relative_power"] <= 1.0 for point in spectral["psd_curve"])

    monkeypatch.setattr(
        signal_analysis,
        "get_recording_info",
        lambda path: {
            "input_path": path,
            "sample_freq": 512.0,
            "duration_seconds": 300.0,
            "channel_names": FakeRaw.ch_names,
        },
    )
    plan = signal_analysis.recommend_detector_plan({"input_path": "case.edf"})
    assert [run["biomarker"] for run in plan["recommended_runs"]] == ["HFO", "Spike"]
    assert plan["recommended_runs"][0]["filter"] == {"fp": 80, "fs": 243.2}
    assert set(plan["excluded_channels"]) == {"EKG1-EKG2", "X1-X2"}

    hfo_csv = tmp_path / "hfo.csv"
    hfo_csv.write_text("channel_names,start,end\nA1-A2,0,1\nEKG1-EKG2,0,1\nA1-A2,2,3\n", encoding="utf-8")
    spike_csv = tmp_path / "spike.csv"
    spike_csv.write_text("channel,start,end\nA2-A3,0,1\nA1-A2,1,2\nX1-X2,1,2\n", encoding="utf-8")
    ranked = signal_analysis.rank_candidate_channels(
        {
            "runs": [
                {"biomarker": "HFO", "events_csv": str(hfo_csv)},
                {"biomarker": "Spike", "events_csv": str(spike_csv)},
            ],
            "duration_seconds": 120.0,
            "exclude_channels": ["EKG1-EKG2", "X1-X2"],
            "top_n": 2,
        }
    )
    assert ranked["candidate_channel_count"] == 2
    assert ranked["top_channels"][0]["channel"] == "A1-A2"
    assert ranked["top_channels"][0]["hfo_count"] == 2
    assert ranked["top_channels"][0]["spike_count"] == 1
    assert all(item["channel"] not in {"EKG1-EKG2", "X1-X2"} for item in ranked["top_channels"])


def test_load_recording_resets_recording_scoped_review_and_project_state(monkeypatch):
    paths = ["/tmp/one.fif", "/tmp/two.fif"]
    def fake_info(path):
        return {"input_path": path, "sample_freq": 200.0, "channel_names": ["A1"]}
    monkeypatch.setattr(sidecar, "get_recording_info", fake_info)
    session = PyHFOSidecarSession()
    session.dispatch("load_recording", {"input_path": paths[0]})
    session.review_events = [{"event_id": "evt"}]
    session.review_events_by_run = {"run-a": [{"event_id": "evt-a", "run_id": "run-a"}]}
    session.review_active_run_id = "run-a"
    session.review_accepted_run_id = "run-a"
    session.review_visible_run_ids = ["run-a"]
    session.review_annotations = {"evt": {"label": "accepted"}}
    session.annotation_audit_trail = [{"event_id": "evt"}]
    session.review_cursor_event_id = "evt"
    session.parameter_plan_history = [{"plan": True}]
    session.project_state = {"project_id": "old"}

    session.dispatch("load_recording", {"input_path": paths[1]})

    assert session.review_events == []
    assert session.review_events_by_run == {}
    assert session.review_active_run_id is None
    assert session.review_accepted_run_id is None
    assert session.review_visible_run_ids == []
    assert session.review_annotations == {}
    assert session.annotation_audit_trail == []
    assert session.review_cursor_event_id is None
    assert session.parameter_plan_history == []
    assert session.project_state is None


def test_dispatch_annotate_event_stages_without_confirmation():
    session = PyHFOSidecarSession()
    events = [{"event_id": "evt-1", "channel": "A1", "start_seconds": 0.0, "end_seconds": 0.1}]

    staged = session.dispatch("annotate_event", {"event_id": "evt-1", "label": "accepted", "events": events})

    assert staged["result"]["status"] == "staged"
    assert session.review_annotations == {}

    applied = session.dispatch("annotate_event", {"event_id": "evt-1", "label": "accepted", "confirmed": True, "events": events})
    assert applied["result"]["status"] == "complete"
    assert session.review_annotations["evt-1"]["label"] == "accepted"


def test_dispatch_annotate_event_preserves_display_label():
    session = PyHFOSidecarSession()
    events = [{"event_id": "evt-1", "channel": "A1", "start_seconds": 0.0, "end_seconds": 0.1}]

    applied = session.dispatch("annotate_event", {"event_id": "evt-1", "label": "accepted", "display_label": "Physiological", "confirmed": True, "events": events})

    assert applied["result"]["annotation"]["label"] == "accepted"
    assert applied["result"]["annotation"]["display_label"] == "Physiological"
    assert session.review_annotations["evt-1"]["display_label"] == "Physiological"


def test_dispatch_annotate_event_requires_boolean_true_confirmation():
    session = PyHFOSidecarSession()
    events = [{"event_id": "evt-1", "channel": "A1", "start_seconds": 0.0, "end_seconds": 0.1}]

    for value in ["false", "0", "true", 1]:
        staged = session.dispatch("annotate_event", {"event_id": "evt-1", "label": "artifact", "confirmed": value, "events": events})
        assert staged["result"]["status"] == "staged"
        assert session.review_annotations == {}

    applied = session.dispatch("annotate_event", {"event_id": "evt-1", "label": "artifact", "confirmed": True, "events": events})
    assert applied["result"]["status"] == "complete"


def test_sidecar_persists_run_scoped_review_state(monkeypatch, tmp_path):
    monkeypatch.setenv("PYHFO_AGENT_SESSION_DIR", str(tmp_path))
    session = PyHFOSidecarSession(session_id="run-scoped")
    session.review_events = [{"event_id": "fallback", "run_id": "fallback"}]
    session.review_events_by_run = {"run-a": [{"event_id": "a-1", "run_id": "run-a"}]}
    session.review_active_run_id = "run-a"
    session.review_accepted_run_id = "run-a"
    session.review_visible_run_ids = ["run-a"]
    session.review_cursor_event_id = "a-1"

    save_sidecar_session(session)
    restored = load_sidecar_session("run-scoped")

    assert restored.review_events_by_run == {"run-a": [{"event_id": "a-1", "run_id": "run-a"}]}
    assert restored.review_active_run_id == "run-a"
    assert restored.review_accepted_run_id == "run-a"
    assert restored.review_visible_run_ids == ["run-a"]
    assert restored.review_cursor_event_id == "a-1"


def test_sidecar_dispatch_review_actions_are_run_scoped():
    session = PyHFOSidecarSession()
    session.review_events_by_run = {
        "run-a": [{"event_id": "a-1", "run_id": "run-a", "channel": "A1", "start_seconds": 0.0, "end_seconds": 0.1}],
        "run-b": [{"event_id": "b-1", "run_id": "run-b", "channel": "B1", "start_seconds": 1.0, "end_seconds": 1.1}],
    }
    session.review_active_run_id = "run-b"

    listed = session.dispatch("list_events", {"run_id": "run-a"})["result"]
    cursor = session.dispatch("review_event_cursor", {"action": "current", "run_id": "run-a"})["result"]
    annotation = session.dispatch("annotate_event", {"event_id": "a-1", "label": "accepted", "confirmed": True, "run_id": "run-a"})["result"]
    overlap = session.dispatch("cross_channel_overlap_review", {"run_id": "run-a"})["result"]

    assert listed["run_id"] == "run-a"
    assert listed["events"][0]["event_id"] == "a-1"
    assert cursor["event"]["run_id"] == "run-a"
    assert annotation["annotation"]["run_id"] == "run-a"
    assert overlap["status"] == "complete"
    assert "b-1" not in {event["event_id"] for event in session.dispatch("list_events", {"run_id": "run-a"})["result"]["events"]}
