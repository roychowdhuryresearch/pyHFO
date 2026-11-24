from __future__ import annotations

import json
import zipfile

from pyhfo.agent_api.sessions import export_report, export_session, export_workbook, summarize_saved_session
from pyhfo.utils.session_store import load_session_checkpoint


def _electron_parity_session_state() -> dict:
    event = {
        "event_id": "evt-hfo-001",
        "run_id": "run-ste",
        "channel": "A1",
        "start_seconds": 1.0,
        "end_seconds": 1.18,
        "duration_seconds": 0.18,
        "biomarker": "HFO",
        "features": {"peak_frequency_hz": 120, "confidence": 0.91},
        "annotation": {"label": "accepted", "display_label": "Pathological", "reviewer": "human", "source": "electron-ui", "notes": "fixture review"},
    }
    return {
        "format": "electron-pyqt-parity-fixture",
        "biomarker_type": "HFO",
        "recording_path": "/synthetic/parity_fixture.edf",
        "recording_info": {"filename": "parity_fixture.edf", "sample_freq": 1000, "channels": ["A1", "A2"], "duration_seconds": 5},
        "run_records": [
            {
                "runId": "run-ste",
                "biomarker": "HFO",
                "detector": "STE",
                "events": 1,
                "channels": 1,
                "status": "complete",
                "classifier": "HF CPU classifier",
                "parameters": {
                    "detector_params": json.dumps({"rms_window": 0.003, "min_gap": 0.01, "peak_thres": 3.0}),
                    "classifier_params": json.dumps({"model_type": "default_cpu", "source_preference": "huggingface", "steps": ["artifact", "spike", "ehfo"]}),
                },
            }
        ],
        "review_events": [event],
        "review_events_by_run": {"run-ste": [event]},
        "review_annotations": {"evt-hfo-001": event["annotation"]},
        "active_run_id": "run-ste",
        "accepted_run_id": "run-ste",
        "visible_run_ids": ["run-ste"],
        "selected_review_event_id": "evt-hfo-001",
        "prepare_controls": {"fp": "80", "fs": "250", "rp": "0.5", "rs": "40", "nJobs": 2, "lineNoiseNotchHz": "60"},
        "classifier_controls": {"mode": "HF CPU", "device": "cpu", "batchSize": "32", "hubArtifact": "roychowdhuryresearch/HFO-artifact"},
        "waveform_controls": {"visibleChannels": 8, "timeWindowSeconds": 5, "startSeconds": 0, "montageMode": "referential"},
        "runtime_log": [{"timestamp": "2026-06-01T00:00:00Z", "level": "success", "message": "Fixture complete"}],
    }


def test_electron_parity_fixture_round_trips_session_report_and_workbook(tmp_path, monkeypatch):
    monkeypatch.setenv("PYHFO_AGENT_EXPORT_DIR", str(tmp_path))
    state = _electron_parity_session_state()

    session = export_session({"output_path": str(tmp_path / "parity-fixture.pybrain"), "session_state": state})
    assert session["artifact_type"] == "pybrain_session"

    checkpoint = load_session_checkpoint(session["artifact_path"])
    assert checkpoint["run_records"][0]["runId"] == "run-ste"
    assert checkpoint["review_events_by_run"]["run-ste"][0]["event_id"] == "evt-hfo-001"
    assert checkpoint["waveform_controls"]["visibleChannels"] == 8

    loaded = summarize_saved_session(session["artifact_path"])
    assert loaded["detection_runs"][0]["detector"] == "STE"
    assert loaded["review_events"][0]["annotation"]["display_label"] == "Pathological"
    assert loaded["review_events_by_run"]["run-ste"][0]["channel"] == "A1"
    assert loaded["review_annotations"]["evt-hfo-001"]["source"] == "electron-ui"
    assert loaded["active_run_id"] == "run-ste"
    assert loaded["accepted_run_id"] == "run-ste"
    assert loaded["visible_run_ids"] == ["run-ste"]
    assert loaded["prepare_controls"]["lineNoiseNotchHz"] == "60"
    assert loaded["classifier_controls"]["batchSize"] == "32"

    review_state = {"events": state["review_events"], "annotations": state["review_annotations"]}
    report_payload = {
        "output_path": str(tmp_path / "parity-report.html"),
        "summary": {"run_count": 1, "active_run_id": "run-ste"},
        "runs": state["run_records"],
        "active_run": state["run_records"][0],
        "selected_run": state["run_records"][0],
        "accepted_run_id": "run-ste",
        "visible_run_ids": ["run-ste"],
        "channel_ranking": [{"channel": "A1", "events": 1, "accepted": 1, "total": 1, "priority": "High"}],
        "review_state": review_state,
        "runtime_log": state["runtime_log"],
    }
    report = export_report(report_payload)
    assert report["artifact_type"] == "report_html"
    html = (tmp_path / "parity-report.html").read_text(encoding="utf-8")
    assert "Run Inventory" in html
    assert "Channel Ranking" in html
    assert "Event Preview" in html
    assert "Pathological" in html
    assert "STE" in html
    assert (tmp_path / "parity-report_files" / "review_annotations.csv").exists()
    assert "Fixture complete" in (tmp_path / "parity-report_files" / "runtime_log.json").read_text(encoding="utf-8")

    workbook = export_workbook({
        "output_path": str(tmp_path / "parity-workbook.xlsx"),
        "summary": report_payload["summary"],
        "runs": state["run_records"],
        "channel_ranking": report_payload["channel_ranking"],
        "review_state": review_state,
    })
    assert workbook["artifact_type"] == "workbook_xlsx"
    with zipfile.ZipFile(workbook["artifact_path"]) as zf:
        assert "xl/workbook.xml" in zf.namelist()
        workbook_xml = zf.read("xl/worksheets/sheet4.xml").decode("utf-8")
        assert "Pathological" in workbook_xml
        assert "electron-ui" in workbook_xml
