from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np

from pyhfo.agent_api.parameters import recommend_parameters
from pyhfo.agent_api.sessions import create_project, export_report, export_session, export_workbook, open_project, save_project, summarize_saved_session
from pyhfo.hfo_feature import HFO_Feature
from pyhfo.param.param_filter import ParamFilter
from pyhfo.utils.analysis_session import AnalysisSession, DetectionRun
from pyhfo.utils.session_store import load_session_checkpoint, save_session_checkpoint


def _assert_artifact_proof(result, path_key="artifact_path"):
    artifact_path = Path(result[path_key])
    assert result["artifact_exists"] is True
    assert artifact_path.is_file()
    assert result["artifact_sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()


def test_project_create_open_save_and_exports(tmp_path, monkeypatch):
    monkeypatch.setenv("PYHFO_AGENT_EXPORT_DIR", str(tmp_path))
    created = create_project({"base_dir": str(tmp_path), "project_id": "case", "recording_path": "/data/case.edf", "parameter_plan_history": [{"id": "plan"}]})
    project_path = created["project_path"]
    opened = open_project(project_path)
    assert opened["state"]["recording_path"] == "/data/case.edf"
    assert opened["state"]["parameter_plan_history"][0]["id"] == "plan"

    saved = save_project({"project_path": project_path, "state": {"project_id": "case", "review_labels": {"evt": "accepted"}}})
    assert saved["state"]["review_labels"]["evt"] == "accepted"
    _assert_artifact_proof(saved, "project_path")

    session = export_session({"output_path": str(tmp_path / "session.json"), "session_state": saved["state"]})
    pybrain_session = export_session({
        "output_path": str(tmp_path / "session.pybrain"),
        "session_state": {
            "biomarker_type": "HFO",
            "recording_path": "/data/case.edf",
            "run_records": [{"runId": "run", "detector": "STE", "events": 1, "channels": 1, "status": "complete"}],
            "review_events": [{"event_id": "evt", "run_id": "run", "channel": "A1"}],
            "review_events_by_run": {"run": [{"event_id": "evt", "run_id": "run", "channel": "A1"}]},
            "review_annotations": {"evt": {"label": "accepted", "display_label": "Pathological", "source": "agent-confirmed"}},
            "active_run_id": "run",
            "visible_run_ids": ["run"],
            "prepare_controls": {"fp": "80", "fs": "250", "nJobs": 2},
            "classifier_controls": {"mode": "HF CPU", "device": "CPU", "batchSize": "64"},
        },
    })
    review_state = {
        "annotations": {"evt": {"label": "accepted", "display_label": "Pathological", "reviewer": "agent", "source": "agent-confirmed", "notes": "confirmed subtype"}},
        "events": [{"event_id": "evt", "run_id": "run", "channel": "A1", "start_seconds": 1.0, "end_seconds": 1.25, "duration_seconds": 0.25, "biomarker": "HFO", "annotation": {"label": "accepted", "display_label": "Pathological"}}],
    }
    report_payload = {
        "summary": {"run_count": 1, "active_run_id": "run", "selected_run_id": "run"},
        "artifacts": {"csv": "events.csv"},
        "runs": [{"runId": "run", "biomarker": "HFO", "detector": "STE", "events": 1, "channels": 1, "status": "complete", "classifier": "not classified", "createdAt": "2026-06-01T00:00:00Z"}],
        "active_run": {"runId": "run"},
        "selected_run": {"runId": "run"},
        "accepted_run_id": "run",
        "visible_run_ids": ["run"],
        "channel_ranking": [{"channel": "A1", "events": 1, "accepted": 1, "total": 1, "priority": "High"}],
        "review_state": review_state,
        "runtime_log": [{"timestamp": "2026-06-01T00:00:01Z", "level": "success", "message": "Exported report"}],
    }
    report = export_report({"output_path": str(tmp_path / "report.json"), **report_payload})
    report_csv = export_report({"output_path": str(tmp_path / "report.csv"), **report_payload})
    png_data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    report_html = export_report({"output_path": str(tmp_path / "report.html"), **report_payload, "waveform_snapshot": {"data_url": png_data_url, "filename": "waveform_snapshot.png", "source": "test"}})
    workbook = export_workbook({"output_path": str(tmp_path / "workbook.json"), "runs": [{"runId": "run"}], "channel_ranking": [{"channel": "A1", "events": 2}], "review_state": review_state})
    xlsx = export_workbook({"output_path": str(tmp_path / "workbook.xlsx"), "summary": {"run_count": 1}, "runs": [{"runId": "run"}], "channel_ranking": [{"channel": "A1", "events": 2}], "review_state": review_state})
    for artifact in (session, pybrain_session, report, report_csv, report_html, workbook, xlsx):
        _assert_artifact_proof(artifact)
    assert session["artifact_path"].endswith("session.json")
    assert pybrain_session["artifact_type"] == "pybrain_session"
    pybrain_checkpoint = load_session_checkpoint(pybrain_session["artifact_path"])
    assert pybrain_checkpoint["active_run_id"] == "run"
    assert pybrain_checkpoint["review_annotations"]["evt"]["display_label"] == "Pathological"
    loaded_pybrain = summarize_saved_session(pybrain_session["artifact_path"])
    assert loaded_pybrain["detection_runs"][0]["runId"] == "run"
    assert loaded_pybrain["review_events"][0]["event_id"] == "evt"
    assert loaded_pybrain["review_annotations"]["evt"]["display_label"] == "Pathological"
    assert loaded_pybrain["prepare_controls"]["fp"] == "80"
    assert loaded_pybrain["classifier_controls"]["batchSize"] == "64"
    assert report["report"]["limits"]
    assert report["report"]["review_annotations"][0]["display_label"] == "Pathological"
    assert report["report"]["review_annotations"][0]["label"] == "accepted"
    assert report["report"]["runs"][0]["runId"] == "run"
    assert report["report"]["channel_ranking"][0]["channel"] == "A1"
    assert report_csv["artifact_type"] == "report_csv"
    report_csv_text = (tmp_path / "report.csv").read_text(encoding="utf-8")
    assert "section,key,value" in report_csv_text
    assert "display_label" in report_csv_text
    assert "Pathological" in report_csv_text
    assert "run_id,biomarker,detector" in report_csv_text
    assert "channel,events,accepted,total,priority" in report_csv_text
    assert "event_id,run_id,channel,start_seconds" in report_csv_text
    assert report_html["artifact_type"] == "report_html"
    report_html_text = (tmp_path / "report.html").read_text(encoding="utf-8")
    report_assets_dir = tmp_path / "report_files"
    assert "<!DOCTYPE html>" in report_html_text
    assert "Analysis Report" in report_html_text
    assert "Report Package Assets" in report_html_text
    assert "Run Inventory" in report_html_text
    assert "Channel Ranking" in report_html_text
    assert "Event Preview" in report_html_text
    assert "Runtime Log" in report_html_text
    assert "Review Annotations" in report_html_text
    assert "Pathological" in report_html_text
    assert "STE" in report_html_text
    assert (report_assets_dir / "metadata.json").exists()
    assert (report_assets_dir / "summary.json").exists()
    assert (report_assets_dir / "artifacts.json").exists()
    assert (report_assets_dir / "runs.json").exists()
    assert (report_assets_dir / "channel_ranking.csv").exists()
    assert (report_assets_dir / "events_preview.csv").exists()
    assert (report_assets_dir / "runtime_log.json").exists()
    assert (report_assets_dir / "review_annotations.csv").exists()
    assert (report_assets_dir / "waveform_snapshot.png").exists()
    assert "Waveform Snapshot" in report_html_text
    assert "data_url_omitted" in (report_assets_dir / "metadata.json").read_text(encoding="utf-8")
    assert "Pathological" in (report_assets_dir / "review_annotations.csv").read_text(encoding="utf-8")
    assert "STE" in (report_assets_dir / "runs.json").read_text(encoding="utf-8")
    assert "A1" in (report_assets_dir / "channel_ranking.csv").read_text(encoding="utf-8")
    assert "Pathological" in (report_assets_dir / "events_preview.csv").read_text(encoding="utf-8")
    assert "Exported report" in (report_assets_dir / "runtime_log.json").read_text(encoding="utf-8")
    assert workbook["artifact_type"] == "workbook_json"
    assert workbook["workbook"]["runs"][0]["runId"] == "run"
    assert workbook["workbook"]["review_annotations"][0]["display_label"] == "Pathological"
    assert xlsx["artifact_type"] == "workbook_xlsx"
    with zipfile.ZipFile(xlsx["artifact_path"]) as zf:
        assert "xl/workbook.xml" in zf.namelist()
        assert "xl/worksheets/sheet1.xml" in zf.namelist()
        assert "xl/worksheets/sheet2.xml" in zf.namelist()
        assert "Pathological" in zf.read("xl/worksheets/sheet4.xml").decode("utf-8")


def test_project_paths_are_scoped_to_project_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PYHFO_AGENT_PROJECT_DIR", str(tmp_path / "projects"))
    created = create_project({"project_id": "safe"})
    assert str(tmp_path / "projects") in created["project_path"]

    from pyhfo.agent_api.errors import PyHFOAgentError
    import pytest
    with pytest.raises(PyHFOAgentError) as exc_info:
        save_project({"project_path": str(tmp_path / "outside.json"), "state": {"project_id": "bad"}})
    assert exc_info.value.code == "invalid_project_path"


def test_first_project_save_returns_proof_for_created_project(tmp_path, monkeypatch):
    monkeypatch.setenv("PYHFO_AGENT_PROJECT_DIR", str(tmp_path / "projects"))

    saved = save_project({"project_id": "first-save", "state": {"review_labels": {"evt": "accepted"}}})

    _assert_artifact_proof(saved, "project_path")


def test_create_project_sanitizes_parent_directory_project_id(tmp_path, monkeypatch):
    monkeypatch.setenv("PYHFO_AGENT_PROJECT_DIR", str(tmp_path / "projects"))
    created = create_project({"project_id": ".."})
    assert str(tmp_path / "projects") in created["project_path"]
    assert ".." not in created["project_id"]


def test_exports_require_scoped_paths_unless_user_selected(tmp_path, monkeypatch):
    monkeypatch.setenv("PYHFO_AGENT_EXPORT_DIR", str(tmp_path / "exports"))
    outside = tmp_path / "chosen" / "session.pybrain"
    from pyhfo.agent_api.errors import PyHFOAgentError
    import pytest
    with pytest.raises(PyHFOAgentError):
        export_session({"output_path": str(outside), "session_state": {"biomarker_type": "HFO"}})
    exported = export_session({"output_path": str(outside), "allow_absolute_path": True, "session_state": {"biomarker_type": "HFO"}})
    assert exported["artifact_type"] == "pybrain_session"
    assert outside.exists()


def test_saved_session_summary_restores_analysis_events_and_cursor(tmp_path):
    analysis = AnalysisSession("HFO")
    run = DetectionRun.create(
        biomarker_type="HFO",
        detector_name="STE",
        selected_channels=np.array(["A1", "A2"]),
        param_filter=ParamFilter(fp=40, fs=80, rp=0.5, rs=93, sample_freq=200),
        event_features=HFO_Feature(np.array(["A1", "A2"]), np.array([[20, 40], [80, 120]]), np.array([]), "STE", 200),
    )
    analysis.add_run(run)
    # Exact suffix is generated from run id; summary must preserve the explicit cursor once known.
    checkpoint_preview = {"biomarker_type": "HFO", "analysis_session": analysis.to_dict()}
    path_preview = tmp_path / "preview.pybrain"
    save_session_checkpoint(path_preview, checkpoint_preview)
    preview = summarize_saved_session(path_preview)
    analysis.review_cursor_event_id = preview["review_events"][1]["event_id"]

    path = tmp_path / "case.pybrain"
    save_session_checkpoint(path, {"biomarker_type": "HFO", "sample_freq": 200, "analysis_session": analysis.to_dict()})

    summary = summarize_saved_session(path)

    assert summary["event_count"] == 2
    assert summary["active_run_id"] == run.run_id
    assert summary["visible_run_ids"] == [run.run_id]
    assert summary["detection_runs"][0]["runId"] == run.run_id
    assert summary["review_events"][0]["channel"] == "A1"
    assert summary["review_events"][0]["start_seconds"] == 0.1
    assert summary["review_events_by_run"][run.run_id][1]["channel"] == "A2"
    assert summary["review_run_filters"][run.run_id]["fp"] == 40
    assert summary["review_run_filters"][run.run_id]["fs"] == 80
    assert summary["cursor_event_id"] == summary["review_events"][1]["event_id"]
    assert summary["analysis_session"]["review_cursor_event_id"] == summary["cursor_event_id"]
    assert summary["recording_info"]["sample_freq"] == 200
    assert recommend_parameters({"biomarker": "HFO", "recording_info": summary["recording_info"]})["validation"]["valid"] is True


def test_saved_session_summary_preserves_multi_run_registry_and_visibility(tmp_path):
    analysis = AnalysisSession("HFO")
    first = DetectionRun.create(
        biomarker_type="HFO",
        detector_name="STE",
        selected_channels=np.array(["A1"]),
        event_features=HFO_Feature(np.array(["A1"]), np.array([[20, 40]]), np.array([]), "STE", 200),
    )
    second = DetectionRun.create(
        biomarker_type="HFO",
        detector_name="MNI",
        selected_channels=np.array(["A2"]),
        event_features=HFO_Feature(np.array(["A2", "A2"]), np.array([[80, 120], [140, 180]]), np.array([]), "MNI", 200),
    )
    analysis.add_run(first)
    analysis.add_run(second)
    analysis.accept_run(first.run_id)
    analysis.set_run_visible(first.run_id, False)
    path = tmp_path / "multi.pybrain"
    save_session_checkpoint(path, {"biomarker_type": "HFO", "analysis_session": analysis.to_dict()})

    summary = summarize_saved_session(path)

    assert [run["runId"] for run in summary["detection_runs"]] == [first.run_id, second.run_id]
    assert summary["active_run_id"] == second.run_id
    assert summary["accepted_run_id"] == first.run_id
    assert summary["visible_run_ids"] == [second.run_id]
    assert len(summary["review_events_by_run"][first.run_id]) == 1
    assert len(summary["review_events_by_run"][second.run_id]) == 2
    assert summary["review_events"][0]["run_id"] == second.run_id
