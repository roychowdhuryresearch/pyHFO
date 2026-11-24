from __future__ import annotations

import json
import os
import threading
import zipfile

import numpy as np
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pyhfo.agent_api.server import build_server
from pyhfo.agent_api.sidecar import PyHFOSidecarSession
from pyhfo.hfo_feature import HFO_Feature
from pyhfo.param.param_filter import ParamFilter
from pyhfo.utils.analysis_session import AnalysisSession, DetectionRun
from pyhfo.utils.session_store import save_session_checkpoint


def _json(url, *, payload=None):
    if payload is None:
        with urlopen(url, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    req = Request(url, data=json.dumps(payload).encode("utf-8"), headers={"content-type": "application/json"}, method="POST")
    with urlopen(req, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def test_backend_server_core_contracts(tiny_fif_path, tmp_path, monkeypatch):
    monkeypatch.setenv("PYHFO_AGENT_EXPORT_DIR", str(tmp_path))
    # /runs/classify now routes through run_pipeline_isolated, which (inline, the
    # test default) resolves the function from the pipeline module — so fake it there.
    monkeypatch.setattr("pyhfo.agent_api.pipeline.classify_existing_run", lambda payload: {"status": "complete", "active_run_id": payload.get("run_id"), "biomarker": payload.get("biomarker"), "detector": "STE", "num_events": 1, "artifacts": {"session": payload.get("session_path")}, "classifier_metadata": {"classification_mode": "active_run_session"}})
    session = PyHFOSidecarSession(session_id="server-test")
    session.review_events = [{"event_id": "evt-1", "run_id": "run", "channel": "B10Ref", "start_seconds": 0.1, "end_seconds": 0.2, "duration_seconds": 0.1}]
    server = build_server("127.0.0.1", 0, session)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = server.server_address
    host, port = str(address[0]), int(address[1])
    base = f"http://{host}:{port}"
    try:
        assert _json(base + "/health")["status"] == "ok"
        assert _json(base + "/manifest")["schema_version"] == "pyhfo-sidecar.v1"
        loaded = _json(base + "/recordings/load", payload={"input_path": str(tiny_fif_path)})
        assert loaded["result"]["sample_freq"] == 200.0
        qs = urlencode({"input_path": str(tiny_fif_path), "channels": "B10Ref", "duration_seconds": 1, "target_samples_per_channel": 20})
        window = _json(base + "/waveform/window?" + qs)
        assert window["returned_samples_per_channel"] <= 20
        plan = _json(base + "/parameters/recommend", payload={"biomarker": "Spike", "sample_freq": 200.0})
        assert plan["ui_card"]["type"] == "parameter_plan"
        defaults = _json(base + "/parameters/detector-defaults", payload={"biomarker": "Spike", "detector": "RMS/LL", "sample_freq": 200.0, "n_jobs": 2})
        assert defaults["ui_values"]["rms_window"] == "0.02"
        assert defaults["detector_params"]["n_jobs"] == 2
        events = _json(base + "/events")
        assert events["count"] == 1
        session.review_events_by_run["other-run"] = [{"event_id": "other-1", "run_id": "other-run", "channel": "C1", "start_seconds": 0.5, "end_seconds": 0.6}]
        run_scoped_events = _json(base + "/events?" + urlencode({"run_id": "other-run"}))
        assert run_scoped_events["run_id"] == "other-run"
        assert run_scoped_events["events"][0]["channel"] == "C1"
        feature_summary = _json(base + "/events/evt-1")
        assert feature_summary["summary"]["event_id"] == "evt-1"
        assert feature_summary["summary"]["channel"] == "B10Ref"
        posted_summary = _json(base + "/events/feature-summary", payload={"event_id": "evt-project", "run_id": "project-run", "events": [{"event_id": "evt-project", "run_id": "project-run", "channel": "A1", "duration_seconds": 0.3, "features": {"score": "0.7"}}]})
        assert posted_summary["summary"]["channel"] == "A1"
        assert posted_summary["summary"]["features"]["score"] == "0.7"
        line_noise = _json(base + "/recordings/line-noise", payload={"input_path": str(tiny_fif_path), "candidate_frequencies_hz": [60], "max_duration_seconds": 1.0})
        assert line_noise["status"] == "complete"
        assert line_noise["candidate_frequencies_hz"] == [60.0]
        posted_snippet = _json(base + "/events/snippet", payload={"event_id": "evt-1", "run_id": "run", "events": session.review_events, "input_path": str(tiny_fif_path), "padding_seconds": 0.05, "target_samples_per_channel": 32})
        assert posted_snippet["event"]["event_id"] == "evt-1"
        assert posted_snippet["waveform"]["returned_samples_per_channel"] <= 32
        staged = _json(base + "/events/evt-1/annotation", payload={"label": "accepted", "events": session.review_events})
        assert staged["status"] == "staged"
        annotation = _json(base + "/events/evt-1/annotation", payload={"label": "accepted", "confirmed": True, "events": session.review_events})
        assert annotation["annotation"]["label"] == "accepted"
        exported = _json(base + "/exports/session", payload={"output_path": str(tmp_path / "session.json")})
        assert exported["artifact_type"] == "session_json"
        pybrain = _json(base + "/exports/session", payload={"output_path": str(tmp_path / "session.pybrain"), "session_state": {"biomarker_type": "HFO", "run_records": [{"runId": "run"}], "review_events": session.review_events, "active_run_id": "run"}})
        assert pybrain["artifact_type"] == "pybrain_session"
        loaded_pybrain = _json(base + "/sessions/open-saved", payload={"session_path": pybrain["artifact_path"]})
        assert loaded_pybrain["detection_runs"][0]["runId"] == "run"
        report = _json(base + "/exports/report", payload={"output_path": str(tmp_path / "report.json"), "summary": {"run_count": 1}})
        assert report["artifact_type"] == "report_json"
        report_csv = _json(base + "/exports/report", payload={"output_path": str(tmp_path / "report.csv"), "summary": {"run_count": 1}})
        assert report_csv["artifact_type"] == "report_csv"
        assert "section,key,value" in (tmp_path / "report.csv").read_text(encoding="utf-8")
        report_html = _json(base + "/exports/report", payload={"output_path": str(tmp_path / "report.html"), "summary": {"run_count": 1}})
        assert report_html["artifact_type"] == "report_html"
        assert "Analysis Report" in (tmp_path / "report.html").read_text(encoding="utf-8")
        assert (tmp_path / "report_files" / "metadata.json").exists()
        classified = _json(base + "/runs/classify", payload={"session_path": pybrain["artifact_path"], "run_id": "run", "biomarker": "HFO", "classifier": {"enabled": True}})
        assert classified["active_run_id"] == "run"
        assert classified["classifier_metadata"]["classification_mode"] == "active_run_session"
        explicit_overlap = _json(base + "/events/overlap-review", payload={"run_id": "project-run", "events": [{"event_id": "left", "run_id": "project-run", "channel": "A1", "start_seconds": 0.1, "end_seconds": 0.4}, {"event_id": "right", "run_id": "project-run", "channel": "A2", "start_seconds": 0.2, "end_seconds": 0.3}]})
        assert explicit_overlap["overlap_count"] == 1
        assert explicit_overlap["overlaps"][0]["left_event_id"] == "left"
        consensus = _json(base + "/runs/consensus", payload={"strategy": "union", "run_ids": ["run-a", "run-b"], "events_by_run": {"run-a": [{"event_id": "a", "run_id": "run-a", "channel": "B10Ref", "start_seconds": 0.1, "end_seconds": 0.2}], "run-b": [{"event_id": "b", "run_id": "run-b", "channel": "B10Ref", "start_seconds": 0.3, "end_seconds": 0.4}]}})
        assert consensus["run"]["detector"] == "Consensus Union"
        assert consensus["run"]["parameters"]["backend_consensus"] is True
        assert len(consensus["events"]) == 2
        comparison = _json(base + "/runs/compare", payload={"run_ids": ["run-a", "run-b"], "events_by_run": {"run-a": [{"event_id": "a", "run_id": "run-a", "channel": "B10Ref", "start_seconds": 0.1, "end_seconds": 0.2}], "run-b": [{"event_id": "b", "run_id": "run-b", "channel": "B10Ref", "start_seconds": 0.15, "end_seconds": 0.25}]}})
        assert comparison["stable_channels"][0]["channel"] == "B10Ref"
        assert comparison["temporal_overlap_count"] == 1
        candidate_csv = tmp_path / "candidate_events.csv"
        candidate_csv.write_text("channel,start,end\nB10Ref,0,1\nB10Ref,2,3\n", encoding="utf-8")
        ranking = _json(base + "/detections/rank-candidates", payload={"runs": [{"biomarker": "HFO", "events_csv": str(candidate_csv)}], "duration_seconds": 120, "top_n": 5})["result"]
        assert ranking["status"] == "complete"
        assert ranking["top_channels"][0]["channel"] == "B10Ref"
        assert ranking["top_channels"][0]["hfo_count"] == 2
        workbook = _json(base + "/exports/workbook", payload={"output_path": str(tmp_path / "workbook.json"), "runs": [{"runId": "run"}]})
        assert workbook["artifact_type"] == "workbook_json"
        assert os.path.exists(workbook["artifact_path"])
        xlsx = _json(base + "/exports/workbook", payload={"output_path": str(tmp_path / "workbook.xlsx"), "runs": [{"runId": "run"}], "summary": {"run_count": 1}})
        assert xlsx["artifact_type"] == "workbook_xlsx"
        with zipfile.ZipFile(xlsx["artifact_path"]) as zf:
            assert "xl/workbook.xml" in zf.namelist()
    finally:
        server.shutdown(); server.server_close()


def test_backend_server_structured_error(tiny_fif_path):
    server = build_server("127.0.0.1", 0, PyHFOSidecarSession(session_id="errors"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = server.server_address
    host, port = str(address[0]), int(address[1])
    try:
        try:
            _json(f"http://{host}:{port}/waveform/window?" + urlencode({"input_path": str(tiny_fif_path), "channels": "missing"}))
        except HTTPError as exc:
            body = json.loads(exc.read().decode("utf-8"))
        else:  # pragma: no cover
            raise AssertionError("expected HTTPError")
    finally:
        server.shutdown(); server.server_close()
    assert body["status"] == "error"
    assert body["error"]["code"] == "channel_not_found"


def test_backend_server_rejects_missing_token_when_configured(monkeypatch):
    monkeypatch.setenv("PYHFO_ELECTRON_TOKEN", "secret-token")
    server = build_server("127.0.0.1", 0, PyHFOSidecarSession(session_id="auth"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = server.server_address
    host, port = str(address[0]), int(address[1])
    try:
        try:
            _json(f"http://{host}:{port}/health")
        except HTTPError as exc:
            body = json.loads(exc.read().decode("utf-8"))
            assert exc.code == 401
        else:  # pragma: no cover
            raise AssertionError("expected unauthorized")
        req = Request(f"http://{host}:{port}/health", headers={"authorization": "Bearer secret-token"})
        with urlopen(req, timeout=5) as response:
            ok = json.loads(response.read().decode("utf-8"))
        assert ok["status"] == "ok"
    finally:
        server.shutdown(); server.server_close()


def test_backend_server_malformed_json_returns_400(monkeypatch):
    server = build_server("127.0.0.1", 0, PyHFOSidecarSession(session_id="bad-json"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = server.server_address
    host, port = str(address[0]), int(address[1])
    try:
        req = Request(f"http://{host}:{port}/parameters/recommend", data=b"{bad", headers={"content-type": "application/json"}, method="POST")
        try:
            urlopen(req, timeout=5)
        except HTTPError as exc:
            body = json.loads(exc.read().decode("utf-8"))
            assert exc.code == 400
        else:  # pragma: no cover
            raise AssertionError("expected malformed JSON failure")
    finally:
        server.shutdown(); server.server_close()
    assert body["error"]["code"] == "invalid_request"


def test_backend_open_saved_session_hydrates_events_and_cursor(tmp_path):
    analysis = AnalysisSession("HFO")
    run = DetectionRun.create(
        biomarker_type="HFO",
        detector_name="STE",
        selected_channels=np.array(["A1", "A2"]),
        param_filter=ParamFilter(fp=40, fs=80, rp=0.5, rs=93, sample_freq=200),
        event_features=HFO_Feature(np.array(["A1", "A2"]), np.array([[20, 40], [80, 120]]), np.array([]), "STE", 200),
    )
    analysis.add_run(run)
    preview_path = tmp_path / "preview.pybrain"
    save_session_checkpoint(preview_path, {"biomarker_type": "HFO", "analysis_session": analysis.to_dict()})

    preview_session = PyHFOSidecarSession(session_id="preview")
    preview_server = build_server("127.0.0.1", 0, preview_session)
    preview_thread = threading.Thread(target=preview_server.serve_forever, daemon=True)
    preview_thread.start()
    preview_host, preview_port = str(preview_server.server_address[0]), int(preview_server.server_address[1])
    try:
        preview = _json(f"http://{preview_host}:{preview_port}/sessions/open-saved", payload={"session_path": str(preview_path)})
        analysis.review_cursor_event_id = preview["review_events"][1]["event_id"]
    finally:
        preview_server.shutdown(); preview_server.server_close()

    session_path = tmp_path / "case.pybrain"
    save_session_checkpoint(session_path, {"biomarker_type": "HFO", "analysis_session": analysis.to_dict()})
    session = PyHFOSidecarSession(session_id="saved-session")
    server = build_server("127.0.0.1", 0, session)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = str(server.server_address[0]), int(server.server_address[1])
    try:
        loaded = _json(f"http://{host}:{port}/sessions/open-saved", payload={"session_path": str(session_path)})
        events = _json(f"http://{host}:{port}/events")
        first_run_id = loaded["detection_runs"][0]["runId"]
        active_run_id = loaded["active_run_id"]
        first_run_events = _json(f"http://{host}:{port}/events?" + urlencode({"run_id": first_run_id}))
        active_run_events = _json(f"http://{host}:{port}/events?" + urlencode({"run_id": active_run_id}))
        cursor = _json(f"http://{host}:{port}/events/cursor", payload={"run_id": first_run_id, "action": "current"})
        annotation = _json(f"http://{host}:{port}/events/{first_run_events['events'][0]['event_id']}/annotation", payload={"run_id": first_run_id, "label": "accepted", "confirmed": True})
    finally:
        server.shutdown(); server.server_close()

    assert loaded["event_count"] == 2
    assert loaded["review_run_filters"][first_run_id]["fp"] == 40
    assert session.saved_session_path == str(session_path)
    assert session.review_run_filters[first_run_id]["fs"] == 80
    assert session.last_detection["request"]["filter"]["fp"] == 40
    assert session.review_cursor_event_id == loaded["cursor_event_id"]
    assert session.review_events_by_run[first_run_id]
    assert events["cursor_event_id"] == loaded["cursor_event_id"]
    assert events["count"] == 2
    assert first_run_events["count"] == 2
    assert active_run_events["count"] == 2
    assert cursor["run_id"] == first_run_id
    assert annotation["annotation"]["run_id"] == first_run_id
