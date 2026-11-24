import json
import threading
from urllib.request import Request, urlopen

import pytest

from pyhfo.agent_api import sidecar as sidecar_module
from pyhfo.agent_api.sidecar import PyHFOSidecarSession
from pyhfo.agent_api import sidecar_web
from pyhfo.agent_api.sidecar_web import HTML, build_agent_message, build_agent_payload, build_server


def _read_json(url):
    with urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def test_sidecar_html_contains_control_and_nlp_chat_surface():
    assert "PyHFO Agent Side Panel" in HTML
    assert "NLP Agent Chat" in HTML
    assert "Send to ieeg-agent" in HTML
    assert "Visible Trace" in HTML
    assert "visible decision trace" in HTML
    assert "/manifest" in HTML
    assert "/action" in HTML
    assert "/chat" in HTML


def test_build_agent_payload_injects_sidecar_context_and_focused_skills():
    session = PyHFOSidecarSession(session_id="ctx")
    session.recording_info = {"input_path": "/data/case.edf", "sample_freq": 1000.0}

    payload = build_agent_payload("Please clean and run HFO", session, conversation_id="conv", recording_path="/override.edf")

    assert payload["active_skill"] is None
    allowed_skill_names = [item["name"] for item in payload["allowed_skills"]]
    assert allowed_skill_names == sidecar_web.DEFAULT_AGENT_ALLOWED_SKILLS
    assert "eeg-ieeg-pyhfo" not in allowed_skill_names
    assert payload["skill_recommendation_candidates"][0]["name"] == "eeg-format-understanding"
    assert payload["conversation_id"] == "conv"
    assert "Session data" in payload["message"]
    assert "/override.edf" in payload["message"]
    assert "Doctor request" in payload["message"]
    assert "Evidence read" in payload["message"]
    assert "audit summary" in payload["message"]
    assert "clinical_label_context" in payload["message"]
    assert "brain_signal_roi" in payload["message"]
    assert "detector evidence" in payload["message"]


def test_build_agent_message_uses_session_recording_when_no_override():
    session = PyHFOSidecarSession()
    session.recording_info = {"input_path": "/data/current.edf"}

    message = build_agent_message("what should I do?", session)

    assert "/data/current.edf" in message
    assert "Do not give diagnosis" in message
    assert "audit summary" in message
    assert "real brain-signal ROI" in message


def test_sidecar_http_routes_and_chat_proxy(monkeypatch):
    captured = {}

    def fake_call(payload):
        captured.update(payload)
        return {"content": "Agent answer", "active_skill": "eeg-signal-roi", "model_used": "fake"}

    monkeypatch.setattr(sidecar_web, "call_ieeg_agent", fake_call)
    monkeypatch.setattr(sidecar_module, "get_recording_info", lambda path: {"input_path": path, "sample_freq": 1000.0, "channel_names": ["LG01-LG02", "EKG1-EKG2"]})
    server = build_server("127.0.0.1", 0, PyHFOSidecarSession(session_id="http-test"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        manifest = _read_json(f"http://{host}:{port}/manifest")
        state = _read_json(f"http://{host}:{port}/state")
        req = Request(
            f"http://{host}:{port}/action",
            data=json.dumps({"action": "get_state", "arguments": {}}).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=5) as response:
            action = json.loads(response.read().decode("utf-8"))
        chat_req = Request(
            f"http://{host}:{port}/chat",
            data=json.dumps({"message": "Use this real iEEG", "recording_path": "/data/case.edf"}).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urlopen(chat_req, timeout=5) as response:
            chat = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()

    assert manifest["schema_version"] == "pyhfo-sidecar.v1"
    assert state["session_id"] == "http-test"
    assert action["action"] == "get_state"
    assert chat["content"] == "Agent answer"
    assert chat["active_skill"] == "eeg-signal-roi"
    assert chat["ui_commands"][0]["type"] == "agent.show_trace_step"
    assert chat["ui_commands"][0]["requires_confirmation"] is False
    assert captured["active_skill"] is None
    assert "eeg-signal-roi" in [item["name"] for item in captured["allowed_skills"]]
    assert "eeg-ieeg-pyhfo" not in [item["name"] for item in captured["allowed_skills"]]
    assert "/data/case.edf" in captured["message"]
    assert '"clinical_label_context": "not_available_for_ordinary_edf_chat"' in captured["message"]
    assert '"brain_signal_roi_loaded": true' in captured["message"]
    assert '"EKG1-EKG2"' in captured["message"]


def test_chat_response_normalizes_backend_ui_commands(monkeypatch):
    def fake_call(_payload):
        return {
            "content": "Focus the suspicious event and stage a label.",
            "active_skill": "eeg-result-interpretation",
            "ui_commands": [
                {
                    "command_id": "focus-evt",
                    "type": "waveform.center_event",
                    "reason": "Review strongest candidate",
                    "payload": {"event_id": "evt-1"},
                },
                {
                    "command_id": "stage-label",
                    "type": "events.stage_annotation",
                    "payload": {"event_id": "evt-1", "label": "Pathological"},
                },
                {"command_id": "unsafe", "type": "filesystem.delete", "payload": {"path": "/tmp/case"}},
            ],
        }

    monkeypatch.setattr(sidecar_web, "call_ieeg_agent", fake_call)
    response = sidecar_web.handle_chat(
        {"message": "focus and annotate", "ui_state": {"session_id": "renderer-session"}},
        PyHFOSidecarSession(session_id="backend-session"),
    )

    commands = response["ui_commands"]
    assert [command["command_id"] for command in commands] == ["focus-evt", "stage-label"]
    assert commands[0]["type"] == "waveform.center_event"
    assert commands[0]["requires_confirmation"] is False
    assert commands[0]["session_id"] == "renderer-session"
    assert commands[1]["type"] == "events.stage_annotation"
    assert commands[1]["requires_confirmation"] is True
    assert commands[1]["reason"] == "focus and annotate"
    assert all(command["type"] != "filesystem.delete" for command in commands)


def test_chat_response_allows_workflow_commands_and_applies_safety_defaults(monkeypatch):
    def fake_call(_payload):
        return {
            "content": "Run the requested workflow.",
            "ui_commands": [
                {"command_id": "open", "type": "recording.open", "payload": {"input_path": "/data/new.edf"}, "requires_confirmation": False},
                {"command_id": "qc", "type": "analysis.run_qc", "payload": {}, "requires_confirmation": False},
                {"command_id": "detect", "type": "detection.run", "payload": {"biomarker": "HFO", "detector": "STE"}},
                {"command_id": "rank", "type": "detection.rank_candidates", "payload": {}},
                {"command_id": "classify", "type": "classification.run", "payload": {"biomarker": "HFO", "mode": "HF CPU"}},
                {"command_id": "accept", "type": "review.accept_run", "payload": {"run_id": "run-1"}, "requires_confirmation": False},
                {"command_id": "export", "type": "export.workbook", "payload": {"run_id": "run-1"}, "requires_confirmation": False},
                {"command_id": "batch-create", "type": "batch.create", "payload": {"input_dir": "/data/cases"}},
                {"command_id": "batch-run", "type": "batch.run", "payload": {"manifest_path": "/data/cases/batch_project.json"}},
            ],
        }

    monkeypatch.setattr(sidecar_web, "call_ieeg_agent", fake_call)
    response = sidecar_web.handle_chat(
        {"message": "run workflow", "ui_state": {"session_id": "renderer-session"}},
        PyHFOSidecarSession(session_id="backend-session"),
    )

    commands = {command["command_id"]: command for command in response["ui_commands"]}
    assert set(commands) == {
        "open",
        "qc",
        "detect",
        "rank",
        "classify",
        "accept",
        "export",
        "batch-create",
        "batch-run",
    }
    assert commands["open"]["requires_confirmation"] is True
    assert commands["accept"]["requires_confirmation"] is True
    assert commands["export"]["requires_confirmation"] is True
    assert commands["qc"]["requires_confirmation"] is False
    assert commands["detect"]["requires_confirmation"] is True
    assert commands["classify"]["requires_confirmation"] is True
    assert commands["batch-create"]["requires_confirmation"] is True
    assert commands["batch-run"]["requires_confirmation"] is True
    assert all(command["session_id"] == "renderer-session" for command in commands.values())



def test_sidecar_async_chat_reports_running_then_result(monkeypatch):
    def fake_handle(payload, session):
        return {"status": "complete", "content": payload["message"], "active_skill": "eeg-format-understanding"}

    monkeypatch.setattr(sidecar_web, "handle_chat", fake_handle)
    server = build_server("127.0.0.1", 0, PyHFOSidecarSession(session_id="async-test"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        chat_req = Request(
            f"http://{host}:{port}/chat-async",
            data=json.dumps({"message": "hello"}).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urlopen(chat_req, timeout=5) as response:
            submitted = json.loads(response.read().decode("utf-8"))
        assert submitted["status"] == "running"
        result = None
        for _ in range(20):
            result = _read_json(f"http://{host}:{port}/chat-result?job_id={submitted['job_id']}")
            if result["status"] == "complete":
                break
            import time
            time.sleep(0.05)
    finally:
        server.shutdown()
        server.server_close()

    assert result["status"] == "complete"
    assert result["result"]["content"] == "hello"

def test_pathological_request_does_not_load_bids_labels_without_explicit_label_source(monkeypatch):
    captured = {}

    def fake_call(payload):
        captured.update(payload)
        return {"content": "No EDF-only label answer", "active_skill": "eeg-ieeg-pyhfo", "model_used": "fake"}

    monkeypatch.setattr(sidecar_web, "call_ieeg_agent", fake_call)
    monkeypatch.setattr(sidecar_web, "run_signal_candidate_analysis", lambda session, path, message: setattr(session, "candidate_analysis", {"status": "complete", "candidate_ranking": {"top_channels": []}}) or session.candidate_analysis)
    monkeypatch.setattr(sidecar_module, "get_recording_info", lambda path: {"input_path": path, "sample_freq": 512.0, "channel_names": ["LG01-LG02", "EKG1-EKG2"]})
    monkeypatch.setattr(
        sidecar_module,
        "load_bids_channel_annotations",
        lambda path: pytest.fail("pathological alone must not load BIDS labels"),
    )
    server = build_server("127.0.0.1", 0, PyHFOSidecarSession(session_id="pathology-no-label-test"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        chat_req = Request(
            f"http://{host}:{port}/chat",
            data=json.dumps({"message": "How many pathological channels?", "recording_path": "/data/case.edf"}).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urlopen(chat_req, timeout=5) as response:
            chat = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()

    assert chat["content"] == "No EDF-only label answer"
    assert chat["sidecar_state"]["bids_channel_annotations_loaded"] is False
    assert chat["sidecar_state"]["bids_channel_annotations"] is None
    assert '"clinical_label_context": "not_available_for_ordinary_edf_chat"' in captured["message"]
    assert '"brain_signal_roi_loaded": true' in captured["message"]
    assert '"soz_channel_count"' not in captured["message"]


def test_sidecar_chat_does_not_load_labels_by_default_even_for_explicit_label_requests(monkeypatch):
    captured = {}

    def fake_call(payload):
        captured.update(payload)
        return {"content": "Agent label answer", "active_skill": "eeg-ieeg-pyhfo", "model_used": "fake"}

    monkeypatch.setattr(sidecar_web, "call_ieeg_agent", fake_call)
    monkeypatch.setattr(sidecar_module, "get_recording_info", lambda path: {"input_path": path, "sample_freq": 512.0, "channel_names": ["A1-A2"]})
    monkeypatch.setattr(
        sidecar_module,
        "load_bids_channel_annotations",
        lambda path: {"channels_tsv": "/data/case_channels.tsv", "soz_channel_count": 2, "soz_channels": ["A1-A2", "A2-A3"], "soz_or_resection_channel_count": 2},
    )
    server = build_server("127.0.0.1", 0, PyHFOSidecarSession(session_id="label-test"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        chat_req = Request(
            f"http://{host}:{port}/chat",
            data=json.dumps({"message": "Use BIDS labels: how many SOZ channels?", "recording_path": "/data/case.edf"}).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urlopen(chat_req, timeout=5) as response:
            chat = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()

    assert chat["content"] == "Agent label answer"
    assert chat["sidecar_state"]["bids_channel_annotations_loaded"] is False
    assert chat["sidecar_state"]["bids_channel_annotations"] is None
    assert chat["sidecar_state"]["label_loading_enabled"] is False
    assert '"clinical_label_context": "not_available_for_ordinary_edf_chat"' in captured["message"]
    assert '"soz_channel_count": 2' not in captured["message"]


def test_build_agent_payload_masks_loaded_labels_unless_explicitly_requested():
    session = PyHFOSidecarSession(session_id="masked")
    session.recording_info = {"input_path": "/data/case.edf", "sample_freq": 1000.0}
    session.bids_channel_annotations = {"soz_channel_count": 4, "soz_channels": ["A1-A2"]}

    masked = build_agent_payload("prepare safely", session, conversation_id="conv", recording_path="/data/case.edf")
    explicit = build_agent_payload(
        "how many pathological channels?",
        session,
        conversation_id="conv",
        recording_path="/data/case.edf",
        include_label_context=True,
    )

    assert '"clinical_label_context": "not_available_for_ordinary_edf_chat"' in masked["message"]
    assert '"soz_channel_count": 4' not in masked["message"]
    assert '"clinical_label_context": "not_available_for_ordinary_edf_chat"' in explicit["message"]
    assert '"soz_channel_count": 4' not in explicit["message"]



def test_pathological_request_runs_signal_candidate_analysis(monkeypatch):
    captured = {}
    analysis_calls = []

    def fake_call(payload):
        captured.update(payload)
        return {"content": "Candidate answer", "active_skill": "eeg-result-interpretation", "model_used": "fake"}

    def fake_analysis(session, path, message):
        analysis_calls.append((path, message))
        session.candidate_analysis = {
            "status": "complete",
            "candidate_ranking": {"top_channels": [{"channel": "A1-A2", "total_events": 3}]},
        }
        return session.candidate_analysis

    monkeypatch.setattr(sidecar_web, "call_ieeg_agent", fake_call)
    monkeypatch.setattr(sidecar_web, "run_signal_candidate_analysis", fake_analysis)
    monkeypatch.setattr(sidecar_module, "get_recording_info", lambda path: {"input_path": path, "sample_freq": 512.0, "channel_names": ["A1-A2"]})

    response = sidecar_web.handle_chat(
        {"message": "What are the pathological candidate channels?", "recording_path": "/data/case.edf"},
        PyHFOSidecarSession(session_id="candidate-test"),
    )

    assert response["content"] == "Candidate answer"
    assert analysis_calls == [("/data/case.edf", "What are the pathological candidate channels?")]
    assert '"candidate_analysis_summary"' in captured["message"]
    assert '"top_candidate_channels"' in captured["message"]
    assert '"A1-A2"' in captured["message"]
    assert captured["active_skill"] == "eeg-result-interpretation"


def test_load_recording_resets_stale_file_specific_state(monkeypatch):
    monkeypatch.setattr(sidecar_module, "get_recording_info", lambda path: {"input_path": path, "sample_freq": 512.0, "channel_names": ["A1-A2"]})
    session = PyHFOSidecarSession(session_id="reset-test")
    session.dispatch("load_recording", {"input_path": "/data/one.edf"})
    session.brain_signal_roi = {"input_path": "/data/one.edf", "brain_signal_channels": ["A1-A2"]}
    session.bids_channel_annotations = {"soz_channel_count": 1}
    session.candidate_analysis = {"status": "complete"}
    session.last_detection = {"status": "complete", "num_events": 1}
    session.prepared_recording = {"output_path": "/tmp/one.fif"}

    session.dispatch("load_recording", {"input_path": "/data/two.edf"})
    state = session.get_state()

    assert state["loaded_recording_path"] == "/data/two.edf"
    assert state["brain_signal_roi"] is None
    assert state["bids_channel_annotations"] is None
    assert state["candidate_analysis"] is None
    assert state["last_detection_summary"] is None
    assert state["prepared_recording"] is None

def test_call_ieeg_agent_requires_configured_token(monkeypatch):
    monkeypatch.delenv("PYHFO_AGENT_SERVICE_TOKEN", raising=False)
    monkeypatch.delenv("AGENT_SERVICE_TOKEN", raising=False)
    with pytest.raises(sidecar_web.PyHFOAgentError) as missing_token:
        sidecar_web.call_ieeg_agent({"message": "hi"})
    assert missing_token.value.code == "agent_token_missing"
    monkeypatch.setenv("AGENT_SERVICE_TOKEN", "test-token")
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"content":"ok"}'

    def fake_urlopen(request, timeout):
        captured["authorization"] = request.get_header("Authorization")
        captured["contract_version"] = request.get_header("X-contract-version")
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(sidecar_web, "urlopen", fake_urlopen)

    response = sidecar_web.call_ieeg_agent({"message": "hi"}, timeout_seconds=0.01)

    assert response == {"content": "ok"}
    assert captured["authorization"] == "Bearer test-token"
    assert captured["contract_version"] == sidecar_web.AGENT_CONTRACT_VERSION
    assert captured["timeout"] == 0.01


def test_missing_chat_service_returns_actionable_error(monkeypatch):
    def unavailable(request, timeout):
        raise sidecar_web.URLError("connection refused")

    monkeypatch.setenv("PYHFO_AGENT_SERVICE_TOKEN", "test-token")
    monkeypatch.setattr(sidecar_web, "urlopen", unavailable)
    with pytest.raises(sidecar_web.PyHFOAgentError) as error:
        sidecar_web.call_ieeg_agent({"message": "hi"})
    assert error.value.code == "agent_unavailable"
    assert "PYHFO_AGENT_CHAT_URL" in error.value.message


def test_missing_ui_catalog_disables_command_forwarding(tmp_path, monkeypatch):
    monkeypatch.setattr(sidecar_web, "UI_COMMAND_CATALOG_PATH", tmp_path / "missing.json")
    assert sidecar_web._load_ui_command_catalog() == {}


def test_label_requests_are_disabled_by_default(monkeypatch):
    monkeypatch.delenv("PYHFO_AGENT_ENABLE_LABEL_LOADING", raising=False)

    assert sidecar_web._requests_labels("Use BIDS labels and SOZ columns") is False
    assert sidecar_web._requests_labels("Do not read BIDS labels or channels.tsv; use only EDF signal") is False
    assert sidecar_web._requests_labels("without ground truth or annotations, what are abnormal channels?") is False


def test_negative_label_mentions_do_not_grant_label_access_when_env_enabled(monkeypatch):
    monkeypatch.setenv("PYHFO_AGENT_ENABLE_LABEL_LOADING", "1")

    assert sidecar_web._requests_labels("Use BIDS labels and SOZ columns") is True
    assert sidecar_web._requests_labels("Do not read BIDS labels or channels.tsv; use only EDF signal") is False
    assert sidecar_web._requests_labels("without ground truth or annotations, what are abnormal channels?") is False
