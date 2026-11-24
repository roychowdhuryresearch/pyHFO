import json

from pyhfo.agent_mcp import tools


def test_list_tools_includes_pyhfo_tools():
    listed = tools.list_tools()
    names = {tool["name"] for tool in listed}

    assert "pyhfo_get_recording_info" in names
    assert "pyhfo_detect_events" in names
    assert "pyhfo_infer_brain_signal_roi" in names
    assert "pyhfo_compute_channel_qc" in names
    assert "pyhfo_compute_spectral_profile" in names
    assert "pyhfo_recommend_detector_plan" in names
    assert "pyhfo_rank_candidate_channels" in names
    assert "pyhfo_prepare_recording" in names
    assert "pyhfo_get_capability_manifest" in names
    assert "pyhfo_get_app_state" not in names
    assert "pyhfo_dispatch_action" not in names
    detect_tool = next(tool for tool in listed if tool["name"] == "pyhfo_detect_events")
    assert detect_tool["inputSchema"]["required"] == ["input_path", "output_dir"]
    assert detect_tool["outputSchema"]["type"] == "object"
    assert "Electron desktop co-pilot" in detect_tool["description"]
    assert "detector_params" in detect_tool["description"]
    rank_tool = next(tool for tool in listed if tool["name"] == "pyhfo_rank_candidate_channels")
    assert "events_csv" in rank_tool["description"]
    prepare_tool = next(tool for tool in listed if tool["name"] == "pyhfo_prepare_recording")
    assert "Nyquist-safe" in prepare_tool["description"]


def test_call_tool_returns_structured_content(monkeypatch):
    monkeypatch.setattr(
        tools,
        "get_recording_info",
        lambda path: {"input_path": path, "sample_freq": 200.0},
    )

    result = tools.call_tool("pyhfo_get_recording_info", {"input_path": "sample.fif"})

    assert result["structuredContent"]["sample_freq"] == 200.0
    assert json.loads(result["content"][0]["text"])["input_path"] == "sample.fif"


def test_call_tool_maps_execution_errors(monkeypatch):
    def fail(_request):
        raise RuntimeError("boom")

    monkeypatch.setattr(tools, "run_detection", fail)

    result = tools.call_tool("pyhfo_detect_events", {"input_path": "a", "output_dir": "b"})

    assert result["isError"] is True
    assert result["structuredContent"]["error"]["code"] == "RuntimeError"


def test_capability_manifest_tool_exposes_action_contract():
    result = tools.call_tool("pyhfo_get_capability_manifest", {"include_missing": False})
    names = {action["name"] for action in result["structuredContent"]["actions"]}

    assert "load_recording" in names
    assert "prepare_recording" in names
    assert "annotate_event" in names
    assert "review_event_cursor" in names

