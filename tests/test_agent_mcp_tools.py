import json

from src.agent_mcp import tools


def test_list_tools_includes_pyhfo_tools():
    listed = tools.list_tools()
    names = {tool["name"] for tool in listed}

    assert "pyhfo_get_recording_info" in names
    assert "pyhfo_detect_events" in names
    detect_tool = next(tool for tool in listed if tool["name"] == "pyhfo_detect_events")
    assert detect_tool["inputSchema"]["required"] == ["input_path", "output_dir"]
    assert detect_tool["outputSchema"]["type"] == "object"


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
