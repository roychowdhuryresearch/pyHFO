import json
import subprocess
import sys

from src.agent_mcp.server import PyHFOMCPServer


def _request(method, params=None, request_id=1):
    payload = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        payload["params"] = params
    return payload


def test_initialize_advertises_tools_and_resources():
    server = PyHFOMCPServer()

    response = server.handle_message(
        _request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        )
    )

    result = response["result"]
    assert result["protocolVersion"] == "2025-06-18"
    assert "tools" in result["capabilities"]
    assert "resources" in result["capabilities"]


def test_tools_list_and_call(monkeypatch):
    server = PyHFOMCPServer()
    monkeypatch.setattr(
        "src.agent_mcp.tools.get_recording_info",
        lambda path: {"input_path": path, "sample_freq": 200.0},
    )

    listed = server.handle_message(_request("tools/list"))
    assert any(tool["name"] == "pyhfo_get_recording_info" for tool in listed["result"]["tools"])

    called = server.handle_message(
        _request(
            "tools/call",
            {"name": "pyhfo_get_recording_info", "arguments": {"input_path": "sample.fif"}},
        )
    )
    assert called["result"]["structuredContent"]["sample_freq"] == 200.0


def test_resources_read_help():
    server = PyHFOMCPServer()

    response = server.handle_message(
        _request("resources/read", {"uri": "pyhfo://server/help"})
    )

    content = response["result"]["contents"][0]
    assert content["mimeType"] == "text/markdown"
    assert "pyhfo_detect_events" in content["text"]


def test_unknown_tool_returns_json_rpc_error():
    server = PyHFOMCPServer()

    response = server.handle_message(_request("tools/call", {"name": "missing"}))

    assert response["error"]["code"] == -32602


def test_stdio_server_handles_initialize_and_tools_list():
    messages = "\n".join(
        [
            json.dumps(_request("initialize", {"protocolVersion": "2025-06-18"}, 1)),
            json.dumps(_request("tools/list", request_id=2)),
            "",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-m", "src.agent_mcp.server", "--transport", "stdio"],
        input=messages,
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )

    assert completed.returncode == 0
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert responses[0]["result"]["serverInfo"]["name"] == "pyhfo-agent"
    assert any(tool["name"] == "pyhfo_detect_events" for tool in responses[1]["result"]["tools"])
    assert completed.stderr == ""
