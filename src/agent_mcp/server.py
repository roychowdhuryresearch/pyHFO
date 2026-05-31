from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from src.agent_api.serialization import json_safe
from src.agent_mcp.resources import list_resource_templates, list_resources, read_resource
from src.agent_mcp.tools import call_tool, list_tools


PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "pyhfo-agent"
SERVER_VERSION = "0.1.0"

JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603
MCP_RESOURCE_NOT_FOUND = -32002


class PyHFOMCPServer:
    def initialize_result(self, requested_version: str | None = None) -> dict[str, Any]:
        return {
            "protocolVersion": requested_version or PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {},
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "title": "PyHFO Agent",
                "version": SERVER_VERSION,
            },
            "instructions": "Use PyHFO tools to inspect EEG recordings, run detections, and read generated artifacts.",
        }

    def handle_request(self, method: str, params: dict[str, Any] | None) -> Any:
        params = params or {}
        if method == "initialize":
            return self.initialize_result(params.get("protocolVersion"))
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": list_tools()}
        if method == "tools/call":
            name = params.get("name")
            if not name:
                raise ValueError("tools/call requires params.name")
            return call_tool(str(name), params.get("arguments") or {})
        if method == "resources/list":
            return {"resources": list_resources()}
        if method == "resources/templates/list":
            return {"resourceTemplates": list_resource_templates()}
        if method == "resources/read":
            uri = params.get("uri")
            if not uri:
                raise ValueError("resources/read requires params.uri")
            return {"contents": [read_resource(str(uri))]}
        raise NotImplementedError(method)

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return self._error_response(None, JSONRPC_INVALID_REQUEST, "Invalid JSON-RPC request")
        request_id = message.get("id")
        method = message.get("method")
        if not method:
            return self._error_response(request_id, JSONRPC_INVALID_REQUEST, "Missing method")
        if request_id is None:
            # Notifications do not receive responses. The initialized notification is accepted.
            return None
        try:
            result = self.handle_request(str(method), message.get("params") or {})
            return {"jsonrpc": "2.0", "id": request_id, "result": json_safe(result)}
        except NotImplementedError:
            return self._error_response(request_id, JSONRPC_METHOD_NOT_FOUND, f"Method not found: {method}")
        except KeyError as exc:
            return self._error_response(request_id, JSONRPC_INVALID_PARAMS, str(exc).strip("'"))
        except FileNotFoundError as exc:
            return self._error_response(request_id, MCP_RESOURCE_NOT_FOUND, str(exc))
        except ValueError as exc:
            return self._error_response(request_id, JSONRPC_INVALID_PARAMS, str(exc))
        except Exception as exc:
            return self._error_response(request_id, JSONRPC_INTERNAL_ERROR, str(exc))

    def handle_line(self, line: str) -> dict[str, Any] | None:
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            return self._error_response(None, JSONRPC_PARSE_ERROR, f"Parse error: {exc}")
        return self.handle_message(message)

    @staticmethod
    def _error_response(request_id, code: int, message: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        error = {"code": code, "message": message}
        if data:
            error["data"] = data
        return {"jsonrpc": "2.0", "id": request_id, "error": error}

    def run_stdio(self, stdin=None, stdout=None):
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            response = self.handle_line(line)
            if response is None:
                continue
            stdout.write(json.dumps(json_safe(response), separators=(",", ":")) + "\n")
            stdout.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyHFO MCP server.")
    parser.add_argument("--transport", choices=["stdio"], default="stdio")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    server = PyHFOMCPServer()
    if args.transport == "stdio":
        server.run_stdio()
        return 0
    parser.error(f"Unsupported transport: {args.transport}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
