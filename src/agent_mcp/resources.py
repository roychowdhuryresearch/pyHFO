from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from src.agent_mcp.app_resources import UI_RESOURCE_URI, read_ui_resource, ui_resource_descriptor


HELP_URI = "pyhfo://server/help"


def list_resources() -> list[dict[str, Any]]:
    return [
        {
            "uri": HELP_URI,
            "name": "PyHFO Agent Help",
            "title": "PyHFO Agent Help",
            "description": "Available PyHFO MCP tools and artifact conventions.",
            "mimeType": "text/markdown",
            "annotations": {"audience": ["user", "assistant"], "priority": 0.8},
        },
        ui_resource_descriptor(),
    ]


def list_resource_templates() -> list[dict[str, Any]]:
    return [
        {
            "uriTemplate": "pyhfo://artifact?path={absolute_path}",
            "name": "PyHFO artifact",
            "title": "PyHFO Artifact",
            "description": "Read a text/JSON PyHFO artifact by absolute path.",
            "mimeType": "application/json",
        }
    ]


def _read_help() -> dict[str, Any]:
    text = """# PyHFO Agent MCP Server

Use `pyhfo_get_recording_info` to inspect an EEG recording, `pyhfo_detect_events` to run a headless detection, and batch tools for resumable folder workflows.

Detection tools return artifact paths. Large EEG arrays are never embedded in tool responses.
"""
    return {"uri": HELP_URI, "mimeType": "text/markdown", "text": text}


def _read_artifact(parsed) -> dict[str, Any]:
    query = parse_qs(parsed.query)
    raw_path = (query.get("path") or [""])[0]
    if not raw_path:
        raise FileNotFoundError("Artifact path is required.")
    path = Path(raw_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Artifact not found: {path}")
    text = path.read_text(encoding="utf-8")
    mime_type = "application/json" if path.suffix.lower() == ".json" else "text/plain"
    return {"uri": f"pyhfo://artifact?path={path}", "mimeType": mime_type, "text": text}


def read_resource(uri: str) -> dict[str, Any]:
    if uri == UI_RESOURCE_URI:
        return read_ui_resource(uri)
    if uri == HELP_URI:
        return _read_help()
    parsed = urlparse(uri)
    if parsed.scheme == "pyhfo" and parsed.netloc == "artifact":
        return _read_artifact(parsed)
    raise FileNotFoundError(f"Resource not found: {uri}")


def json_resource(uri: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "uri": uri,
        "mimeType": "application/json",
        "text": json.dumps(payload, indent=2, sort_keys=True),
    }
