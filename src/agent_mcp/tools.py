from __future__ import annotations

import json
from typing import Any, Callable

from src.agent_api import create_batch, get_recording_info, run_batch, run_detection
from src.agent_api.errors import error_response
from src.agent_api.serialization import json_safe
from src.agent_mcp.app_resources import UI_RESOURCE_URI


JSON_OBJECT_SCHEMA = {"type": "object", "additionalProperties": True}


def _tool(
    name: str,
    title: str,
    description: str,
    input_schema: dict[str, Any],
    handler: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    ui_resource: str | None = None,
) -> dict[str, Any]:
    definition = {
            "name": name,
            "title": title,
            "description": description,
            "inputSchema": input_schema,
            "outputSchema": JSON_OBJECT_SCHEMA,
    }
    if ui_resource:
        definition["_meta"] = {"ui": {"resourceUri": ui_resource}}
    return {
        "definition": definition,
        "handler": handler,
    }


def _recording_info_tool(arguments: dict[str, Any]) -> dict:
    return get_recording_info(arguments["input_path"])


def _detect_events_tool(arguments: dict[str, Any]) -> dict:
    return run_detection(arguments)


def _create_batch_tool(arguments: dict[str, Any]) -> dict:
    return create_batch(
        arguments["input_dir"],
        arguments.get("output_dir"),
        recursive=bool(arguments.get("recursive", True)),
    )


def _run_batch_tool(arguments: dict[str, Any]) -> dict:
    return run_batch(arguments)


def _session_summary_tool(arguments: dict[str, Any]) -> dict:
    # This tool intentionally reads structured summaries, not large raw EEG/session payloads.
    path = arguments.get("summary_path") or arguments.get("detection_result_path")
    if not path:
        raise ValueError("summary_path is required.")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


TOOL_REGISTRY = {
    item["definition"]["name"]: item
    for item in [
        _tool(
            "pyhfo_get_recording_info",
            "Get Recording Info",
            "Read EEG recording metadata without loading the PyQt GUI.",
            {
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "description": "Path to an EEG recording."},
                },
                "required": ["input_path"],
                "additionalProperties": False,
            },
            _recording_info_tool,
        ),
        _tool(
            "pyhfo_detect_events",
            "Detect Events",
            "Run a headless PyHFO detection pipeline and return artifact paths.",
            {
                "type": "object",
                "properties": {
                    "input_path": {"type": "string"},
                    "output_dir": {"type": "string"},
                    "biomarker": {"type": "string", "default": "HFO"},
                    "detector": {"type": "string", "default": "STE"},
                    "filter": JSON_OBJECT_SCHEMA,
                    "detector_params": JSON_OBJECT_SCHEMA,
                    "classifier": JSON_OBJECT_SCHEMA,
                    "n_jobs": {"type": "integer", "minimum": 1, "default": 4},
                    "exports": {"type": "array", "items": {"type": "string"}},
                    "fail_on_error": {"type": "boolean", "default": True},
                },
                "required": ["input_path", "output_dir"],
                "additionalProperties": False,
            },
            _detect_events_tool,
            ui_resource=UI_RESOURCE_URI,
        ),
        _tool(
            "pyhfo_create_batch_project",
            "Create Batch Project",
            "Create a resumable PyHFO batch manifest from an input folder.",
            {
                "type": "object",
                "properties": {
                    "input_dir": {"type": "string"},
                    "output_dir": {"type": "string"},
                    "recursive": {"type": "boolean", "default": True},
                },
                "required": ["input_dir"],
                "additionalProperties": False,
            },
            _create_batch_tool,
        ),
        _tool(
            "pyhfo_run_batch",
            "Run Batch",
            "Run PyHFO detection for a batch manifest or input folder.",
            {
                "type": "object",
                "properties": {
                    "manifest_path": {"type": "string"},
                    "input_dir": {"type": "string"},
                    "output_dir": {"type": "string"},
                    "recursive": {"type": "boolean", "default": True},
                    "retry_failed": {"type": "boolean", "default": False},
                    "fail_fast": {"type": "boolean", "default": False},
                    "detection": JSON_OBJECT_SCHEMA,
                },
                "additionalProperties": False,
            },
            _run_batch_tool,
            ui_resource=UI_RESOURCE_URI,
        ),
        _tool(
            "pyhfo_export_session_summary",
            "Read Session Summary",
            "Read a PyHFO summary JSON artifact as structured content.",
            {
                "type": "object",
                "properties": {
                    "summary_path": {"type": "string"},
                    "detection_result_path": {"type": "string"},
                },
                "additionalProperties": False,
            },
            _session_summary_tool,
        ),
    ]
}


def list_tools() -> list[dict[str, Any]]:
    return [item["definition"] for item in TOOL_REGISTRY.values()]


def call_tool(name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Unknown tool: {name}")
    try:
        structured = json_safe(TOOL_REGISTRY[name]["handler"](dict(arguments or {})))
        text = json.dumps(structured, sort_keys=True)
        return {
            "content": [{"type": "text", "text": text}],
            "structuredContent": structured,
        }
    except Exception as exc:
        error = error_response(exc)
        return {
            "content": [{"type": "text", "text": json.dumps(error, sort_keys=True)}],
            "structuredContent": {"error": error},
            "isError": True,
        }
