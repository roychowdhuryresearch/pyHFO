from __future__ import annotations

import json
from typing import Any, Callable

from pyhfo.agent_api import (
    create_batch,
    get_capability_manifest,
    get_recording_info,
    infer_brain_signal_roi,
    compute_channel_qc,
    compute_spectral_profile,
    rank_candidate_channels,
    recommend_detector_plan,
    prepare_recording,
    run_batch,
    run_detection,
)
from pyhfo.agent_api.errors import error_response
from pyhfo.agent_api.serialization import json_safe
from pyhfo.agent_mcp.app_resources import UI_RESOURCE_URI


JSON_OBJECT_SCHEMA = {"type": "object", "additionalProperties": True}

PREPARE_RECORDING_DESCRIPTION = (
    "Create a cleaned PyHFO-ready FIF/FIF.GZ copy. Use after reading metadata/ROI/QC when the user asks to "
    "prepare or clean a recording, not merely to view the desktop. Explicitly pass output_path. Use "
    "line_noise_notch_hz/notch_freqs only when requested or supported by spectral evidence; use bandpass with "
    "Nyquist-safe h_freq for HFO work. Bad/flat/outlier channel handling is algorithmic QC, not a clinical label."
)

RANK_CANDIDATES_DESCRIPTION = (
    "Rank candidate channels from detector CSV outputs after detection. Each runs[] item should include "
    '{"biomarker":"HFO","events_csv":"/path/events.csv"}; run_id is optional. Use this after pyhfo_detect_events '
    "when the user asks which channels have the most HFO/spike/spindle activity, counts, rates, or candidate "
    "epileptiform signal evidence. Do not infer clinical/pathological labels from this ranking alone."
)

DETECT_EVENTS_DESCRIPTION = (
    "Run a headless PyHFO detection pipeline and return artifact paths. Use this MCP tool for backend/batch/headless "
    "work. In the Electron desktop co-pilot, prefer emitting UI command detection.run so the visible workspace stages "
    "and applies the run. Required: input_path and output_dir. For HFO all-channel STE, use biomarker='HFO', "
    "detector='STE', include_channels omitted, exports including csv/session/summary_json, and filter such as "
    '{"fp":80,"fs":500} clamped below Nyquist. Preserve explicit user parameters in detector_params; for '
    '"RMS = 2 detect STE", keep detector="STE" and pass detector_params={"rms_thres":2}. Outputs are algorithmic '
    "candidate events requiring expert review."
)


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


def _infer_roi_tool(arguments: dict[str, Any]) -> dict:
    return infer_brain_signal_roi(arguments["input_path"])


def _channel_qc_tool(arguments: dict[str, Any]) -> dict:
    return compute_channel_qc(arguments)


def _spectral_profile_tool(arguments: dict[str, Any]) -> dict:
    return compute_spectral_profile(arguments)


def _detector_plan_tool(arguments: dict[str, Any]) -> dict:
    return recommend_detector_plan(arguments)


def _rank_candidates_tool(arguments: dict[str, Any]) -> dict:
    return rank_candidate_channels(arguments)


def _detect_events_tool(arguments: dict[str, Any]) -> dict:
    return run_detection(arguments)


def _prepare_recording_tool(arguments: dict[str, Any]) -> dict:
    return prepare_recording(arguments)


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


def _capability_manifest_tool(arguments: dict[str, Any]) -> dict:
    return get_capability_manifest(include_missing=bool(arguments.get("include_missing", True)))


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
            "pyhfo_infer_brain_signal_roi",
            "Infer Brain-Signal ROI",
            "Infer a label-free brain-signal ROI from EDF/MNE channel names and identify auxiliary/uncertain channels to exclude from default detection.",
            {
                "type": "object",
                "properties": {"input_path": {"type": "string"}},
                "required": ["input_path"],
                "additionalProperties": False,
            },
            _infer_roi_tool,
        ),
        _tool(
            "pyhfo_compute_channel_qc",
            "Compute Channel QC",
            "Compute label-free channel QC metrics on selected channels: RMS, peak-to-peak, flat fraction, NaN fraction, and robust outliers.",
            {
                "type": "object",
                "properties": {
                    "input_path": {"type": "string"},
                    "exclude_channels": {"type": "array", "items": {"type": "string"}},
                    "outlier_z_threshold": {"type": "number", "default": 8.0},
                    "max_duration_seconds": {"type": "number", "default": 60.0},
                },
                "required": ["input_path"],
                "additionalProperties": False,
            },
            _channel_qc_tool,
        ),
        _tool(
            "pyhfo_compute_spectral_profile",
            "Compute Spectral Profile",
            "Compute compact FFT spectral profile for planning: relative band power, line-noise candidates, Nyquist, and recommended HFO band.",
            {
                "type": "object",
                "properties": {
                    "input_path": {"type": "string"},
                    "exclude_channels": {"type": "array", "items": {"type": "string"}},
                    "max_channels": {"type": "integer", "default": 32},
                    "max_duration_seconds": {"type": "number", "default": 60.0},
                },
                "required": ["input_path"],
                "additionalProperties": False,
            },
            _spectral_profile_tool,
        ),
        _tool(
            "pyhfo_recommend_detector_plan",
            "Recommend Detector Plan",
            "Recommend label-free Spike/HFO detector runs from recording metadata and brain ROI, including Nyquist-safe HFO filters.",
            {
                "type": "object",
                "properties": {
                    "input_path": {"type": "string"},
                    "recording_info": JSON_OBJECT_SCHEMA,
                    "roi": JSON_OBJECT_SCHEMA,
                },
                "additionalProperties": False,
            },
            _detector_plan_tool,
        ),
        _tool(
            "pyhfo_rank_candidate_channels",
            "Rank Candidate Channels",
            RANK_CANDIDATES_DESCRIPTION,
            {
                "type": "object",
                "properties": {
                    "runs": {"type": "array", "items": JSON_OBJECT_SCHEMA},
                    "duration_seconds": {"type": "number"},
                    "exclude_channels": {"type": "array", "items": {"type": "string"}},
                    "top_n": {"type": "integer", "default": 20},
                },
                "required": ["runs"],
                "additionalProperties": False,
            },
            _rank_candidates_tool,
        ),
        _tool(
            "pyhfo_detect_events",
            "Detect Events",
            DETECT_EVENTS_DESCRIPTION,
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
                    "include_channels": {"type": "array", "items": {"type": "string"}},
                    "exclude_channels": {"type": "array", "items": {"type": "string"}},
                    "fail_on_error": {"type": "boolean", "default": True},
                },
                "required": ["input_path", "output_dir"],
                "additionalProperties": False,
            },
            _detect_events_tool,
            ui_resource=UI_RESOURCE_URI,
        ),
        _tool(
            "pyhfo_prepare_recording",
            "Prepare Recording",
            PREPARE_RECORDING_DESCRIPTION,
            {
                "type": "object",
                "properties": {
                    "input_path": {"type": "string"},
                    "output_path": {"type": "string"},
                    "extra_bad_channels": {"type": "array", "items": {"type": "string"}},
                    "drop_bad_channels": {"type": "boolean", "default": True},
                    "drop_flat_channels": {"type": "boolean", "default": True},
                    "mark_outlier_channels": {"type": "boolean", "default": True},
                    "drop_outlier_channels": {"type": "boolean", "default": False},
                    "outlier_z_threshold": {"type": "number", "default": 8.0},
                    "bipolar_montage": {"type": "string", "enum": ["none", "auto_adjacent", "pairs"], "default": "none"},
                    "bipolar_pairs": {"type": "array", "items": JSON_OBJECT_SCHEMA},
                    "drop_original_after_bipolar": {"type": "boolean", "default": False},
                    "line_noise_notch_hz": {"type": "number"},
                    "notch_freqs": {"type": "array", "items": {"type": "number"}},
                    "bandpass": JSON_OBJECT_SCHEMA,
                    "resample_sfreq": {"type": "number"},
                },
                "required": ["input_path", "output_path"],
                "additionalProperties": False,
            },
            _prepare_recording_tool,
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
        _tool(
            "pyhfo_get_capability_manifest",
            "Get Capability Manifest",
            "Return the decoupled PyHFO sidecar manifest: explicit actions, schemas, workflow phases, safety boundaries, and missing doctor-review wrappers.",
            {
                "type": "object",
                "properties": {
                    "include_missing": {"type": "boolean", "default": True},
                },
                "additionalProperties": False,
            },
            _capability_manifest_tool,
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
