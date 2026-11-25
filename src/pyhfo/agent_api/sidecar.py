from __future__ import annotations

import json
import csv
import importlib.util
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pyhfo import __version__
from pyhfo.paths import pyhfo_home
from pyhfo.agent_api.errors import PyHFOAgentError
from pyhfo.agent_api.pipeline import create_batch, get_recording_info
from pyhfo.agent_api.subprocess_runner import run_pipeline_isolated
from pyhfo.agent_api.signal_analysis import (
    compute_channel_qc,
    compute_spectral_profile,
    detect_line_noise,
    infer_brain_signal_roi,
    infer_brain_signal_roi_from_channels,
    rank_candidate_channels,
    recommend_detector_plan,
)
from pyhfo.agent_api.serialization import json_safe, utc_now_iso
from pyhfo.agent_api.parameters import detector_defaults, recommend_parameters, sweep_parameters, validate_parameters
from pyhfo.agent_api.review import (
    annotate_event,
    compare_detector_runs,
    create_consensus_run,
    event_cursor,
    event_feature_summary,
    event_snippet,
    export_review_state,
    list_events,
    overlap_review,
)
from pyhfo.agent_api.sessions import create_project, export_report, export_session, export_workbook, open_project, save_project
from pyhfo.agent_api.waveform import waveform_channels, waveform_overview, waveform_window


def _torch_gpu_capability() -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {
            "available": False,
            "backend": "torch",
            "install_hint": "Install PyTorch with CUDA support to enable Hugging Face GPU classifier presets.",
            "ui_behavior": "Disable GPU classifier presets; keep CPU and Custom classifier paths available.",
        }
    try:
        import torch  # type: ignore

        available = bool(torch.cuda.is_available())
        return {
            "available": available,
            "backend": "torch.cuda",
            "device_count": int(torch.cuda.device_count()) if available else 0,
            "install_hint": "" if available else "CUDA is not available to PyTorch; use the CPU preset or install a CUDA-enabled PyTorch build.",
            "ui_behavior": "Enable GPU classifier preset when available; otherwise disable it and explain CPU fallback.",
        }
    except Exception as exc:
        return {
            "available": False,
            "backend": "torch.cuda",
            "install_hint": f"Could not inspect PyTorch CUDA availability: {exc}",
            "ui_behavior": "Disable GPU classifier presets; keep CPU and Custom classifier paths available.",
        }


def get_detector_availability() -> dict[str, Any]:
    """Return detector support in a form an agent can plan against."""

    try:
        from pyhfo.utils.utils_detector import get_yasa_import_error, has_hil, has_yasa

        hil_available = bool(has_hil())
        yasa_available = bool(has_yasa())
        probe_error = "" if yasa_available else get_yasa_import_error()
    except Exception as exc:  # pragma: no cover - defensive dependency probe
        hil_available = False
        yasa_available = False
        probe_error = str(exc)

    def optional_status(available: bool, install_hint: str) -> dict[str, Any]:
        return {
            "status": "callable" if available else "unavailable_optional_dependency",
            "can_run_without_extra_config": bool(available),
            "install_hint": "" if available else install_hint,
        }

    return json_safe(
        {
            "schema_version": "pyhfo-detectors.v1",
            "probe_error": probe_error,
            "recommended_smoke_detectors": {
                "HFO": "STE",
                "Spindle": "MOLLE",
                "Spike": "RMS/LL",
            },
            "preferred_detectors_when_available": {
                "Spindle": "YASA",
            },
            "detectors_by_biomarker": {
                "HFO": [
                    {
                        "detector": "STE",
                        "status": "callable",
                        "dependency": "HFODetector.ste",
                        "can_run_without_extra_config": True,
                        "smoke_test_role": "default_hfo_smoke",
                    },
                    {
                        "detector": "MNI",
                        "status": "callable",
                        "dependency": "HFODetector.mni",
                        "can_run_without_extra_config": True,
                    },
                    {
                        "detector": "HIL",
                        "dependency": "HFODetector.hil",
                        "optional_dependency": "HFODetector.hil",
                        **optional_status(
                            hil_available,
                            "Install an HFODetector build that includes the optional HIL detector.",
                        ),
                    },
                    {
                        "detector": "RMS",
                        "status": "callable",
                        "dependency": "built_in_threshold",
                        "can_run_without_extra_config": True,
                    },
                    {
                        "detector": "LineLength",
                        "status": "callable",
                        "dependency": "built_in_threshold",
                        "can_run_without_extra_config": True,
                    },
                ],
                "Spindle": [
                    {
                        "detector": "YASA",
                        "dependency": "yasa",
                        "optional_dependency": "yasa",
                        **optional_status(
                            yasa_available,
                            "Install the optional 'yasa' package to run YASA spindle detection.",
                        ),
                    },
                    {
                        "detector": "MOLLE",
                        "status": "callable",
                        "dependency": "built_in_threshold",
                        "can_run_without_extra_config": True,
                        "smoke_test_role": "spindle_smoke_without_optional_dependency",
                    },
                    {
                        "detector": "A7",
                        "status": "callable",
                        "dependency": "built_in_threshold",
                        "can_run_without_extra_config": True,
                    },
                    {
                        "detector": "LSM",
                        "status": "callable_requires_configuration",
                        "dependency": "built_in_kramer_lsm",
                        "can_run_without_extra_config": False,
                        "required_configuration": ["parameter_file", "model_parameters"],
                        "llm_usage": "Use only when a Kramer LSM parameter file, bundled preset, or explicit model_parameters are provided.",
                    },
                ],
                "Spike": [
                    {
                        "detector": "RMS/LL",
                        "status": "callable",
                        "dependency": "built_in_threshold",
                        "can_run_without_extra_config": True,
                        "smoke_test_role": "default_spike_smoke",
                    }
                ],
            },
            "interpretation_policy": [
                "Detector availability is an execution capability, not clinical validity.",
                "Use MOLLE or A7 for no-extra-dependency spindle smoke tests when YASA is unavailable.",
                "Report unavailable optional detectors explicitly instead of presenting a failed dependency as a successful run.",
                "Keep detector outputs as signal-derived candidates requiring expert review.",
            ],
        }
    )


def labels_enabled() -> bool:
    """Gate all BIDS/clinical-label loading; disabled unless explicitly enabled."""

    return (os.getenv("PYHFO_AGENT_ENABLE_LABEL_LOADING") or "").strip().lower() in {"1", "true", "yes", "on"}


def _coerce_int(value: Any) -> int | None:
    try:
        if value in (None, "", "n/a", "N/A"):
            return None
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _candidate_channels_tsv(input_path: str | Path) -> Path | None:
    path = Path(str(input_path)).expanduser().resolve()
    candidates = []
    name = path.name
    if name.endswith("_ieeg.edf"):
        candidates.append(path.with_name(name.replace("_ieeg.edf", "_channels.tsv")))
    if name.endswith("_ieeg.fif"):
        candidates.append(path.with_name(name.replace("_ieeg.fif", "_channels.tsv")))
    if name.endswith("_ieeg.fif.gz"):
        candidates.append(path.with_name(name.replace("_ieeg.fif.gz", "_channels.tsv")))
    candidates.append(path.with_name(f"{path.stem}_channels.tsv"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None

def load_bids_channel_annotations(input_path: str | Path) -> dict[str, Any] | None:
    """Read BIDS channels.tsv annotations that agents need for evidence-backed answers.

    This deliberately stays small and structured. The raw TSV can be large; the side
    panel only needs auditable counts/lists for SOZ, resection, and good/bad status.
    """

    channels_tsv = _candidate_channels_tsv(input_path)
    if channels_tsv is None:
        return None

    with channels_tsv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    def selected(column: str, value: int) -> list[str]:
        if not rows or column not in rows[0]:
            return []
        return [str(row.get("name") or "") for row in rows if _coerce_int(row.get(column)) == value and row.get("name")]

    columns = list(rows[0].keys()) if rows else []
    soz_channels = selected("soz", 1)
    resection_channels = selected("resection", 1)
    bad_channels = selected("good", 0)
    soz_or_resection_channels = sorted(set(soz_channels) | set(resection_channels))
    return json_safe(
        {
            "channels_tsv": str(channels_tsv),
            "columns": columns,
            "n_rows": len(rows),
            "definitions": {
                "soz": "Seizure onset zone. 1 for yes, 0 for no, -1 for unknown.",
                "resection": "Resection status. 1 for resected, 0 for not, -1 for unknown.",
                "good": "Good channel status. 1 for good, 0 for bad.",
            },
            "soz_channel_count": len(soz_channels),
            "soz_channels": soz_channels,
            "resection_channel_count": len(resection_channels),
            "resection_channels": resection_channels,
            "soz_or_resection_channel_count": len(soz_or_resection_channels),
            "soz_or_resection_channels": soz_or_resection_channels,
            "bad_channel_count": len(bad_channels),
            "bad_channels": bad_channels,
        }
    )


def _channel_set(values: Any) -> set[str]:
    if not values:
        return set()
    if isinstance(values, str):
        values = [values]
    result: set[str] = set()
    for value in values:
        channel = str(value or "").strip()
        if channel:
            result.add(channel)
    return result


def _candidate_channels_from_payload(payload: dict[str, Any]) -> set[str]:
    explicit = payload.get("candidate_channels")
    if explicit is not None:
        return _channel_set(explicit)

    candidates: set[str] = set()
    analysis = payload.get("candidate_analysis") or {}
    if isinstance(analysis, dict):
        for key in ("top_channels", "channels", "candidates"):
            rows = analysis.get(key)
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict):
                        candidates.update(_channel_set(row.get("channel") or row.get("channel_name") or row.get("name")))
                    else:
                        candidates.update(_channel_set(row))

    runs = payload.get("runs")
    if isinstance(runs, list):
        for run in runs:
            if isinstance(run, dict):
                rows = run.get("top_channels") or run.get("channels") or run.get("candidates")
                if isinstance(rows, list):
                    for row in rows:
                        if isinstance(row, dict):
                            candidates.update(_channel_set(row.get("channel") or row.get("channel_name") or row.get("name")))
                        else:
                            candidates.update(_channel_set(row))

    return candidates


def _case_insensitive_intersection(left: set[str], right: set[str]) -> list[str]:
    right_by_lower = {item.lower(): item for item in right}
    matches = [item for item in left if item.lower() in right_by_lower]
    return sorted(matches)


def _case_insensitive_difference(left: set[str], right: set[str]) -> list[str]:
    right_lower = {item.lower() for item in right}
    return sorted(item for item in left if item.lower() not in right_lower)


def compare_candidate_channels_to_bids_labels(
    session: Any,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare signal-derived candidate channels to gated BIDS labels.

    This is evaluation-only. It never loads labels unless the environment gate is
    enabled, and it never treats labels as ordinary metadata or detector input.
    """

    if not labels_enabled():
        raise PyHFOAgentError(
            "label_loading_disabled",
            "BIDS/clinical label comparison is disabled. Use signal-derived evidence, or set PYHFO_AGENT_ENABLE_LABEL_LOADING=1 for an explicit evaluation run.",
        )

    request = dict(payload or {})
    labels = request.get("bids_channel_annotations") or getattr(session, "bids_channel_annotations", None)
    if labels is None:
        input_path = request.get("input_path") or (getattr(session, "recording_info", None) or {}).get("input_path")
        if not input_path:
            raise PyHFOAgentError("invalid_request", "input_path is required when BIDS labels are not already loaded.")
        labels = load_bids_channel_annotations(str(input_path))
        if labels is None:
            raise PyHFOAgentError("file_not_found", "No matching BIDS channels.tsv found for this recording.")
        session.bids_channel_annotations = labels

    candidate_channels = _candidate_channels_from_payload(request)
    if not candidate_channels and getattr(session, "candidate_analysis", None):
        candidate_channels = _candidate_channels_from_payload({"candidate_analysis": session.candidate_analysis})
    if not candidate_channels:
        raise PyHFOAgentError(
            "invalid_request",
            "candidate_channels or candidate_analysis is required for ground-truth comparison.",
        )

    soz_channels = _channel_set(labels.get("soz_channels"))
    resection_channels = _channel_set(labels.get("resection_channels"))
    bad_channels = _channel_set(labels.get("bad_channels"))
    soz_or_resection = _channel_set(labels.get("soz_or_resection_channels")) or (soz_channels | resection_channels)

    matches = {
        "soz": _case_insensitive_intersection(candidate_channels, soz_channels),
        "resection": _case_insensitive_intersection(candidate_channels, resection_channels),
        "soz_or_resection": _case_insensitive_intersection(candidate_channels, soz_or_resection),
        "bad_channel": _case_insensitive_intersection(candidate_channels, bad_channels),
    }
    label_sets = {
        "soz": soz_channels,
        "resection": resection_channels,
        "soz_or_resection": soz_or_resection,
        "bad_channel": bad_channels,
    }
    missing_label_channels = {
        key: _case_insensitive_difference(values, candidate_channels)
        for key, values in label_sets.items()
    }
    candidates_without_label_overlap = _case_insensitive_difference(candidate_channels, soz_or_resection | bad_channels)

    result = {
        "status": "complete",
        "schema_version": "pyhfo-label-comparison.v1",
        "gate": "PYHFO_AGENT_ENABLE_LABEL_LOADING",
        "label_source": labels.get("channels_tsv"),
        "candidate_channel_count": len(candidate_channels),
        "candidate_channels": sorted(candidate_channels),
        "label_counts": {
            "soz": len(soz_channels),
            "resection": len(resection_channels),
            "soz_or_resection": len(soz_or_resection),
            "bad_channel": len(bad_channels),
        },
        "matches": matches,
        "missing_label_channels": missing_label_channels,
        "candidates_without_label_overlap": candidates_without_label_overlap,
        "summary": {
            "candidate_matches_soz_or_resection_count": len(matches["soz_or_resection"]),
            "candidate_matches_bad_channel_count": len(matches["bad_channel"]),
            "candidate_without_label_overlap_count": len(candidates_without_label_overlap),
        },
        "interpretation_policy": [
            "This is gated evaluation evidence only, not ordinary EDF chat metadata.",
            "Do not use BIDS labels to create or alter signal-derived candidate rankings.",
            "Agreement/disagreement is a study comparison and is not a clinical diagnosis.",
        ],
    }
    session.candidate_analysis = {
        **(getattr(session, "candidate_analysis", None) or {}),
        "label_comparison": result,
    }
    return json_safe(result)


def _schema_ref(description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "description": description,
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def get_capability_manifest(include_missing: bool = True) -> dict[str, Any]:
    """Return the contract a generic agent should plan against.

    The manifest is the coupling boundary: the generic agent sees explicit state,
    actions, schemas, safety constraints, and clinical-review limits; it does not
    need PyHFO-specific system prompt text or PyQt internals.
    """

    actions: list[dict[str, Any]] = [
        {
            "name": "get_state",
            "phase": "state",
            "status": "callable",
            "description": "Read the current sidecar session state: loaded recording, prepared copy, last detection, batch manifest, artifacts, and pending review gaps.",
            "input_schema": _schema_ref("No arguments.", {}, []),
            "read_only": True,
            "writes_artifacts": False,
            "doctor_approval_required": False,
            "llm_usage": "Use before selecting a next action when a session may already contain a loaded/prepared recording or prior results.",
        },
        {
            "name": "get_detector_availability",
            "phase": "planning",
            "status": "callable",
            "description": "Return biomarker detector availability, optional dependency status, and no-extra-dependency smoke-test recommendations.",
            "input_schema": _schema_ref("No arguments.", {}, []),
            "read_only": True,
            "writes_artifacts": False,
            "doctor_approval_required": False,
            "llm_usage": "Use before spindle/YASA/LSM planning. If YASA is unavailable, choose callable built-in paths such as MOLLE/A7 or report the dependency gap.",
        },
        {
            "name": "load_recording",
            "phase": "metadata",
            "status": "callable",
            "description": "Read recording signal metadata without starting the PyQt GUI, running detection, or loading ground-truth BIDS labels.",
            "input_schema": _schema_ref("Recording metadata request.", {"input_path": {"type": "string", "description": "EDF/BDF/FIF/FIF.GZ/BrainVision path readable by PyHFO/MNE."}}, ["input_path"]),
            "read_only": True,
            "writes_artifacts": False,
            "doctor_approval_required": False,
            "llm_usage": "Always use first for unknown files; use sample_freq/duration/channel names/line_freq to choose safe preprocessing and detector parameters. This intentionally does not load SOZ/resection labels.",
        },
        {
            "name": "infer_brain_signal_roi",
            "phase": "metadata",
            "status": "callable",
            "description": "Infer a label-free brain-signal ROI from EDF/MNE channel names and identify auxiliary/uncertain channels to exclude from default detection.",
            "input_schema": _schema_ref("Brain-signal ROI request.", {"input_path": {"type": "string", "description": "EDF/BDF/FIF/FIF.GZ/BrainVision path readable by PyHFO/MNE."}}, ["input_path"]),
            "read_only": True,
            "writes_artifacts": False,
            "doctor_approval_required": False,
            "llm_usage": "Use after load_recording and before preprocessing/detection to select real brain-signal channels. Exclude EKG/ECG/EOG/EMG/AUX/X/uncertain channels from default candidate pathology analysis unless the user confirms they are neural.",
        },
        {
            "name": "compute_channel_qc",
            "phase": "quality_control",
            "status": "callable",
            "description": "Compute label-free channel QC metrics on the selected brain-signal ROI: RMS, peak-to-peak, flat fraction, NaN fraction, and robust amplitude outliers.",
            "input_schema": _schema_ref("Channel QC request.", {"input_path": {"type": "string"}, "exclude_channels": {"type": "array", "items": {"type": "string"}}, "outlier_z_threshold": {"type": "number", "default": 8.0}, "max_duration_seconds": {"type": "number", "default": 60.0}}, ["input_path"]),
            "read_only": True,
            "writes_artifacts": False,
            "doctor_approval_required": False,
            "llm_usage": "Use after infer_brain_signal_roi and before detection to identify flat/outlier/noisy candidates. Treat output as heuristic QC requiring review.",
        },
        {
            "name": "compute_spectral_profile",
            "phase": "quality_control",
            "status": "callable",
            "description": "Compute a compact label-free spectral profile for detector planning: relative band power, line-noise candidates, Nyquist, and recommended HFO band.",
            "input_schema": _schema_ref("Spectral profile request.", {"input_path": {"type": "string"}, "exclude_channels": {"type": "array", "items": {"type": "string"}}, "max_channels": {"type": "integer", "default": 32}, "max_duration_seconds": {"type": "number", "default": 60.0}}, ["input_path"]),
            "read_only": True,
            "writes_artifacts": False,
            "doctor_approval_required": False,
            "llm_usage": "Use to choose Nyquist-safe detector filters and decide whether line-noise handling should be discussed before detection.",
        },
        {
            "name": "detect_line_noise",
            "phase": "quality_control",
            "status": "callable",
            "description": "Detect candidate 50/60 Hz line-noise peaks and harmonics on a bounded preview for notch-filter decisions.",
            "input_schema": _schema_ref(
                "Line-noise detection request.",
                {
                    "input_path": {"type": "string"},
                    "exclude_channels": {"type": "array", "items": {"type": "string"}},
                    "candidate_frequencies_hz": {"type": "array", "items": {"type": "number"}},
                    "site_line_freq_hz": {"type": "number"},
                    "max_channels": {"type": "integer", "default": 32},
                    "max_harmonics": {"type": "integer", "default": 4},
                    "ratio_threshold": {"type": "number", "default": 5.0},
                    "max_duration_seconds": {"type": "number", "default": 60.0},
                },
                ["input_path"],
            ),
            "read_only": True,
            "writes_artifacts": False,
            "doctor_approval_required": False,
            "llm_usage": "Use before prepare_recording when deciding whether to apply a 50/60 Hz notch. Do not guess mains frequency from geography.",
        },
        {
            "name": "recommend_detector_plan",
            "phase": "planning",
            "status": "callable",
            "description": "Recommend Spike/HFO detector runs from recording metadata and label-free brain ROI, including Nyquist-safe HFO filters and interpretation limits.",
            "input_schema": _schema_ref("Detector planning request.", {"input_path": {"type": "string"}, "recording_info": {"type": "object", "additionalProperties": True}, "roi": {"type": "object", "additionalProperties": True}}, []),
            "read_only": True,
            "writes_artifacts": False,
            "doctor_approval_required": False,
            "llm_usage": "Use before pyhfo_detect_events to pick conservative smoke-test or analysis runs and explain why each detector/filter is appropriate.",
        },
        {
            "name": "rank_candidate_channels",
            "phase": "review",
            "status": "callable",
            "description": "Rank candidate channels from one or more detector CSV outputs after ROI exclusions, returning per-channel event counts/rates by biomarker.",
            "input_schema": _schema_ref("Candidate ranking request.", {"runs": {"type": "array", "items": {"type": "object"}}, "duration_seconds": {"type": "number"}, "exclude_channels": {"type": "array", "items": {"type": "string"}}, "top_n": {"type": "integer", "default": 20}}, ["runs"]),
            "read_only": True,
            "writes_artifacts": False,
            "doctor_approval_required": False,
            "llm_usage": "Use after Spike/HFO detection to produce signal-derived candidate rankings. Do not call these clinical pathological labels.",
        },
        {
            "name": "load_bids_channel_annotations",
            "phase": "labels",
            "status": "disabled_by_default" if not labels_enabled() else "callable",
            "description": "Clinical/BIDS sidecar labels are disabled by default. When explicitly enabled for evaluation, this reads BIDS channels.tsv labels such as soz, resection, and good/bad status. This may expose ground-truth labels and must not be used as ordinary signal metadata.",
            "input_schema": _schema_ref("BIDS channel annotation request.", {"input_path": {"type": "string", "description": "Recording path whose sibling channels.tsv should be read."}}, ["input_path"]),
            "read_only": True,
            "writes_artifacts": False,
            "doctor_approval_required": False,
            "llm_usage": "Do not use in normal agent operation. Label loading is disabled unless PYHFO_AGENT_ENABLE_LABEL_LOADING=1 is set for an explicit evaluation run.",
        },
        {
            "name": "compare_candidate_channels_to_bids_labels",
            "phase": "labels",
            "status": "disabled_by_default" if not labels_enabled() else "callable",
            "description": "Evaluation-only comparison between signal-derived candidate channels and gated BIDS channels.tsv labels such as SOZ/resection.",
            "input_schema": _schema_ref(
                "Gated label-comparison request.",
                {
                    "input_path": {"type": "string", "description": "Recording path whose sibling channels.tsv should be read if labels are not already loaded."},
                    "candidate_channels": {"type": "array", "items": {"type": "string"}},
                    "candidate_analysis": {"type": "object", "additionalProperties": True},
                    "bids_channel_annotations": {"type": "object", "additionalProperties": True},
                },
                [],
            ),
            "read_only": True,
            "writes_artifacts": False,
            "doctor_approval_required": False,
            "llm_usage": "Use only when PYHFO_AGENT_ENABLE_LABEL_LOADING=1 is set for an explicit evaluation study. Keep labels separated from detector outputs and report agreement/disagreement only.",
        },
        {
            "name": "prepare_recording",
            "phase": "preprocess",
            "status": "callable",
            "description": "Create a reproducible cleaned FIF/FIF.GZ copy with explicit bad-channel, flat-channel, outlier, notch, broad-filter, resampling, and bipolar-montage settings.",
            "input_schema": _schema_ref(
                "Preprocessing request.",
                {
                    "input_path": {"type": "string"},
                    "output_path": {"type": "string", "description": "Must end with .fif or .fif.gz."},
                    "extra_bad_channels": {"type": "array", "items": {"type": "string"}},
                    "drop_bad_channels": {"type": "boolean", "default": True},
                    "drop_flat_channels": {"type": "boolean", "default": True},
                    "mark_outlier_channels": {"type": "boolean", "default": True},
                    "drop_outlier_channels": {"type": "boolean", "default": False},
                    "outlier_z_threshold": {"type": "number", "default": 8.0},
                    "bipolar_montage": {"type": "string", "enum": ["none", "auto_adjacent", "pairs"], "default": "none"},
                    "bipolar_pairs": {"type": "array", "items": {"type": "object"}},
                    "drop_original_after_bipolar": {"type": "boolean", "default": False},
                    "line_noise_notch_hz": {"type": "number", "description": "Use 50 or 60 only when known from user/site/header."},
                    "notch_freqs": {"type": "array", "items": {"type": "number"}},
                    "bandpass": {"type": "object", "additionalProperties": True},
                    "resample_sfreq": {"type": "number"},
                },
                ["input_path", "output_path"],
            ),
            "read_only": False,
            "writes_artifacts": True,
            "doctor_approval_required": False,
            "llm_usage": "Use for real analysis after metadata. Report every cleaning/montage/filter action and use output_path for detection if enough neural channels remain.",
        },
        {
            "name": "run_detection",
            "phase": "analysis",
            "status": "callable",
            "description": "Run headless HFO/spindle/spike detection and export structured artifacts.",
            "input_schema": _schema_ref(
                "Detection request.",
                {
                    "input_path": {"type": "string", "description": "Prefer a prepared FIF/FIF.GZ copy."},
                    "output_dir": {"type": "string"},
                    "biomarker": {"type": "string", "enum": ["HFO", "Spindle", "Spike"], "default": "HFO"},
                    "detector": {"type": "string", "default": "STE"},
                    "filter": {"type": "object", "additionalProperties": True},
                    "detector_params": {"type": "object", "additionalProperties": True},
                    "classifier": {"type": "object", "additionalProperties": True},
                    "n_jobs": {"type": "integer", "minimum": 1, "default": 1},
                    "exports": {"type": "array", "items": {"type": "string"}},
                    "include_channels": {"type": "array", "items": {"type": "string"}},
                    "exclude_channels": {"type": "array", "items": {"type": "string"}},
                },
                ["input_path", "output_dir"],
            ),
            "read_only": False,
            "writes_artifacts": True,
            "doctor_approval_required": False,
            "llm_usage": "Smoke-test first with n_jobs=1 and Nyquist-safe filters. Treat counts as algorithmic candidates requiring expert review.",
        },
        {
            "name": "run_severity_scoring",
            "phase": "analysis",
            "status": "callable",
            "description": "Score a scalp EEG recording's overall severity with the BASED transformer model, in 15-second segments on a 0-5 scale.",
            "input_schema": _schema_ref(
                "Severity scoring request.",
                {
                    "input_path": {"type": "string", "description": "Defaults to the currently loaded recording if omitted."},
                    "model_dir": {"type": "string", "description": "Local config.json and model.safetensors directory. Defaults to PYHFO_SEVERITY_MODEL_DIR or the pinned model release."},
                    "export_csv": {"type": "string", "description": "Optional path to also write per-segment scores as CSV."},
                },
                [],
            ),
            "read_only": False,
            "writes_artifacts": False,
            "doctor_approval_required": False,
            "llm_usage": "Only meaningful for verified scalp EEG (conventional 10-20 layout). Report the 0-5 score as a research signal, not a clinical severity diagnosis. Use clean_mean_score / clean_peak_score / clean_peak_time_sec only when artifact_quality.status is assessed. If artifact screening is partial or unavailable, disclose that clean summaries were withheld; raw mean_score/peak_score may be reported only when explicitly labelled as all-segment raw scores.",
        },
        {
            "name": "classify_existing_run",
            "phase": "analysis",
            "status": "callable",
            "description": "Classify an existing saved PyBrain/NPZ session active run without rerunning detector discovery.",
            "input_schema": _schema_ref(
                "Active-run classification request.",
                {
                    "session_path": {"type": "string"},
                    "biomarker": {"type": "string", "enum": ["HFO", "Spindle"], "default": "HFO"},
                    "run_id": {"type": "string"},
                    "classifier": {"type": "object", "additionalProperties": True},
                    "exports": {"type": "array", "items": {"type": "string"}},
                },
                ["session_path", "classifier"],
            ),
            "read_only": False,
            "writes_artifacts": True,
            "doctor_approval_required": False,
            "llm_usage": "Use after run_detection/session export when the user wants PyQt-style post-hoc classification of the active run without detector rerun.",
        },
        {
            "name": "create_batch_project",
            "phase": "batch",
            "status": "callable",
            "description": "Create a resumable manifest from a folder/cohort without running detection.",
            "input_schema": _schema_ref("Batch manifest request.", {"input_dir": {"type": "string"}, "output_dir": {"type": "string"}, "recursive": {"type": "boolean", "default": True}}, ["input_dir"]),
            "read_only": False,
            "writes_artifacts": True,
            "doctor_approval_required": False,
            "llm_usage": "Use before broad batch runs; inspect discovered/skipped cases before running detection.",
        },
        {
            "name": "run_batch",
            "phase": "batch",
            "status": "callable",
            "description": "Run detection over a batch manifest or input directory.",
            "input_schema": _schema_ref("Batch run request.", {"manifest_path": {"type": "string"}, "input_dir": {"type": "string"}, "output_dir": {"type": "string"}, "recursive": {"type": "boolean", "default": True}, "retry_failed": {"type": "boolean", "default": False}, "fail_fast": {"type": "boolean", "default": False}, "detection": {"type": "object", "additionalProperties": True}}, []),
            "read_only": False,
            "writes_artifacts": True,
            "doctor_approval_required": False,
            "llm_usage": "Use fail_fast/n_jobs=1 for first validation. Summarize failures separately.",
        },
        {"name": "waveform_window", "phase": "waveform", "status": "callable", "description": "Return a bounded, optionally downsampled waveform window for selected channels.", "input_schema": _schema_ref("Waveform window request.", {"input_path": {"type": "string"}, "start_seconds": {"type": "number"}, "duration_seconds": {"type": "number"}, "channels": {"type": "array", "items": {"type": "string"}}, "target_samples_per_channel": {"type": "integer"}, "show_filtered": {"type": "boolean"}, "filter_60hz": {"type": "boolean"}, "filter_low": {"type": "number"}, "filter_high": {"type": "number"}}, []), "read_only": True, "writes_artifacts": False, "doctor_approval_required": False, "llm_usage": "Use for renderer waveform views; never request full recordings."},
        {"name": "waveform_overview", "phase": "waveform", "status": "callable", "description": "Return bounded overview min/max/rms bins for selected channels.", "input_schema": _schema_ref("Waveform overview request.", {"input_path": {"type": "string"}, "bins": {"type": "integer"}}, []), "read_only": True, "writes_artifacts": False, "doctor_approval_required": False, "llm_usage": "Use for minimap/overview rendering."},
        {"name": "waveform_channels", "phase": "waveform", "status": "callable", "description": "Return channel metadata for a recording.", "input_schema": _schema_ref("Waveform channels request.", {"input_path": {"type": "string"}}, []), "read_only": True, "writes_artifacts": False, "doctor_approval_required": False, "llm_usage": "Use before selecting waveform channels."},
        {"name": "detector_defaults", "phase": "planning", "status": "callable", "description": "Instantiate PyHFO detector default parameters with optional biomarker/detector/sample-frequency context.", "input_schema": _schema_ref("Parameter defaults request.", {"biomarker": {"type": "string"}, "detector": {"type": "string"}, "sample_freq": {"type": "number"}, "preset": {"type": "string"}}, []), "read_only": True, "writes_artifacts": False, "doctor_approval_required": False, "llm_usage": "Use to inspect baseline detector fields before recommendation, validation, or manual overrides."},
        {"name": "recommend_parameters", "phase": "planning", "status": "callable", "description": "Recommend goal-profile-aware detector parameters with validation and UI diff card.", "input_schema": _schema_ref("Parameter recommendation request.", {"goal_profile": {"type": "string"}, "biomarker": {"type": "string"}, "recording_info": {"type": "object"}}, []), "read_only": False, "writes_artifacts": False, "doctor_approval_required": False, "llm_usage": "Use before detection to explain parameter adjustments and validation status."},
        {"name": "validate_parameters", "phase": "planning", "status": "callable", "description": "Validate detector/filter parameters, including Nyquist safety.", "input_schema": _schema_ref("Parameter validation request.", {"sample_freq": {"type": "number"}, "biomarker": {"type": "string"}, "filter": {"type": "object"}}, []), "read_only": True, "writes_artifacts": False, "doctor_approval_required": False, "llm_usage": "Reject unsafe filters with concrete metadata reasons."},
        {"name": "sweep_parameters", "phase": "planning", "status": "callable", "description": "Create a bounded local parameter sweep plan around recommended settings.", "input_schema": _schema_ref("Parameter sweep request.", {"goal_profile": {"type": "string"}, "biomarker": {"type": "string"}}, []), "read_only": False, "writes_artifacts": False, "doctor_approval_required": False, "llm_usage": "Use for explicit exploration/comparison goals."},
        {"name": "list_events", "phase": "doctor_review", "status": "callable", "description": "List current or run-scoped candidate events and review annotations.", "input_schema": _schema_ref("Event list request.", {"run_id": {"type": "string"}, "events": {"type": "array"}}, []), "read_only": True, "writes_artifacts": False, "doctor_approval_required": False, "llm_usage": "Use after detection to inspect the review queue before focusing or annotating events."},
        {"name": "review_event_cursor", "phase": "doctor_review", "status": "callable", "description": "Navigate candidate events without PyQt widgets.", "input_schema": _schema_ref("Event cursor request.", {"action": {"type": "string"}, "event_id": {"type": "string"}}, []), "read_only": False, "writes_artifacts": False, "doctor_approval_required": True, "llm_usage": "Use to focus events for human review."},
        {"name": "event_snippet", "phase": "doctor_review", "status": "callable", "description": "Return bounded waveform, filtered-trace, and time-frequency payloads around one candidate event.", "input_schema": _schema_ref("Event snippet request.", {"event_id": {"type": "string"}, "input_path": {"type": "string"}, "padding_seconds": {"type": "number"}, "target_samples_per_channel": {"type": "integer"}, "run_id": {"type": "string"}}, ["event_id"]), "read_only": True, "writes_artifacts": False, "doctor_approval_required": False, "llm_usage": "Use for focused physician review packets; never request full recordings."},
        {"name": "event_feature_summary", "phase": "doctor_review", "status": "callable", "description": "Summarize structured features for one candidate event.", "input_schema": _schema_ref("Event feature summary request.", {"event_id": {"type": "string"}, "run_id": {"type": "string"}, "events": {"type": "array"}}, ["event_id"]), "read_only": True, "writes_artifacts": False, "doctor_approval_required": False, "llm_usage": "Use to inspect event features before asking a doctor to review a label."},
        {"name": "annotate_event", "phase": "doctor_review", "status": "callable", "description": "Apply/clear human review labels with provenance and audit trail.", "input_schema": _schema_ref("Annotation request.", {"event_id": {"type": "string"}, "label": {"type": "string"}, "notes": {"type": "string"}, "display_label": {"type": "string"}}, ["event_id", "label"]), "read_only": False, "writes_artifacts": False, "doctor_approval_required": True, "llm_usage": "Stage suggestions; persistent labels require human confirmation."},
        {"name": "cross_channel_overlap_review", "phase": "doctor_review", "status": "callable", "description": "Inspect cross-channel temporal overlaps for candidate events.", "input_schema": _schema_ref("Overlap review request.", {"events": {"type": "array"}}, []), "read_only": True, "writes_artifacts": False, "doctor_approval_required": True, "llm_usage": "Use for overlap review summaries, not diagnosis."},
        {"name": "compare_detector_runs", "phase": "review", "status": "callable", "description": "Compare multiple detector runs for channel support, run-specific candidates, and temporal overlaps.", "input_schema": _schema_ref("Run comparison request.", {"events_by_run": {"type": "object", "additionalProperties": True}, "run_ids": {"type": "array", "items": {"type": "string"}}, "tolerance_seconds": {"type": "number"}, "max_overlap_examples": {"type": "integer"}}, []), "read_only": True, "writes_artifacts": False, "doctor_approval_required": False, "llm_usage": "Use after two or more runs to identify stable versus parameter-sensitive signal-derived candidates."},
        {"name": "create_consensus_run", "phase": "review", "status": "callable", "description": "Create a consensus candidate run from multiple source runs.", "input_schema": _schema_ref("Consensus run request.", {"events_by_run": {"type": "object", "additionalProperties": True}, "run_ids": {"type": "array", "items": {"type": "string"}}, "strategy": {"type": "string", "default": "majority"}, "biomarker": {"type": "string"}}, []), "read_only": False, "writes_artifacts": False, "doctor_approval_required": True, "llm_usage": "Use for review-layer consensus only; consensus events are algorithmic candidates, not ground truth."},
        {"name": "export_review_state", "phase": "review", "status": "callable", "description": "Export current review cursor, visible runs, events, annotations, and audit trail as structured JSON.", "input_schema": _schema_ref("No arguments.", {}, []), "read_only": True, "writes_artifacts": False, "doctor_approval_required": False, "llm_usage": "Use before report/session/workbook exports or when summarizing current review state."},
        {
            "name": "read_summary",
            "phase": "review",
            "status": "callable",
            "description": "Read a small PyHFO summary JSON artifact into structured content.",
            "input_schema": _schema_ref("Summary read request.", {"summary_path": {"type": "string"}}, ["summary_path"]),
            "read_only": True,
            "writes_artifacts": False,
            "doctor_approval_required": False,
            "llm_usage": "Use after detection to inspect counts/settings/warnings without loading large EEG arrays.",
        },
        {"name": "create_project", "phase": "project", "status": "callable", "description": "Create a resumable project state from recording, artifacts, runs, labels, and action log.", "input_schema": _schema_ref("Project creation request.", {"recording_path": {"type": "string"}, "derived_artifacts": {"type": "object"}, "detection_runs": {"type": "array"}, "review_labels": {"type": "object"}, "action_log": {"type": "array"}}, []), "read_only": False, "writes_artifacts": False, "doctor_approval_required": False, "llm_usage": "Use when starting a durable review project record."},
        {"name": "open_project", "phase": "project", "status": "callable", "description": "Open a saved PyHFO project JSON artifact into sidecar state.", "input_schema": _schema_ref("Project open request.", {"project_path": {"type": "string"}, "path": {"type": "string"}}, []), "read_only": True, "writes_artifacts": False, "doctor_approval_required": False, "llm_usage": "Use to resume an existing review project."},
        {"name": "save_project", "phase": "project", "status": "callable", "description": "Write the current or supplied project state to a JSON artifact.", "input_schema": _schema_ref("Project save request.", {"project_path": {"type": "string"}, "output_path": {"type": "string"}, "state": {"type": "object"}}, []), "read_only": False, "writes_artifacts": True, "doctor_approval_required": False, "llm_usage": "Use to persist case review progress without embedding raw EEG data."},
        {"name": "export_session", "phase": "export", "status": "callable", "description": "Export current sidecar/session state as a JSON artifact.", "input_schema": _schema_ref("Session export request.", {"output_path": {"type": "string"}, "session_state": {"type": "object"}}, []), "read_only": False, "writes_artifacts": True, "doctor_approval_required": False, "llm_usage": "Use for reproducible handoff of state and artifact pointers."},
        {"name": "export_report", "phase": "export", "status": "callable", "description": "Export a physician-reviewable HTML report with metadata, runs, review state, artifacts, and limitations.", "input_schema": _schema_ref("Report export request.", {"output_path": {"type": "string"}, "review_state": {"type": "object"}, "allow_absolute_path": {"type": "boolean"}}, []), "read_only": False, "writes_artifacts": True, "doctor_approval_required": False, "llm_usage": "Use for review/audit reports; state limitations that signal-derived candidates are not clinical diagnoses."},
        {"name": "export_workbook", "phase": "export", "status": "callable", "description": "Export review state and summaries as a workbook artifact.", "input_schema": _schema_ref("Workbook export request.", {"output_path": {"type": "string"}, "review_state": {"type": "object"}, "allow_absolute_path": {"type": "boolean"}}, []), "read_only": False, "writes_artifacts": True, "doctor_approval_required": False, "llm_usage": "Use when the doctor/coordinator wants spreadsheet-ready review state."},
    ]

    missing_actions: list[dict[str, Any]] = []
    if include_missing:
        actions.extend(missing_actions)

    detector_availability = get_detector_availability()
    spindle_detectors = {
        detector["detector"]: detector
        for detector in detector_availability["detectors_by_biomarker"]["Spindle"]
    }
    hfo_detectors = {
        detector["detector"]: detector
        for detector in detector_availability["detectors_by_biomarker"]["HFO"]
    }
    yasa_status = spindle_detectors["YASA"]
    hil_status = hfo_detectors["HIL"]

    return json_safe(
        {
            "schema_version": "pyhfo-sidecar.v1",
            "software_version": __version__,
            "purpose": "Decoupled PyHFO sidecar contract for generic ReAct agents and optional web UI control.",
            "architecture": {
                "generic_agent_role": "Plan and call explicit actions from this manifest; keep PyHFO domain rules in a skill, not the core system prompt.",
                "pyhfo_sidecar_role": "Own PyHFO imports, state, preprocessing/detection execution, artifacts, and UI-control-safe action semantics.",
                "frontend_role": "Render state/actions/results; avoid embedding analysis rules that belong in the sidecar manifest or agent skill.",
            },
            "recommended_pipeline": [
                "load_recording",
                "infer_brain_signal_roi to select real brain-signal channels and exclude EKG/aux/uncertain channels",
                "get_detector_availability before spindle/YASA/LSM planning",
                "detect_line_noise when notch handling is uncertain; do not guess 50/60 Hz from geography",
                "prepare_recording when real analysis/cleaning/montage/notch is needed",
                "run_detection as smoke test before full/batch analysis",
                "read_summary and surface warnings/artifacts",
                "doctor_review wrappers before any clinical annotation/curation claims",
            ],
            "actions": actions,
            "optional_dependencies": {
                "yasa": {
                    "available": yasa_status["status"] == "callable",
                    "detectors": [{"biomarker": "Spindle", "detector": "YASA"}],
                    "install_hint": yasa_status["install_hint"],
                    "ui_behavior": "Disable YASA Run actions when unavailable, while keeping parameters visible for parity/review.",
                },
                "hfo_hil": {
                    "available": hil_status["status"] == "callable",
                    "detectors": [{"biomarker": "HFO", "detector": "HIL"}],
                    "install_hint": hil_status["install_hint"],
                    "ui_behavior": "Disable HIL-specific run actions when the optional HIL detector is unavailable; keep STE/MNI/RMS/LineLength available.",
                },
            },
            "detector_availability": detector_availability,
            "classifier": {
                "gpu": _torch_gpu_capability(),
                "cpu": {
                    "available": True,
                    "backend": "torch" if importlib.util.find_spec("torch") else "python",
                    "ui_behavior": "Keep Hugging Face CPU classifier preset available for HFO/Spindle classifier workflows.",
                },
            },
            "safety": {
                "clinical_boundary": "Outputs are research/algorithmic artifacts requiring expert review, not diagnosis or treatment advice.",
                "artifact_policy": "Write cleaned copies/results to explicit output paths; never overwrite raw recordings.",
                "language_policy": "English-first internals; user-facing agent may answer in the user's language.",
                "label_loading_policy": "BIDS/clinical labels are disabled by default. Chat workflows must use EDF signal-derived evidence only unless PYHFO_AGENT_ENABLE_LABEL_LOADING=1 is set for evaluation.",
            },
        }
    )


def _severity_cache_key(input_path: Any) -> str:
    """Recording identity for the severity cache: absolute, case-folded, forward-slashed.

    The same file reaches us as both C:\\data\\a.edf and C:/data/a.edf depending on who asked, and
    two spellings of one recording must not look like two recordings.
    """
    try:
        return str(Path(str(input_path or "")).expanduser().resolve()).replace("\\", "/").lower()
    except Exception:
        return str(input_path or "").replace("\\", "/").lower()


@dataclass
class PyHFOSidecarSession:
    session_id: str = "default"
    created_at: str = field(default_factory=utc_now_iso)
    recording_info: dict[str, Any] | None = None
    prepared_recording: dict[str, Any] | None = None
    saved_session_path: str | None = None
    last_detection: dict[str, Any] | None = None
    last_severity: dict[str, Any] | None = None
    # Severity results keyed by the recording they were computed from. Scoring a long study takes
    # many minutes, so switching away and back must not mean recomputing -- and `last_severity`
    # alone could only ever describe ONE recording, which is how a previous file's score ended up
    # shown against a newly loaded one.
    severity_by_recording: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_batch: dict[str, Any] | None = None
    last_summary: dict[str, Any] | None = None
    candidate_analysis: dict[str, Any] | None = None
    bids_channel_annotations: dict[str, Any] | None = None
    brain_signal_roi: dict[str, Any] | None = None
    action_log: list[dict[str, Any]] = field(default_factory=list)
    review_events: list[dict[str, Any]] = field(default_factory=list)
    review_events_by_run: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    review_run_filters: dict[str, dict[str, Any]] = field(default_factory=dict)
    review_active_run_id: str | None = None
    review_accepted_run_id: str | None = None
    review_visible_run_ids: list[str] = field(default_factory=list)
    review_annotations: dict[str, Any] = field(default_factory=dict)
    annotation_audit_trail: list[dict[str, Any]] = field(default_factory=list)
    review_cursor_event_id: str | None = None
    parameter_plan_history: list[dict[str, Any]] = field(default_factory=list)
    project_state: dict[str, Any] | None = None

    def get_state(self) -> dict[str, Any]:
        return json_safe(
            {
                "status": "ready",
                "session_id": self.session_id,
                "created_at": self.created_at,
                "loaded_recording_path": (self.recording_info or {}).get("input_path"),
                "recording_info": self.recording_info,
                "bids_channel_annotations_loaded": self.bids_channel_annotations is not None,
                "bids_channel_annotations": self.bids_channel_annotations,
                "brain_signal_roi_loaded": self.brain_signal_roi is not None,
                "brain_signal_roi": self.brain_signal_roi,
                "prepared_recording_path": (self.prepared_recording or {}).get("output_path"),
                "prepared_recording": self.prepared_recording,
                "last_detection_output_dir": (self.last_detection or {}).get("output_dir"),
                "last_detection_summary": {
                    "status": (self.last_detection or {}).get("status"),
                    "biomarker": (self.last_detection or {}).get("biomarker"),
                    "detector": (self.last_detection or {}).get("detector"),
                    "num_events": (self.last_detection or {}).get("num_events"),
                    "num_channels": (self.last_detection or {}).get("num_channels"),
                    "artifacts": (self.last_detection or {}).get("artifacts"),
                    "warnings": (self.last_detection or {}).get("warnings"),
                }
                if self.last_detection
                else None,
                "last_severity_summary": {
                    "status": (self.last_severity or {}).get("status"),
                    "eeg_type": (self.last_severity or {}).get("eeg_type"),
                    "num_segments": (self.last_severity or {}).get("num_segments"),
                    "mean_score": (self.last_severity or {}).get("mean_score"),
                    "peak_score": (self.last_severity or {}).get("peak_score"),
                    "peak_time_sec": (self.last_severity or {}).get("peak_time_sec"),
                    "artifacts": (self.last_severity or {}).get("artifacts"),
                    "warnings": (self.last_severity or {}).get("warnings"),
                }
                if self.last_severity
                else None,
                "candidate_analysis": self.candidate_analysis,
                "last_batch_manifest_path": (self.last_batch or {}).get("manifest_path"),
                "action_log_tail": self.action_log[-10:],
                "review_state": export_review_state(self),
                "parameter_plan_history": self.parameter_plan_history,
                "project_state": self.project_state,
                "missing_review_wrappers": [],
            }
        )

    def dispatch(self, action: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        args = dict(arguments or {})
        normalized = str(action or "").strip()
        started_at = utc_now_iso()
        if not normalized:
            raise PyHFOAgentError("invalid_request", "action name is required")

        if normalized == "get_capability_manifest":
            result = get_capability_manifest(include_missing=bool(args.get("include_missing", True)))
        elif normalized == "get_detector_availability":
            result = get_detector_availability()
        elif normalized == "get_state":
            result = self.get_state()
        elif normalized == "load_recording":
            previous_path = (self.recording_info or {}).get("input_path")
            result = get_recording_info(str(args.get("input_path") or ""))
            new_path = result.get("input_path")
            if previous_path and new_path and previous_path != new_path:
                # Avoid leaking stale ROI/detections/labels from the previous patient/file.
                self.prepared_recording = None
                self.last_detection = None
                self.last_summary = None
                self.candidate_analysis = None
                self.bids_channel_annotations = None
                self.brain_signal_roi = None
                self.review_events = []
                self.review_events_by_run = {}
                self.review_active_run_id = None
                self.review_accepted_run_id = None
                self.review_visible_run_ids = []
                self.review_annotations = {}
                self.annotation_audit_trail = []
                self.review_cursor_event_id = None
                self.parameter_plan_history = []
                self.project_state = None
            self.recording_info = result
            # Severity follows the recording: show the score already computed for THIS file (free,
            # and it survives switching back and forth), and never the previous file's.
            self.last_severity = self.severity_by_recording.get(_severity_cache_key(result.get("input_path")))
        elif normalized == "infer_brain_signal_roi":
            if args.get("channel_names"):
                result = infer_brain_signal_roi_from_channels([str(channel) for channel in args.get("channel_names") or []])
            elif self.recording_info and self.recording_info.get("channel_names") and (
                not args.get("input_path") or args.get("input_path") == self.recording_info.get("input_path")
            ):
                result = infer_brain_signal_roi_from_channels([str(channel) for channel in self.recording_info.get("channel_names") or []])
            else:
                result = infer_brain_signal_roi(str(args.get("input_path") or (self.recording_info or {}).get("input_path") or ""))
            result = {**result, "input_path": str(args.get("input_path") or (self.recording_info or {}).get("input_path") or "")}
            self.brain_signal_roi = result
        elif normalized == "compute_channel_qc":
            result = compute_channel_qc({**args, "input_path": args.get("input_path") or (self.recording_info or {}).get("input_path")})
        elif normalized == "compute_spectral_profile":
            result = compute_spectral_profile({**args, "input_path": args.get("input_path") or (self.recording_info or {}).get("input_path")})
        elif normalized == "detect_line_noise":
            result = detect_line_noise({**args, "input_path": args.get("input_path") or (self.recording_info or {}).get("input_path")})
        elif normalized == "recommend_detector_plan":
            payload = {**args}
            if not payload.get("recording_info") and self.recording_info:
                payload["recording_info"] = self.recording_info
            if not payload.get("roi") and self.brain_signal_roi:
                payload["roi"] = self.brain_signal_roi
            result = recommend_detector_plan(payload)
        elif normalized == "rank_candidate_channels":
            result = rank_candidate_channels(args)
        elif normalized == "load_bids_channel_annotations":
            if not labels_enabled():
                raise PyHFOAgentError(
                    "label_loading_disabled",
                    "BIDS/clinical label loading is disabled. Use EDF signal-derived tools instead, or start with PYHFO_AGENT_ENABLE_LABEL_LOADING=1 for explicit evaluation.",
                )
            result = load_bids_channel_annotations(str(args.get("input_path") or (self.recording_info or {}).get("input_path") or ""))
            if result is None:
                raise PyHFOAgentError("file_not_found", "No matching BIDS channels.tsv found for this recording.")
            self.bids_channel_annotations = result
        elif normalized == "compare_candidate_channels_to_bids_labels":
            result = compare_candidate_channels_to_bids_labels(self, {**args, "input_path": args.get("input_path") or (self.recording_info or {}).get("input_path")})
        elif normalized == "prepare_recording":
            result = run_pipeline_isolated("prepare_recording", args)
            self.prepared_recording = result
            self.last_detection = None
            self.last_summary = None
            self.candidate_analysis = None
            if args.get("bipolar_montage"):
                # A bipolar montage renames channels (A1,A2 -> A1-A2) and drops the
                # monopolar originals, so a brain_signal_roi computed for the original
                # channel set no longer matches the prepared file. Leaving it would make
                # the next run_detection auto-inject all-missing channels and fail with
                # invalid_channel_selection. Invalidate it.
                self.brain_signal_roi = None
        elif normalized == "run_detection":
            if self.brain_signal_roi:
                args.setdefault("include_channels", self.brain_signal_roi.get("brain_signal_channels") or [])
                args.setdefault("exclude_channels", self.brain_signal_roi.get("default_detection_exclusions") or [])
            # Snapshot which recording the SESSION is on at launch. Detection legitimately
            # runs on a prepared file whose path differs from recording_info (load A ->
            # prepare A->clean -> detect clean), so we compare the session recording
            # before vs after the run — NOT the detection input against recording_info.
            launch_recording_path = (self.recording_info or {}).get("input_path")
            result = run_pipeline_isolated("run_detection", args)
            # If a concurrent load_recording switched the session's recording while this
            # detection ran in its subprocess, don't overwrite last_detection with a result
            # for the previous file (cross-file/cross-patient mix). The result is still
            # returned to the caller that requested it.
            if (self.recording_info or {}).get("input_path") == launch_recording_path:
                self.last_detection = {**result, "request": json_safe(args)} if isinstance(result, dict) else result
        elif normalized == "run_severity_scoring":
            payload = {**args}
            if not payload.get("input_path"):
                payload["input_path"] = (self.recording_info or {}).get("input_path")
            # Same cross-file safety check as run_detection: don't let a slow severity
            # pass overwrite last_severity if the session moved on to a different recording.
            launch_recording_path = (self.recording_info or {}).get("input_path")
            result = run_pipeline_isolated("run_severity_scoring", payload)
            # Cache under the recording that was actually scored, so it is reusable later even if the
            # session has already moved on; only SHOW it if that is still the loaded recording.
            if isinstance(result, dict) and result.get("status") == "complete":
                self.severity_by_recording[_severity_cache_key(payload.get("input_path"))] = result
            if (self.recording_info or {}).get("input_path") == launch_recording_path:
                self.last_severity = result
        elif normalized == "classify_existing_run":
            result = run_pipeline_isolated("classify_existing_run", args)
            self.last_detection = {**result, "request": json_safe(args)} if isinstance(result, dict) else result
        elif normalized == "create_batch_project":
            result = create_batch(str(args.get("input_dir") or ""), args.get("output_dir"), recursive=bool(args.get("recursive", True)))
            self.last_batch = result
        elif normalized == "run_batch":
            result = run_pipeline_isolated("run_batch", args)
            self.last_batch = result
        elif normalized == "waveform_window":
            result = waveform_window(str(args.get("input_path") or (self.recording_info or {}).get("input_path") or ""), start_seconds=float(args.get("start_seconds", 0.0)), duration_seconds=float(args.get("duration_seconds", 10.0)), channels=args.get("channels"), target_samples_per_channel=int(args.get("target_samples_per_channel", 1200)), units=str(args.get("units", "uV")), show_filtered=bool(args.get("show_filtered", False)), filter_60hz=bool(args.get("filter_60hz", False)), filter_low=args.get("filter_low"), filter_high=args.get("filter_high"))
        elif normalized == "waveform_overview":
            result = waveform_overview(str(args.get("input_path") or (self.recording_info or {}).get("input_path") or ""), channels=args.get("channels"), bins=int(args.get("bins", 512)), units=str(args.get("units", "uV")))
        elif normalized == "waveform_channels":
            result = waveform_channels(str(args.get("input_path") or (self.recording_info or {}).get("input_path") or ""))
        elif normalized == "detector_defaults":
            payload = {**args}
            payload.setdefault("recording_info", self.recording_info or {})
            result = detector_defaults(payload)
        elif normalized == "recommend_parameters":
            payload = {**args}
            payload.setdefault("recording_info", self.recording_info or {})
            payload.setdefault("roi", self.brain_signal_roi or {})
            result = recommend_parameters(payload)
            self.parameter_plan_history.append(result)
        elif normalized == "validate_parameters":
            payload = {**args}
            payload.setdefault("recording_info", self.recording_info or {})
            result = validate_parameters(payload)
        elif normalized == "sweep_parameters":
            payload = {**args}
            payload.setdefault("recording_info", self.recording_info or {})
            payload.setdefault("roi", self.brain_signal_roi or {})
            result = sweep_parameters(payload)
            self.parameter_plan_history.append(result)
        elif normalized == "list_events":
            result = list_events(self, events=args.get("events"), run_id=args.get("run_id"))
        elif normalized == "review_event_cursor":
            result = event_cursor(self, action=str(args.get("action", "current")), event_id=args.get("event_id"), events=args.get("events"), run_id=args.get("run_id"))
        elif normalized == "event_snippet":
            result = event_snippet(self, str(args.get("event_id")), input_path=str(args.get("input_path") or (self.recording_info or {}).get("input_path") or ""), padding_seconds=float(args.get("padding_seconds", 1.0)), target_samples_per_channel=int(args.get("target_samples_per_channel", 600)), events=args.get("events"), run_id=args.get("run_id"))
        elif normalized == "event_feature_summary":
            result = event_feature_summary(self, str(args.get("event_id")), events=args.get("events"), run_id=args.get("run_id"))
        elif normalized == "annotate_event":
            result = annotate_event(self, str(args.get("event_id")), label=str(args.get("label")), notes=args.get("notes"), reviewer=str(args.get("reviewer", "human")), source=str(args.get("source", "agent")), confirmed=(args.get("confirmed") is True), display_label=args.get("display_label"), events=args.get("events"), run_id=args.get("run_id"))
        elif normalized == "cross_channel_overlap_review":
            result = overlap_review(self, events=args.get("events"), tolerance_seconds=float(args.get("tolerance_seconds", 0.0)), run_id=args.get("run_id"))
        elif normalized == "compare_detector_runs":
            result = compare_detector_runs(self, events_by_run=args.get("events_by_run"), run_ids=args.get("run_ids"), tolerance_seconds=float(args.get("tolerance_seconds", 0.0)), max_overlap_examples=int(args.get("max_overlap_examples", 100)))
        elif normalized == "create_consensus_run":
            result = create_consensus_run(self, events_by_run=args.get("events_by_run"), run_ids=args.get("run_ids"), strategy=str(args.get("strategy", "majority")), biomarker=args.get("biomarker"))
        elif normalized == "export_review_state":
            result = export_review_state(self)
        elif normalized == "create_project":
            result = create_project(args)
            self.project_state = result.get("state")
        elif normalized == "open_project":
            result = open_project(str(args.get("project_path") or args.get("path") or ""))
            self.project_state = result.get("state")
        elif normalized == "save_project":
            payload = {**args}
            payload.setdefault("state", self.project_state or {})
            result = save_project(payload)
            self.project_state = result.get("state")
        elif normalized == "export_session":
            payload = {**args}
            payload.setdefault("session_state", self.get_state())
            result = export_session(payload)
        elif normalized == "export_report":
            payload = {**args}
            payload.setdefault("review_state", export_review_state(self))
            result = export_report(payload)
        elif normalized == "export_workbook":
            payload = {**args}
            payload.setdefault("review_state", export_review_state(self))
            result = export_workbook(payload)
        elif normalized == "read_summary":
            summary_path = Path(str(args.get("summary_path") or "")).expanduser().resolve()
            if not summary_path.exists():
                raise PyHFOAgentError("file_not_found", f"Summary file does not exist: {summary_path}")
            with open(summary_path, "r", encoding="utf-8") as handle:
                result = json.load(handle)
            self.last_summary = result
        else:
            raise PyHFOAgentError("invalid_action", f"Unknown sidecar action: {normalized}")

        self.action_log.append({"action": normalized, "started_at": started_at, "finished_at": utc_now_iso(), "status": "complete"})
        return json_safe({"status": "complete", "action": normalized, "result": result, "state": self.get_state()})


def _session_store_dir() -> Path:
    configured = os.getenv("PYHFO_AGENT_SESSION_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return pyhfo_home() / "sessions"


def _session_path(session_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(session_id or "default").strip())[:120] or "default"
    return _session_store_dir() / f"{safe_id}.json"


def load_sidecar_session(session_id: str = "default") -> PyHFOSidecarSession:
    path = _session_path(session_id)
    if not path.exists():
        return PyHFOSidecarSession(session_id=session_id or "default")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return PyHFOSidecarSession(session_id=session_id or "default")
    if not isinstance(payload, dict):
        return PyHFOSidecarSession(session_id=session_id or "default")
    allowed_fields = {
        "session_id",
        "created_at",
        "recording_info",
        "prepared_recording",
        "saved_session_path",
        "last_detection",
        "last_batch",
        "last_summary",
        "candidate_analysis",
        "bids_channel_annotations",
        "brain_signal_roi",
        "action_log",
        "review_events",
        "review_events_by_run",
        "review_run_filters",
        "review_active_run_id",
        "review_accepted_run_id",
        "review_visible_run_ids",
        "review_annotations",
        "annotation_audit_trail",
        "review_cursor_event_id",
        "parameter_plan_history",
        "project_state",
    }
    kwargs: dict[str, Any] = {key: payload.get(key) for key in allowed_fields if key in payload}
    kwargs["session_id"] = str(kwargs.get("session_id") or session_id or "default")
    kwargs["created_at"] = str(kwargs.get("created_at") or utc_now_iso())
    for list_field in ["action_log", "review_events", "review_visible_run_ids", "annotation_audit_trail", "parameter_plan_history"]:
        if not isinstance(kwargs.get(list_field), list):
            kwargs[list_field] = []
    for dict_field in ["review_events_by_run", "review_run_filters", "review_annotations"]:
        if not isinstance(kwargs.get(dict_field), dict):
            kwargs[dict_field] = {}
    return PyHFOSidecarSession(**kwargs)


def save_sidecar_session(session: PyHFOSidecarSession) -> Path:
    path = _session_path(session.session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": session.session_id,
        "created_at": session.created_at,
        "recording_info": session.recording_info,
        "prepared_recording": session.prepared_recording,
        "saved_session_path": session.saved_session_path,
        "last_detection": session.last_detection,
        "last_batch": session.last_batch,
        "last_summary": session.last_summary,
        "candidate_analysis": session.candidate_analysis,
        "bids_channel_annotations": session.bids_channel_annotations,
        "brain_signal_roi": session.brain_signal_roi,
        "action_log": session.action_log[-200:],
        "review_events": session.review_events,
        "review_events_by_run": session.review_events_by_run,
        "review_run_filters": session.review_run_filters,
        "review_active_run_id": session.review_active_run_id,
        "review_accepted_run_id": session.review_accepted_run_id,
        "review_visible_run_ids": session.review_visible_run_ids,
        "review_annotations": session.review_annotations,
        "annotation_audit_trail": session.annotation_audit_trail[-500:],
        "review_cursor_event_id": session.review_cursor_event_id,
        "parameter_plan_history": session.parameter_plan_history[-100:],
        "project_state": session.project_state,
    }
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def dispatch_action(action: str, arguments: dict[str, Any] | None = None, session: PyHFOSidecarSession | None = None) -> dict[str, Any]:
    return (session or PyHFOSidecarSession()).dispatch(action, arguments or {})
