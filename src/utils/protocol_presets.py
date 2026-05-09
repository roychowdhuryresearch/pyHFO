from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.param.param_classifier import ParamClassifier
from src.param.param_detector import ParamDetector
from src.param.param_filter import ParamFilter, ParamFilterSpindle


PROTOCOL_PRESET_VERSION = 1


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _param_to_dict(param):
    if param is None:
        return None
    if hasattr(param, "to_dict"):
        return param.to_dict()
    return dict(param)


def build_protocol_preset(
    *,
    name: str,
    biomarker_type: str,
    param_filter=None,
    param_detector=None,
    param_classifier=None,
    notes: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    name = str(name or "").strip()
    biomarker_type = str(biomarker_type or "").strip()
    if not name:
        raise ValueError("Protocol preset name is required.")
    if biomarker_type not in {"HFO", "Spindle", "Spike"}:
        raise ValueError("Protocol preset biomarker_type must be HFO, Spindle, or Spike.")
    return {
        "format": "pybrain-protocol-preset",
        "version": PROTOCOL_PRESET_VERSION,
        "name": name,
        "biomarker_type": biomarker_type,
        "created_at": _now_iso(),
        "notes": str(notes or ""),
        "metadata": dict(metadata or {}),
        "filter": _param_to_dict(param_filter),
        "detector": _param_to_dict(param_detector),
        "classifier": _param_to_dict(param_classifier),
    }


def protocol_from_run(run, *, name: str | None = None, notes: str = "") -> dict[str, Any]:
    if run is None:
        raise ValueError("A detection run is required to build a protocol preset.")
    preset_name = name or f"{getattr(run, 'biomarker_type', 'Event')} {getattr(run, 'detector_name', 'Protocol')}"
    return build_protocol_preset(
        name=preset_name,
        biomarker_type=getattr(run, "biomarker_type", ""),
        param_filter=getattr(run, "param_filter", None),
        param_detector=getattr(run, "param_detector", None),
        param_classifier=getattr(run, "param_classifier", None),
        notes=notes,
        metadata={
            "source_run_id": getattr(run, "run_id", ""),
            "source_detector": getattr(run, "detector_name", ""),
            "source_created_at": getattr(run, "created_at", ""),
        },
    )


def save_protocol_preset(path, preset: dict[str, Any]):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(preset, indent=2), encoding="utf-8")
    return target


def load_protocol_preset(path) -> dict[str, Any]:
    preset = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_protocol_preset(preset)
    return preset


def validate_protocol_preset(preset: dict[str, Any]):
    if preset.get("format") != "pybrain-protocol-preset":
        raise ValueError("Unsupported protocol preset format.")
    if int(preset.get("version", 0) or 0) > PROTOCOL_PRESET_VERSION:
        raise ValueError("Protocol preset was created by a newer PyBrain version.")
    if preset.get("biomarker_type") not in {"HFO", "Spindle", "Spike"}:
        raise ValueError("Protocol preset has an unsupported biomarker type.")
    if not preset.get("name"):
        raise ValueError("Protocol preset is missing a name.")
    return True


def restore_protocol_params(preset: dict[str, Any]):
    validate_protocol_preset(preset)
    biomarker_type = preset["biomarker_type"]
    filter_payload = preset.get("filter")
    detector_payload = preset.get("detector")
    classifier_payload = preset.get("classifier")
    if filter_payload:
        param_filter = ParamFilterSpindle.from_dict(dict(filter_payload)) if biomarker_type == "Spindle" else ParamFilter.from_dict(dict(filter_payload))
    else:
        param_filter = None
    return {
        "biomarker_type": biomarker_type,
        "param_filter": param_filter,
        "param_detector": ParamDetector.from_dict(detector_payload) if detector_payload else None,
        "param_classifier": ParamClassifier.from_dict(classifier_payload) if classifier_payload else None,
    }
