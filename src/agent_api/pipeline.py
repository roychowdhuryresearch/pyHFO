from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from src.agent_api.errors import PyHFOAgentError, error_response
from src.agent_api.schemas import (
    BatchResult,
    DetectionRequest,
    DetectionResult,
    RecordingInfo,
    normalize_biomarker,
    normalize_detector,
)
from src.agent_api.serialization import file_metadata, json_safe, utc_now_iso, write_json
from src.hfo_app import HFO_App
from src.param.param_classifier import ParamClassifier
from src.param.param_detector import (
    ParamHIL,
    ParamHFORMS,
    ParamHFOLineLength,
    ParamMNI,
    ParamDetector,
    ParamSpikeRMSLL,
    ParamSpindleA7,
    ParamSpindleLSM,
    ParamSpindleRMS,
    ParamSTE,
    ParamYASA,
)
from src.param.param_filter import ParamFilter, ParamFilterSpindle
from src.spike_app import SpikeApp
from src.spindle_app import SpindleApp
from src.utils.batch_project import BatchProject, create_batch_project, load_batch_project
from src.utils.utils_io import load_mne_raw


SUMMARY_EXPORT = "summary_json"


def _coerce_request(payload: DetectionRequest | dict[str, Any]) -> DetectionRequest:
    if isinstance(payload, DetectionRequest):
        payload.biomarker = normalize_biomarker(payload.biomarker)
        payload.detector = normalize_detector(payload.biomarker, payload.detector)
        return payload
    return DetectionRequest.from_dict(payload)


def _backend_for_biomarker(biomarker: str):
    canonical = normalize_biomarker(biomarker)
    if canonical == "HFO":
        return HFO_App()
    if canonical == "Spindle":
        return SpindleApp()
    if canonical == "Spike":
        return SpikeApp()
    raise PyHFOAgentError("invalid_biomarker", f"Unsupported biomarker: {biomarker}")


def _set_attrs(obj, overrides: dict[str, Any]):
    for key, value in overrides.items():
        if hasattr(obj, key):
            setattr(obj, key, value)
        else:
            raise PyHFOAgentError(
                "invalid_parameter",
                f"Unsupported parameter for {obj.__class__.__name__}: {key}",
            )
    return obj


def _filter_param_for(biomarker: str, sample_freq: float, payload: dict[str, Any]):
    canonical = normalize_biomarker(biomarker)
    defaults = {
        "HFO": {"fp": 80, "fs": 500},
        "Spindle": {"fp": 1, "fs": 30},
        "Spike": {"fp": 4, "fs": 80},
    }[canonical]
    values = {**defaults, **dict(payload or {})}
    cls = ParamFilterSpindle if canonical == "Spindle" else ParamFilter
    param = cls(
        fp=values.get("fp", values.get("pass_band", defaults["fp"])),
        fs=values.get("fs", values.get("stop_band", defaults["fs"])),
        rp=values.get("rp", 0.5),
        rs=values.get("rs", 93),
        space=values.get("space", 0.5),
        sample_freq=sample_freq,
    )
    return param


def _detector_param_for(
    biomarker: str,
    detector: str,
    sample_freq: float,
    n_jobs: int,
    payload: dict[str, Any],
) -> ParamDetector:
    canonical_biomarker = normalize_biomarker(biomarker)
    canonical_detector = normalize_detector(canonical_biomarker, detector)
    values = dict(payload or {})
    values.setdefault("sample_freq", sample_freq)
    values.setdefault("n_jobs", n_jobs)

    if canonical_biomarker == "HFO":
        if canonical_detector == "STE":
            detector_param = ParamSTE(sample_freq=sample_freq, n_jobs=n_jobs)
        elif canonical_detector == "MNI":
            detector_param = ParamMNI(sample_freq=sample_freq, n_jobs=n_jobs)
        elif canonical_detector == "HIL":
            detector_param = ParamHIL(sample_freq=sample_freq, n_jobs=n_jobs)
        elif canonical_detector in {"RMS", "HFO RMS"}:
            detector_param = ParamHFORMS(sample_freq=sample_freq, n_jobs=n_jobs)
        else:
            detector_param = ParamHFOLineLength(sample_freq=sample_freq, n_jobs=n_jobs)
    elif canonical_biomarker == "Spindle":
        if canonical_detector == "YASA":
            detector_param = ParamYASA(sample_freq=sample_freq, n_jobs=n_jobs)
        elif canonical_detector in {"LSM", "Kramer LSM"}:
            detector_param = ParamSpindleLSM(sample_freq=sample_freq, n_jobs=n_jobs)
        elif canonical_detector == "A7":
            detector_param = ParamSpindleA7(sample_freq=sample_freq, n_jobs=n_jobs)
        else:
            detector_param = ParamSpindleRMS(sample_freq=sample_freq, n_jobs=n_jobs)
    else:
        detector_param = ParamSpikeRMSLL(sample_freq=sample_freq, n_jobs=n_jobs)

    _set_attrs(detector_param, values)
    return ParamDetector(detector_param, detector_type=canonical_detector)


def _classifier_param(payload: dict[str, Any]) -> ParamClassifier:
    values = dict(payload or {})
    values.pop("enabled", None)
    defaults = ParamClassifier().to_dict()
    defaults.update(values)
    return ParamClassifier.from_dict(defaults)


def _recording_format(path: Path) -> str:
    if path.name.lower().endswith(".fif.gz"):
        return ".fif.gz"
    return path.suffix.lower()


def get_recording_info(input_path: str | Path) -> dict:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise PyHFOAgentError("file_not_found", f"Input file does not exist: {path}")
    raw = load_mne_raw(str(path))
    sample_freq = float(raw.info["sfreq"])
    n_times = int(getattr(raw, "n_times", 0) or 0)
    duration_seconds = float(n_times / sample_freq) if sample_freq else 0.0
    channel_names = [str(channel) for channel in raw.info.get("ch_names", [])]
    info = RecordingInfo(
        input_path=str(path),
        format=_recording_format(path),
        sample_freq=sample_freq,
        duration_seconds=duration_seconds,
        channel_names=channel_names,
        recording_channel_names=channel_names,
        line_freq=raw.info.get("line_freq"),
        highpass=raw.info.get("highpass"),
        lowpass=raw.info.get("lowpass"),
        n_channels=int(raw.info.get("nchan", len(channel_names))),
    )
    return json_safe(info.to_dict())


def _event_count(event_features) -> int:
    if event_features is None:
        return 0
    if hasattr(event_features, "get_num_biomarker"):
        return int(event_features.get_num_biomarker())
    starts = getattr(event_features, "starts", [])
    return int(len(starts))


def _event_channel_count(event_features) -> int:
    if event_features is None:
        return 0
    channel_names = getattr(event_features, "channel_names", [])
    if channel_names is None:
        return 0
    return int(len(np.unique(np.asarray(channel_names))))


def _export_requested(exports: list[str], name: str) -> bool:
    normalized = {item.lower().replace("-", "_") for item in exports}
    return name in normalized


def _write_summary(output_dir: Path, result: DetectionResult, request: DetectionRequest, app) -> Path:
    summary = result.to_dict()
    summary["request"] = request.to_dict()
    summary["input_file"] = file_metadata(request.resolved_input_path())
    summary["created_at"] = utc_now_iso()
    summary["recording_info"] = getattr(app, "edf_param", None) or {}
    return write_json(output_dir / "summary.json", summary)


def _run_classifier(app, biomarker: str, payload: dict[str, Any], warnings: list[str]):
    if not payload or not payload.get("enabled", False):
        return
    classifier_param = _classifier_param(payload)
    app.set_classifier(classifier_param)
    classifier_steps = payload.get("steps")
    if classifier_steps is None:
        classifier_steps = ["artifact", "spike", "ehfo"] if biomarker == "HFO" else ["artifact", "spike"]
    normalized_steps = {str(step).lower() for step in classifier_steps}
    if "artifact" in normalized_steps and hasattr(app, "classify_artifacts"):
        app.classify_artifacts()
    if "spike" in normalized_steps and hasattr(app, "classify_spikes"):
        app.classify_spikes()
    if "ehfo" in normalized_steps and hasattr(app, "classify_ehfos"):
        app.classify_ehfos()
    if normalized_steps - {"artifact", "spike", "ehfo"}:
        warnings.append(f"Ignored unknown classifier steps: {sorted(normalized_steps - {'artifact', 'spike', 'ehfo'})}")


def run_detection(payload: DetectionRequest | dict[str, Any]) -> dict:
    request = _coerce_request(payload)
    start_time = time.monotonic()
    input_path = request.resolved_input_path()
    output_dir = request.resolved_output_dir()
    artifacts: dict[str, str] = {}
    warnings: list[str] = []

    if not input_path.exists():
        raise PyHFOAgentError("file_not_found", f"Input file does not exist: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    app = _backend_for_biomarker(request.biomarker)
    try:
        app.set_n_jobs(request.n_jobs)
        app.load_edf(str(input_path))
        filter_param = _filter_param_for(request.biomarker, app.sample_freq, request.filter)
        app.set_filter_parameter(filter_param)
        if request.biomarker in {"HFO", "Spike"} or request.filter.get("force", False):
            app.filter_eeg_data()
        detector_param = _detector_param_for(
            request.biomarker,
            request.detector,
            app.sample_freq,
            request.n_jobs,
            request.detector_params,
        )
        app.set_detector(detector_param)
        app.detect_biomarker()
        _run_classifier(app, request.biomarker, request.classifier, warnings)

        if _export_requested(request.exports, "excel"):
            excel_path = output_dir / "events.xlsx"
            app.export_excel(excel_path)
            artifacts["excel"] = str(excel_path)
        if _export_requested(request.exports, "csv"):
            csv_path = output_dir / "events.csv"
            app.export_report(csv_path)
            artifacts["csv"] = str(csv_path)
        if _export_requested(request.exports, "session"):
            session_path = output_dir / "session.pybrain"
            app.export_app(session_path)
            artifacts["session"] = str(session_path)
        if _export_requested(request.exports, "clinical_summary") and hasattr(app, "export_clinical_summary"):
            clinical_path = output_dir / "clinical_summary.xlsx"
            app.export_clinical_summary(clinical_path)
            artifacts["clinical_summary"] = str(clinical_path)

        active_run_id = getattr(getattr(app, "analysis_session", None), "active_run_id", None)
        result = DetectionResult(
            status="complete",
            biomarker=request.biomarker,
            detector=request.detector,
            input_path=str(input_path),
            output_dir=str(output_dir),
            num_events=_event_count(app.event_features),
            num_channels=_event_channel_count(app.event_features),
            active_run_id=active_run_id,
            artifacts=artifacts,
            warnings=warnings,
            timing={"elapsed_seconds": round(time.monotonic() - start_time, 6)},
            run_summaries=app.get_run_summaries() if hasattr(app, "get_run_summaries") else [],
            decision_summary=app.get_decision_summary() if hasattr(app, "get_decision_summary") else {},
        )
        if _export_requested(request.exports, SUMMARY_EXPORT) or not request.exports:
            summary_path = _write_summary(output_dir, result, request, app)
            result.artifacts["summary_json"] = str(summary_path)
            _write_summary(output_dir, result, request, app)
        return json_safe(result.to_dict())
    except Exception as exc:
        error = error_response(exc)
        result = DetectionResult(
            status="failed",
            biomarker=request.biomarker,
            detector=request.detector,
            input_path=str(input_path),
            output_dir=str(output_dir),
            artifacts=artifacts,
            warnings=warnings,
            timing={"elapsed_seconds": round(time.monotonic() - start_time, 6)},
            error=error,
        )
        write_json(output_dir / "summary.json", result.to_dict())
        if request.fail_on_error:
            if isinstance(exc, PyHFOAgentError):
                raise exc
            raise PyHFOAgentError(error["code"], error["message"], details=error.get("details", {})) from exc
        return json_safe(result.to_dict())


def create_batch(input_dir: str | Path, output_dir: str | Path | None = None, *, recursive: bool = True) -> dict:
    project = create_batch_project(input_dir, output_dir, recursive=recursive)
    manifest_path = project.save()
    return {
        "manifest_path": str(manifest_path),
        "summary": project.progress_summary(),
        "cases": [case.to_dict() for case in project.cases],
    }


def _load_or_create_batch(payload: dict[str, Any]) -> tuple[BatchProject, Path]:
    manifest_path = payload.get("manifest_path") or payload.get("manifest")
    if manifest_path:
        manifest = Path(manifest_path).expanduser().resolve()
        return load_batch_project(manifest), manifest
    input_dir = payload.get("input_dir") or payload.get("input")
    if not input_dir:
        raise PyHFOAgentError("invalid_request", "manifest_path or input_dir is required for batch runs.")
    output_dir = payload.get("output_dir") or payload.get("output")
    project = create_batch_project(input_dir, output_dir, recursive=bool(payload.get("recursive", True)))
    manifest = project.save()
    return project, manifest


def run_batch(payload: dict[str, Any]) -> dict:
    if not isinstance(payload, dict):
        raise PyHFOAgentError("invalid_request", "Batch request must be an object.")
    project, manifest_path = _load_or_create_batch(payload)
    retry_failed = bool(payload.get("retry_failed", False))
    fail_fast = bool(payload.get("fail_fast", False))
    detection_config = dict(payload.get("detection") or {})
    errors: list[dict[str, Any]] = []

    while True:
        case = project.next_case(retry_failed=retry_failed)
        if case is None:
            break
        project.mark_running(case.case_id)
        project.save(manifest_path)
        case_request = {
            **detection_config,
            "input_path": case.input_path,
            "output_dir": case.output_dir,
            "fail_on_error": False,
        }
        try:
            result = run_detection(case_request)
            if result.get("status") == "complete":
                project.mark_complete(case.case_id, result)
            else:
                error = result.get("error") or {"code": "detection_failed", "message": "Detection failed."}
                project.mark_failed(case.case_id, error.get("message", "Detection failed."))
                errors.append({"case_id": case.case_id, **error})
                if fail_fast:
                    break
        except Exception as exc:
            error = error_response(exc)
            project.mark_failed(case.case_id, error["message"])
            errors.append({"case_id": case.case_id, **error})
            if fail_fast:
                break
        finally:
            project.save(manifest_path)

    result = BatchResult(
        status="complete" if not errors else "failed" if fail_fast else "complete_with_errors",
        manifest_path=str(manifest_path),
        summary=project.progress_summary(),
        cases=[case.to_dict() for case in project.cases],
        errors=errors,
    )
    return json_safe(result.to_dict())
