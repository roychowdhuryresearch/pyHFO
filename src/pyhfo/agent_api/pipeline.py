from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from pyhfo.agent_api.errors import PyHFOAgentError, error_response
from pyhfo.agent_api.schemas import (
    BatchResult,
    DetectionRequest,
    DetectionResult,
    RecordingInfo,
    SeverityScoringResult,
    normalize_biomarker,
    normalize_detector,
)
from pyhfo.agent_api.serialization import file_fingerprint, file_metadata, json_safe, utc_now_iso, write_json
from pyhfo.hfo_app import HFO_App
from pyhfo.paths import new_run_directory
from pyhfo.param.param_classifier import ParamClassifier
from pyhfo.param.param_detector import (
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
from pyhfo.param.param_filter import ParamFilter, ParamFilterSpindle
from pyhfo.spike_app import SpikeApp
from pyhfo.spindle_app import SpindleApp
from pyhfo.utils.batch_project import BatchProject, create_batch_project, load_batch_project
from pyhfo.utils.utils_io import get_edf_info, load_mne_raw, read_eeg_data
from pyhfo.utils.utils_montage import infer_eeg_type


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
    nyquist = float(sample_freq) / 2.0 if sample_freq else 0.0

    def _clamp_fs(target: float) -> float:
        # Keep the default upper edge below Nyquist so an empty/absent filter on a
        # common clinical rate (e.g. HFO default fs=500 on a 1000 Hz recording,
        # Nyquist 500) doesn't fail the >=Nyquist guard below.
        return round(min(target, 0.95 * nyquist), 1) if nyquist else target

    defaults = {
        "HFO": {"fp": 80, "fs": _clamp_fs(500.0)},
        "Spindle": {"fp": 1, "fs": _clamp_fs(30.0)},
        "Spike": {"fp": 4, "fs": _clamp_fs(80.0)},
    }[canonical]
    raw = dict(payload or {})
    cls = ParamFilterSpindle if canonical == "Spindle" else ParamFilter
    param = cls(
        # Read fp/fs (preferred) or the pass_band/stop_band aliases directly from the
        # raw payload. Resolving them from a defaults-merged dict made the alias branch
        # dead code (fp/fs are always present post-merge), silently dropping a caller's
        # pass_band/stop_band band in favor of the defaults.
        fp=raw.get("fp", raw.get("pass_band", defaults["fp"])),
        fs=raw.get("fs", raw.get("stop_band", defaults["fs"])),
        rp=raw.get("rp", 0.5),
        rs=raw.get("rs", 93),
        space=raw.get("space", 0.5),
        sample_freq=sample_freq,
    )
    fp = float(param.fp)
    fs = float(param.fs)
    if fp <= 0 or fs <= 0 or fs <= fp:
        raise PyHFOAgentError(
            "invalid_filter",
            f"Invalid {canonical} filter band: fp={fp}, fs={fs}. Expected 0 < fp < fs.",
            details={"fp": fp, "fs": fs, "sample_freq": sample_freq},
        )
    if nyquist and fs >= nyquist:
        raise PyHFOAgentError(
            "invalid_filter",
            f"Invalid {canonical} filter upper edge {fs} Hz for sampling rate {sample_freq} Hz; it must be below Nyquist ({nyquist} Hz).",
            details={"fp": fp, "fs": fs, "sample_freq": sample_freq, "nyquist_hz": nyquist},
        )
    return param


def _apply_channel_selection(app, request: DetectionRequest, warnings: list[str]) -> None:
    include = [str(channel) for channel in request.include_channels if str(channel)]
    exclude = {str(channel) for channel in request.exclude_channels if str(channel)}
    if not include and not exclude:
        return
    raw = getattr(app, "raw", None)
    if raw is None:
        warnings.append("Channel selection was requested but backend raw object is unavailable; detection used all loaded channels.")
        return
    available = [str(channel) for channel in raw.ch_names]
    available_set = set(available)
    missing_include = [channel for channel in include if channel not in available_set]
    missing_exclude = [channel for channel in exclude if channel not in available_set]
    if missing_include:
        warnings.append(f"Ignored missing include_channels: {missing_include}")
    if missing_exclude:
        warnings.append(f"Ignored missing exclude_channels: {missing_exclude}")
    selected = [channel for channel in (include or available) if channel in available_set and channel not in exclude]
    if not selected:
        raise PyHFOAgentError(
            "invalid_channel_selection",
            "Channel selection removed all channels; provide valid include_channels or fewer exclude_channels.",
            details={"include_channels": include, "exclude_channels": sorted(exclude), "available_channels": available},
        )
    if selected == available:
        return
    raw.pick_channels(selected, ordered=True)
    app.raw = raw
    app.edf_param = get_edf_info(raw)
    app.sample_freq = int(app.edf_param["sfreq"])
    app.eeg_data, app.channel_names = read_eeg_data(raw)
    app.recording_channel_names = np.array(app.channel_names).copy()
    app.eeg_data_un60 = app.eeg_data
    app.eeg_data_60 = None
    app.filter_data = None
    app.filtered = False
    warnings.append(f"Detection restricted to {len(selected)} selected channels.")


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
        eeg_type=infer_eeg_type(channel_names),
        recording_fingerprint=file_fingerprint(path)["sha256"],
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


# HFO detectors (STE/MNI) estimate detection thresholds over epochs of signal, so a recording shorter
# than ~1 minute commonly yields 0 events even when the filter band and channel selection are valid.
# Clinical HFO analysis uses at least ~1 min of interictal data.
MIN_DETECTION_DURATION_SECONDS = 60.0


def short_recording_warning(duration_seconds: float | None, num_events: int) -> str | None:
    """Return an explanation when a detector found 0 events because the recording is too short, else None.

    Returns None when events were found, the duration is unknown/invalid, or the recording is long
    enough, so callers can append the result unconditionally to a warnings list.
    """
    if num_events > 0:
        return None
    if not duration_seconds or duration_seconds <= 0:
        return None
    if duration_seconds >= MIN_DETECTION_DURATION_SECONDS:
        return None
    return (
        f"Recording is only {duration_seconds:.1f}s long; HFO detectors need a longer segment "
        f"(at least {int(MIN_DETECTION_DURATION_SECONDS)}s of interictal data is recommended) to "
        "estimate detection thresholds, so 0 events is expected. Load a longer recording."
    )


def _recording_duration_seconds(app) -> float | None:
    eeg_data = getattr(app, "eeg_data", None)
    sample_freq = getattr(app, "sample_freq", 0)
    if eeg_data is None or not sample_freq:
        return None
    try:
        return float(eeg_data.shape[1]) / float(sample_freq)
    except (IndexError, TypeError, ZeroDivisionError, ValueError):
        return None


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


def _classifier_metadata(payload: dict[str, Any], biomarker: str, classifier_steps: list[Any] | None = None, executed_steps: list[str] | None = None) -> dict[str, Any]:
    if not payload or not payload.get("enabled", False):
        return {"enabled": False}
    steps = classifier_steps if classifier_steps is not None else payload.get("steps")
    if steps is None:
        steps = ["artifact", "spike", "ehfo"] if biomarker == "HFO" else ["artifact", "spike"]
    return json_safe({
        "enabled": True,
        "biomarker": biomarker,
        "steps": [str(step).lower() for step in steps],
        "executed_steps": executed_steps or [],
        "device": payload.get("device"),
        "batch_size": payload.get("batch_size"),
        "model_type": payload.get("model_type"),
        "source_preference": payload.get("source_preference"),
        "artifact_path": payload.get("artifact_path"),
        "spike_path": payload.get("spike_path"),
        "ehfo_path": payload.get("ehfo_path"),
        "artifact_onnx": payload.get("artifact_onnx"),
        "spike_onnx": payload.get("spike_onnx"),
        "ehfo_onnx": payload.get("ehfo_onnx"),
        "artifact_card": payload.get("artifact_card"),
        "spike_card": payload.get("spike_card"),
        "ehfo_card": payload.get("ehfo_card"),
        "use_spike": payload.get("use_spike"),
        "use_ehfo": payload.get("use_ehfo"),
        "ignore_region": payload.get("ignore_region") or [payload.get("ignore_before", 1), payload.get("ignore_after", 1)],
        "artifact_threshold": payload.get("artifact_threshold", payload.get("threshold", 0.5)),
    })


_DEFAULT_CLASSIFIER_MODEL_TYPES = {"default_cpu", "default_gpu", "onnx_cpu"}
_EXPLICIT_CLASSIFIER_MODEL_KEYS = (
    "artifact_path", "spike_path", "ehfo_path",
    "artifact_onnx", "spike_onnx", "ehfo_onnx",
    "artifact_card", "spike_card", "ehfo_card",
)


def _classifier_preset(payload: dict[str, Any]) -> str | None:
    if any(payload.get(key) for key in _EXPLICIT_CLASSIFIER_MODEL_KEYS):
        return None
    preset = str(payload.get("model_type") or "").strip().lower() or "default_cpu"
    return preset if preset in _DEFAULT_CLASSIFIER_MODEL_TYPES else None


def _classifier_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Include the selected preset in the request used for hashing and export."""
    if payload.get("enabled") and (preset := _classifier_preset(payload)):
        return {**payload, "model_type": preset}
    return payload


def _load_default_classifier_if_requested(app, payload: dict[str, Any]) -> bool:
    """Load the selected preset and retain its resolved model sources for provenance."""
    model_type = _classifier_preset(payload)
    if model_type is None:
        return False
    loader = None
    if model_type == "onnx_cpu":
        loader = getattr(app, "set_default_onnx_classifier", None)
    elif model_type == "default_gpu":
        loader = getattr(app, "set_default_gpu_classifier", None)
    if loader is None:
        loader = getattr(app, "set_default_cpu_classifier", None)
    if loader is None:
        return False
    payload["model_type"] = model_type
    loader()
    param = getattr(app, "param_classifier", None)
    if param is not None:
        batch_size = payload.get("batch_size")
        if batch_size:
            try:
                param.batch_size = int(batch_size)
            except (TypeError, ValueError):
                pass
        # Surface the bundled defaults back into the payload so classifier
        # metadata reflects what actually ran instead of the null UI values.
        for attr in (
            "artifact_path", "spike_path", "ehfo_path",
            "artifact_onnx", "spike_onnx", "ehfo_onnx",
            "artifact_card", "spike_card", "ehfo_card",
        ):
            value = getattr(param, attr, None)
            if value and not payload.get(attr):
                payload[attr] = value
    return True


def _run_classifier(app, biomarker: str, payload: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    if not payload or not payload.get("enabled", False):
        return {"enabled": False}
    loaded_preset = _load_default_classifier_if_requested(app, payload)
    if not loaded_preset:
        classifier_param = _classifier_param(payload)
        app.set_classifier(classifier_param)
    classifier_steps = payload.get("steps")
    if classifier_steps is None:
        classifier_steps = ["artifact", "spike", "ehfo"] if biomarker == "HFO" else ["artifact", "spike"]
    normalized_steps = {str(step).lower() for step in classifier_steps}
    executed_steps: list[str] = []

    # Run each classify step in isolation. A single model whose preprocessing
    # config is out of range (e.g. an eHFO model whose selected_freq_range_hz
    # exceeds its freq_range_hz -> PreProcessing.check_bound) used to raise and
    # abort the whole classify run, dropping the other models' results with an
    # opaque error. Now a failing step is recorded as a warning and the rest
    # still run, so partial classification succeeds and the cause is visible.
    def _run_step(name: str, fn) -> None:
        try:
            fn()
            executed_steps.append(name)
        except Exception as exc:  # noqa: BLE001 - surface per-step, keep the run going
            warnings.append(f"Classifier step '{name}' skipped: {exc}")

    if "artifact" in normalized_steps and hasattr(app, "classify_artifacts"):
        ignore_region = payload.get("ignore_region") or [payload.get("ignore_before", 1), payload.get("ignore_after", 1)]
        threshold = payload.get("artifact_threshold", payload.get("threshold", 0.5))
        def _classify_artifact() -> None:
            try:
                app.classify_artifacts(ignore_region, threshold)
            except TypeError:
                app.classify_artifacts()
        _run_step("artifact", _classify_artifact)
    if "spike" in normalized_steps and hasattr(app, "classify_spikes"):
        _run_step("spike", app.classify_spikes)
    if "ehfo" in normalized_steps and hasattr(app, "classify_ehfos"):
        _run_step("ehfo", app.classify_ehfos)
    if normalized_steps - {"artifact", "spike", "ehfo"}:
        warnings.append(f"Ignored unknown classifier steps: {sorted(normalized_steps - {'artifact', 'spike', 'ehfo'})}")
    metadata = _classifier_metadata(payload, biomarker, list(classifier_steps), executed_steps)
    if loaded_preset:
        metadata["model_preset"] = payload["model_type"]
        params = getattr(app, "param_classifier", None)
        metadata["model_type"] = getattr(params, "model_type", None) or payload["model_type"]
    return metadata



def _import_app_from_session(biomarker: str, session_path: Path):
    canonical = normalize_biomarker(biomarker)
    if canonical == "HFO":
        return HFO_App.import_app(session_path)
    if canonical == "Spindle":
        return SpindleApp.import_app(session_path)
    if canonical == "Spike":
        return SpikeApp.import_app(session_path)
    raise PyHFOAgentError("invalid_biomarker", f"Unsupported biomarker: {biomarker}")


def classify_existing_run(payload: dict[str, Any]) -> dict:
    request = dict(payload or {})
    session_path = Path(str(request.get("session_path") or request.get("path") or "")).expanduser().resolve()
    if not session_path.exists():
        raise PyHFOAgentError("file_not_found", f"Session file does not exist: {session_path}")
    biomarker = normalize_biomarker(request.get("biomarker"))
    classifier = dict(request.get("classifier") or {})
    classifier.setdefault("enabled", True)
    classifier = _classifier_request(classifier)
    exports = [str(item) for item in request.get("exports") or ["csv", "session", SUMMARY_EXPORT]]
    output_dir = Path(request.get("output_dir") or new_run_directory("classified")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    artifacts: dict[str, str] = {}
    start_time = time.monotonic()

    app = _import_app_from_session(biomarker, session_path)
    run_id = request.get("run_id") or getattr(getattr(app, "analysis_session", None), "active_run_id", None)
    if run_id and hasattr(app, "activate_run"):
        app.activate_run(str(run_id))
    active_run_id = getattr(getattr(app, "analysis_session", None), "active_run_id", None) or str(run_id or "") or None
    classifier_metadata = _run_classifier(app, biomarker, classifier.copy(), warnings)

    if _export_requested(exports, "excel"):
        excel_path = output_dir / "events.xlsx"
        app.export_excel(excel_path)
        artifacts["excel"] = str(excel_path)
    if _export_requested(exports, "csv"):
        csv_path = output_dir / "events.csv"
        app.export_report(csv_path)
        artifacts["csv"] = str(csv_path)
    if _export_requested(exports, "session"):
        session_format = str(request.get("session_format") or request.get("sessionFormat") or session_path.suffix.lstrip(".") or "pybrain").lower()
        session_suffix = ".npz" if session_format == "npz" else ".pybrain"
        classified_session_path = output_dir / f"classified_session{session_suffix}"
        app.export_app(classified_session_path)
        artifacts["session"] = str(classified_session_path)
    if _export_requested(exports, "clinical_summary") and hasattr(app, "export_clinical_summary"):
        clinical_path = output_dir / "clinical_summary.xlsx"
        app.export_clinical_summary(clinical_path)
        artifacts["clinical_summary"] = str(clinical_path)

    result = DetectionResult(
        status="complete",
        biomarker=biomarker,
        detector=str(getattr(getattr(app, "param_detector", None), "detector_type", request.get("detector") or "active-run")),
        input_path=str(session_path),
        output_dir=str(output_dir),
        num_events=_event_count(getattr(app, "event_features", None)),
        num_channels=_event_channel_count(getattr(app, "event_features", None)),
        active_run_id=active_run_id,
        artifacts=artifacts,
        warnings=warnings,
        timing={"elapsed_seconds": round(time.monotonic() - start_time, 6)},
        run_summaries=app.get_run_summaries() if hasattr(app, "get_run_summaries") else [],
        decision_summary=app.get_decision_summary() if hasattr(app, "get_decision_summary") else {},
        classifier_metadata={**classifier_metadata, "source_run_id": str(run_id or active_run_id or ""), "classification_mode": "active_run_session"},
    )
    if _export_requested(exports, SUMMARY_EXPORT) or not exports:
        summary = result.to_dict()
        summary["request"] = json_safe({**request, "session_path": str(session_path), "classifier": classifier})
        summary["input_file"] = file_metadata(session_path)
        summary["created_at"] = utc_now_iso()
        summary["recording_info"] = getattr(app, "edf_param", None) or {}
        summary_path = write_json(output_dir / "summary.json", summary)
        result.artifacts["summary_json"] = str(summary_path)
    return json_safe(result.to_dict())

def run_detection(payload: DetectionRequest | dict[str, Any]) -> dict:
    request = _coerce_request(payload)
    request = replace(request, classifier=_classifier_request(request.classifier or {}))
    start_time = time.monotonic()
    input_path = request.resolved_input_path()
    output_dir = request.resolved_output_dir()
    artifacts: dict[str, str] = {}
    warnings: list[str] = []

    if not input_path.exists():
        raise PyHFOAgentError("file_not_found", f"Input file does not exist: {input_path}")
    recording_fingerprint = file_fingerprint(input_path)["sha256"]
    parameters_sha256 = _canonical_payload_sha256(request.to_dict())
    output_dir.mkdir(parents=True, exist_ok=True)

    app = _backend_for_biomarker(request.biomarker)
    try:
        app.set_n_jobs(request.n_jobs)
        app.load_edf(str(input_path))
        _apply_channel_selection(app, request, warnings)
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
        classifier_metadata = _run_classifier(app, request.biomarker, request.classifier.copy(), warnings)

        if _export_requested(request.exports, "excel"):
            excel_path = output_dir / "events.xlsx"
            app.export_excel(excel_path)
            artifacts["excel"] = str(excel_path)
        if _export_requested(request.exports, "csv"):
            csv_path = output_dir / "events.csv"
            app.export_report(csv_path)
            artifacts["csv"] = str(csv_path)
        if _export_requested(request.exports, "session"):
            session_suffix = ".npz" if request.session_format == "npz" else ".pybrain"
            session_path = output_dir / f"session{session_suffix}"
            app.export_app(session_path)
            artifacts["session"] = str(session_path)
        if _export_requested(request.exports, "clinical_summary") and hasattr(app, "export_clinical_summary"):
            clinical_path = output_dir / "clinical_summary.xlsx"
            app.export_clinical_summary(clinical_path)
            artifacts["clinical_summary"] = str(clinical_path)

        short_warning = short_recording_warning(_recording_duration_seconds(app), _event_count(app.event_features))
        if short_warning:
            warnings.append(short_warning)

        active_run_id = getattr(getattr(app, "analysis_session", None), "active_run_id", None)
        result = DetectionResult(
            status="complete",
            biomarker=request.biomarker,
            detector=request.detector,
            input_path=str(input_path),
            output_dir=str(output_dir),
            recording_fingerprint=recording_fingerprint,
            parameters_sha256=parameters_sha256,
            num_events=_event_count(app.event_features),
            num_channels=_event_channel_count(app.event_features),
            active_run_id=active_run_id,
            artifacts=artifacts,
            warnings=warnings,
            timing={"elapsed_seconds": round(time.monotonic() - start_time, 6)},
            run_summaries=app.get_run_summaries() if hasattr(app, "get_run_summaries") else [],
            decision_summary=app.get_decision_summary() if hasattr(app, "get_decision_summary") else {},
            classifier_metadata=classifier_metadata,
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
            recording_fingerprint=recording_fingerprint,
            parameters_sha256=parameters_sha256,
            artifacts=artifacts,
            warnings=warnings,
            timing={"elapsed_seconds": round(time.monotonic() - start_time, 6)},
            classifier_metadata=_classifier_metadata(request.classifier, request.biomarker),
            error=error,
        )
        write_json(output_dir / "summary.json", result.to_dict())
        if request.fail_on_error:
            if isinstance(exc, PyHFOAgentError):
                raise exc
            raise PyHFOAgentError(error["code"], error["message"], details=error.get("details", {})) from exc
        return json_safe(result.to_dict())


def _nan_to_none(value: float) -> float | None:
    return None if value is None or math.isnan(value) else float(value)


# Matches the PyQt app: a segment where any scalp channel swings past this is movement or
# electrode artifact, not severe EEG. Severity navigation and the headline mean/peak skip those,
# so a 3000 uV burst cannot present itself as the most severe part of a recording.
SEVERITY_ARTIFACT_THRESHOLD_UV = 1000.0
SEVERITY_SEGMENT_SECONDS = 15.0
SEVERITY_TUEG_19 = frozenset({
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ",
})


def _clean_severity_channel_name(value: str) -> str:
    cleaned = str(value).upper()
    for token in ("-REF", "EEG ", "POL ", " "):
        cleaned = cleaned.replace(token, "")
    return cleaned.strip()


def _canonical_payload_sha256(value: Any) -> str:
    encoded = json.dumps(json_safe(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _model_directory_sha256(model_dir: str | Path) -> str:
    """Digest the exact model bundle, including stable relative file names."""
    root = Path(model_dir).expanduser().resolve()
    files = sorted((path for path in root.rglob("*") if path.is_file()), key=lambda path: path.relative_to(root).as_posix())
    if not root.is_dir() or not files:
        raise PyHFOAgentError("model_not_found", f"Severity model directory is missing or empty: {root}")
    digest = hashlib.sha256()
    for file_path in files:
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(file_path.stat().st_size.to_bytes(8, "big"))
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _severity_artifact_flags(
    app,
    scores: list[dict[str, Any]],
) -> tuple[list[bool | None], dict[str, Any], list[str]]:
    """Assess per-segment artifacts without treating missing evidence as clean signal."""
    if not scores:
        return [], {
            "status": "not_applicable",
            "method": "tueg19_peak_amplitude",
            "threshold_uv": SEVERITY_ARTIFACT_THRESHOLD_UV,
            "assessed_segments": 0,
            "total_segments": 0,
            "channel_count": 0,
        }, []

    def unavailable(reason: str, *, channel_count: int = 0) -> tuple[list[bool | None], dict[str, Any], list[str]]:
        quality = {
            "status": "unavailable",
            "method": "tueg19_peak_amplitude",
            "threshold_uv": SEVERITY_ARTIFACT_THRESHOLD_UV,
            "assessed_segments": 0,
            "total_segments": len(scores),
            "channel_count": channel_count,
            "reason": reason,
        }
        warning = f"Artifact screening unavailable: {reason} Raw severity segments were not classified as clean."
        return [None] * len(scores), quality, [warning]

    channel_count = 0
    try:
        rows = [
            index
            for index, channel in enumerate(app.channel_names)
            if _clean_severity_channel_name(str(channel)) in SEVERITY_TUEG_19
        ]
        channel_count = len(rows)
        if not rows:
            return unavailable("no recording channels matched the TUEG-19 scalp layout")
        sample_freq = float(app.sample_freq)
        if not math.isfinite(sample_freq) or sample_freq <= 0:
            return unavailable("the recording sampling frequency is invalid", channel_count=len(rows))
        eeg = np.asarray(app.eeg_data)
        if eeg.ndim != 2 or max(rows) >= eeg.shape[0]:
            return unavailable("the loaded EEG array does not match the reported channel layout", channel_count=len(rows))

        flags: list[bool | None] = []
        for record in scores:
            start_sample = int(round(float(record["start_sec"]) * sample_freq))
            end_sample = int(round(float(record["end_sec"]) * sample_freq))
            if start_sample < 0 or end_sample <= start_sample or end_sample > eeg.shape[1]:
                flags.append(None)
                continue
            segment = eeg[rows, start_sample:end_sample]
            if segment.shape != (len(rows), end_sample - start_sample) or segment.size == 0:
                flags.append(None)
                continue
            if not np.isfinite(segment).all():
                flags.append(None)
                continue
            peak = float(np.abs(segment).max())
            flags.append(bool(peak >= SEVERITY_ARTIFACT_THRESHOLD_UV))

        assessed_segments = sum(flag is not None for flag in flags)
        if assessed_segments == 0:
            return unavailable("no score segment could be matched to readable EEG samples", channel_count=len(rows))
        status = "assessed" if assessed_segments == len(scores) else "partial"
        quality = {
            "status": status,
            "method": "tueg19_peak_amplitude",
            "threshold_uv": SEVERITY_ARTIFACT_THRESHOLD_UV,
            "assessed_segments": assessed_segments,
            "total_segments": len(scores),
            "channel_count": len(rows),
        }
        warnings = [] if status == "assessed" else [
            f"Artifact screening was unavailable for {len(scores) - assessed_segments} of {len(scores)} severity segments; "
            "clean severity summaries were withheld."
        ]
        return flags, quality, warnings
    except Exception as exc:
        return unavailable(
            f"EEG samples could not be read ({exc.__class__.__name__}: {exc})",
            channel_count=channel_count,
        )


def run_severity_scoring(payload: dict[str, Any]) -> dict:
    """Score a scalp EEG recording's severity with the BASED transformer model.

    payload:
      input_path: str (required)
      model_dir: str (optional; defaults to HFO_App.default_severity_model_dir())
      export_csv: str (optional path) -- if set, also writes the per-segment scores there
    """
    payload = dict(payload or {})
    start_time = time.monotonic()
    input_path = Path(str(payload.get("input_path") or "")).expanduser().resolve()
    warnings: list[str] = []

    if not input_path.exists():
        raise PyHFOAgentError("file_not_found", f"Input file does not exist: {input_path}")

    app = HFO_App()
    app.load_edf(str(input_path))

    if app.eeg_type != "scalp":
        raise PyHFOAgentError(
            "invalid_recording",
            "Severity scoring requires a conventional scalp EEG recording; "
            f"this recording was detected as '{app.eeg_type}'.",
        )

    eeg_shape = getattr(getattr(app, "eeg_data", None), "shape", ())
    sample_freq = float(getattr(app, "sample_freq", 0) or 0)
    duration_seconds = float(eeg_shape[1] / sample_freq) if len(eeg_shape) == 2 and sample_freq > 0 else None
    if duration_seconds is not None and duration_seconds < SEVERITY_SEGMENT_SECONDS:
        raise PyHFOAgentError(
            "no_data",
            f"Severity scoring requires at least one complete {SEVERITY_SEGMENT_SECONDS:g}-second segment; "
            f"the loaded recording is {duration_seconds:.3f} seconds.",
            details={
                "duration_seconds": round(duration_seconds, 6),
                "minimum_segment_seconds": SEVERITY_SEGMENT_SECONDS,
                "num_segments": 0,
            },
        )

    from pyhfo.utils.torch_models import severity_model_dir

    model_dir = severity_model_dir(payload["model_dir"]) if payload.get("model_dir") else app.default_severity_model_dir()
    try:
        from pyhfo.severity_app import SeverityScorer
    except ImportError as exc:
        raise PyHFOAgentError("missing_dependency", "Install pyhfo[severity] to score scalp EEG.") from exc
    model_sha256 = _model_directory_sha256(model_dir)
    effective_parameters = {
        "artifact_threshold_uv": SEVERITY_ARTIFACT_THRESHOLD_UV,
        "model_sha256": model_sha256,
    }
    scorer = SeverityScorer(model_dir)
    try:
        severity_result = app.run_severity_scoring(scorer)
    except ValueError as exc:
        # e.g. zero channels overlap the TUEG-19 scalp layout -- a bad-input condition,
        # not a server error.
        raise PyHFOAgentError("invalid_recording", str(exc)) from exc

    scores = severity_result.scores_df.to_dict("records") if len(severity_result.scores_df) else []
    if not scores:
        raise PyHFOAgentError(
            "no_data",
            f"Severity scoring produced no complete {SEVERITY_SEGMENT_SECONDS:g}-second segments.",
            details={"minimum_segment_seconds": SEVERITY_SEGMENT_SECONDS, "num_segments": 0},
        )

    artifact_flags, artifact_quality, artifact_warnings = _severity_artifact_flags(app, scores)
    warnings.extend(artifact_warnings)
    for record, artifactual in zip(scores, artifact_flags):
        record["artifact"] = artifactual
    # A clean headline is valid only when every scored segment had readable artifact evidence.
    # Partial/unavailable screening remains visible in scores but cannot promote raw segments to clean.
    clean = [record for record in scores if record.get("artifact") is False] if artifact_quality["status"] == "assessed" else []
    clean_peak = max(clean, key=lambda record: float(record["score"])) if clean else None

    artifacts: dict[str, str] = {}
    export_csv = payload.get("export_csv")
    if export_csv:
        app.export_severity_csv(str(export_csv))
        artifacts["csv"] = str(export_csv)

    result = SeverityScoringResult(
        status="complete",
        input_path=str(input_path),
        eeg_type=app.eeg_type,
        recording_fingerprint=file_fingerprint(input_path)["sha256"],
        model_name="BASED",
        model_version=f"based-severity-{model_sha256[:12]}",
        model_sha256=model_sha256,
        parameters_sha256=_canonical_payload_sha256(effective_parameters),
        num_segments=len(scores),
        mean_score=_nan_to_none(severity_result.mean_score),
        peak_score=_nan_to_none(severity_result.peak_score),
        peak_time_sec=_nan_to_none(severity_result.peak_time_sec),
        num_clean_segments=len(clean),
        artifact_threshold_uv=SEVERITY_ARTIFACT_THRESHOLD_UV,
        artifact_quality=artifact_quality,
        clean_mean_score=_nan_to_none(sum(float(r["score"]) for r in clean) / len(clean)) if clean else None,
        clean_peak_score=_nan_to_none(float(clean_peak["score"])) if clean_peak else None,
        clean_peak_time_sec=_nan_to_none(float(clean_peak["start_sec"])) if clean_peak else None,
        scores=scores,
        artifacts=artifacts,
        warnings=warnings,
        timing={"elapsed_seconds": round(time.monotonic() - start_time, 6)},
    )
    return json_safe(result.to_dict())


def create_batch(input_dir: str | Path, output_dir: str | Path | None = None, *, recursive: bool = True) -> dict:
    project = create_batch_project(input_dir, output_dir, recursive=recursive)
    manifest_path = project.save()
    manifest_fingerprint = file_fingerprint(manifest_path)
    return {
        "status": "complete",
        "manifest_path": str(manifest_path),
        "artifact_exists": True,
        "artifact_sha256": manifest_fingerprint["sha256"],
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
    manifest_fingerprint = file_fingerprint(manifest_path)
    return json_safe({
        **result.to_dict(),
        "artifact_exists": True,
        "artifact_sha256": manifest_fingerprint["sha256"],
    })
