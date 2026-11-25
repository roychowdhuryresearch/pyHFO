from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from pyhfo.agent_api.errors import PyHFOAgentError
from pyhfo.agent_api.schemas import DETECTORS_BY_BIOMARKER, normalize_biomarker, normalize_detector
from pyhfo.agent_api.serialization import json_safe, utc_now_iso
from pyhfo.param.param_detector import (
    ParamHIL,
    ParamHFORMS,
    ParamHFOLineLength,
    ParamMNI,
    ParamSpikeRMSLL,
    ParamSpindleA7,
    ParamSpindleLSM,
    ParamSpindleRMS,
    ParamSTE,
    ParamYASA,
)

GOAL_PROFILES: dict[str, dict[str, Any]] = {
    "qc": {"version": "2026.06", "n_jobs": 1, "duration_limit_seconds": 60, "description": "Quick quality-control planning without broad detection."},
    "exploratory": {"version": "2026.06", "n_jobs": 1, "duration_limit_seconds": 120, "description": "Conservative exploratory smoke run."},
    "specificity": {"version": "2026.06", "n_jobs": 2, "duration_limit_seconds": None, "description": "Favor stricter candidate output for review."},
    "comparison": {"version": "2026.06", "n_jobs": 2, "duration_limit_seconds": None, "description": "Comparable parameter plan across runs."},
    "batch": {"version": "2026.06", "n_jobs": 1, "duration_limit_seconds": None, "description": "Stable batch-safe defaults."},
    "reproduction": {"version": "2026.06", "n_jobs": 1, "duration_limit_seconds": None, "description": "Preserve explicit prior settings for reproducibility."},
    "smoke": {"version": "2026.06", "n_jobs": 1, "duration_limit_seconds": 30, "description": "Minimal smoke-test run."},
    "full": {"version": "2026.06", "n_jobs": 4, "duration_limit_seconds": None, "description": "Full local analysis plan."},
}
GOAL_ALIASES = {"smoke-test": "smoke", "full-run": "full", "specificity-focused": "specificity"}

DEFAULT_SAMPLE_FREQ_BY_DETECTOR: dict[str, float] = {
    "STE": 2000.0,
    "MNI": 200.0,
    "HIL": 2000.0,
    "RMS": 2000.0,
    "HFO RMS": 2000.0,
    "LineLength": 2000.0,
    "LL": 2000.0,
    "HFO LL": 2000.0,
    "YASA": 2000.0,
    "LSM": 2000.0,
    "Kramer LSM": 2000.0,
    "A7": 2000.0,
    "MOLLE": 2000.0,
    "FASST": 2000.0,
    "Spindle RMS": 2000.0,
    "RMS/LL": 2000.0,
}

LSM_PRESETS: tuple[tuple[str, str], ...] = (
    ("AllAges (Kwon 2023)", "all_ages_kwon_2023.json"),
    ("Original (Kramer 2021)", "original_kramer_2021.json"),
    ("ESES (McLaren 2023)", "eses_mclaren_2023.json"),
    ("Infant (Berja 2024)", "infant_berja_2024.json"),
)


def _lsm_preset_paths() -> dict[str, str]:
    preset_dir = files("pyhfo").joinpath("resources", "spindle_lsm")
    return {label: str(preset_dir / filename) for label, filename in LSM_PRESETS if (preset_dir / filename).is_file()}


def _lsm_preset_ui_values(preset_path: str) -> dict[str, str]:
    if not preset_path:
        return {}
    try:
        payload = json.loads(Path(preset_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    params = payload.get("params", {}) or {}
    mu = payload.get("mu", {}) or {}
    sigma = payload.get("sigma", {}) or {}
    transition_matrix = payload.get("transition_matrix", [[0.5, 0.5], [0.5, 0.5]]) or [[0.5, 0.5], [0.5, 0.5]]
    index_base = int(payload.get("index_base", 0) or 0)

    def display_indexes(values: Any) -> str:
        if not isinstance(values, (list, tuple)):
            values = [values]
        indexes = [int(value) for value in values]
        if index_base == 0:
            indexes = [value + 1 for value in indexes]
        return ",".join(str(value) for value in indexes)

    return {
        "lsm_window_duration": _as_text(params.get("window_duration", 0.5)),
        "lsm_step_duration": _as_text(params.get("step_duration", 0.1)),
        "lsm_theta_index": display_indexes(params.get("theta_index", [7])),
        "lsm_nine_15_index": display_indexes(params.get("nine_15_index", [12, 14])),
        "lsm_mu_log_p_theta1": _as_text(mu.get("log_P_theta1")),
        "lsm_mu_log_p_theta0": _as_text(mu.get("log_P_theta0")),
        "lsm_mu_log_p_9_15_1": _as_text(mu.get("log_P_9_15_1")),
        "lsm_mu_log_p_9_15_0": _as_text(mu.get("log_P_9_15_0")),
        "lsm_mu_f1": _as_text(mu.get("F1")),
        "lsm_mu_f0": _as_text(mu.get("F0")),
        "lsm_sigma_log_p_theta1": _as_text(sigma.get("log_P_theta1")),
        "lsm_sigma_log_p_theta0": _as_text(sigma.get("log_P_theta0")),
        "lsm_sigma_log_p_9_15_1": _as_text(sigma.get("log_P_9_15_1")),
        "lsm_sigma_log_p_9_15_0": _as_text(sigma.get("log_P_9_15_0")),
        "lsm_sigma_f1": _as_text(sigma.get("F1")),
        "lsm_sigma_f0": _as_text(sigma.get("F0")),
        "lsm_transition_00": _as_text(transition_matrix[0][0]),
        "lsm_transition_01": _as_text(transition_matrix[0][1]),
        "lsm_transition_10": _as_text(transition_matrix[1][0]),
        "lsm_transition_11": _as_text(transition_matrix[1][1]),
    }


def _sample_freq_value(payload: dict[str, Any], info: dict[str, Any]) -> Any:
    for key in ("sample_freq", "sfreq", "sampling_frequency", "sample_frequency_hz"):
        value = payload.get(key)
        if value not in (None, ""):
            return value
    for key in ("sample_freq", "sfreq", "sampling_frequency", "sample_frequency_hz"):
        value = info.get(key)
        if value not in (None, ""):
            return value
    return 0.0


def _detector_sample_freq(payload: dict[str, Any], detector: str) -> float:
    info = _recording_info(payload)
    try:
        sfreq = float(_sample_freq_value(payload, info))
    except (TypeError, ValueError) as exc:
        raise PyHFOAgentError("invalid_request", "sample frequency must be numeric.") from exc
    if sfreq > 0:
        return sfreq
    return DEFAULT_SAMPLE_FREQ_BY_DETECTOR.get(detector, 2000.0)


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    raw = payload.get(key)
    if raw in (None, ""):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise PyHFOAgentError("invalid_request", f"{key} must be an integer.") from exc
    if value < 1:
        raise PyHFOAgentError("invalid_request", f"{key} must be positive.")
    return value


def _detector_default_param(biomarker: str, detector: str, sample_freq: float, n_jobs: int | None):
    canonical_biomarker = normalize_biomarker(biomarker)
    canonical_detector = normalize_detector(canonical_biomarker, detector)
    common = {"sample_freq": sample_freq}
    if n_jobs is not None:
        common["n_jobs"] = n_jobs
    if canonical_biomarker == "HFO":
        if canonical_detector == "STE":
            return ParamSTE(**common)
        if canonical_detector == "MNI":
            return ParamMNI(**common)
        if canonical_detector == "HIL":
            return ParamHIL(**common)
        if canonical_detector in {"RMS", "HFO RMS"}:
            return ParamHFORMS(**common)
        return ParamHFOLineLength(**common)
    if canonical_biomarker == "Spindle":
        if canonical_detector == "YASA":
            return ParamYASA(**common)
        if canonical_detector in {"LSM", "Kramer LSM"}:
            return ParamSpindleLSM(**common)
        if canonical_detector == "A7":
            return ParamSpindleA7(**common)
        return ParamSpindleRMS(**common)
    return ParamSpikeRMSLL(**common)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _pair_items(value: Any) -> tuple[Any, Any]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return value[0], value[1]
    return "", ""


def _detector_ui_values(detector: str, params: dict[str, Any], request: dict[str, Any] | None = None) -> dict[str, str]:
    canonical = detector
    if canonical == "STE":
        return {
            "rms_window": _as_text(params.get("rms_window")),
            "min_window": _as_text(params.get("min_window")),
            "min_gap": _as_text(params.get("min_gap")),
            "min_oscillation": _as_text(params.get("min_osc")),
            "peak_threshold": _as_text(params.get("peak_thres")),
            "rms_threshold": _as_text(params.get("rms_thres")),
            "epoch_length": _as_text(params.get("epoch_len")),
        }
    if canonical == "MNI":
        return {
            "epoch_time": _as_text(params.get("epoch_time")),
            "min_window": _as_text(params.get("min_win")),
            "chf_percentage": _as_text(params.get("per_CHF")),
            "threshold_percentage": _as_text(float(params.get("thrd_perc")) * 100 if params.get("thrd_perc") is not None else ""),
            "min_gap_time": _as_text(params.get("min_gap")),
            "epoch_chf": _as_text(params.get("epo_CHF")),
            "baseline_window": _as_text(params.get("base_seg")),
            "baseline_shift": _as_text(params.get("base_shift")),
            "baseline_threshold": _as_text(params.get("base_thrd")),
            "baseline_min_time": _as_text(params.get("base_min")),
        }
    if canonical == "HIL":
        return {
            "pass_band": _as_text(params.get("pass_band")),
            "stop_band": _as_text(params.get("stop_band")),
            "sd_threshold": _as_text(params.get("sd_threshold")),
            "min_window": _as_text(params.get("min_window")),
            "sample_freq": _as_text(params.get("sample_freq")),
            "epoch_time": _as_text(params.get("epoch_time")),
        }
    if canonical in {"RMS", "HFO RMS"}:
        return {
            "window": _as_text(params.get("rms_window")),
            "min_duration": _as_text(params.get("min_window")),
            "max_duration": _as_text(params.get("max_window")),
            "min_gap": _as_text(params.get("min_gap")),
            "threshold": _as_text(params.get("threshold")),
            "peak_threshold": _as_text(params.get("peak_threshold")),
        }
    if canonical in {"LineLength", "LL", "HFO LL"}:
        return {
            "window": _as_text(params.get("ll_window")),
            "min_duration": _as_text(params.get("min_window")),
            "max_duration": _as_text(params.get("max_window")),
            "min_gap": _as_text(params.get("min_gap")),
            "threshold": _as_text(params.get("threshold")),
            "peak_threshold": _as_text(params.get("peak_threshold")),
        }
    if canonical == "YASA":
        freq_sp_low, freq_sp_high = _pair_items(params.get("freq_sp"))
        freq_broad_low, freq_broad_high = _pair_items(params.get("freq_broad"))
        duration_low, duration_high = _pair_items(params.get("duration"))
        return {
            "freq_sp_low": _as_text(freq_sp_low),
            "freq_sp_high": _as_text(freq_sp_high),
            "freq_broad_low": _as_text(freq_broad_low),
            "freq_broad_high": _as_text(freq_broad_high),
            "duration_low": _as_text(duration_low),
            "duration_high": _as_text(duration_high),
            "min_distance": _as_text(params.get("min_distance")),
            "rel_power": _as_text(params.get("rel_pow")),
            "correlation": _as_text(params.get("corr")),
            "rms": _as_text(params.get("rms")),
        }
    if canonical in {"LSM", "Kramer LSM"}:
        preset_paths = _lsm_preset_paths()
        requested_preset = str((request or {}).get("preset") or "").strip()
        default_preset = requested_preset if requested_preset in preset_paths else next(iter(preset_paths), "Custom file / manual")
        preset_path = preset_paths.get(default_preset, "")
        return {
            "preset": default_preset,
            "preset_path": preset_path,
            "preset_paths_json": json_safe(preset_paths),
            "parameter_file": _as_text(params.get("parameter_file")),
            "probability": _as_text(params.get("prob_threshold")),
            "min_duration": _as_text(params.get("min_spindle_duration")),
            "separation": _as_text(params.get("spindle_separation_threshold")),
            "min_prominence": _as_text(params.get("min_peak_prominence")),
            "start_frequency": _as_text(params.get("start_frequency")),
            "stop_frequency": _as_text(params.get("stop_frequency")),
            "lsm_model_parameters_enabled": "false",
            **_lsm_preset_ui_values(preset_path),
        }
    if canonical == "A7":
        freq_low, freq_high = _pair_items(params.get("freq_sp"))
        broad_low, broad_high = _pair_items(params.get("freq_broad"))
        duration_low, duration_high = _pair_items(params.get("duration"))
        return {
            "freq_low": _as_text(freq_low),
            "freq_high": _as_text(freq_high),
            "duration_low": _as_text(duration_low),
            "duration_high": _as_text(duration_high),
            "min_distance": _as_text(params.get("min_distance")),
            "smooth_window": _as_text(params.get("smooth_window")),
            "rms_threshold": _as_text(params.get("rms_threshold")),
            "broad_low": _as_text(broad_low),
            "broad_high": _as_text(broad_high),
            "relative_power_threshold": _as_text(params.get("relative_power_threshold")),
            "correlation_threshold": _as_text(params.get("correlation_threshold")),
        }
    if canonical in {"MOLLE", "FASST", "Spindle RMS"}:
        freq_low, freq_high = _pair_items(params.get("freq_sp"))
        duration_low, duration_high = _pair_items(params.get("duration"))
        return {
            "freq_low": _as_text(freq_low),
            "freq_high": _as_text(freq_high),
            "duration_low": _as_text(duration_low),
            "duration_high": _as_text(duration_high),
            "min_distance": _as_text(params.get("min_distance")),
            "smooth_window": _as_text(params.get("smooth_window")),
            "rms_threshold": _as_text(params.get("rms_threshold")),
        }
    return {
        "rms_window": _as_text(params.get("rms_window")),
        "ll_window": _as_text(params.get("ll_window")),
        "min_duration": _as_text(params.get("min_window")),
        "max_duration": _as_text(params.get("max_window")),
        "min_gap": _as_text(params.get("min_gap")),
        "rms_threshold": _as_text(params.get("rms_thres")),
        "ll_threshold": _as_text(params.get("ll_thres")),
        "peak_threshold": _as_text(params.get("peak_thres")),
    }


def _goal(value: str | None) -> str:
    raw = (value or "smoke").strip().lower().replace("_", "-")
    goal = GOAL_ALIASES.get(raw, raw)
    if goal not in GOAL_PROFILES:
        raise PyHFOAgentError("invalid_goal_profile", f"Unsupported goal profile: {value}", details={"supported": sorted(GOAL_PROFILES)})
    return goal


def _recording_info(payload: dict[str, Any]) -> dict[str, Any]:
    info = payload.get("recording_info") or {}
    if not isinstance(info, dict):
        raise PyHFOAgentError("invalid_request", "recording_info must be an object.")
    return info


def _sample_freq(payload: dict[str, Any]) -> float:
    info = _recording_info(payload)
    try:
        sfreq = float(_sample_freq_value(payload, info))
    except (TypeError, ValueError) as exc:
        raise PyHFOAgentError("invalid_request", "sample frequency must be numeric.") from exc
    if sfreq <= 0:
        raise PyHFOAgentError("invalid_recording", "sample frequency is required for parameter planning.")
    return sfreq


def _default_filter(biomarker: str, sfreq: float, goal: str) -> dict[str, float]:
    nyquist = sfreq / 2.0
    if biomarker == "HFO":
        upper = min(500.0, round(nyquist * 0.95, 1))
        lower = 80.0
        if upper <= lower:
            raise PyHFOAgentError("invalid_filter", "Sampling rate is too low for an HFO band above 80 Hz.", details={"sample_freq": sfreq, "nyquist_hz": nyquist})
        if goal == "specificity":
            lower = min(120.0, max(80.0, upper * 0.5))
        return {"fp": float(lower), "fs": float(upper)}
    if biomarker == "Spike":
        return {"fp": 4.0, "fs": min(80.0, round(nyquist * 0.8, 1))}
    return {"fp": 1.0, "fs": min(30.0, round(nyquist * 0.5, 1))}


def validate_parameters(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise PyHFOAgentError("invalid_request", "Parameter payload must be an object.")
    sfreq = _sample_freq(payload)
    biomarker = normalize_biomarker(payload.get("biomarker"))
    detector = normalize_detector(biomarker, payload.get("detector"))
    filt = dict(payload.get("filter") or _default_filter(biomarker, sfreq, _goal(payload.get("goal_profile"))))
    try:
        fp_raw = filt.get("fp")
        fs_raw = filt.get("fs")
        if fp_raw is None or fs_raw is None:
            raise ValueError("missing filter edge")
        fp = float(fp_raw)
        fs = float(fs_raw)
    except (TypeError, ValueError) as exc:
        raise PyHFOAgentError("invalid_filter", "filter.fp and filter.fs must be numeric.") from exc
    nyquist = sfreq / 2.0
    errors = []
    warnings = []
    if fp <= 0 or fs <= fp:
        errors.append({"code": "invalid_filter_band", "message": "Expected 0 < fp < fs."})
    if fs >= nyquist:
        errors.append({"code": "nyquist_violation", "message": f"Filter upper edge {fs} Hz must be below Nyquist {nyquist} Hz."})
    include = payload.get("include_channels") or []
    exclude = payload.get("exclude_channels") or []
    if isinstance(include, str):
        include = [include]
    if isinstance(exclude, str):
        exclude = [exclude]
    if include is not None and not isinstance(include, list):
        errors.append({"code": "invalid_channel_selection", "message": "include_channels must be a list."})
    if include == [] and payload.get("require_channels"):
        errors.append({"code": "empty_channel_selection", "message": "At least one included channel is required."})
    status = "invalid" if errors else "valid"
    if biomarker == "HFO" and sfreq < 250:
        warnings.append("Low sampling rate limits HFO interpretation even when filters are Nyquist-safe.")
    result = {"status": status, "valid": not errors, "errors": errors, "warnings": warnings, "biomarker": biomarker, "detector": detector, "filter": {"fp": fp, "fs": fs}, "nyquist_hz": nyquist}
    if errors:
        return json_safe(result)
    return json_safe(result)


def detector_defaults(payload: dict[str, Any]) -> dict[str, Any]:
    """Return PyQt/PyHFO detector parameter-object defaults in renderer UI shape."""
    if not isinstance(payload, dict):
        raise PyHFOAgentError("invalid_request", "Detector defaults request must be an object.")
    biomarker = normalize_biomarker(payload.get("biomarker"))
    detector = normalize_detector(biomarker, payload.get("detector"))
    sample_freq = _detector_sample_freq(payload, detector)
    n_jobs = _optional_int(payload, "n_jobs")
    param = _detector_default_param(biomarker, detector, sample_freq, n_jobs)
    raw_defaults = dict(param.to_dict())
    ui_values = _detector_ui_values(detector, raw_defaults, payload)
    return json_safe({
        "status": "complete",
        "source": "pyhfo_param_detector_defaults",
        "biomarker": biomarker,
        "detector": detector,
        "sample_freq": raw_defaults.get("sample_freq", sample_freq),
        "n_jobs": raw_defaults.get("n_jobs"),
        "detector_params": raw_defaults,
        "ui_values": ui_values,
        "rationale": [
            "Values are instantiated from pyhfo.param.param_detector classes, matching the PyQt parameter objects.",
            "ui_values maps backend parameter names to the Electron/PyQt form field names.",
        ],
    })


def recommend_parameters(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise PyHFOAgentError("invalid_request", "Parameter request must be an object.")
    goal = _goal(payload.get("goal_profile") or payload.get("goal"))
    profile = GOAL_PROFILES[goal]
    sfreq = _sample_freq(payload)
    biomarker = normalize_biomarker(payload.get("biomarker"))
    detector = normalize_detector(biomarker, payload.get("detector"))
    previous = dict(payload.get("previous_parameters") or {})
    recommended = {
        "biomarker": biomarker,
        "detector": detector,
        "filter": _default_filter(biomarker, sfreq, goal),
        "n_jobs": int(payload.get("n_jobs") or profile["n_jobs"]),
        "goal_profile": {"name": goal, **profile},
        "include_channels": payload.get("include_channels") or (payload.get("roi") or {}).get("brain_signal_channels") or [],
        "exclude_channels": payload.get("exclude_channels") or (payload.get("roi") or {}).get("default_detection_exclusions") or [],
    }
    validation = validate_parameters({**payload, **recommended, "sample_freq": sfreq})
    diff = []
    for key in ["biomarker", "detector", "filter", "n_jobs", "include_channels", "exclude_channels"]:
        before = previous.get(key)
        after = recommended.get(key)
        if before != after:
            diff.append({"field": key, "before": before, "after": after, "state": "changed" if before is not None else "added"})
    card = {"type": "parameter_plan", "title": f"{biomarker} {detector} parameter plan", "goal_profile": goal, "validation_status": validation["status"], "diff": diff, "warnings": validation.get("warnings", []), "requires_confirmation": True}
    plan = {"status": "complete", "created_at": utc_now_iso(), "parameters": recommended, "validation": validation, "diff": diff, "rationale": [profile["description"], "Filter edges are constrained below Nyquist.", "Clinical/BIDS labels are not used for ordinary parameter planning."], "ui_card": card}
    return json_safe(plan)


def sweep_parameters(payload: dict[str, Any]) -> dict[str, Any]:
    base = recommend_parameters(payload)
    params = base["parameters"]
    filt = params["filter"]
    span = float(filt["fs"]) - float(filt["fp"])
    candidates = []
    for idx, factor in enumerate([0.9, 1.0, 1.1]):
        fp = max(1.0, round(float(filt["fp"]) * factor, 1))
        fs = round(fp + span, 1)
        candidate = {**params, "filter": {"fp": fp, "fs": fs}, "sweep_index": idx}
        validation = validate_parameters({**payload, **candidate})
        if validation["valid"]:
            candidates.append({"parameters": candidate, "validation": validation})
    return json_safe({"status": "complete", "base_plan": base, "candidates": candidates, "rationale": "Bounded local sweep around the recommended pass band; unsafe candidates are omitted."})
