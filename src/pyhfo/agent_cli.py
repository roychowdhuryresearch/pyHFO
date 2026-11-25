from __future__ import annotations

import argparse
import contextlib
import json
import sys
from typing import Any

from pyhfo import __version__
from pyhfo.agent_api import (
    PyHFOAgentError,
    classify_existing_run,
    create_batch,
    detector_defaults,
    get_capability_manifest,
    get_recording_info,
    infer_brain_signal_roi,
    compute_channel_qc,
    compute_spectral_profile,
    detect_line_noise,
    rank_candidate_channels,
    recommend_parameters,
    recommend_detector_plan,
    prepare_recording,
    load_sidecar_session,
    run_batch,
    run_detection,
    run_severity_scoring,
    save_sidecar_session,
    sweep_parameters,
    validate_parameters,
)
from pyhfo.agent_api.errors import error_response
from pyhfo.agent_api.serialization import json_safe


_DETECTION_CLASSIFIER_HELP = (
    "Use --classifier-param enabled=true to classify detected events. With no model_type or model sources, "
    "PyHFO uses default_cpu (ONNX) and downloads the pinned models on first use."
)


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        stripped = value.strip()
        # JSON arrays/objects (e.g. a list-valued detector param "[11, 16]") arrive as
        # strings; parse them back so float(v) over the elements works downstream.
        if stripped[:1] in "[{":
            try:
                return json.loads(stripped)
            except Exception:
                pass
        return value


def _parse_key_values(items: list[str] | None, *, list_keys: set[str] | None = None) -> dict[str, Any]:
    values: dict[str, Any] = {}
    list_keys = list_keys or set()
    for item in items or []:
        if "=" not in item:
            raise PyHFOAgentError("invalid_argument", f"Expected KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise PyHFOAgentError("invalid_argument", f"Empty key in argument: {item}")
        parsed = _parse_scalar(value.strip())
        if key in list_keys:
            values.setdefault(key, []).append(parsed)
        else:
            values[key] = parsed
    return values


def _parse_json_object(raw: str | None, field_name: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PyHFOAgentError("invalid_argument", f"{field_name} must be valid JSON.") from exc
    if not isinstance(payload, dict):
        raise PyHFOAgentError("invalid_argument", f"{field_name} must be a JSON object.")
    return payload


def _print_json(payload: dict):
    print(json.dumps(json_safe(payload), indent=2, sort_keys=True))


def _add_detection_args(parser: argparse.ArgumentParser):
    parser.add_argument("--biomarker", default="HFO", help="Biomarker type: HFO, Spindle, or Spike.")
    parser.add_argument("--detector", default=None, help="Detector name, e.g. STE, MNI, YASA, RMS/LL.")
    parser.add_argument("--n-jobs", type=int, default=4, help="Worker count for supported backend operations.")
    parser.add_argument("--filter", action="append", default=[], metavar="KEY=VALUE", help="Filter parameter override.")
    parser.add_argument(
        "--detector-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Detector parameter override.",
    )
    parser.add_argument(
        "--classifier-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=_DETECTION_CLASSIFIER_HELP,
    )
    parser.add_argument(
        "--export",
        action="append",
        default=[],
        help="Export to create. May be repeated. Examples: summary-json, excel, csv, session.",
    )
    parser.add_argument("--include-channel", action="append", default=[], help="Exact channel to include in detection; may be repeated.")
    parser.add_argument("--exclude-channel", action="append", default=[], help="Exact channel to exclude from detection; may be repeated.")


def _detection_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    config = {
        "biomarker": args.biomarker,
        "detector": args.detector,
        "filter": _parse_key_values(args.filter),
        "detector_params": _parse_key_values(args.detector_param),
        "classifier": _parse_key_values(args.classifier_param, list_keys={"steps"}),
        "n_jobs": args.n_jobs,
        "include_channels": args.include_channel,
        "exclude_channels": args.exclude_channel,
    }
    if args.export:
        config["exports"] = args.export
    return config


def _optional_json_object(raw: str | None, field_name: str) -> dict[str, Any] | None:
    parsed = _parse_json_object(raw, field_name)
    return parsed if parsed else None


def _add_parameter_common_args(parser: argparse.ArgumentParser, *, goal_default: str | None = None):
    parser.add_argument("--biomarker", default="HFO", help="Biomarker type: HFO, Spindle, or Spike.")
    parser.add_argument("--detector", default=None, help="Detector name, e.g. STE, MNI, YASA, RMS/LL.")
    parser.add_argument("--sample-freq", type=float, help="Sample frequency in Hz.")
    parser.add_argument("--n-jobs", type=int, help="Worker count for parameter planning/defaults.")
    parser.add_argument("--goal-profile", default=goal_default, help="Goal profile: qc, smoke, exploratory, specificity, comparison, batch, reproduction, full.")
    parser.add_argument("--filter", action="append", default=[], metavar="KEY=VALUE", help="Filter parameter value, e.g. fp=80. May be repeated.")
    parser.add_argument("--include-channel", action="append", default=[], help="Exact channel to include; may be repeated.")
    parser.add_argument("--exclude-channel", action="append", default=[], help="Exact channel to exclude; may be repeated.")
    parser.add_argument("--recording-info-json", help="Recording metadata JSON object.")
    parser.add_argument("--roi-json", help="ROI JSON object, typically from the roi command.")
    parser.add_argument("--previous-json", help="Previous parameter JSON object for diff generation.")


def _parameter_payload_from_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "biomarker": args.biomarker,
        "detector": args.detector,
    }
    if args.sample_freq is not None:
        payload["sample_freq"] = args.sample_freq
    if args.n_jobs is not None:
        payload["n_jobs"] = args.n_jobs
    if args.goal_profile:
        payload["goal_profile"] = args.goal_profile
    if args.filter:
        payload["filter"] = _parse_key_values(args.filter)
    if args.include_channel:
        payload["include_channels"] = args.include_channel
    if args.exclude_channel:
        payload["exclude_channels"] = args.exclude_channel
    recording_info = _optional_json_object(getattr(args, "recording_info_json", None), "recording-info-json")
    if recording_info:
        payload["recording_info"] = recording_info
    roi = _optional_json_object(getattr(args, "roi_json", None), "roi-json")
    if roi:
        payload["roi"] = roi
    previous = _optional_json_object(getattr(args, "previous_json", None), "previous-json")
    if previous:
        payload["previous_parameters"] = previous
    if getattr(args, "require_channels", False):
        payload["require_channels"] = True
    if getattr(args, "preset", None):
        payload["preset"] = args.preset
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyhfo", description="Agent-friendly PyHFO CLI.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="Read recording metadata.")
    info.add_argument("--input", required=True, help="Path to an EEG recording.")
    info.add_argument("--json", action="store_true", help="Accepted for explicit JSON mode; JSON is always emitted.")

    roi = subparsers.add_parser("roi", help="Infer label-free brain-signal ROI from channel names.")
    roi.add_argument("--input", required=True, help="Path to an EEG recording.")

    severity = subparsers.add_parser("severity", help="Score whole-recording severity (0-5 per 15s segment) for a scalp EEG recording.")
    severity.add_argument("--input", required=True, help="Path to a scalp EEG recording.")
    severity.add_argument("--model-dir", help="Directory containing local config.json and model.safetensors (also PYHFO_SEVERITY_MODEL_DIR).")
    severity.add_argument("--export-csv", help="Optional path to also write per-segment scores as CSV.")

    qc = subparsers.add_parser("qc", help="Compute label-free channel QC metrics.")
    qc.add_argument("--input", required=True, help="Path to an EEG recording.")
    qc.add_argument("--exclude-channel", action="append", default=[], help="Channel to exclude from QC; may be repeated.")
    qc.add_argument("--outlier-z-threshold", type=float, default=8.0)
    qc.add_argument("--max-duration-seconds", type=float, default=60.0, help="Bounded preview duration for memory-safe QC.")

    spectral = subparsers.add_parser("spectral", help="Compute compact spectral profile for detector planning.")
    spectral.add_argument("--input", required=True, help="Path to an EEG recording.")
    spectral.add_argument("--exclude-channel", action="append", default=[], help="Channel to exclude from spectral profile; may be repeated.")
    spectral.add_argument("--max-channels", type=int, default=32)
    spectral.add_argument("--max-duration-seconds", type=float, default=60.0, help="Bounded preview duration for memory-safe spectral planning.")

    line_noise = subparsers.add_parser("line-noise", help="Detect candidate 50/60 Hz line-noise peaks for preprocessing decisions.")
    line_noise.add_argument("--input", required=True, help="Path to an EEG recording.")
    line_noise.add_argument("--exclude-channel", action="append", default=[], help="Channel to exclude from line-noise detection; may be repeated.")
    line_noise.add_argument("--candidate-frequency", action="append", type=float, default=[], help="Candidate base mains frequency, usually 50 or 60; may be repeated.")
    line_noise.add_argument("--site-line-freq-hz", type=float, help="Explicit site/header line frequency supplied by user or protocol.")
    line_noise.add_argument("--max-channels", type=int, default=32)
    line_noise.add_argument("--max-harmonics", type=int, default=4)
    line_noise.add_argument("--bandwidth-hz", type=float, default=1.0)
    line_noise.add_argument("--shoulder-width-hz", type=float, default=5.0)
    line_noise.add_argument("--ratio-threshold", type=float, default=5.0)
    line_noise.add_argument("--max-duration-seconds", type=float, default=60.0, help="Bounded preview duration for memory-safe line-noise detection.")

    plan = subparsers.add_parser("detector-plan", help="Recommend label-free Spike/HFO detector plan.")
    plan.add_argument("--input", required=True, help="Path to an EEG recording.")
    plan.add_argument("--recording-info-json", help="Recording metadata as a JSON object (reuse precomputed info).")
    plan.add_argument("--roi-json", help="Region-of-interest as a JSON object (curated channel selection).")

    rank = subparsers.add_parser("rank-candidates", help="Rank candidate channels from detector CSV outputs.")
    rank.add_argument("--run", action="append", required=True, metavar="BIOMARKER=CSV", help="Detector run artifact; may be repeated, e.g. HFO=/tmp/events.csv")
    rank.add_argument("--duration-seconds", type=float, default=0.0)
    rank.add_argument("--exclude-channel", action="append", default=[], help="Channel to exclude from ranking; may be repeated.")
    rank.add_argument("--top-n", type=int, default=20)

    detect = subparsers.add_parser("detect", help="Run a PyHFO detection pipeline.")
    detect.add_argument("--input", required=True, help="Path to an EEG recording.")
    detect.add_argument("--output", required=True, help="Output directory.")
    _add_detection_args(detect)

    quick_detect = subparsers.add_parser("quick-detect", help="Run metadata-backed quick detection: recommend parameters, validate them, then run detection.")
    quick_detect.add_argument("--input", required=True, help="Path to an EEG recording.")
    quick_detect.add_argument("--output", required=True, help="Output directory.")
    quick_detect.add_argument("--biomarker", default="HFO", help="Biomarker type: HFO, Spindle, or Spike.")
    quick_detect.add_argument("--detector", required=True, help="Detector name, e.g. STE, MNI, YASA, RMS/LL.")
    quick_detect.add_argument("--goal-profile", default="smoke", help="Goal profile for automatic parameter recommendation.")
    quick_detect.add_argument("--sample-freq", type=float, help="Override sample frequency in Hz.")
    quick_detect.add_argument("--n-jobs", type=int, help="Worker count override.")
    quick_detect.add_argument("--filter", action="append", default=[], metavar="KEY=VALUE", help="Filter override, e.g. fp=80. May be repeated.")
    quick_detect.add_argument("--detector-param", action="append", default=[], metavar="KEY=VALUE", help="Detector parameter override.")
    quick_detect.add_argument("--classifier-param", action="append", default=[], metavar="KEY=VALUE", help=_DETECTION_CLASSIFIER_HELP)
    quick_detect.add_argument("--export", action="append", default=[], help="Export to create. May be repeated. Examples: summary-json, csv, session.")
    quick_detect.add_argument("--include-channel", action="append", default=[], help="Exact channel to include; may be repeated.")
    quick_detect.add_argument("--exclude-channel", action="append", default=[], help="Exact channel to exclude; may be repeated.")

    classify = subparsers.add_parser("classify", help="Classify an existing saved PyHFO/PyBrain session without rerunning detection.")
    classify.add_argument("--session", "--session-path", dest="session_path", required=True, help="Saved .pybrain/.npz session path from a prior detection/export.")
    classify.add_argument("--output", help="Output directory for classified artifacts. Defaults next to the session.")
    classify.add_argument("--biomarker", default="HFO", help="Biomarker type: HFO or Spindle.")
    classify.add_argument("--run-id", help="Run id to activate before classification.")
    classify.add_argument(
        "--classifier-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Classifier parameter override. Examples: model_type=default_cpu, device=cpu, batch_size=16, steps=artifact.",
    )
    classify.add_argument(
        "--export",
        action="append",
        default=[],
        help="Export to create. May be repeated. Examples: summary-json, csv, excel, session, clinical_summary.",
    )
    classify.add_argument("--session-format", choices=["pybrain", "npz"], help="Format for exported classified session artifact.")

    batch_create = subparsers.add_parser("batch-create", help="Create a resumable batch manifest.")
    batch_create.add_argument("--input", required=True, help="Input directory containing recordings.")
    batch_create.add_argument("--output", required=True, help="Output directory for batch artifacts.")
    batch_create.add_argument("--no-recursive", action="store_true", help="Only scan the input directory.")

    batch_run = subparsers.add_parser("batch-run", help="Run detection for a batch manifest or directory.")
    batch_run.add_argument("--manifest", help="Existing batch_project.json path.")
    batch_run.add_argument("--input", help="Input directory; used when --manifest is omitted.")
    batch_run.add_argument("--output", help="Output directory; used when --manifest is omitted.")
    batch_run.add_argument("--retry-failed", action="store_true", help="Retry failed cases.")
    batch_run.add_argument("--fail-fast", action="store_true", help="Stop after the first failed case.")
    batch_run.add_argument("--no-recursive", action="store_true", help="Only scan the input directory.")
    _add_detection_args(batch_run)

    parameters = subparsers.add_parser("parameters", help="Plan, validate, and inspect detector parameters.")
    parameter_subparsers = parameters.add_subparsers(dest="parameters_command", required=True)

    defaults = parameter_subparsers.add_parser("defaults", help="Return PyHFO detector default parameters.")
    _add_parameter_common_args(defaults)
    defaults.add_argument("--preset", help="Optional detector preset, e.g. an LSM preset label.")

    recommend = parameter_subparsers.add_parser("recommend", help="Recommend goal-profile-aware parameters.")
    _add_parameter_common_args(recommend, goal_default="smoke")

    validate = parameter_subparsers.add_parser("validate", help="Validate parameters, including Nyquist safety.")
    _add_parameter_common_args(validate)
    validate.add_argument("--require-channels", action="store_true", help="Require at least one include-channel.")

    sweep = parameter_subparsers.add_parser("sweep", help="Create a bounded local parameter sweep.")
    _add_parameter_common_args(sweep, goal_default="exploratory")

    prepare = subparsers.add_parser("prepare", help="Create a cleaned PyHFO-ready FIF/FIF.GZ copy.")
    prepare.add_argument("--input", required=True, help="Path to an EEG recording.")
    prepare.add_argument("--output", required=True, help="Destination .fif or .fif.gz path.")
    prepare.add_argument("--extra-bad-channel", action="append", default=[], help="Additional channel to drop/mark bad.")
    prepare.add_argument("--keep-bad-channels", action="store_true", help="Do not drop file-declared or extra bad channels.")
    prepare.add_argument("--keep-flat-channels", action="store_true", help="Do not drop flat channels.")
    prepare.add_argument("--drop-outlier-channels", action="store_true", help="Aggressively drop robust amplitude outlier channels.")
    prepare.add_argument("--outlier-z-threshold", type=float, default=8.0)
    prepare.add_argument("--bipolar-montage", default="none", choices=["none", "auto_adjacent", "pairs"])
    prepare.add_argument("--bipolar-pairs-json", help="JSON array of {anode,cathode,name?} objects.")
    prepare.add_argument("--drop-original-after-bipolar", action="store_true")
    prepare.add_argument("--line-noise-notch-hz", type=float)
    prepare.add_argument("--notch", action="append", type=float, default=[], help="Notch frequency; may be repeated.")
    prepare.add_argument("--bandpass-json", help="JSON object, e.g. '{\"l_freq\":1,\"h_freq\":300}'.")
    prepare.add_argument("--resample-sfreq", type=float)

    manifest = subparsers.add_parser("manifest", help="Print the LLM/action capability manifest.")
    manifest.add_argument("--callable-only", action="store_true", help="Hide planned actions that still need wrappers.")

    state = subparsers.add_parser("state", help="Print an empty sidecar session state snapshot.")
    state.add_argument("--session-id", default="cli")

    action = subparsers.add_parser("action", help="Dispatch a sidecar action as JSON.")
    action.add_argument("--name", required=True, help="Action name from the capability manifest.")
    action.add_argument("--args-json", default="{}", help="Action arguments as a JSON object.")
    action.add_argument("--session-id", default="cli")

    serve = subparsers.add_parser("serve", help="Start the minimal HTTP sidecar control surface.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    return parser


def _run(args: argparse.Namespace) -> dict:
    if args.command == "info":
        return get_recording_info(args.input)
    if args.command == "roi":
        return infer_brain_signal_roi(args.input)
    if args.command == "severity":
        return run_severity_scoring({
            "input_path": args.input,
            "model_dir": args.model_dir,
            "export_csv": args.export_csv,
        })
    if args.command == "qc":
        return compute_channel_qc({
            "input_path": args.input,
            "exclude_channels": args.exclude_channel,
            "outlier_z_threshold": args.outlier_z_threshold,
            "max_duration_seconds": args.max_duration_seconds,
        })
    if args.command == "spectral":
        return compute_spectral_profile({
            "input_path": args.input,
            "exclude_channels": args.exclude_channel,
            "max_channels": args.max_channels,
            "max_duration_seconds": args.max_duration_seconds,
        })
    if args.command == "line-noise":
        return detect_line_noise({
            "input_path": args.input,
            "exclude_channels": args.exclude_channel,
            "candidate_frequencies_hz": args.candidate_frequency or [50.0, 60.0],
            "site_line_freq_hz": args.site_line_freq_hz,
            "max_channels": args.max_channels,
            "max_harmonics": args.max_harmonics,
            "bandwidth_hz": args.bandwidth_hz,
            "shoulder_width_hz": args.shoulder_width_hz,
            "ratio_threshold": args.ratio_threshold,
            "max_duration_seconds": args.max_duration_seconds,
        })
    if args.command == "detector-plan":
        payload = {"input_path": args.input}
        recording_info = _parse_json_object(args.recording_info_json, "recording-info-json")
        if recording_info:
            payload["recording_info"] = recording_info
        roi = _parse_json_object(args.roi_json, "roi-json")
        if roi:
            payload["roi"] = roi
        return recommend_detector_plan(payload)
    if args.command == "rank-candidates":
        runs = []
        for item in args.run:
            if "=" not in item:
                raise PyHFOAgentError("invalid_argument", f"Expected BIOMARKER=CSV, got: {item}")
            biomarker, events_csv = item.split("=", 1)
            runs.append({"biomarker": biomarker.strip(), "events_csv": events_csv.strip()})
        return rank_candidate_channels({"runs": runs, "duration_seconds": args.duration_seconds, "exclude_channels": args.exclude_channel, "top_n": args.top_n})
    if args.command == "detect":
        request = {
            **_detection_config_from_args(args),
            "input_path": args.input,
            "output_dir": args.output,
        }
        return run_detection(request)
    if args.command == "quick-detect":
        recording_info = get_recording_info(args.input)
        plan_payload = {
            "goal_profile": args.goal_profile,
            "biomarker": args.biomarker,
            "detector": args.detector,
            "recording_info": recording_info,
            "include_channels": args.include_channel,
            "exclude_channels": args.exclude_channel,
        }
        if args.sample_freq is not None:
            plan_payload["sample_freq"] = args.sample_freq
        if args.n_jobs is not None:
            plan_payload["n_jobs"] = args.n_jobs
        if args.filter:
            plan_payload["filter"] = _parse_key_values(args.filter)
        parameter_plan = recommend_parameters(plan_payload)
        planned = dict(parameter_plan.get("parameters") or {})
        request = {
            **planned,
            "input_path": args.input,
            "output_dir": args.output,
            "biomarker": args.biomarker,
            "detector": args.detector,
            "filter": {**dict(planned.get("filter") or {}), **_parse_key_values(args.filter)},
            "detector_params": _parse_key_values(args.detector_param),
            "classifier": _parse_key_values(args.classifier_param, list_keys={"steps"}),
            "include_channels": args.include_channel or planned.get("include_channels") or [],
            "exclude_channels": args.exclude_channel or planned.get("exclude_channels") or [],
        }
        if args.n_jobs is not None:
            request["n_jobs"] = args.n_jobs
        if args.export:
            request["exports"] = args.export
        final_validation = validate_parameters(
            {
                **plan_payload,
                "biomarker": args.biomarker,
                "detector": args.detector,
                "filter": request["filter"],
                "detector_params": request["detector_params"],
                "recording_info": recording_info,
            }
        )
        if not final_validation.get("valid", False):
            return {
                "status": "invalid",
                "mode": "quick_detection",
                "parameter_plan": parameter_plan,
                "final_validation": final_validation,
            }
        detection = run_detection(request)
        return {
            "status": detection.get("status", "complete"),
            "mode": "quick_detection",
            "parameter_plan": parameter_plan,
            "final_validation": final_validation,
            "detection": detection,
        }
    if args.command == "classify":
        request = {
            "session_path": args.session_path,
            "output_dir": args.output,
            "biomarker": args.biomarker,
            "run_id": args.run_id,
            "classifier": _parse_key_values(args.classifier_param, list_keys={"steps"}),
        }
        if args.export:
            request["exports"] = args.export
        if args.session_format:
            request["session_format"] = args.session_format
        return classify_existing_run(request)
    if args.command == "parameters":
        payload = _parameter_payload_from_args(args)
        if args.parameters_command == "defaults":
            return detector_defaults(payload)
        if args.parameters_command == "recommend":
            return recommend_parameters(payload)
        if args.parameters_command == "validate":
            return validate_parameters(payload)
        if args.parameters_command == "sweep":
            return sweep_parameters(payload)
        raise PyHFOAgentError("invalid_argument", f"Unknown parameters command: {args.parameters_command}")
    if args.command == "batch-create":
        return create_batch(args.input, args.output, recursive=not args.no_recursive)
    if args.command == "batch-run":
        request = {
            "manifest_path": args.manifest,
            "input_dir": args.input,
            "output_dir": args.output,
            "recursive": not args.no_recursive,
            "retry_failed": args.retry_failed,
            "fail_fast": args.fail_fast,
            "detection": _detection_config_from_args(args),
        }
        return run_batch(request)
    if args.command == "prepare":
        pairs = []
        if args.bipolar_pairs_json:
            try:
                pairs = json.loads(args.bipolar_pairs_json)
            except json.JSONDecodeError as exc:
                raise PyHFOAgentError("invalid_argument", "bipolar-pairs-json must be valid JSON.") from exc
            if not isinstance(pairs, list):
                raise PyHFOAgentError("invalid_argument", "bipolar-pairs-json must be a JSON array.")
        request = {
            "input_path": args.input,
            "output_path": args.output,
            "extra_bad_channels": args.extra_bad_channel,
            "drop_bad_channels": not args.keep_bad_channels,
            "drop_flat_channels": not args.keep_flat_channels,
            "drop_outlier_channels": args.drop_outlier_channels,
            "outlier_z_threshold": args.outlier_z_threshold,
            "bipolar_montage": args.bipolar_montage,
            "bipolar_pairs": pairs,
            "drop_original_after_bipolar": args.drop_original_after_bipolar,
            "line_noise_notch_hz": args.line_noise_notch_hz,
            "notch_freqs": args.notch,
            "bandpass": _parse_json_object(args.bandpass_json, "bandpass-json"),
            "resample_sfreq": args.resample_sfreq,
        }
        return prepare_recording(request)
    if args.command == "manifest":
        return get_capability_manifest(include_missing=not args.callable_only)
    if args.command == "state":
        return load_sidecar_session(args.session_id).get_state()
    if args.command == "action":
        session = load_sidecar_session(args.session_id)
        result = session.dispatch(
            args.name,
            _parse_json_object(args.args_json, "args-json"),
        )
        save_sidecar_session(session)
        return result
    if args.command == "serve":
        from pyhfo.agent_api.sidecar_web import main as web_main

        raise SystemExit(web_main(["--host", args.host, "--port", str(args.port)]))
    raise PyHFOAgentError("invalid_argument", f"Unknown command: {args.command}")


def _exit_code_for(error: dict) -> int:
    code = str(error.get("code", ""))
    if code.startswith("invalid_") or code == "file_not_found":
        return 2
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        # stdout is this CLI's machine-readable JSON channel, but the pipeline (and its
        # deps: HFO_App.load_edf, tqdm bars, MNE) print progress there. Route anything the
        # command itself emits to stderr so stdout carries only the result JSON below --
        # same split pipeline_worker.py already uses for its own stdout contract.
        with contextlib.redirect_stdout(sys.stderr):
            result = _run(args)
        _print_json(result)
        return 0
    except SystemExit as exc:
        return int(exc.code or 0)
    except Exception as exc:
        error = error_response(exc)
        print(json.dumps(json_safe({"status": "error", "error": error}), sort_keys=True), file=sys.stderr)
        return _exit_code_for(error)


if __name__ == "__main__":
    raise SystemExit(main())
