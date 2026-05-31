from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from src.agent_api import PyHFOAgentError, create_batch, get_recording_info, run_batch, run_detection
from src.agent_api.errors import error_response
from src.agent_api.serialization import json_safe


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
        return value


def _parse_key_values(items: list[str] | None) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise PyHFOAgentError("invalid_argument", f"Expected KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise PyHFOAgentError("invalid_argument", f"Empty key in argument: {item}")
        values[key] = _parse_scalar(value.strip())
    return values


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
        help="Classifier parameter override. Add enabled=true to run classification.",
    )
    parser.add_argument(
        "--export",
        action="append",
        default=[],
        help="Export to create. May be repeated. Examples: summary-json, excel, csv, session.",
    )


def _detection_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    config = {
        "biomarker": args.biomarker,
        "detector": args.detector,
        "filter": _parse_key_values(args.filter),
        "detector_params": _parse_key_values(args.detector_param),
        "classifier": _parse_key_values(args.classifier_param),
        "n_jobs": args.n_jobs,
    }
    if args.export:
        config["exports"] = args.export
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyhfo-agent", description="Agent-friendly PyHFO CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="Read recording metadata.")
    info.add_argument("--input", required=True, help="Path to an EEG recording.")
    info.add_argument("--json", action="store_true", help="Accepted for explicit JSON mode; JSON is always emitted.")

    detect = subparsers.add_parser("detect", help="Run a PyHFO detection pipeline.")
    detect.add_argument("--input", required=True, help="Path to an EEG recording.")
    detect.add_argument("--output", required=True, help="Output directory.")
    _add_detection_args(detect)

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

    return parser


def _run(args: argparse.Namespace) -> dict:
    if args.command == "info":
        return get_recording_info(args.input)
    if args.command == "detect":
        request = {
            **_detection_config_from_args(args),
            "input_path": args.input,
            "output_dir": args.output,
        }
        return run_detection(request)
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
        _print_json(_run(args))
        return 0
    except SystemExit as exc:
        return int(exc.code or 0)
    except Exception as exc:
        error = error_response(exc)
        print(json.dumps(json_safe({"status": "error", "error": error}), sort_keys=True), file=sys.stderr)
        return _exit_code_for(error)


if __name__ == "__main__":
    raise SystemExit(main())
