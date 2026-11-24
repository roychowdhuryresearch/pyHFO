"""Out-of-process worker for heavy PyHFO pipeline calls.

The backend serves HTTP on a single-process ThreadingHTTPServer. Detection /
classification / batch are CPU- and GIL-bound (PyTorch inference + p_map feature
extraction); running them inline in a request-handler thread starves latency-
sensitive endpoints (notably /waveform/window, which carries a hard 20s client
abort) until the heavy job finishes. This module runs ONE such call in a fresh
interpreter so the server process keeps a free GIL and stays responsive.

Contract (kept deliberately dumb so the parent runner owns all policy):
  argv[1] = path to a JSON file: {"fn": <name>, "payload": {...}}
  - stdout: EXACTLY one machine-readable JSON payload: json_safe(result) on
    success, or {"status": "error", "error": ...} on failure
  - stderr: tqdm progress + any stray library prints
  - exit 0 on success; non-zero if the pipeline raised (e.g. fail_on_error)

Stdout is kept clean by redirecting any prints emitted during compute to stderr,
so the parent can json.loads(stdout) without scraping past tqdm/log noise.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time

# Imported lazily-at-module-load (this IS the heavy interpreter): pulling in the
# pipeline here is the point — it happens in the child, off the server's GIL.
from pyhfo.agent_api.errors import error_response
from pyhfo.agent_api.pipeline import classify_existing_run, run_batch, run_detection, run_severity_scoring
from pyhfo.agent_api.preprocessing import prepare_recording
from pyhfo.agent_api.serialization import json_safe

_FUNCTIONS = {
    "run_detection": run_detection,
    "classify_existing_run": classify_existing_run,
    "run_batch": run_batch,
    "prepare_recording": prepare_recording,
    "run_severity_scoring": run_severity_scoring,
}


def _write_stdout_json(value: dict) -> None:
    sys.stdout.write(json.dumps(json_safe(value), sort_keys=True))
    sys.stdout.flush()


def _install_parent_death_guard() -> None:
    """Hard-exit if the spawning backend dies, so a backend SIGKILL mid-detection
    can never leave an orphaned compute process running (mirrors the JS
    PARENT_DEATH_GUARD that guards the backend itself). POSIX: when the parent
    dies the child is reparented (ppid changes), which this thread detects."""
    expected = os.environ.get("IEEG_PARENT_PID")
    if not expected:
        return
    try:
        expected_pid = int(expected)
    except (TypeError, ValueError):
        return

    def _watch() -> None:
        while True:
            try:
                if os.getppid() != expected_pid:
                    os._exit(137)
            except Exception:
                os._exit(137)
            time.sleep(1.0)

    threading.Thread(target=_watch, name="parent-death-guard", daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    _install_parent_death_guard()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        _write_stdout_json({"status": "error", "error": {"code": "invalid_argument", "message": "pipeline_worker requires a payload-file path."}})
        return 2
    try:
        with open(argv[0], "r", encoding="utf-8") as handle:
            request = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        _write_stdout_json({"status": "error", "error": {"code": "invalid_argument", "message": f"Unreadable payload file: {exc}"}})
        return 2

    fn_name = str(request.get("fn") or "")
    payload = request.get("payload") or {}
    function = _FUNCTIONS.get(fn_name)
    if function is None:
        _write_stdout_json({"status": "error", "error": {"code": "invalid_argument", "message": f"Unknown pipeline fn: {fn_name}"}})
        return 2

    real_stdout = sys.stdout
    try:
        # Any print() the pipeline (or its deps) emits during compute goes to
        # stderr; the real stdout is reserved for the single result JSON below.
        with contextlib.redirect_stdout(sys.stderr):
            result = function(payload)
    except Exception as exc:  # the pipeline raised (e.g. fail_on_error=True)
        real_stdout.write(json.dumps(json_safe({"status": "error", "error": error_response(exc)}), sort_keys=True))
        real_stdout.flush()
        return 1
    real_stdout.write(json.dumps(json_safe(result)))
    real_stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
