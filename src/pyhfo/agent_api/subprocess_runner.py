"""Run heavy PyHFO pipeline calls out-of-process to free the server's GIL.

`run_pipeline_isolated(fn, payload)` is a drop-in for calling
prepare_recording/run_detection/classify_existing_run/run_batch directly. When subprocess mode is
enabled (env PYHFO_PIPELINE_SUBPROCESS truthy — set by the server's main()), it
spawns pipeline_worker in a fresh interpreter and blocks the calling HTTP handler
thread on it. Because the handler blocks on os read()/wait() (which release the
GIL) while the CPU/GIL-bound work runs in a SEPARATE interpreter, every other
server thread — notably /waveform/window — stays responsive. When disabled
(CLI/MCP/tests, the default), it just calls the function inline.

Behaviour is kept identical to the inline call:
  - success -> the same json_safe result dict the function returns
  - the function raising (e.g. fail_on_error=True) -> a PyHFOAgentError is raised
    here too, reconstructed from the worker's on-disk summary.json when available
  - a child killed by a signal (OOM) -> PyHFOAgentError, and crucially the server
    process SURVIVES (inline, an OOM SIGKILL would take down the whole backend).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import signal
import tempfile
import threading
from pathlib import Path
from typing import Any

import pyhfo
from pyhfo.agent_api.errors import PyHFOAgentError
from pyhfo.agent_api.serialization import json_safe

_WORKER_MODULE = "pyhfo.agent_api.pipeline_worker"
_VALID_FNS = ("prepare_recording", "run_detection", "classify_existing_run", "run_batch", "run_severity_scoring")

# Last-call telemetry (also used by tests to assert the call really forked). Tiny
# and overwritten each call; never part of the returned result payload.
_last_isolation: dict[str, Any] = {}


def last_isolation() -> dict[str, Any]:
    return dict(_last_isolation)


def subprocess_enabled() -> bool:
    return str(os.environ.get("PYHFO_PIPELINE_SUBPROCESS", "")).strip().lower() in {"1", "true", "yes", "on"}


def _max_procs() -> int:
    try:
        return max(1, int(os.environ.get("PYHFO_PIPELINE_MAX_PROCS", "1")))
    except (TypeError, ValueError):
        return 1


# Bound concurrent compute children so overlapping ops (manual + agent-staged)
# can't fork-bomb the box. Default 1 == today's effective single-detection-at-once.
_SEMAPHORE = threading.BoundedSemaphore(_max_procs())


def _inline_call(fn_name: str, payload: dict[str, Any]) -> dict:
    if fn_name == "prepare_recording":
        from pyhfo.agent_api.preprocessing import prepare_recording  # lazy: avoid import cost when only spawning

        return prepare_recording(payload)
    from pyhfo.agent_api import pipeline  # lazy: avoid import cost when only spawning

    return getattr(pipeline, fn_name)(payload)


def run_pipeline_isolated(fn_name: str, payload: dict[str, Any]) -> dict:
    if fn_name not in _VALID_FNS:
        raise PyHFOAgentError("invalid_argument", f"Unknown pipeline fn: {fn_name}")
    payload = payload or {}
    if not subprocess_enabled():
        _last_isolation.clear()
        _last_isolation.update({"mode": "inline", "pid": os.getpid()})
        return _inline_call(fn_name, payload)
    return _run_in_subprocess(fn_name, payload)


# Live isolated children, so a long pass can be cancelled instead of only abandoned. A severity
# score over a multi-hour study runs for a quarter of an hour; without this the only way out is to
# kill the app.
_ACTIVE_LOCK = threading.Lock()
_ACTIVE: dict[str, subprocess.Popen] = {}
# Runs the user asked to stop. A cancelled child exits non-zero like a crashed one; without this
# the user is told their own Cancel was a failure, and the agent reports a crash that never happened.
_CANCELLED: set[str] = set()


def _worker_command(payload_path: str) -> list[str]:
    """How to start one pipeline job in its own process.

    Frozen, `sys.executable` is the app itself and there is no interpreter to accept `-m`, so the app
    re-invokes itself with a marker its entry point dispatches on. Without this, subprocess mode is
    impossible in a packaged build -- which is why it used to fall back to running inline, taking
    working cancellation and a responsive UI down with it.
    """
    if getattr(sys, "frozen", False):
        return [sys.executable, "--pipeline-worker", payload_path]
    return [sys.executable, "-m", _WORKER_MODULE, payload_path]


def cancel_pipeline(fn_name: str) -> bool:
    """Terminate the running child for fn_name. True when something was actually killed."""
    with _ACTIVE_LOCK:
        proc = _ACTIVE.get(fn_name)
        if proc is not None and proc.poll() is None:
            # Marked BEFORE the kill so the waiting thread cannot mistake the exit for a crash.
            _CANCELLED.add(fn_name)
    if proc is None or proc.poll() is not None:
        return False
    try:
        if os.name == 'nt':
            # The child is its own process group (start_new_session); taskkill /T takes the workers
            # it may have spawned with it, which terminate() alone would orphan.
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)], capture_output=True, check=False)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.kill()
        except Exception:
            return False
    return True


def _run_in_subprocess(fn_name: str, payload: dict[str, Any]) -> dict:
    request = {"fn": fn_name, "payload": json_safe(payload)}
    output_dir = payload.get("output_dir") if isinstance(payload, dict) else None
    with _SEMAPHORE:
        tmp_path: str | None = None
        try:
            handle, tmp_path = tempfile.mkstemp(prefix="pyhfo-pipeline-", suffix=".json")
            with os.fdopen(handle, "w", encoding="utf-8") as fh:
                json.dump(request, fh)
            env = {
                **os.environ,
                "IEEG_PARENT_PID": str(os.getpid()),
                "PYHFO_PIPELINE_SUBPROCESS": "0",  # the child runs the fn inline; never re-fork
            }
            if not getattr(sys, "frozen", False):
                # Forward the parent's package location for source checkouts as well as wheels.
                # Keep the caller's working directory so relative input and output paths still work.
                package_parent = str(Path(pyhfo.__file__).resolve().parent.parent)
                env["PYTHONPATH"] = os.pathsep.join(filter(None, (package_parent, env.get("PYTHONPATH"))))
            proc = subprocess.Popen(
                _worker_command(tmp_path),
                stdout=subprocess.PIPE,
                stderr=None,  # inherit: tqdm/progress flow to the backend's stderr (log parity)
                env=env,
                start_new_session=True,
            )
            with _ACTIVE_LOCK:
                _ACTIVE[fn_name] = proc
            # Only stdout is piped, so there is no second-pipe deadlock. read()
            # drains continuously until the child closes stdout (on exit).
            out_bytes = proc.stdout.read() if proc.stdout else b""
            returncode = proc.wait()
            _last_isolation.clear()
            _last_isolation.update({"mode": "subprocess", "pid": proc.pid, "returncode": returncode})
            with _ACTIVE_LOCK:
                was_cancelled = fn_name in _CANCELLED
                _CANCELLED.discard(fn_name)
            if was_cancelled:
                raise PyHFOAgentError('cancelled', 'The run was cancelled.')
            return _result_from_child(out_bytes, returncode, output_dir, payload)
        finally:
            with _ACTIVE_LOCK:
                if _ACTIVE.get(fn_name) is not None and _ACTIVE[fn_name].pid == getattr(locals().get('proc', None), 'pid', None):
                    _ACTIVE.pop(fn_name, None)
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


def _result_from_child(out_bytes: bytes, returncode: int, output_dir: Any, payload: dict[str, Any]) -> dict:
    text = (out_bytes or b"").decode("utf-8", errors="replace").strip()
    if text:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            pass  # fall through to the on-disk / error paths
        else:
            if returncode != 0 and isinstance(parsed, dict) and parsed.get("error"):
                _raise_pipeline_error(parsed["error"])
            if returncode == 0 and isinstance(parsed, dict):
                return parsed

    # Non-success: reproduce the inline behaviour. run_detection writes summary.json
    # (status=failed, with an `error`) before it raises under fail_on_error, so a
    # non-zero exit + summary lets us re-raise the SAME PyHFOAgentError inline would.
    summary = _read_summary(output_dir)
    if isinstance(summary, dict) and summary.get("error"):
        _raise_pipeline_error(summary["error"])
    if returncode == 0 and isinstance(summary, dict):
        return summary  # clean exit, unparseable stdout, but a usable summary on disk

    if returncode < 0:
        raise PyHFOAgentError(
            "detection_subprocess_killed",
            f"The detection process was killed (signal {-returncode}; most likely out of memory). The backend stayed up — reduce channels/duration or free memory and retry.",
        )
    raise PyHFOAgentError(
        "detection_subprocess_failed",
        f"The analysis process exited with code {returncode} without a usable result. See the status log for details.",
    )


def _raise_pipeline_error(error: Any) -> None:
    payload = error if isinstance(error, dict) else {}
    raise PyHFOAgentError(
        str(payload.get("code") or "detection_failed"),
        str(payload.get("message") or "Detection failed."),
        details=payload.get("details") or {},
    )


def _read_summary(output_dir: Any) -> dict | None:
    if not output_dir:
        return None
    try:
        path = Path(str(output_dir)).expanduser() / "summary.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return None
