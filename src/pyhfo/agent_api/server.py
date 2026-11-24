from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from pyhfo import __version__
from pyhfo.agent_api.errors import PyHFOAgentError, error_response
from pyhfo.agent_api.parameters import detector_defaults, recommend_parameters, sweep_parameters, validate_parameters
from pyhfo.agent_api.pipeline import create_batch
from pyhfo.agent_api.subprocess_runner import run_pipeline_isolated
from pyhfo.agent_api.review import annotate_event, compare_detector_runs, create_consensus_run, event_cursor, event_feature_summary, event_snippet, export_review_state, list_events, overlap_review
from pyhfo.agent_api.serialization import json_safe
from pyhfo.agent_api.signal_analysis import detect_line_noise
from pyhfo.agent_api.sessions import create_project, export_report, export_session, export_workbook, open_project, save_project, summarize_saved_session
from pyhfo.agent_api.sidecar import PyHFOSidecarSession, get_capability_manifest
from pyhfo.agent_api.sidecar_web import get_chat_job, submit_chat_job
from pyhfo.agent_api.waveform import waveform_channels, waveform_overview, waveform_window


def _status_for_error(exc: Exception) -> int:
    if isinstance(exc, PyHFOAgentError):
        if exc.code in {"file_not_found", "event_not_found", "job_not_found", "run_not_found", "not_found"}:
            return 404
        if exc.code in {
            "invalid_request", "invalid_time_window", "invalid_filter", "invalid_goal_profile",
            "channel_not_found", "no_events", "invalid_annotation", "invalid_cursor_action",
            "unsafe_bind_host", "invalid_export_path", "invalid_project_path", "unauthorized", "invalid_channel_selection",
            "invalid_recording", "invalid_biomarker", "invalid_detector", "request_too_large",
        }:
            return 400 if exc.code != "unauthorized" else 401
    return 500


def _expected_token() -> str | None:
    return (os.getenv("PYHFO_ELECTRON_TOKEN") or os.getenv("PYHFO_BACKEND_TOKEN") or "").strip() or None


def _query(url: str) -> dict[str, Any]:
    parsed = parse_qs(urlparse(url).query)
    return {key: values[-1] if len(values) == 1 else values for key, values in parsed.items()}


def _float(payload: dict[str, Any], key: str, default: float) -> float:
    return float(payload.get(key, default))


def _int(payload: dict[str, Any], key: str, default: int) -> int:
    return int(payload.get(key, default))


def _bool(payload: dict[str, Any], key: str, default: bool = False) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _channels(value: Any) -> Any:
    if isinstance(value, str) and "," in value:
        return [item for item in value.split(",") if item]
    return value


def make_handler(session: PyHFOSidecarSession):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def _authorized(self) -> bool:
            token = _expected_token()
            if not token:
                return True
            provided = self.headers.get("authorization") or ""
            return hmac.compare_digest(provided, f"Bearer {token}")

        def _send_json(self, payload: Any, status: int = 200):
            # Compact separators, no indent/sort: this is the waveform-scroll hot path
            # (a 32-channel window serializes ~2x faster compact than indent=2+sort_keys,
            # with a smaller payload over IPC). The renderer always json.loads() it.
            body = json.dumps(json_safe(payload), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _payload(self) -> dict[str, Any]:
            length = int(self.headers.get("content-length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            try:
                return json.loads(raw or "{}")
            except json.JSONDecodeError as exc:
                raise PyHFOAgentError("invalid_request", "Request body must be valid JSON.") from exc

        def _handle_get(self, path: str, query: dict[str, Any]):
            if path == "/health":
                return {"status": "ok", "service": "pyhfo-agent-backend", "version": __version__, "local_only": True, "session_id": session.session_id}
            if path == "/manifest":
                return get_capability_manifest()
            if path == "/state":
                return session.get_state()
            if path == "/waveform/window":
                return waveform_window(str(query.get("input_path") or (session.recording_info or {}).get("input_path") or ""), start_seconds=_float(query, "start_seconds", 0), duration_seconds=_float(query, "duration_seconds", 10), channels=_channels(query.get("channels")), target_samples_per_channel=_int(query, "target_samples_per_channel", 1200), units=str(query.get("units", "uV")), show_filtered=_bool(query, "show_filtered"), filter_60hz=_bool(query, "filter_60hz"), filter_low=_float(query, "filter_low", 0.0) or None, filter_high=_float(query, "filter_high", 0.0) or None)
            if path == "/waveform/overview":
                return waveform_overview(str(query.get("input_path") or (session.recording_info or {}).get("input_path") or ""), channels=_channels(query.get("channels")), bins=_int(query, "bins", 512), units=str(query.get("units", "uV")))
            if path == "/waveform/channels":
                return waveform_channels(str(query.get("input_path") or (session.recording_info or {}).get("input_path") or ""))
            if path == "/detections":
                return {"status": "complete", "runs": [session.last_detection] if session.last_detection else []}
            if path.startswith("/detections/"):
                run_id = path.rsplit("/", 1)[-1]
                if session.last_detection and run_id in {str(session.last_detection.get("active_run_id")), str(session.last_detection.get("run_id")), "latest"}:
                    return {"status": "complete", "run": session.last_detection}
                raise PyHFOAgentError("run_not_found", f"Unknown run_id: {run_id}")
            if path == "/severity":
                if session.last_severity:
                    return {"status": "complete", "result": session.last_severity}
                return {"status": "not_run", "result": None}
            if path == "/events":
                return list_events(session, run_id=query.get("run_id"))
            if path.startswith("/events/") and path.endswith("/snippet"):
                event_id = path.split("/")[2]
                return event_snippet(session, event_id, input_path=query.get("input_path") or (session.recording_info or {}).get("input_path"), padding_seconds=_float(query, "padding_seconds", 1.0), target_samples_per_channel=_int(query, "target_samples_per_channel", 600), run_id=query.get("run_id"))
            if path.startswith("/events/"):
                event_id = path.rsplit("/", 1)[-1]
                return event_feature_summary(session, event_id, run_id=query.get("run_id"))
            if path == "/agent/chat-result":
                return get_chat_job(str(query.get("job_id") or ""))
            raise PyHFOAgentError("not_found", f"Unknown endpoint: {path}")

        def _handle_post(self, path: str, payload: dict[str, Any]):
            if path == "/projects/create":
                result = create_project(payload); session.project_state = result.get("state"); return result
            if path == "/projects/open":
                result = open_project(str(payload.get("project_path") or payload.get("path") or "")); session.project_state = result.get("state"); return result
            if path == "/sessions/open-saved":
                result = summarize_saved_session(str(payload.get("session_path") or payload.get("path") or ""))
                session.saved_session_path = result.get("session_path")
                session.recording_info = result.get("recording_info")
                session.review_events = result.get("review_events") or []
                session.review_events_by_run = result.get("review_events_by_run") or {}
                session.review_run_filters = result.get("review_run_filters") or {}
                session.review_active_run_id = result.get("active_run_id")
                session.review_accepted_run_id = result.get("accepted_run_id")
                session.review_visible_run_ids = result.get("visible_run_ids") or []
                session.review_cursor_event_id = result.get("cursor_event_id")
                session.last_detection = {
                    "status": "loaded" if result.get("detected") else "not_detected",
                    "biomarker": result.get("biomarker"),
                    "detector": "saved-session",
                    "num_events": result.get("event_count"),
                    "num_channels": (result.get("recording_info") or {}).get("n_channels"),
                    "artifacts": {"session_path": result.get("session_path"), "format": result.get("format")},
                    "request": {"filter": (result.get("review_run_filters") or {}).get(result.get("active_run_id") or "") or {}},
                    "warnings": ["Loaded saved-session summary with run filter context; full PyQt backend object restore remains a compatibility slice."],
                }
                return result
            if path == "/projects/save":
                result = save_project(payload); session.project_state = result.get("state"); return result
            if path == "/batch/create":
                result = create_batch(str(payload.get("input_dir") or payload.get("input") or ""), payload.get("output_dir") or payload.get("output"), recursive=bool(payload.get("recursive", True))); session.last_batch = result; return result
            if path == "/batch/run":
                result = run_pipeline_isolated("run_batch", payload); session.last_batch = result; return result
            if path == "/recordings/load":
                return session.dispatch("load_recording", payload)
            if path == "/recordings/roi":
                return session.dispatch("infer_brain_signal_roi", payload)
            if path == "/recordings/qc":
                return session.dispatch("compute_channel_qc", payload)
            if path == "/recordings/spectral":
                return session.dispatch("compute_spectral_profile", payload)
            if path == "/recordings/line-noise":
                return detect_line_noise({**payload, "input_path": payload.get("input_path") or (session.recording_info or {}).get("input_path")})
            if path == "/recordings/prepare":
                return session.dispatch("prepare_recording", payload)
            if path == "/detections/plan":
                return session.dispatch("recommend_detector_plan", payload)
            if path == "/detections/run":
                return session.dispatch("run_detection", payload)
            if path == "/severity/run":
                return session.dispatch("run_severity_scoring", payload)
            if path == "/severity/cancel":
                from pyhfo.agent_api.subprocess_runner import cancel_pipeline
                cancelled = cancel_pipeline("run_severity_scoring")
                return {"status": "cancelled" if cancelled else "not_running"}
            if path == "/severity/adopt":
                # A severity result computed OUTSIDE this session (the agent runs the scorer in its
                # own process). Storing it here is what lets the desktop panel show a score it did
                # not run itself -- otherwise the number exists only in the agent transcript.
                result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
                if not isinstance(result, dict) or result.get("status") != "complete":
                    return {"status": "rejected", "reason": "expected a completed severity result"}
                from pyhfo.agent_api.sidecar import _severity_cache_key
                session.severity_by_recording[_severity_cache_key(result.get("input_path"))] = result
                # Only show it when it belongs to the recording on screen; the cache keeps it either way.
                if not session.recording_info or _severity_cache_key(session.recording_info.get("input_path")) == _severity_cache_key(result.get("input_path")):
                    session.last_severity = result
                return {"status": "ok", "num_segments": result.get("num_segments")}
            if path == "/detections/rank-candidates":
                return session.dispatch("rank_candidate_channels", payload)
            if path == "/parameters/recommend":
                result = recommend_parameters({**payload, "recording_info": payload.get("recording_info") or session.recording_info or {}}); session.parameter_plan_history.append(result); return result
            if path == "/parameters/detector-defaults":
                return detector_defaults({**payload, "recording_info": payload.get("recording_info") or session.recording_info or {}})
            if path == "/parameters/validate":
                return validate_parameters({**payload, "recording_info": payload.get("recording_info") or session.recording_info or {}})
            if path == "/parameters/sweep":
                result = sweep_parameters({**payload, "recording_info": payload.get("recording_info") or session.recording_info or {}}); session.parameter_plan_history.append(result); return result
            if path == "/events/snippet":
                return event_snippet(session, str(payload.get("event_id") or ""), input_path=payload.get("input_path") or (session.recording_info or {}).get("input_path"), padding_seconds=float(payload.get("padding_seconds", 1.0)), target_samples_per_channel=int(payload.get("target_samples_per_channel", 600)), events=payload.get("events"), run_id=payload.get("run_id"))
            if path == "/events/feature-summary":
                return event_feature_summary(session, str(payload.get("event_id") or ""), events=payload.get("events"), run_id=payload.get("run_id"))
            if path.startswith("/events/") and path.endswith("/annotation"):
                event_id = path.split("/")[2]
                return annotate_event(session, event_id, label=str(payload.get("label")), notes=payload.get("notes"), reviewer=str(payload.get("reviewer", "human")), source=str(payload.get("source", "user")), confirmed=(payload.get("confirmed") is True), display_label=payload.get("display_label"), events=payload.get("events"), run_id=payload.get("run_id"))
            if path == "/events/cursor":
                return event_cursor(session, action=str(payload.get("action", "current")), event_id=payload.get("event_id"), events=payload.get("events"), run_id=payload.get("run_id"))
            if path == "/events/overlap-review":
                return overlap_review(session, events=payload.get("events"), tolerance_seconds=float(payload.get("tolerance_seconds", 0.0)), run_id=payload.get("run_id"))
            if path == "/runs/compare":
                return compare_detector_runs(session, events_by_run=payload.get("events_by_run"), run_ids=payload.get("run_ids"), tolerance_seconds=float(payload.get("tolerance_seconds", 0.0)), max_overlap_examples=int(payload.get("max_overlap_examples", 100)))
            if path == "/runs/consensus":
                return create_consensus_run(session, events_by_run=payload.get("events_by_run"), run_ids=payload.get("run_ids"), strategy=str(payload.get("strategy", "majority")), biomarker=payload.get("biomarker"))
            if path == "/runs/classify":
                result = run_pipeline_isolated("classify_existing_run", payload)
                session.last_detection = {**result, "request": json_safe(payload)} if isinstance(result, dict) else result
                return result
            if path == "/exports/report":
                return export_report({**payload, "review_state": payload.get("review_state") or export_review_state(session)})
            if path == "/exports/session":
                return export_session({**payload, "session_state": payload.get("session_state") or session.get_state()})
            if path == "/exports/workbook":
                return export_workbook({**payload, "review_state": payload.get("review_state") or export_review_state(session)})
            if path == "/agent/chat":
                return submit_chat_job(payload, session)
            raise PyHFOAgentError("not_found", f"Unknown endpoint: {path}")

        def do_GET(self):
            try:
                if not self._authorized():
                    raise PyHFOAgentError("unauthorized", "Missing or invalid backend authorization token.")
                self._send_json(self._handle_get(urlparse(self.path).path, _query(self.path)))
            except Exception as exc:
                self._send_json({"status": "error", "error": error_response(exc)}, _status_for_error(exc))

        def do_POST(self):
            try:
                if not self._authorized():
                    raise PyHFOAgentError("unauthorized", "Missing or invalid backend authorization token.")
                self._send_json(self._handle_post(urlparse(self.path).path, self._payload()))
            except Exception as exc:
                self._send_json({"status": "error", "error": error_response(exc)}, _status_for_error(exc))

    return Handler


class _LocalThreadingHTTPServer(ThreadingHTTPServer):
    # ThreadingHTTPServer inherits socketserver.TCPServer's tiny default listen
    # backlog of 5. Rapid waveform scrolling fires bursts of concurrent
    # /waveform/window connections; with only 5 queued, the kernel resets the
    # overflow and the renderer's fetch() fails before any response. A larger
    # backlog absorbs the burst while the accept loop drains it. daemon_threads
    # keeps in-flight handlers from blocking shutdown.
    daemon_threads = True
    request_queue_size = 128


def build_server(host: str = "127.0.0.1", port: int = 8765, session: PyHFOSidecarSession | None = None) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost"}:
        raise PyHFOAgentError("unsafe_bind_host", "PyHFO backend binds to 127.0.0.1/localhost only by default.")
    return _LocalThreadingHTTPServer((host, port), make_handler(session or PyHFOSidecarSession()))


def main(argv: list[str] | None = None) -> int:
    # The real server runs heavy pipeline calls (detection/classify/batch) in a
    # subprocess so they can't GIL-starve latency-sensitive endpoints like
    # /waveform/window. Opt-in here (not in build_server) so tests/CLI stay inline
    # by default; override with PYHFO_PIPELINE_SUBPROCESS=0 to force inline.
    # On in a packaged build too, now that the frozen app can act as its own pipeline worker. It used
    # to be forced off when frozen, which quietly cost the release the two things subprocess mode
    # exists for: a responsive UI during a long run, and a Cancel button that actually stops one.
    os.environ.setdefault("PYHFO_PIPELINE_SUBPROCESS", "1")
    parser = argparse.ArgumentParser(description="Local PyHFO HTTP agent API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = build_server(args.host, args.port)
    address = server.server_address
    host, port = str(address[0]), int(address[1])
    print(json.dumps({"status": "ready", "url": f"http://{host}:{port}", "local_only": True}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
