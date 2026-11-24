from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import uuid
from importlib.resources import files
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from pyhfo.paths import pyhfo_home
from pyhfo.agent_api.errors import PyHFOAgentError, error_response
from pyhfo.agent_api.serialization import json_safe, utc_now_iso
from pyhfo.agent_api.sidecar import PyHFOSidecarSession, get_capability_manifest

DEFAULT_AGENT_CHAT_URL = "http://127.0.0.1:8001/internal/agent/v1/chat"
AGENT_CONTRACT_VERSION = "ieeg-agent.v1"
DEFAULT_AGENT_SKILL: str | None = None
DEFAULT_AGENT_ALLOWED_SKILLS = [
    "eeg-format-understanding",
    "eeg-signal-roi",
    "eeg-qc-cleaning",
    "eeg-preprocessing",
    "pyhfo-detection",
    "eeg-result-interpretation",
    "eeg-iteration-optimization",
    "eeg-clinical-label-evaluation",
]

WRITE_RISKS = {"compute", "artifact_write", "review_write", "destructive"}
UI_COMMAND_CATALOG_PATH = files("pyhfo").joinpath("resources", "ui-command-catalog.json")


def _load_ui_command_catalog() -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(UI_COMMAND_CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {
        str(entry["type"]): entry
        for entry in data.get("commands", [])
        if isinstance(entry, dict) and isinstance(entry.get("type"), str) and entry.get("type")
    }


AGENT_UI_COMMAND_CATALOG = _load_ui_command_catalog()
AGENT_UI_COMMAND_TYPES = set(AGENT_UI_COMMAND_CATALOG)
AGENT_UI_COMMANDS_REQUIRING_CONFIRMATION = {
    command_type
    for command_type, spec in AGENT_UI_COMMAND_CATALOG.items()
    if bool(spec.get("requires_confirmation")) or spec.get("risk") in WRITE_RISKS
}


_CHAT_JOBS: dict[str, dict[str, Any]] = {}
_CHAT_JOBS_LOCK = threading.Lock()
_CHAT_JOB_TTL_SECONDS = 1800.0


def _prune_chat_jobs() -> None:
    now = time.time()
    with _CHAT_JOBS_LOCK:
        stale = [job_id for job_id, job in _CHAT_JOBS.items() if now - float(job.get("created_at") or now) > _CHAT_JOB_TTL_SECONDS]
        for job_id in stale:
            _CHAT_JOBS.pop(job_id, None)


def submit_chat_job(payload: dict[str, Any], session: PyHFOSidecarSession) -> dict[str, Any]:
    """Run chat in the background so the browser can poll progress."""

    _prune_chat_jobs()
    job_id = f"chat-{uuid.uuid4().hex}"
    now = time.time()
    with _CHAT_JOBS_LOCK:
        _CHAT_JOBS[job_id] = {"job_id": job_id, "status": "running", "created_at": now, "updated_at": now}

    def run() -> None:
        try:
            result = handle_chat(payload, session)
            status = "complete"
            update = {"result": result}
        except Exception as exc:  # pragma: no cover - defensive status reporting
            status = "error"
            update = {"error": error_response(exc)}
        with _CHAT_JOBS_LOCK:
            job = _CHAT_JOBS.get(job_id)
            if job is not None:
                job.update(update)
                job["status"] = status
                job["updated_at"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return {"status": "running", "job_id": job_id}


def get_chat_job(job_id: str) -> dict[str, Any]:
    _prune_chat_jobs()
    with _CHAT_JOBS_LOCK:
        job = dict(_CHAT_JOBS.get(job_id) or {})
    if not job:
        raise PyHFOAgentError("job_not_found", f"Unknown chat job: {job_id}")
    elapsed = max(0.0, time.time() - float(job.get("created_at") or time.time()))
    payload = {"job_id": job_id, "status": job.get("status", "unknown"), "elapsed_seconds": round(elapsed, 1)}
    if job.get("status") == "complete":
        payload["result"] = job.get("result")
    elif job.get("status") == "error":
        payload["error"] = job.get("error")
    return payload


def _coerce_ui_command(raw: Any, *, fallback_reason: str, session_id: str | None = None) -> dict[str, Any] | None:
    """Normalize agent-supplied UI commands into the renderer command contract.

    The backend never executes these commands. It only allowlists and normalizes
    them so the Electron renderer can validate against live UI state, apply safe
    view commands, and keep persistent/review writes staged for doctor approval.
    """

    if not isinstance(raw, dict):
        return None
    command_type = str(raw.get("type") or "").strip()
    if command_type not in AGENT_UI_COMMAND_TYPES:
        return None
    payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else {}
    requires_confirmation = raw.get("requires_confirmation")
    if command_type in AGENT_UI_COMMANDS_REQUIRING_CONFIRMATION:
        requires_confirmation = True
    elif requires_confirmation is None:
        requires_confirmation = False
    command = {
        "command_id": str(raw.get("command_id") or f"agent-cmd-{uuid.uuid4().hex}"),
        "type": command_type,
        "reason": str(raw.get("reason") or fallback_reason or command_type),
        "payload": json_safe(payload),
        "requires_confirmation": bool(requires_confirmation),
        "created_at": str(raw.get("created_at") or utc_now_iso()),
    }
    command_session_id = raw.get("session_id") or session_id
    if command_session_id:
        command["session_id"] = str(command_session_id)
    return command


def _agent_response_ui_command_candidates(agent_response: dict[str, Any]) -> list[Any]:
    candidates: list[Any] = []
    for key in ("ui_commands", "commands"):
        value = agent_response.get(key)
        if isinstance(value, list):
            candidates.extend(value)
    nested = agent_response.get("result")
    if isinstance(nested, dict):
        for key in ("ui_commands", "commands"):
            value = nested.get(key)
            if isinstance(value, list):
                candidates.extend(value)
    return candidates


def extract_agent_ui_commands(
    agent_response: dict[str, Any],
    *,
    fallback_reason: str,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    commands = [
        command
        for command in (
            _coerce_ui_command(candidate, fallback_reason=fallback_reason, session_id=session_id)
            for candidate in _agent_response_ui_command_candidates(agent_response)
        )
        if command is not None
    ]
    if commands:
        return commands
    content = str(agent_response.get("content") or "").strip()
    if not content:
        return []
    trace = _coerce_ui_command(
        {
            "type": "agent.show_trace_step",
            "reason": fallback_reason or "Agent response received",
            "payload": {"panel": "Chat", "message": content[:2000]},
            "requires_confirmation": False,
        },
        fallback_reason=fallback_reason or "Agent response received",
        session_id=session_id,
    )
    return [trace] if trace else []


def labels_enabled() -> bool:
    """Gate all BIDS/clinical-label exposure for the doctor-side NLP panel."""

    return (os.getenv("PYHFO_AGENT_ENABLE_LABEL_LOADING") or "").strip().lower() in {"1", "true", "yes", "on"}

HTML = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>PyHFO Agent Side Panel</title>
  <style>
    :root { color-scheme: light dark; }
    body { font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.4; }
    textarea, input, button { font: inherit; width: 100%; box-sizing: border-box; margin: .25rem 0 .75rem; }
    textarea { min-height: 7rem; }
    button { cursor: pointer; padding: .55rem .75rem; }
    pre { background: #111827; color: #e5e7eb; padding: 1rem; border-radius: .5rem; overflow: auto; max-height: 38rem; }
    .grid { display: grid; grid-template-columns: minmax(320px, 1fr) minmax(380px, 1.2fr); gap: 1rem; align-items: start; }
    .card { border: 1px solid #9ca3af55; border-radius: .75rem; padding: 1rem; }
    .primary { background: #2563eb; color: white; border: 0; border-radius: .4rem; }
    .muted { color: #6b7280; font-size: .92rem; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: .75rem; }
  </style>
</head>
<body>
  <h1>PyHFO Agent Side Panel</h1>
  <p class=\"muted\">Natural language goes to <strong>ieeg-agent</strong>. Explicit state/actions stay in the decoupled <strong>PyHFO sidecar</strong>. Research support only; not clinical diagnosis. Answers should include a visible decision trace: evidence, rule/metadata interpretation, conclusion, and uncertainty.</p>
  <div class=\"grid\">
    <section class=\"card\">
      <h2>NLP Agent Chat</h2>
      <label>Current/target recording path <input id=\"recordingPath\" placeholder=\"/path/to/recording.edf\" /></label>
      <label>Conversation id <input id=\"conversationId\" value=\"pyhfo-side-panel\" /></label>
      <label>Ask the agent <textarea id=\"message\">Use the current real iEEG recording. Inspect metadata, decide if it is already bipolar, prepare it safely, and include a visible decision trace with evidence and uncertainty.</textarea></label>
      <button class=\"primary\" onclick=\"sendChat()\">Send to ieeg-agent</button>
      <div class=\"row\">
        <button onclick=\"loadManifest()\">Load manifest</button>
        <button onclick=\"loadState()\">Load state</button>
      </div>
      <p class=\"muted\">To enable chat, configure <code>PYHFO_AGENT_SERVICE_TOKEN</code> or <code>AGENT_SERVICE_TOKEN</code> for your ieeg-agent service. Default chat URL: <code>http://127.0.0.1:8001/internal/agent/v1/chat</code>.</p>
    </section>
    <section class=\"card\">
      <h2>Agent Result + Visible Trace</h2>
      <pre id=\"out\">Ready.</pre>
    </section>
    <section class=\"card\">
      <h2>Direct Sidecar Action</h2>
      <label>Action name <input id=\"action\" value=\"get_state\" /></label>
      <label>Arguments JSON <textarea id=\"args\">{}</textarea></label>
      <button onclick=\"callAction()\">Dispatch action</button>
    </section>
    <section class=\"card\">
      <h2>Suggested real-data prompt</h2>
      <pre>Load this real EDF, determine whether channels are already bipolar, avoid unsafe notch assumptions if line frequency is missing, prepare a cleaned FIF, then run a Nyquist-safe HFO smoke test only if metadata supports it.</pre>
    </section>
  </div>
  <script>
    const out = document.getElementById('out');
    const show = x => { out.textContent = typeof x === 'string' ? x : JSON.stringify(x, null, 2); };
    const showAgentResult = result => {
      if(result && result.content){
        out.textContent = result.content + "\\n\\n--- details ---\\n" + JSON.stringify({
          active_skill: result.active_skill,
          model_used: result.model_used,
          sidecar_state: result.sidecar_state
        }, null, 2);
      } else {
        show(result);
      }
    };
    async function parseResponse(resp){
      const body = await resp.json().catch(async () => ({status:'error', raw: await resp.text()}));
      if(!resp.ok) body.http_status = resp.status;
      return body;
    }
    async function loadManifest(){ show(await parseResponse(await fetch('/manifest'))); }
    async function loadState(){ show(await parseResponse(await fetch('/state'))); }
    async function callAction(){
      const action = document.getElementById('action').value;
      const args = JSON.parse(document.getElementById('args').value || '{}');
      show(await parseResponse(await fetch('/action', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify({action, arguments: args})})));
    }
    let activeChatJob = null;
    async function pollChatJob(jobId, startedAt){
      const body = await parseResponse(await fetch('/chat-result?job_id=' + encodeURIComponent(jobId)));
      const elapsed = Math.round((Date.now() - startedAt) / 1000);
      if(body.status === 'running'){
        show({status: 'thinking', job_id: jobId, elapsed_seconds: elapsed, note: 'The agent is still working. You can keep this tab open; results will appear here automatically.'});
        setTimeout(() => pollChatJob(jobId, startedAt), 2000);
        return;
      }
      activeChatJob = null;
      showAgentResult(body.result || body);
    }
    async function sendChat(){
      if(activeChatJob){ show({status: 'thinking', job_id: activeChatJob, note: 'A request is already running; waiting for its result.'}); return; }
      const payload = {
        message: document.getElementById('message').value,
        conversation_id: document.getElementById('conversationId').value || ('pyhfo-side-panel-' + Date.now()),
        recording_path: document.getElementById('recordingPath').value || undefined
      };
      show({status: 'submitting', note: 'Starting background ieeg-agent request...'});
      const submitted = await parseResponse(await fetch('/chat-async', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(payload)}));
      if(!submitted.job_id){ show(submitted); return; }
      activeChatJob = submitted.job_id;
      show({status: 'thinking', job_id: activeChatJob, elapsed_seconds: 0});
      pollChatJob(activeChatJob, Date.now());
    }
  </script>
</body>
</html>
"""


def agent_chat_url() -> str:
    return os.getenv("PYHFO_AGENT_CHAT_URL") or os.getenv("IEEG_AGENT_CHAT_URL") or DEFAULT_AGENT_CHAT_URL


def agent_service_token() -> str | None:
    return os.getenv("PYHFO_AGENT_SERVICE_TOKEN") or os.getenv("AGENT_SERVICE_TOKEN")


def agent_chat_timeout_seconds() -> float:
    raw = os.getenv("PYHFO_AGENT_CHAT_TIMEOUT_SECONDS") or os.getenv("IEEG_AGENT_CHAT_TIMEOUT_SECONDS")
    if not raw:
        return 300.0
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 300.0



def _sidecar_output_root() -> Path:
    configured = os.getenv("PYHFO_SIDE_PANEL_OUTPUT_DIR") or os.getenv("PYHFO_AGENT_OUTPUT_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return pyhfo_home() / "runs"


def _safe_slug(value: str) -> str:
    stem = Path(str(value or "recording")).stem or "recording"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", stem)[:80] or "recording"


def _requests_candidate_signal_analysis(message: str) -> bool:
    lowered = " ".join(str(message or "").lower().split())
    candidate_terms = [
        "pathological channel",
        "pathological candidate",
        "abnormal channel",
        "candidate channel",
        "epileptiform",
        "hfo count",
        "spike count",
        "event count",
        "病理",
        "异常通道",
        "候选通道",
    ]
    return any(term in lowered for term in candidate_terms)


def _detection_runs_requested(message: str) -> bool:
    lowered = " ".join(str(message or "").lower().split())
    return _requests_candidate_signal_analysis(lowered) or any(
        term in lowered
        for term in ["run detection", "detect", "hfo", "spike", "ripple", "count events", "事件", "检测"]
    )


def _runs_from_detection_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs = []
    for summary in summaries:
        artifacts = summary.get("artifacts") if isinstance(summary, dict) else {}
        csv_path = (artifacts or {}).get("csv")
        if csv_path:
            runs.append({"biomarker": summary.get("biomarker") or "event", "events_csv": csv_path})
    return runs


def _compact_candidate_analysis(analysis: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the compact, answer-relevant candidate evidence for the chat agent."""

    if not analysis:
        return None
    ranking = analysis.get("candidate_ranking") or {}
    detector_runs = []
    for summary in analysis.get("detection_summaries") or []:
        detector_runs.append(
            {
                "biomarker": summary.get("biomarker"),
                "detector": summary.get("detector"),
                "status": summary.get("status"),
                "num_events": summary.get("num_events"),
                "num_channels": summary.get("num_channels"),
                "artifacts": summary.get("artifacts"),
            }
        )
    return {
        "status": analysis.get("status") or ranking.get("status"),
        "interpretation_policy": ranking.get("interpretation_policy")
        or "Rankings are signal-derived algorithmic candidates; they are not clinical labels or diagnosis.",
        "candidate_channel_count": ranking.get("candidate_channel_count"),
        "duration_seconds": ranking.get("duration_seconds"),
        "excluded_channels": ranking.get("excluded_channels"),
        "top_candidate_channels": (ranking.get("top_channels") or [])[:12],
        "detector_runs": detector_runs,
        "artifact_paths": ranking.get("artifact_paths"),
        "limits": analysis.get("interpretation_limits"),
    }


def run_signal_candidate_analysis(session: PyHFOSidecarSession, recording_path: str, message: str) -> dict[str, Any]:
    """Build deterministic signal-derived evidence before the LLM writes the answer.

    This prevents the chat agent from stopping at metadata/ROI for prompts like
    "pathological channels". The analysis is still label-free and algorithmic.
    """

    if not recording_path:
        raise PyHFOAgentError("invalid_request", "recording_path is required for signal candidate analysis")
    state = session.get_state()
    if state.get("loaded_recording_path") != recording_path:
        session.dispatch("load_recording", {"input_path": recording_path})
    if not session.get_state().get("brain_signal_roi_loaded"):
        session.dispatch("infer_brain_signal_roi", {"input_path": recording_path})
    state = session.get_state()
    info = state.get("recording_info") or {}
    roi = state.get("brain_signal_roi") or {}
    exclude_channels = list(roi.get("default_detection_exclusions") or [])

    qc = session.dispatch(
        "compute_channel_qc",
        {"input_path": recording_path, "exclude_channels": exclude_channels, "outlier_z_threshold": 8.0},
    )["result"]
    spectral = session.dispatch(
        "compute_spectral_profile",
        {"input_path": recording_path, "exclude_channels": exclude_channels, "max_channels": 32},
    )["result"]
    plan = session.dispatch(
        "recommend_detector_plan",
        {"input_path": recording_path, "recording_info": info, "roi": roi},
    )["result"]

    detection_summaries: list[dict[str, Any]] = []
    ranking = None
    if _detection_runs_requested(message):
        run_root = _sidecar_output_root() / _safe_slug(recording_path) / time.strftime("%Y%m%d-%H%M%S")
        run_root.mkdir(parents=True, exist_ok=True)
        # Keep automatic UI runs bounded: execute only the first HFO and first Spike plan.
        seen: set[str] = set()
        for run in plan.get("recommended_runs") or []:
            biomarker = str(run.get("biomarker") or "").strip() or "event"
            key = biomarker.lower()
            if key in seen or key not in {"hfo", "spike"}:
                continue
            seen.add(key)
            output_dir = run_root / key
            result = session.dispatch(
                "run_detection",
                {
                    "input_path": recording_path,
                    "output_dir": str(output_dir),
                    "biomarker": biomarker,
                    "detector": run.get("detector"),
                    "filter": run.get("filter") or {},
                    "n_jobs": int(run.get("n_jobs") or 1),
                    "exports": run.get("exports") or ["summary-json", "csv"],
                    "include_channels": roi.get("brain_signal_channels") or [],
                    "exclude_channels": exclude_channels,
                },
            )["result"]
            detection_summaries.append(result)
        runs = _runs_from_detection_summaries(detection_summaries)
        if runs:
            ranking = session.dispatch(
                "rank_candidate_channels",
                {
                    "runs": runs,
                    "duration_seconds": info.get("duration_seconds") or 0,
                    "exclude_channels": exclude_channels,
                    "top_n": 20,
                },
            )["result"]

    analysis = json_safe(
        {
            "status": "complete",
            "policy": "Signal-derived algorithmic evidence only; clinical/BIDS labels were not loaded.",
            "recording_path": recording_path,
            "recording_info": {
                "sample_freq": info.get("sample_freq"),
                "duration_seconds": info.get("duration_seconds"),
                "n_channels": info.get("n_channels"),
                "line_freq": info.get("line_freq"),
            },
            "roi": roi,
            "qc_summary": {
                "roi_channel_count": qc.get("roi_channel_count"),
                "flat_channels": qc.get("flat_channels"),
                "outlier_channels": qc.get("outlier_channels"),
                "excluded_channels": qc.get("excluded_channels"),
                "warnings": qc.get("warnings"),
            },
            "spectral_summary": {
                "nyquist_hz": spectral.get("nyquist_hz"),
                "recommended_hfo_band": spectral.get("recommended_hfo_band"),
                "line_noise_candidates": spectral.get("line_noise_candidates"),
            },
            "detector_plan": plan,
            "detection_summaries": detection_summaries,
            "candidate_ranking": ranking,
            "interpretation_limits": [
                "These are algorithmic signal-derived candidate channels, not clinical pathological labels.",
                "Visual review and expert interpretation are required before clinical use.",
            ],
        }
    )
    session.candidate_analysis = analysis
    return analysis

def build_agent_message(
    user_message: str,
    session: PyHFOSidecarSession,
    recording_path: str | None = None,
    *,
    include_label_context: bool = False,
) -> str:
    state = session.get_state()
    manifest = get_capability_manifest(include_missing=True)
    selected_path = recording_path or state.get("loaded_recording_path") or state.get("prepared_recording_path")
    labels_included = bool(include_label_context and labels_enabled() and state.get("bids_channel_annotations"))
    candidate_summary = _compact_candidate_analysis(state.get("candidate_analysis"))
    context = {
        "selected_recording_path": selected_path,
        "loaded_recording_path": state.get("loaded_recording_path"),
        "recording_info": state.get("recording_info"),
        "brain_signal_roi_loaded": state.get("brain_signal_roi_loaded"),
        "brain_signal_roi": state.get("brain_signal_roi"),
        "prepared_recording_path": state.get("prepared_recording_path"),
        "last_detection_summary": state.get("last_detection_summary"),
        "candidate_analysis_summary": candidate_summary,
        "recommended_pipeline": manifest.get("recommended_pipeline"),
        "clinical_label_context": "included" if labels_included else "not_available_for_ordinary_edf_chat",
    }
    if labels_included:
        context["bids_channel_annotations"] = state.get("bids_channel_annotations")
    return (
        "You are helping with EEG/iEEG research workflow support from a PyHFO side panel. "
        "Use the session data below as ordinary user-visible recording context. "
        "For EDF-only abnormal/pathological-channel questions, do not use clinical labels; first use real brain-signal ROI, "
        "then QC/spectral/detector evidence when available. "
        "If candidate_analysis_summary.status is complete, answer from top_candidate_channels and detector_runs; "
        "do not answer with ROI-only language such as 'no definitive pathological channels were identified at this stage'. "
        "Instead say these are signal-derived algorithmic candidate channels, not clinical pathology labels. "
        "Do not give diagnosis, treatment advice, or definitive clinical localization. "
        "Reply with an audit summary using these headings when useful: Evidence read, interpretation rule, conclusion/action, uncertainty/limits.\n\n"
        f"Session data:\n{json.dumps(json_safe(context), ensure_ascii=False, indent=2)}\n\n"
        f"Doctor request:\n{user_message}"
    )


def build_agent_payload(
    user_message: str,
    session: PyHFOSidecarSession,
    *,
    conversation_id: str | None = None,
    recording_path: str | None = None,
    include_label_context: bool = False,
    ui_state: dict | None = None,
) -> dict[str, Any]:
    request_id = f"pyhfo-side-panel-{uuid.uuid4().hex}"
    active_skill = DEFAULT_AGENT_SKILL
    if _requests_candidate_signal_analysis(user_message) and session.get_state().get("candidate_analysis"):
        active_skill = "eeg-result-interpretation"
    return {
        "request_id": request_id,
        "conversation_id": conversation_id or "pyhfo-side-panel",
        "message": build_agent_message(user_message, session, recording_path, include_label_context=include_label_context),
        "knowledge_base_id": None,
        "use_search": False,
        # Co-pilot turns must reliably emit tool calls (emit_ui_command / get_ui_state).
        # qwen-turbo and qwen3.5-flash are too weak — in live tests they reply text-only or
        # claim the tool doesn't exist, so the agent never drives the UI. qwen3.5-plus does
        # it correctly. Override per-deploy with PYHFO_AGENT_MODEL / IEEG_AGENT_MODEL.
        "model": os.getenv("PYHFO_AGENT_MODEL") or os.getenv("IEEG_AGENT_MODEL") or "qwen3.5-plus",
        # Do not pin the legacy eeg-ieeg-pyhfo router/alias. The ieeg-agent is
        # a ReAct agent; the sidecar only whitelists focused EEG/iEEG skills so
        # bootstrap can pick the right step from the user request and context.
        "active_skill": active_skill,
        "skill_environment": "prod",
        "response_mode": "default",
        "user_context": {
            "user_id": "pyhfo-side-panel",
            "username": "doctor-side-panel",
            "role": "USER",
            "tenant_id": "default",
            "team_ids": [],
            "allowed_mcp_servers": None,
        },
        "auth_context": {"permissions": [], "session_id": "pyhfo-side-panel"},
        "allowed_skills": [{"name": name, "config_ref": "prod"} for name in DEFAULT_AGENT_ALLOWED_SKILLS],
        "skill_recommendation_candidates": [
            {
                "name": "eeg-format-understanding",
                "score": 0.95,
                "reason": "Start by reading recording metadata and constraints.",
            },
            {
                "name": "eeg-signal-roi",
                "score": 0.9,
                "reason": "EDF-only pathology/abnormal-channel requests need label-free brain-signal ROI first.",
            },
            {
                "name": "pyhfo-detection",
                "score": 0.85,
                "reason": "Use Nyquist-safe Spike/HFO detection after metadata, ROI, and QC support it.",
            },
            {
                "name": "eeg-result-interpretation",
                "score": 0.8,
                "reason": "Report signal-derived candidate channels and uncertainty after tool outputs exist.",
            },
        ],
        "allowed_mcps": [],
        # Forward the desktop UI snapshot so the agent can read what the clinician is viewing and
        # target emit_ui_command at the right channel/event/time (co-pilot in-seam).
        "ui_state": ui_state or None,
        "request_metadata": {"source": "pyhfo-side-panel", "request_id": request_id, "client_version": "sidecar-web.v1"},
    }


def call_ieeg_agent(payload: dict[str, Any], *, timeout_seconds: float | None = None) -> dict[str, Any]:
    token = agent_service_token()
    if not token:
        raise PyHFOAgentError(
            "agent_token_missing",
            "Set PYHFO_AGENT_SERVICE_TOKEN or AGENT_SERVICE_TOKEN before using the NLP chat panel.",
        )
    request = Request(
        agent_chat_url(),
        data=json.dumps(json_safe(payload)).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {token}",
            "X-Contract-Version": AGENT_CONTRACT_VERSION,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds or agent_chat_timeout_seconds()) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise PyHFOAgentError("agent_http_error", f"ieeg-agent returned HTTP {exc.code}. Check the chat URL and service token.") from exc
    except (URLError, TimeoutError) as exc:
        raise PyHFOAgentError(
            "agent_unavailable",
            "Cannot reach the chat service. Start ieeg-agent or set PYHFO_AGENT_CHAT_URL to its chat endpoint.",
        ) from exc


def handle_chat(payload: dict[str, Any], session: PyHFOSidecarSession) -> dict[str, Any]:
    message = str(payload.get("message") or "").strip()
    if not message:
        raise PyHFOAgentError("invalid_request", "message is required")
    recording_path = str(payload.get("recording_path") or "").strip() or None
    if recording_path and session.get_state().get("loaded_recording_path") != recording_path:
        # Make the NLP panel context evidence-backed for the selected file before
        # asking the generic agent to reason about it. This loads signal metadata
        # only; BIDS ground-truth labels remain behind an explicit action.
        session.dispatch("load_recording", {"input_path": recording_path})
    current_state = session.get_state()
    if recording_path and (not current_state.get("brain_signal_roi_loaded") or (current_state.get("brain_signal_roi") or {}).get("input_path") != recording_path):
        session.dispatch("infer_brain_signal_roi", {"input_path": recording_path})
    if recording_path and _requests_candidate_signal_analysis(message):
        run_signal_candidate_analysis(session, recording_path, message)
    label_requested = bool(labels_enabled() and _requests_labels(message))
    # Label loading is disabled by default. The NLP panel should stay EDF/signal-only
    # unless a future evaluation run explicitly starts the sidecar with
    # PYHFO_AGENT_ENABLE_LABEL_LOADING=1.
    if label_requested:
        session.dispatch("load_bids_channel_annotations", {"input_path": recording_path})
    ui_state = payload.get("ui_state") if isinstance(payload.get("ui_state"), dict) else {}
    agent_payload = build_agent_payload(
        message,
        session,
        conversation_id=payload.get("conversation_id"),
        recording_path=recording_path,
        include_label_context=label_requested,
        ui_state=ui_state,
    )
    agent_response = call_ieeg_agent(agent_payload)
    ui_commands = extract_agent_ui_commands(
        agent_response,
        fallback_reason=message[:120] or "Agent response received",
        session_id=str(ui_state.get("session_id") or "") or None,
    )
    return json_safe(
        {
            "status": "complete",
            "agent_chat_url": agent_chat_url(),
            "conversation_id": agent_payload["conversation_id"],
            "request_id": agent_payload["request_id"],
            "content": agent_response.get("content"),
            "active_skill": agent_response.get("active_skill") or agent_response.get("used_skill"),
            "model_used": agent_response.get("model_used"),
            "ui_commands": ui_commands,
            "raw_agent_response": agent_response,
            "sidecar_state": _chat_response_state(session, include_label_context=label_requested),
        }
    )


def _chat_response_state(session: PyHFOSidecarSession, *, include_label_context: bool) -> dict[str, Any]:
    """Return chat state without leaking label details into non-label replies."""

    state = session.get_state()
    include = bool(labels_enabled() and include_label_context and state.get("bids_channel_annotations"))
    state["bids_channel_annotations_context_included"] = include
    state["label_loading_enabled"] = labels_enabled()
    if not include:
        state["bids_channel_annotations"] = None
    return state


def _requests_labels(message: str) -> bool:
    """Return true only when label loading is explicitly enabled and requested.

    Default product behavior is signal-only: no BIDS/clinical labels are loaded
    from chat. This helper exists only for future evaluation runs started with
    PYHFO_AGENT_ENABLE_LABEL_LOADING=1.
    """

    if not labels_enabled():
        return False

    import re

    lowered = " ".join(str(message or "").lower().split())
    label_family_pattern = r"(bids label|channels\.tsv|channel tsv|sidecar label|metadata label|ground[- ]truth|soz|seizure onset|resection|resected|annotation)"
    if re.search(r"(do not|don't|dont|without|no|not|avoid|exclude).{0,160}" + label_family_pattern, lowered):
        return False
    negative_markers = [
        "do not read",
        "do not read or load",
        "don't read",
        "do not load",
        "don't load",
        "do not use",
        "don't use",
        "without",
        "no ",
        "not ",
        "avoid",
        "exclude",
    ]
    label_terms = [
        "channels.tsv",
        "channel tsv",
        "bids label",
        "bids labels",
        "bids annotation",
        "bids annotations",
        "metadata label",
        "metadata labels",
        "sidecar label",
        "sidecar labels",
        "ground truth",
        "ground-truth",
        "soz",
        "seizure onset",
        "resection",
        "resected",
        "annotation",
        "annotations",
    ]
    for term in label_terms:
        index = lowered.find(term)
        if index < 0:
            continue
        window = lowered[max(0, index - 48): index]
        if any(marker in window for marker in negative_markers):
            continue
        return True
    return False


def make_handler(session: PyHFOSidecarSession):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: Any, status: int = 200):
            body = json.dumps(json_safe(payload), indent=2, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self):
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *args):
            return

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path in {"/", "/index.html"}:
                    self._send_html()
                elif path == "/manifest":
                    self._send_json(get_capability_manifest())
                elif path == "/state":
                    self._send_json(session.get_state())
                elif path == "/chat-result":
                    query = parse_qs(urlparse(self.path).query)
                    self._send_json(get_chat_job(str((query.get("job_id") or [""])[0])))
                else:
                    self._send_json({"status": "error", "error": {"code": "not_found", "message": path}}, 404)
            except Exception as exc:
                self._send_json({"status": "error", "error": error_response(exc)}, 500)

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                length = int(self.headers.get("content-length") or 0)
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                if path == "/action":
                    self._send_json(session.dispatch(payload.get("action"), payload.get("arguments") or {}))
                    return
                if path == "/chat":
                    self._send_json(handle_chat(payload, session))
                    return
                if path == "/chat-async":
                    self._send_json(submit_chat_job(payload, session), 202)
                    return
                self._send_json({"status": "error", "error": {"code": "not_found", "message": path}}, 404)
            except Exception as exc:
                self._send_json({"status": "error", "error": error_response(exc)}, 400)

    return Handler


def build_server(host: str = "127.0.0.1", port: int = 8765, session: PyHFOSidecarSession | None = None) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(session or PyHFOSidecarSession()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Minimal HTTP control surface for the PyHFO sidecar and ieeg-agent chat bridge.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = build_server(args.host, args.port)
    print(json.dumps({"status": "ready", "url": f"http://{args.host}:{args.port}/", "agent_chat_url": agent_chat_url()}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
