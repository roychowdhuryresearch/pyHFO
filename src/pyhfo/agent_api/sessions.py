from __future__ import annotations

import base64
import csv
import html as html_lib
import json
import os
import re
import uuid
import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from pyhfo.paths import export_root, pyhfo_home
from pyhfo.agent_api.errors import PyHFOAgentError
from pyhfo.agent_api.serialization import file_fingerprint, json_safe, utc_now_iso, write_json
from pyhfo.utils.app_state import checkpoint_array, checkpoint_get, checkpoint_version
from pyhfo.utils.analysis_session import AnalysisSession, _feature_array, _feature_event_rows, review_event_id
from pyhfo.utils.session_store import SESSION_FORMAT_VERSION, is_session_path, load_session_checkpoint, save_session_checkpoint


def _safe_id(value: str | None) -> str:
    raw = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value or "project").strip()).strip(".-_")
    return raw[:120] or "project"


def project_root(base_dir: str | Path | None = None) -> Path:
    configured = os.getenv("PYHFO_AGENT_PROJECT_DIR") or pyhfo_home() / "projects"
    root = Path(configured).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def create_project(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(payload or {})
    project_id = _safe_id(payload.get("project_id") or f"project-{uuid.uuid4().hex[:10]}")
    base = project_root()
    root = (base / project_id).resolve()
    if base not in [root, *root.parents]:
        raise PyHFOAgentError("invalid_project_path", f"Project path must stay under {base}")
    root.mkdir(parents=True, exist_ok=True)
    # The caller's own workspace state comes FIRST and is preserved wholesale. Saving a project used
    # to drop it on the first save (only later saves, which carry a project_path, went through
    # save_project) -- so the very first save wrote a skeleton with recording_path=null and the
    # project reopened empty: no recording, no runs, no model results.
    caller_state = dict(payload.get("state") or {})
    state = {
        **caller_state,
        "schema_version": caller_state.get("schema_version") or "pyhfo-project.v1",
        "project_id": project_id,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "recording_path": caller_state.get("recording_path") or payload.get("recording_path"),
        "derived_artifacts": caller_state.get("derived_artifacts") or payload.get("derived_artifacts") or {},
        "detection_runs": caller_state.get("detection_runs") or payload.get("detection_runs") or [],
        "review_labels": caller_state.get("review_labels") or payload.get("review_labels") or {},
        "parameter_plan_history": caller_state.get("parameter_plan_history") or payload.get("parameter_plan_history") or [],
        "agent_conversation_id": caller_state.get("agent_conversation_id") or payload.get("agent_conversation_id"),
        "action_log": caller_state.get("action_log") or payload.get("action_log") or [],
    }
    path = write_json(root / "project.json", state)
    return json_safe({"status": "complete", "project_id": project_id, "project_path": str(path), "state": state})


def _safe_project_file(path: str | Path) -> Path:
    root = project_root()
    p = Path(str(path)).expanduser()
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    if p.is_dir():
        p = p / "project.json"
    # A desktop recent-project entry can still point to its pre-4.0 location.
    # An explicit project root or PyHFO home always scopes access to the new root.
    legacy_existing_project = (
        not os.getenv("PYHFO_HOME")
        and not os.getenv("PYHFO_AGENT_PROJECT_DIR")
        and p.name == "project.json"
        and p.is_relative_to(Path.home() / ".pyhfo_agent_projects")
        and p.is_file()
    )
    if root not in [p, *p.parents] and not legacy_existing_project:
        raise PyHFOAgentError("invalid_project_path", f"Project path must stay under {root}")
    return p


def open_project(path: str | Path) -> dict[str, Any]:
    p = _safe_project_file(path)
    if not p.exists():
        raise PyHFOAgentError("file_not_found", f"Project file does not exist: {p}")
    return json_safe({"status": "complete", "project_path": str(p), "state": json.loads(p.read_text(encoding="utf-8"))})


def _analysis_session_from_payload(payload: Any) -> AnalysisSession | None:
    if not isinstance(payload, dict):
        return None
    try:
        return AnalysisSession.from_dict(payload)
    except Exception:
        return None


def _visible_prediction_array(event_features: Any, attr: str, predicted_flag_attr: str | None = None):
    """Return a classifier-prediction array aligned with the VISIBLE event rows
    that _feature_event_rows yields (it reads channel/starts/ends through
    _feature_array -> get_visible_array, which applies the overlap-hide visibility
    mask). Reading predictions through the same _feature_array path keeps per-event
    indexes aligned even when later-overlap events are hidden. Returns None when the
    prediction has not been produced, matching PyQt get_prediction_scope_options,
    which only exposes a scope when artifact_predicted/spike_predicted/etc. is set.
    """
    if predicted_flag_attr is not None and not getattr(event_features, predicted_flag_attr, False):
        return None
    array = _feature_array(event_features, attr)
    return array if len(array) else None


def _review_events_for_run(run: Any) -> list[dict[str, Any]]:
    if run is None or getattr(run, "event_features", None) is None:
        return []
    events = []
    event_features = run.event_features
    # Surface the classifier-prediction columns PyQt uses for review scoping
    # (hfo_feature.get_prediction_scope_options / _matches_prediction_scope) so the
    # Electron Event Review dialog can offer prediction-based scopes (Artifact /
    # Non-artifact / spkHFO / eHFO / Overlap tagged) on real recordings. Raw values
    # pass through unchanged; the Electron side applies the <1 / >0 / ==1 thresholds.
    artifact_preds = _visible_prediction_array(event_features, "artifact_predictions", "artifact_predicted")
    spike_preds = _visible_prediction_array(event_features, "spike_predictions", "spike_predicted")
    ehfo_preds = _visible_prediction_array(event_features, "ehfo_predictions", "ehfo_predicted")
    overlap_tags = _visible_prediction_array(event_features, "overlap_tagged")
    for idx, row in enumerate(_feature_event_rows(event_features)):
        event_id = review_event_id(run.run_id, idx)
        features: dict[str, Any] = {"start_index": row.get("start_index"), "end_index": row.get("end_index")}
        if artifact_preds is not None and idx < len(artifact_preds):
            features["artifact_prediction"] = float(artifact_preds[idx])
        if spike_preds is not None and idx < len(spike_preds):
            features["spike_prediction"] = float(spike_preds[idx])
        if ehfo_preds is not None and idx < len(ehfo_preds):
            features["ehfo_prediction"] = float(ehfo_preds[idx])
        if overlap_tags is not None and idx < len(overlap_tags):
            overlap_tag = int(overlap_tags[idx])
            if overlap_tag > 0:
                # Only present when actually tagged (PyQt overlap_tagged > 0). Absence,
                # not a zero value, signals "no overlap" so the Electron scope's
                # features-text overlap check cannot false-positive on the key name.
                features["overlap_tagged"] = overlap_tag
        events.append({
            "event_id": event_id,
            "run_id": run.run_id,
            "index": idx,
            "channel": row.get("channel_name"),
            "start_seconds": row.get("start_seconds"),
            "end_seconds": row.get("end_seconds"),
            "duration_seconds": max(0.0, float(row.get("end_seconds") or 0.0) - float(row.get("start_seconds") or 0.0)),
            "biomarker": run.biomarker_type,
            "source": "analysis_session",
            "features": features,
        })
    return json_safe(events)


def _review_events_by_run_from_analysis_session(payload: Any) -> dict[str, list[dict[str, Any]]]:
    analysis_session = _analysis_session_from_payload(payload)
    if analysis_session is None:
        return {}
    return {run.run_id: _review_events_for_run(run) for run in analysis_session.list_runs()}


def _review_events_from_analysis_session(payload: Any) -> list[dict[str, Any]]:
    analysis_session = _analysis_session_from_payload(payload)
    if analysis_session is None:
        return []
    return _review_events_for_run(analysis_session.get_active_run())


def _run_records_from_analysis_session(payload: Any) -> list[dict[str, Any]]:
    analysis_session = _analysis_session_from_payload(payload)
    if analysis_session is None:
        return []
    records = []
    events_by_run = _review_events_by_run_from_analysis_session(payload)
    for run in analysis_session.list_runs():
        records.append({
            "runId": run.run_id,
            "run_id": run.run_id,
            "biomarker": run.biomarker_type,
            "detector": run.detector_name,
            "events": len(events_by_run.get(run.run_id, [])),
            "channels": len(run.selected_channels or []),
            "status": "loaded",
            "classifier": "saved classifier" if run.classified else "not classified",
            "createdAt": run.created_at,
            "parameters": {"display_name": run.display_name, "summary": run.summary},
        })
    return json_safe(records)

def _run_filter_params_from_analysis_session(payload: Any) -> dict[str, dict[str, Any]]:
    analysis_session = _analysis_session_from_payload(payload)
    if analysis_session is None:
        return {}
    filters: dict[str, dict[str, Any]] = {}
    for run in analysis_session.list_runs():
        if getattr(run, "param_filter", None) is not None:
            try:
                filters[str(run.run_id)] = dict(run.param_filter.to_dict())
            except Exception:
                continue
    return json_safe(filters)



def _review_cursor_from_analysis_session(payload: Any, events: list[dict[str, Any]]) -> str | None:
    if not events:
        return None
    if isinstance(payload, dict):
        cursor = payload.get("review_cursor_event_id")
        if cursor and any(event.get("event_id") == cursor for event in events):
            return str(cursor)
    return str(events[0]["event_id"])


def _array_length(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(len(value))
    except TypeError:
        return 0


def _session_event_count(checkpoint: Any, biomarker: str) -> int:
    candidates = {
        "HFO": "HFOs",
        "Spindle": "Spindles",
        "Spike": "Spikes",
    }
    primary = candidates.get(str(biomarker), "HFOs")
    for key in (primary, "HFOs", "Spindles", "Spikes"):
        count = _array_length(checkpoint_array(checkpoint, key, None))
        if count:
            return count
    return 0


def summarize_saved_session(path: str | Path) -> dict[str, Any]:
    p = Path(str(path)).expanduser().resolve()
    if not is_session_path(p):
        raise PyHFOAgentError("invalid_project_path", "Saved sessions must be .pybrain or .npz files.")
    if not p.exists():
        raise PyHFOAgentError("file_not_found", f"Saved session does not exist: {p}")

    checkpoint = load_session_checkpoint(p)
    biomarker = str(checkpoint_get(checkpoint, "biomarker_type", "HFO") or "HFO")
    edf_param = checkpoint_get(checkpoint, "edf_param", {}) or {}
    recording_path = str(edf_param.get("edf_fn") or "")
    channel_names = checkpoint_array(checkpoint, "recording_channel_names", None)
    if channel_names is None:
        channel_names = checkpoint_array(checkpoint, "channel_names", [])
    sample_frequency = checkpoint_get(checkpoint, "sample_freq", None)
    eeg_data = checkpoint_array(checkpoint, "eeg_data", None)
    duration_seconds = None
    try:
        if eeg_data is not None and sample_frequency:
            duration_seconds = float(eeg_data.shape[1]) / float(sample_frequency)
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        duration_seconds = None

    analysis_session_payload = checkpoint_get(checkpoint, "analysis_session", None)
    review_events = _review_events_from_analysis_session(analysis_session_payload)
    review_events_by_run = _review_events_by_run_from_analysis_session(analysis_session_payload)
    detection_runs = _run_records_from_analysis_session(analysis_session_payload)
    review_run_filters = _run_filter_params_from_analysis_session(analysis_session_payload)
    cursor_event_id = _review_cursor_from_analysis_session(analysis_session_payload, review_events)
    active_run_id = analysis_session_payload.get("active_run_id") if isinstance(analysis_session_payload, dict) else None
    accepted_run_id = analysis_session_payload.get("accepted_run_id") if isinstance(analysis_session_payload, dict) else None
    visible_run_ids = analysis_session_payload.get("visible_run_ids", []) if isinstance(analysis_session_payload, dict) else []

    if not detection_runs:
        detection_runs = checkpoint_get(checkpoint, "detection_runs", None) or checkpoint_get(checkpoint, "run_records", []) or []
    if not review_events:
        review_events = checkpoint_get(checkpoint, "review_events", []) or []
    if not review_events_by_run:
        review_events_by_run = checkpoint_get(checkpoint, "review_events_by_run", {}) or {}
    if not review_run_filters:
        review_run_filters = checkpoint_get(checkpoint, "review_run_filters", {}) or {}
    cursor_event_id = cursor_event_id or checkpoint_get(checkpoint, "review_cursor_event_id", None) or (review_events[0].get("event_id") if review_events and isinstance(review_events[0], dict) else None)
    active_run_id = active_run_id or checkpoint_get(checkpoint, "active_run_id", None)
    accepted_run_id = accepted_run_id or checkpoint_get(checkpoint, "accepted_run_id", None)
    visible_run_ids = visible_run_ids or checkpoint_get(checkpoint, "visible_run_ids", []) or []

    return json_safe(
        {
            "status": "complete",
            "session_path": str(p),
            "format": "pybrain-session" if p.suffix == ".pybrain" else "legacy-npz",
            "state_version": checkpoint_version(checkpoint),
            "software_version": checkpoint_get(checkpoint, "software_version"),
            "biomarker": biomarker,
            "recording_path": recording_path,
            "recording_info": {
                "input_path": recording_path,
                "sample_freq": sample_frequency,
                "sampling_frequency": sample_frequency,
                "sample_frequency_hz": sample_frequency,
                "n_channels": _array_length(channel_names),
                "channel_names": list(channel_names) if channel_names is not None else [],
                "duration_seconds": duration_seconds,
            },
            "n_jobs": checkpoint_get(checkpoint, "n_jobs", None),
            "filtered": bool(checkpoint_get(checkpoint, "filtered", False)),
            "detected": bool(checkpoint_get(checkpoint, "detected", False)),
            "classified": bool(checkpoint_get(checkpoint, "classified", False)),
            "event_count": len(review_events) or _session_event_count(checkpoint, biomarker),
            "detection_runs": detection_runs,
            "active_run_id": active_run_id,
            "accepted_run_id": accepted_run_id,
            "visible_run_ids": visible_run_ids,
            "review_events": review_events,
            "review_events_by_run": review_events_by_run,
            "review_run_filters": review_run_filters,
            "review_annotations": checkpoint_get(checkpoint, "review_annotations", {}) or {},
            "cursor_event_id": cursor_event_id,
            "prepare_controls": checkpoint_get(checkpoint, "prepare_controls", {}) or {},
            "classifier_controls": checkpoint_get(checkpoint, "classifier_controls", {}) or {},
            "analysis_session": analysis_session_payload,
        }
    )


def save_project(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload or {})
    path = payload.get("project_path")
    if not path:
        created = create_project(payload)
        fingerprint = file_fingerprint(created["project_path"])
        return json_safe({
            **created,
            "artifact_exists": True,
            "artifact_sha256": fingerprint["sha256"],
        })
    p = _safe_project_file(path)
    state = dict(payload.get("state") or {})
    if not state:
        state = {k: v for k, v in payload.items() if k not in {"project_path", "base_dir"}}
    state.setdefault("schema_version", "pyhfo-project.v1")
    state["updated_at"] = utc_now_iso()
    p.parent.mkdir(parents=True, exist_ok=True)
    write_json(p, state)
    fingerprint = file_fingerprint(p)
    return json_safe({
        "status": "complete",
        "project_path": str(p),
        "state": state,
        "artifact_exists": True,
        "artifact_sha256": fingerprint["sha256"],
    })


def _export_root() -> Path:
    root = export_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_export_path(value: Any, default_name: str, allowed_suffixes: set[str], *, allow_absolute_path: bool = False) -> Path:
    root = _export_root()
    raw = str(value or default_name)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    if not allow_absolute_path and root not in [resolved, *resolved.parents]:
        raise PyHFOAgentError("invalid_export_path", f"Export path must stay under {root}")
    if resolved.suffix.lower() not in allowed_suffixes:
        raise PyHFOAgentError("invalid_export_path", f"Export path must end with one of {sorted(allowed_suffixes)}")
    return resolved


def export_session(payload: dict[str, Any]) -> dict[str, Any]:
    output_path = _safe_export_path(payload.get("output_path"), "session-export.pybrain", {".json", ".pybrain", ".npz"}, allow_absolute_path=bool(payload.get("allow_absolute_path")))
    session_state = dict(payload.get("session_state") or {})
    session_state.setdefault("exported_at", utc_now_iso())
    if output_path.suffix.lower() in {".pybrain", ".npz"}:
        session_state.setdefault("state_version", SESSION_FORMAT_VERSION)
        session_state.setdefault("biomarker_type", session_state.get("biomarker") or "HFO")
        recording_info = session_state.get("recording_info") if isinstance(session_state.get("recording_info"), dict) else {}
        recording_path = session_state.get("recording_path") or recording_info.get("input_path") or recording_info.get("filename")
        session_state.setdefault("edf_param", {"edf_fn": recording_path})
        session_state.setdefault("detection_runs", session_state.get("run_records") or [])
        save_session_checkpoint(output_path, session_state)
        fingerprint = file_fingerprint(output_path)
        return json_safe({
            "status": "complete",
            "artifact_path": str(output_path),
            "artifact_type": "pybrain_session" if output_path.suffix.lower() == ".pybrain" else "npz_session",
            "artifact_exists": True,
            "artifact_sha256": fingerprint["sha256"],
        })
    write_json(output_path, session_state)
    fingerprint = file_fingerprint(output_path)
    return json_safe({
        "status": "complete",
        "artifact_path": str(output_path),
        "artifact_type": "session_json",
        "artifact_exists": True,
        "artifact_sha256": fingerprint["sha256"],
    })


def _annotation_display_value(annotation: Any) -> str:
    if isinstance(annotation, dict):
        return str(annotation.get("display_label") or annotation.get("label") or "")
    return str(annotation or "")


def _annotation_rows(review_state: Any) -> list[dict[str, Any]]:
    if not isinstance(review_state, dict):
        return []
    annotations = review_state.get("annotations")
    if not isinstance(annotations, dict):
        return []
    rows: list[dict[str, Any]] = []
    for event_id, annotation in annotations.items():
        if isinstance(annotation, dict):
            rows.append({
                "event_id": event_id,
                "label": annotation.get("label"),
                "display_label": annotation.get("display_label") or annotation.get("label"),
                "reviewer": annotation.get("reviewer"),
                "source": annotation.get("source"),
                "notes": annotation.get("notes"),
                "updated_at": annotation.get("updated_at"),
            })
        else:
            rows.append({"event_id": event_id, "label": annotation, "display_label": _annotation_display_value(annotation)})
    return rows


def _write_report_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "key", "value"])
        for key, value in (report.get("summary") or {}).items():
            writer.writerow(["summary", key, json.dumps(json_safe(value), ensure_ascii=False) if isinstance(value, (dict, list)) else value])
        for key, value in (report.get("artifacts") or {}).items():
            writer.writerow(["artifact", key, value])
        for limit in report.get("limits") or []:
            writer.writerow(["limit", "message", limit])
        run_rows = _report_run_rows(report)
        if run_rows:
            writer.writerow([])
            writer.writerow(["run_id", "biomarker", "detector", "events", "channels", "status", "classifier", "created_at", "active", "selected", "accepted", "visible"])
            for row in run_rows:
                writer.writerow([row.get("run_id"), row.get("biomarker"), row.get("detector"), row.get("events"), row.get("channels"), row.get("status"), row.get("classifier"), row.get("created_at"), row.get("active"), row.get("selected"), row.get("accepted"), row.get("visible")])
        channel_rows = _report_channel_ranking_rows(report)
        if channel_rows:
            writer.writerow([])
            writer.writerow(["channel", "events", "accepted", "total", "priority"])
            for row in channel_rows:
                writer.writerow([row.get("channel"), row.get("events"), row.get("accepted"), row.get("total"), row.get("priority")])
        event_rows = _report_event_rows(report)
        if event_rows:
            writer.writerow([])
            writer.writerow(["event_id", "run_id", "channel", "start_seconds", "end_seconds", "duration_seconds", "biomarker", "annotation"])
            for row in event_rows:
                writer.writerow([row.get("event_id"), row.get("run_id"), row.get("channel"), row.get("start_seconds"), row.get("end_seconds"), row.get("duration_seconds"), row.get("biomarker"), row.get("annotation")])
        annotations = report.get("review_annotations") or []
        if annotations:
            writer.writerow([])
            writer.writerow(["event_id", "label", "display_label", "reviewer", "source", "notes", "updated_at"])
            for row in annotations:
                writer.writerow([row.get("event_id"), row.get("label"), row.get("display_label"), row.get("reviewer"), row.get("source"), row.get("notes"), row.get("updated_at")])


def _html_cell(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(json_safe(value), ensure_ascii=False)
    return html_lib.escape("" if value is None else str(value))


def _html_table(headers: list[str], rows: list[list[Any]], empty_message: str) -> str:
    if not rows:
        return f'<p class="empty">{html_lib.escape(empty_message)}</p>'
    header_html = "".join(f"<th>{html_lib.escape(header)}</th>" for header in headers)
    row_html = []
    for row in rows:
        row_html.append("<tr>" + "".join(f"<td>{_html_cell(value)}</td>" for value in row) + "</tr>")
    return f'<table class="report-table"><thead><tr>{header_html}</tr></thead><tbody>{"".join(row_html)}</tbody></table>'


def _write_dict_rows_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row.get(header) for header in headers])


def _event_annotation_label(event: dict[str, Any]) -> Any:
    annotation = event.get("annotation")
    if isinstance(annotation, dict):
        return annotation.get("display_label") or annotation.get("label")
    return annotation


def _report_event_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    review_state = report.get("review_state") if isinstance(report.get("review_state"), dict) else {}
    events = review_state.get("events") if isinstance(review_state, dict) else []
    if not isinstance(events, list):
        return []
    rows: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        rows.append({
            "event_id": event.get("event_id"),
            "run_id": event.get("run_id"),
            "channel": event.get("channel"),
            "start_seconds": event.get("start_seconds"),
            "end_seconds": event.get("end_seconds"),
            "duration_seconds": event.get("duration_seconds"),
            "biomarker": event.get("biomarker"),
            "annotation": _event_annotation_label(event),
        })
    return rows


def _report_run_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    runs = report.get("runs") if isinstance(report.get("runs"), list) else []
    accepted_run_id = str(report.get("accepted_run_id") or "")
    active_run = report.get("active_run") if isinstance(report.get("active_run"), dict) else {}
    selected_run = report.get("selected_run") if isinstance(report.get("selected_run"), dict) else {}
    active_run_id = str(active_run.get("runId") or active_run.get("run_id") or report.get("active_run_id") or "")
    selected_run_id = str(selected_run.get("runId") or selected_run.get("run_id") or report.get("selected_run_id") or "")
    visible_ids = {str(item) for item in (report.get("visible_run_ids") if isinstance(report.get("visible_run_ids"), list) else [])}
    rows: list[dict[str, Any]] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        run_id = str(run.get("runId") or run.get("run_id") or "")
        rows.append({
            "run_id": run_id,
            "biomarker": run.get("biomarker"),
            "detector": run.get("detector"),
            "events": run.get("events"),
            "channels": run.get("channels"),
            "status": run.get("status"),
            "classifier": run.get("classifier"),
            "created_at": run.get("createdAt") or run.get("created_at"),
            "active": run_id == active_run_id,
            "selected": run_id == selected_run_id,
            "accepted": run_id == accepted_run_id,
            "visible": (run_id in visible_ids) if visible_ids else True,
        })
    return rows


def _report_channel_ranking_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    ranking = report.get("channel_ranking") if isinstance(report.get("channel_ranking"), list) else []
    rows: list[dict[str, Any]] = []
    for row in ranking:
        if not isinstance(row, dict):
            continue
        rows.append({
            "channel": row.get("channel"),
            "events": row.get("events"),
            "accepted": row.get("accepted"),
            "total": row.get("total"),
            "priority": row.get("priority"),
        })
    return rows


def _report_runtime_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    runtime_log = report.get("runtime_log") if isinstance(report.get("runtime_log"), list) else []
    rows: list[dict[str, Any]] = []
    for entry in runtime_log:
        if not isinstance(entry, dict):
            continue
        rows.append({
            "timestamp": entry.get("timestamp") or entry.get("created_at") or entry.get("time"),
            "level": entry.get("level") or entry.get("status") or entry.get("type"),
            "message": entry.get("message") or entry.get("detail") or entry.get("label"),
        })
    return rows


def _decode_png_data_url(value: Any) -> bytes | None:
    match = re.match(r"^data:image/png;base64,([A-Za-z0-9+/=]+)$", str(value or ""))
    if not match:
        return None
    try:
        data = base64.b64decode(match.group(1), validate=True)
    except Exception:
        return None
    png_signature = b"\x89PNG\r\n\x1a\n"
    return data if data.startswith(png_signature) else None


def _report_metadata_payload(report: dict[str, Any]) -> dict[str, Any]:
    payload = dict(report)
    snapshot = payload.get("waveform_snapshot")
    if isinstance(snapshot, dict) and snapshot.get("data_url"):
        payload["waveform_snapshot"] = {k: v for k, v in snapshot.items() if k != "data_url"}
        payload["waveform_snapshot"]["data_url_omitted"] = True
    return payload


def _write_report_package_assets(path: Path, report: dict[str, Any]) -> dict[str, str]:
    assets_dir = path.with_name(f"{path.stem}_files")
    assets_dir.mkdir(parents=True, exist_ok=True)
    links: dict[str, str] = {}

    snapshot = report.get("waveform_snapshot") if isinstance(report.get("waveform_snapshot"), dict) else {}
    snapshot_data = _decode_png_data_url(snapshot.get("data_url") if isinstance(snapshot, dict) else None)
    if snapshot_data:
        snapshot_name = str(snapshot.get("filename") or "waveform_snapshot.png").replace("/", "-").replace("\\", "-")
        if not snapshot_name.lower().endswith(".png"):
            snapshot_name = f"{snapshot_name}.png"
        snapshot_path = assets_dir / snapshot_name
        snapshot_path.write_bytes(snapshot_data)
        links["Waveform snapshot"] = f"{assets_dir.name}/{snapshot_path.name}"

    metadata_path = assets_dir / "metadata.json"
    metadata_path.write_text(json.dumps(json_safe(_report_metadata_payload(report)), indent=2, ensure_ascii=False), encoding="utf-8")
    links["Metadata JSON"] = f"{assets_dir.name}/{metadata_path.name}"

    summary_path = assets_dir / "summary.json"
    summary_path.write_text(json.dumps(json_safe(report.get("summary") or {}), indent=2, ensure_ascii=False), encoding="utf-8")
    links["Summary JSON"] = f"{assets_dir.name}/{summary_path.name}"

    artifacts_path = assets_dir / "artifacts.json"
    artifacts_path.write_text(json.dumps(json_safe(report.get("artifacts") or {}), indent=2, ensure_ascii=False), encoding="utf-8")
    links["Artifacts JSON"] = f"{assets_dir.name}/{artifacts_path.name}"

    runs = report.get("runs") if isinstance(report.get("runs"), list) else []
    if runs:
        runs_path = assets_dir / "runs.json"
        runs_path.write_text(json.dumps(json_safe(runs), indent=2, ensure_ascii=False), encoding="utf-8")
        links["Run inventory JSON"] = f"{assets_dir.name}/{runs_path.name}"

    channel_rows = _report_channel_ranking_rows(report)
    if channel_rows:
        channel_path = assets_dir / "channel_ranking.csv"
        _write_dict_rows_csv(channel_path, ["channel", "events", "accepted", "total", "priority"], channel_rows)
        links["Channel ranking CSV"] = f"{assets_dir.name}/{channel_path.name}"

    event_rows = _report_event_rows(report)
    if event_rows:
        event_path = assets_dir / "events_preview.csv"
        _write_dict_rows_csv(event_path, ["event_id", "run_id", "channel", "start_seconds", "end_seconds", "duration_seconds", "biomarker", "annotation"], event_rows)
        links["Event preview CSV"] = f"{assets_dir.name}/{event_path.name}"

    runtime_log = report.get("runtime_log") if isinstance(report.get("runtime_log"), list) else []
    if runtime_log:
        runtime_path = assets_dir / "runtime_log.json"
        runtime_path.write_text(json.dumps(json_safe(runtime_log), indent=2, ensure_ascii=False), encoding="utf-8")
        links["Runtime log JSON"] = f"{assets_dir.name}/{runtime_path.name}"

    annotations = report.get("review_annotations") or []
    if annotations:
        annotations_path = assets_dir / "review_annotations.csv"
        with annotations_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["event_id", "label", "display_label", "reviewer", "source", "notes", "updated_at"])
            for row in annotations:
                writer.writerow([row.get("event_id"), row.get("label"), row.get("display_label"), row.get("reviewer"), row.get("source"), row.get("notes"), row.get("updated_at")])
        links["Review annotations CSV"] = f"{assets_dir.name}/{annotations_path.name}"

    return links


def _write_report_html(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact_links = _write_report_package_assets(path, report)
    summary_rows = [[key, value] for key, value in (report.get("summary") or {}).items()]
    artifact_rows = [[key, value] for key, value in (report.get("artifacts") or {}).items()]
    run_rows = _report_run_rows(report)
    channel_rows = _report_channel_ranking_rows(report)
    event_rows = _report_event_rows(report)
    runtime_rows = _report_runtime_rows(report)
    annotation_rows = [[row.get("event_id"), row.get("label"), row.get("display_label"), row.get("reviewer"), row.get("source"), row.get("notes"), row.get("updated_at")] for row in (report.get("review_annotations") or [])]
    limit_items = "".join(f"<li>{_html_cell(limit)}</li>" for limit in (report.get("limits") or []))
    package_items = "".join(f'<li><a href="{html_lib.escape(relpath)}">{html_lib.escape(label)}</a></li>' for label, relpath in artifact_links.items())
    waveform_relpath = artifact_links.get("Waveform snapshot")
    waveform_block = f'<section><h2>Waveform Snapshot</h2><img class="waveform-image" src="{html_lib.escape(waveform_relpath)}" alt="Current waveform snapshot"></section>' if waveform_relpath else ""
    path.write_text(f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PyBrain Analysis Report</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #0f172a; background: #eef2f7; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 32px; }}
    header, section {{ background: #fff; border: 1px solid #cbd5e1; border-radius: 16px; padding: 20px; margin-bottom: 18px; box-shadow: 0 12px 28px rgba(15, 23, 42, .06); }}
    h1, h2 {{ margin: 0 0 10px; }}
    .meta {{ color: #475569; }}
    .report-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    .report-table th, .report-table td {{ border-bottom: 1px solid #e2e8f0; padding: 8px 10px; text-align: left; vertical-align: top; }}
    .report-table th {{ background: #f8fafc; color: #334155; }}
    .waveform-image {{ max-width: 100%; border: 1px solid #cbd5e1; border-radius: 12px; background: #020617; }}
    .empty {{ color: #64748b; }}
    .warning {{ color: #92400e; }}
  </style>
</head>
<body>
<main>
  <header>
    <p class="meta">PyBrain export package</p>
    <h1>Analysis Report</h1>
    <p class="meta">Created: {_html_cell(report.get("created_at"))} · Schema: {_html_cell(report.get("schema_version"))}</p>
  </header>
  <section>
    <h2>Summary</h2>
    {_html_table(["Metric", "Value"], summary_rows, "No summary values were provided.")}
  </section>
  <section>
    <h2>Artifacts</h2>
    {_html_table(["Artifact", "Path"], artifact_rows, "No artifacts were provided.")}
  </section>
  <section>
    <h2>Report Package Assets</h2>
    <p class="meta">These files travel with the HTML report, matching the PyQt report-package pattern.</p>
    <ul>{package_items or "<li>No package assets were produced.</li>"}</ul>
  </section>
  {waveform_block}
  <section>
    <h2>Run Inventory</h2>
    {_html_table(["Run ID", "Biomarker", "Detector", "Events", "Channels", "Status", "Classifier", "Created At", "Active", "Selected", "Accepted", "Visible"], [[row.get("run_id"), row.get("biomarker"), row.get("detector"), row.get("events"), row.get("channels"), row.get("status"), row.get("classifier"), row.get("created_at"), row.get("active"), row.get("selected"), row.get("accepted"), row.get("visible")] for row in run_rows], "No run inventory was provided.")}
  </section>
  <section>
    <h2>Channel Ranking</h2>
    {_html_table(["Channel", "Events", "Accepted", "Total", "Priority"], [[row.get("channel"), row.get("events"), row.get("accepted"), row.get("total"), row.get("priority")] for row in channel_rows], "No channel ranking was provided.")}
  </section>
  <section>
    <h2>Event Preview</h2>
    {_html_table(["Event ID", "Run ID", "Channel", "Start", "End", "Duration", "Biomarker", "Annotation"], [[row.get("event_id"), row.get("run_id"), row.get("channel"), row.get("start_seconds"), row.get("end_seconds"), row.get("duration_seconds"), row.get("biomarker"), row.get("annotation")] for row in event_rows[:500]], "No event preview rows were provided.")}
  </section>
  <section>
    <h2>Review Annotations</h2>
    {_html_table(["Event ID", "Backend Label", "Display Label", "Reviewer", "Source", "Notes", "Updated At"], annotation_rows, "No reviewed annotations were provided.")}
  </section>
  <section>
    <h2>Runtime Log</h2>
    {_html_table(["Timestamp", "Level", "Message"], [[row.get("timestamp"), row.get("level"), row.get("message")] for row in runtime_rows[-80:]], "No runtime log entries were provided.")}
  </section>
  <section>
    <h2>Export Notes</h2>
    <ul class="warning">{limit_items or "<li>No export notes.</li>"}</ul>
  </section>
</main>
</body>
</html>
""", encoding="utf-8")


def export_report(payload: dict[str, Any]) -> dict[str, Any]:
    output_path = _safe_export_path(payload.get("output_path"), "report.html", {".json", ".csv", ".html", ".htm"}, allow_absolute_path=bool(payload.get("allow_absolute_path")))
    review_state = payload.get("review_state") or {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    report = {
        "schema_version": "pyhfo-report.v1",
        "created_at": utc_now_iso(),
        "summary": summary,
        "recording_info": payload.get("recording_info") or {},
        "runs": payload.get("runs") or [],
        "active_run": payload.get("active_run") or {},
        "selected_run": payload.get("selected_run") or {},
        "active_run_id": payload.get("active_run_id") or summary.get("active_run_id"),
        "selected_run_id": payload.get("selected_run_id") or summary.get("selected_run_id"),
        "accepted_run_id": payload.get("accepted_run_id"),
        "visible_run_ids": payload.get("visible_run_ids") or [],
        "channel_ranking": payload.get("channel_ranking") or [],
        "review_state": review_state,
        "review_annotations": _annotation_rows(review_state),
        "waveform_snapshot": payload.get("waveform_snapshot") or {},
        "artifacts": payload.get("artifacts") or {},
        "runtime_log": payload.get("runtime_log") or [],
        "limits": ["Signal-derived candidates are not clinical diagnoses."],
    }
    if output_path.suffix.lower() == ".csv":
        _write_report_csv(output_path, report)
        fingerprint = file_fingerprint(output_path)
        return json_safe({"status": "complete", "artifact_path": str(output_path), "artifact_type": "report_csv", "report": report, "artifact_exists": True, "artifact_sha256": fingerprint["sha256"]})
    if output_path.suffix.lower() in {".html", ".htm"}:
        _write_report_html(output_path, report)
        fingerprint = file_fingerprint(output_path)
        return json_safe({"status": "complete", "artifact_path": str(output_path), "artifact_type": "report_html", "report": report, "artifact_exists": True, "artifact_sha256": fingerprint["sha256"]})
    write_json(output_path, report)
    fingerprint = file_fingerprint(output_path)
    return json_safe({"status": "complete", "artifact_path": str(output_path), "artifact_type": "report_json", "report": report, "artifact_exists": True, "artifact_sha256": fingerprint["sha256"]})


def _xlsx_cell_ref(row_index: int, col_index: int) -> str:
    letters = ""
    col = col_index
    while col:
        col, rem = divmod(col - 1, 26)
        letters = chr(65 + rem) + letters
    return f"{letters}{row_index}"


def _xlsx_sheet_xml(rows: list[list[Any]]) -> str:
    row_xml: list[str] = []
    for row_index, row in enumerate(rows, 1):
        cells = []
        for col_index, value in enumerate(row, 1):
            ref = _xlsx_cell_ref(row_index, col_index)
            if value is None:
                text = ""
            elif isinstance(value, (dict, list, tuple)):
                text = json.dumps(json_safe(value), ensure_ascii=False)
            else:
                text = str(value)
            cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(text)}</t></is></c>')
        row_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>' + "".join(row_xml) + "</sheetData></worksheet>"


def _table_rows(items: Any) -> list[list[Any]]:
    if not isinstance(items, list) or not items:
        return [["No rows"]]
    dict_items = [item for item in items if isinstance(item, dict)]
    if not dict_items:
        return [["Value"], *[[item] for item in items]]
    headers: list[str] = []
    for item in dict_items:
        for key in item.keys():
            if key not in headers:
                headers.append(str(key))
    return [headers, *[[item.get(key) for key in headers] for item in dict_items]]


def _summary_rows(summary: Any) -> list[list[Any]]:
    if not isinstance(summary, dict) or not summary:
        return [["Metric", "Value"], ["status", "no summary"]]
    return [["Metric", "Value"], *[[key, value] for key, value in summary.items()]]


def _review_rows(review_state: Any) -> list[list[Any]]:
    if not isinstance(review_state, dict):
        return [["Metric", "Value"], ["status", "no review state"]]
    rows = [["Metric", "Value"]]
    for key in ("cursor_event_id", "review_message"):
        rows.append([key, review_state.get(key)])
    annotation_rows = _annotation_rows(review_state)
    if annotation_rows:
        rows.append(["annotations", len(annotation_rows)])
        rows.append(["Event ID", "Label", "Display Label", "Reviewer", "Source", "Notes", "Updated At"])
        rows.extend([[row.get("event_id"), row.get("label"), row.get("display_label"), row.get("reviewer"), row.get("source"), row.get("notes"), row.get("updated_at")] for row in annotation_rows])
    return rows


def _write_xlsx(path: Path, workbook: dict[str, Any]) -> None:
    sheets = [
        ("Summary", _summary_rows(workbook.get("summary"))),
        ("Runs", _table_rows(workbook.get("runs"))),
        ("Channel Ranking", _table_rows(workbook.get("channel_ranking"))),
        ("Review", _review_rows(workbook.get("review_state"))),
    ]
    sheet_entries = "\n".join(f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>' for idx, (name, _) in enumerate(sheets, 1))
    rel_entries = "\n".join(f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>' for idx, _ in enumerate(sheets, 1))
    rel_entries += f'\n<Relationship Id="rId{len(sheets)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    overrides = "\n".join(f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for idx, _ in enumerate(sheets, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", f'<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>{overrides}</Types>')
        zf.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        zf.writestr("xl/workbook.xml", f'<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>{sheet_entries}</sheets></workbook>')
        zf.writestr("xl/_rels/workbook.xml.rels", f'<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{rel_entries}</Relationships>')
        zf.writestr("xl/styles.xml", '<?xml version="1.0" encoding="UTF-8"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts><fills count="1"><fill><patternFill patternType="none"/></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf/></cellStyleXfs><cellXfs count="1"><xf xfId="0"/></cellXfs></styleSheet>')
        for idx, (_, rows) in enumerate(sheets, 1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", _xlsx_sheet_xml(rows))


def export_workbook(payload: dict[str, Any]) -> dict[str, Any]:
    output_path = _safe_export_path(payload.get("output_path"), "workbook.xlsx", {".json", ".xlsx"}, allow_absolute_path=bool(payload.get("allow_absolute_path")))
    workbook = {
        "schema_version": "pyhfo-workbook.v1",
        "created_at": utc_now_iso(),
        "summary": payload.get("summary") or {},
        "runs": payload.get("runs") or [],
        "active_run": payload.get("active_run") or {},
        "accepted_run_id": payload.get("accepted_run_id"),
        "channel_ranking": payload.get("channel_ranking") or [],
        "review_state": payload.get("review_state") or {},
        "review_annotations": _annotation_rows(payload.get("review_state") or {}),
        "artifacts": payload.get("artifacts") or {},
        "limits": ["Workbook JSON is an Electron compatibility artifact; native XLSX export remains a backend parity slice."],
    }
    if output_path.suffix.lower() == ".xlsx":
        workbook["limits"] = ["Workbook XLSX contains signal-derived candidates for expert review; it is not a clinical diagnosis."]
        _write_xlsx(output_path, workbook)
        fingerprint = file_fingerprint(output_path)
        return json_safe({"status": "complete", "artifact_path": str(output_path), "artifact_type": "workbook_xlsx", "workbook": workbook, "artifact_exists": True, "artifact_sha256": fingerprint["sha256"]})
    write_json(output_path, workbook)
    fingerprint = file_fingerprint(output_path)
    return json_safe({"status": "complete", "artifact_path": str(output_path), "artifact_type": "workbook_json", "workbook": workbook, "artifact_exists": True, "artifact_sha256": fingerprint["sha256"]})
