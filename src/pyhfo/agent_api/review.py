from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from pyhfo.agent_api.errors import PyHFOAgentError
from pyhfo.agent_api.serialization import json_safe, utc_now_iso
from pyhfo.agent_api.waveform import waveform_window
from pyhfo.utils.app_state import checkpoint_array, checkpoint_get
from pyhfo.utils.session_store import load_session_checkpoint


def _event_id(run_id: str, idx: int) -> str:
    digest = hashlib.sha1(f"{run_id}:{idx}".encode("utf-8")).hexdigest()[:10]
    return f"evt-{idx:04d}-{digest}"


def _coerce_float(row: dict[str, Any], names: list[str], default: float = 0.0) -> float:
    for name in names:
        if name in row and row[name] not in (None, ""):
            try:
                return float(row[name])
            except (TypeError, ValueError):
                continue
    return default


def _summary_sample_freq(events_csv: Path) -> float | None:
    summary_path = events_csv.with_name("summary.json")
    if not summary_path.exists():
        return None
    try:
        import json

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    candidates = [
        summary.get("sample_freq"),
        (summary.get("recording_info") or {}).get("sample_freq"),
        (summary.get("recording_info") or {}).get("sfreq"),
        ((summary.get("request") or {}).get("recording_info") or {}).get("sample_freq"),
        ((summary.get("request") or {}).get("recording_info") or {}).get("sfreq"),
    ]
    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _event_times(row: dict[str, Any], *, sample_freq: float | None) -> tuple[float, float]:
    if any(name in row for name in ["start_seconds", "onset", "start_time"]):
        start = _coerce_float(row, ["start_seconds", "onset", "start_time"], 0.0)
        end = _coerce_float(row, ["end_seconds", "offset", "end_time"], start)
        return start, end
    if "starts" in row or "ends" in row:
        start_samples = _coerce_float(row, ["starts"], 0.0)
        end_samples = _coerce_float(row, ["ends"], start_samples)
        if sample_freq and sample_freq > 0:
            return start_samples / sample_freq, end_samples / sample_freq
        return start_samples, end_samples
    start = _coerce_float(row, ["start"], 0.0)
    end = _coerce_float(row, ["end"], start)
    return start, end


def events_from_csv(events_csv: str | Path, *, run_id: str | None = None, biomarker: str | None = None) -> list[dict[str, Any]]:
    path = Path(str(events_csv)).expanduser().resolve()
    if not path.exists():
        raise PyHFOAgentError("file_not_found", f"Events CSV does not exist: {path}")
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    rid = run_id or path.parent.name or path.stem
    sample_freq = _summary_sample_freq(path)
    events = []
    for idx, row in enumerate(rows):
        channel = row.get("channel") or row.get("channel_name") or row.get("channel_names") or row.get("Channel")
        if not channel:
            continue
        start, end = _event_times(row, sample_freq=sample_freq)
        event = {"event_id": _event_id(rid, idx), "run_id": rid, "index": idx, "channel": str(channel), "start_seconds": start, "end_seconds": end, "duration_seconds": max(0.0, end - start), "biomarker": biomarker or row.get("biomarker") or "unknown", "source": str(path)}
        for key, value in row.items():
            if key not in event:
                event.setdefault("features", {})[key] = value
        events.append(event)
    return json_safe(events)


def session_events(session: Any, explicit_events: list[dict[str, Any]] | None = None, run_id: str | None = None) -> list[dict[str, Any]]:
    if explicit_events is not None:
        return json_safe(list(explicit_events))
    events_by_run = getattr(session, "review_events_by_run", None) or {}
    requested_run_id = str(run_id or getattr(session, "review_active_run_id", "") or "")
    if requested_run_id and requested_run_id in events_by_run:
        return json_safe(list(events_by_run[requested_run_id]))
    stored = getattr(session, "review_events", None)
    if stored:
        return json_safe(list(stored))
    last_detection = getattr(session, "last_detection", None) or {}
    artifacts = last_detection.get("artifacts") or {}
    csv_path = artifacts.get("csv") or artifacts.get("events_csv")
    if csv_path:
        return events_from_csv(csv_path, run_id=last_detection.get("active_run_id") or last_detection.get("run_id"), biomarker=last_detection.get("biomarker"))
    return []


def list_events(session: Any, *, events: list[dict[str, Any]] | None = None, annotations: dict[str, Any] | None = None, run_id: str | None = None) -> dict[str, Any]:
    all_events = session_events(session, events, run_id=run_id)
    review_annotations = annotations if annotations is not None else getattr(session, "review_annotations", {}) or {}
    enriched = []
    for event in all_events:
        item = dict(event)
        item["annotation"] = review_annotations.get(item["event_id"])
        enriched.append(item)
    ids = [event["event_id"] for event in all_events]
    cursor_event_id = getattr(session, "review_cursor_event_id", None)
    if cursor_event_id not in ids:
        cursor_event_id = ids[0] if ids else None
    cursor_index = ids.index(cursor_event_id) if cursor_event_id in ids else -1
    return json_safe({"status": "complete", "run_id": run_id or getattr(session, "review_active_run_id", None), "events": enriched, "count": len(enriched), "cursor_event_id": cursor_event_id, "cursor": {"event_id": cursor_event_id, "index": cursor_index, "count": len(ids)}})




def _pyqt_event_navigation_index(events: list[dict[str, Any]], current_index: int, offset: int) -> int:
    if not events:
        return 0
    normalized = min(max(0, int(current_index)), len(events) - 1)
    current = events[normalized]
    if offset < 0 and normalized == 0 and _is_spindle_event(current):
        return 0
    return (normalized + offset + len(events)) % len(events)


def _is_spindle_event(event: dict[str, Any]) -> bool:
    return "spindle" in str(event.get("biomarker") or "").lower()

def find_event(session: Any, event_id: str, *, events: list[dict[str, Any]] | None = None, run_id: str | None = None) -> dict[str, Any]:
    for event in session_events(session, events, run_id=run_id):
        if str(event.get("event_id")) == str(event_id):
            return dict(event)
    raise PyHFOAgentError("event_not_found", f"Unknown event_id: {event_id}")


def event_cursor(session: Any, *, action: str = "current", event_id: str | None = None, events: list[dict[str, Any]] | None = None, run_id: str | None = None) -> dict[str, Any]:
    all_events = session_events(session, events, run_id=run_id)
    if not all_events:
        raise PyHFOAgentError("no_events", "No events are available for review.")
    current = getattr(session, "review_cursor_event_id", None) or all_events[0]["event_id"]
    ids = [event["event_id"] for event in all_events]
    if event_id:
        if event_id not in ids:
            raise PyHFOAgentError("event_not_found", f"Unknown event_id: {event_id}")
        current = event_id
    idx = ids.index(current) if current in ids else 0
    normalized = str(action or "current")
    annotations = getattr(session, "review_annotations", {}) or {}
    if normalized == "next":
        idx = _pyqt_event_navigation_index(all_events, idx, 1)
    elif normalized in {"previous", "prev"}:
        idx = _pyqt_event_navigation_index(all_events, idx, -1)
    elif normalized == "first_unreviewed":
        idx = next((i for i, eid in enumerate(ids) if eid not in annotations), idx)
    elif normalized in {"current", "select"}:
        pass
    else:
        raise PyHFOAgentError("invalid_cursor_action", f"Unsupported cursor action: {action}")
    selected = all_events[idx]
    selected_run_id = str(run_id or selected.get("run_id") or getattr(session, "review_active_run_id", "") or "")
    if hasattr(session, "review_active_run_id") and selected_run_id:
        session.review_active_run_id = selected_run_id
    if hasattr(session, "review_cursor_event_id"):
        session.review_cursor_event_id = selected["event_id"]
    return json_safe({"status": "complete", "run_id": selected_run_id or None, "cursor": {"event_id": selected["event_id"], "index": idx, "count": len(ids), "has_previous": idx > 0, "has_next": idx < len(ids) - 1}, "event": selected})


def annotate_event(session: Any, event_id: str, *, label: str, notes: str | None = None, reviewer: str = "human", source: str = "user", confirmed: bool = False, display_label: str | None = None, events: list[dict[str, Any]] | None = None, run_id: str | None = None) -> dict[str, Any]:
    found_event = find_event(session, event_id, events=events, run_id=run_id)
    normalized = str(label or "").strip().lower()
    if normalized not in {"accepted", "rejected", "artifact", "uncertain", "clear"}:
        raise PyHFOAgentError("invalid_annotation", "label must be accepted, rejected, artifact, uncertain, or clear.")
    if not hasattr(session, "review_annotations") or getattr(session, "review_annotations") is None:
        session.review_annotations = {}
    if not hasattr(session, "annotation_audit_trail") or getattr(session, "annotation_audit_trail") is None:
        session.annotation_audit_trail = []
    previous = session.review_annotations.get(event_id)
    visible_label = str(display_label or label or normalized).strip()
    record = {"event_id": event_id, "run_id": run_id or found_event.get("run_id"), "label": normalized, "display_label": visible_label, "notes": notes or "", "reviewer": reviewer, "source": source, "updated_at": utc_now_iso(), "previous": previous}
    if not confirmed:
        staged = {**record, "status": "staged", "requires_confirmation": True}
        return json_safe({"status": "staged", "annotation": staged, "requires_confirmation": True, "audit_trail": session.annotation_audit_trail})
    record["status"] = "applied"
    record["confirmed"] = True
    if normalized == "clear":
        session.review_annotations.pop(event_id, None)
    else:
        session.review_annotations[event_id] = record
    session.annotation_audit_trail.append(record)
    return json_safe({"status": "complete", "annotation": record, "audit_trail": session.annotation_audit_trail})


def event_snippet(session: Any, event_id: str, *, input_path: str | Path | None = None, padding_seconds: float = 1.0, target_samples_per_channel: int = 600, events: list[dict[str, Any]] | None = None, run_id: str | None = None) -> dict[str, Any]:
    event = find_event(session, event_id, events=events, run_id=run_id)
    path = str(input_path or (getattr(session, "recording_info", None) or {}).get("input_path") or "")
    start = max(0.0, float(event.get("start_seconds") or 0.0) - float(padding_seconds))
    end = float(event.get("end_seconds") or event.get("start_seconds") or 0.0) + float(padding_seconds)
    try:
        window = waveform_window(path, start_seconds=start, duration_seconds=max(0.001, end - start), channels=[event["channel"]], target_samples_per_channel=target_samples_per_channel)
    except Exception:
        window = _saved_session_waveform_window(session, event, start_seconds=start, duration_seconds=max(0.001, end - start), target_samples_per_channel=target_samples_per_channel)
    filtered_trace = review_filtered_trace(session, event, window)
    time_frequency = review_time_frequency(session, event, window, filtered_trace)
    return json_safe({"status": "complete", "event": event, "waveform": window, "filtered_trace": filtered_trace, "time_frequency": time_frequency})


def _saved_session_waveform_window(session: Any, event: dict[str, Any], *, start_seconds: float, duration_seconds: float, target_samples_per_channel: int) -> dict[str, Any]:
    session_path = str(getattr(session, "saved_session_path", "") or ((getattr(session, "last_detection", None) or {}).get("artifacts") or {}).get("session_path") or "")
    if not session_path:
        raise PyHFOAgentError("invalid_request", "input_path is required and no saved-session waveform payload is available.")
    checkpoint = load_session_checkpoint(Path(session_path))
    sample_freq = float(checkpoint_get(checkpoint, "sample_freq", 0.0) or 0.0)
    if sample_freq <= 0:
        raise PyHFOAgentError("invalid_recording", "Saved session has invalid sample frequency.")
    data = checkpoint_array(checkpoint, "eeg_data", None)
    if data is None:
        raise PyHFOAgentError("invalid_recording", "Saved session does not contain eeg_data for review snippets.")
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    channel_names = checkpoint_array(checkpoint, "recording_channel_names", None)
    if channel_names is None:
        channel_names = checkpoint_array(checkpoint, "channel_names", [])
    channels = [str(channel) for channel in np.asarray(channel_names, dtype=object).tolist()]
    channel = str(event.get("channel") or "")
    if channel not in channels:
        raise PyHFOAgentError("channel_not_found", f"Requested event channel is not present in saved session: {channel}")
    first = max(0, int(round(float(start_seconds) * sample_freq)))
    last = min(int(data.shape[1]), int(round((float(start_seconds) + float(duration_seconds)) * sample_freq)))
    if last <= first:
        raise PyHFOAgentError("invalid_time_window", "Requested saved-session waveform window is outside the recording.")
    row = data[channels.index(channel), first:last]
    target = max(1, int(target_samples_per_channel or 600))
    stride = int(np.ceil(row.size / target)) if row.size > target else 1
    downsampled = row[::stride]
    sample_indices = np.arange(first, last, stride, dtype=int)[: downsampled.size]
    return {
        "status": "complete",
        "input_path": str(session_path),
        "source": "saved_session_eeg_data",
        "start_seconds": float(start_seconds),
        "duration_seconds": float((last - first) / sample_freq),
        "sample_freq": sample_freq,
        "source_sample_range": [int(first), int(last)],
        "downsample_stride": int(stride),
        "target_samples_per_channel": target,
        "returned_samples_per_channel": int(downsampled.size),
        "units": "uV",
        "channels": [channel],
        "times_seconds": (sample_indices / sample_freq).astype(float).tolist(),
        "data": [downsampled.astype(float).tolist()],
    }


def review_filtered_trace(session: Any, event: dict[str, Any], waveform: dict[str, Any]) -> dict[str, Any]:
    row = _first_waveform_row(waveform)
    sample_freq = float(waveform.get("sample_freq") or 0.0)
    if row.size < 4 or sample_freq <= 0:
        return {"status": "unavailable", "reason": "waveform snippet is too short for filtering", "data": [], "times_seconds": waveform.get("times_seconds") or []}
    try:
        from pyhfo.utils.utils_filter import construct_filter, filter_data

        filter_config = _review_filter_config(session, event, sample_freq)
        sos = construct_filter(filter_config["fp"], filter_config["fs"], filter_config["rp"], filter_config["rs"], filter_config["space"], sample_freq)
        filtered = np.asarray(filter_data(row, sos), dtype=float)
        return {
            "status": "complete",
            "source": "pyhfo.utils.utils_filter.filter_data",
            "filter": filter_config,
            "sample_freq": sample_freq,
            "units": waveform.get("units") or "uV",
            "times_seconds": waveform.get("times_seconds") or [],
            "data": filtered.tolist(),
        }
    except Exception as exc:
        fallback = row - _moving_average(row, 5)
        return {
            "status": "fallback",
            "source": "moving_average_highpass_fallback",
            "reason": str(exc),
            "sample_freq": sample_freq,
            "units": waveform.get("units") or "uV",
            "times_seconds": waveform.get("times_seconds") or [],
            "data": fallback.tolist(),
        }


def review_time_frequency(session: Any, event: dict[str, Any], waveform: dict[str, Any], filtered_trace: dict[str, Any]) -> dict[str, Any]:
    sample_freq = float(waveform.get("sample_freq") or 0.0)
    trace = np.asarray(filtered_trace.get("data") or _first_waveform_row(waveform), dtype=float)
    if trace.size < 16 or sample_freq <= 0:
        return {"status": "unavailable", "reason": "waveform snippet is too short for time-frequency analysis", "intensity": []}
    try:
        from pyhfo.utils.utils_plotting import calculate_time_frequency

        freq_min, freq_max = _review_frequency_range(session, event, sample_freq, filtered_trace.get("filter") or {})
        tf = np.asarray(calculate_time_frequency(trace, sample_freq, freq_min=freq_min, freq_max=freq_max), dtype=float)
        # compute_spectrum already returns rows in high-to-low frequency order.
        # Do not flip here, or the renderer labels low-frequency energy as high
        # frequency while the y-axis still says high-at-top.
        tf_display = np.log10(np.maximum(tf, np.finfo(float).tiny))
        vmin, vmax = _percentile_limits(tf_display, lower=8.0, upper=99.7)
        normalized = _normalize_display_matrix(tf_display, vmin, vmax)
        reduced, row_idx, col_idx = _downsample_matrix(normalized, max_rows=48, max_cols=160)
        frequencies = np.linspace(freq_max, freq_min, normalized.shape[0])[row_idx] if normalized.ndim == 2 and normalized.shape[0] else np.asarray([], dtype=float)
        times = np.asarray(waveform.get("times_seconds") or [], dtype=float)
        time_bins = times[col_idx] if times.size and col_idx.size else np.asarray([], dtype=float)
        return {
            "status": "complete",
            "source": "pyhfo.utils.utils_plotting.calculate_time_frequency",
            "display_transform": "pyqt_log10_percentile",
            "display_percentiles": [8.0, 99.7],
            "display_limits": [float(vmin), float(vmax)],
            "colormap": "magma",
            "frequency_range": [float(freq_min), float(freq_max)],
            "high_frequency_at_top": True,
            "frequencies_hz": frequencies.astype(float).tolist(),
            "time_bins_seconds": time_bins.astype(float).tolist(),
            "intensity": reduced.astype(float).tolist(),
        }
    except Exception as exc:
        return {"status": "failed", "source": "pyhfo.utils.utils_plotting.calculate_time_frequency", "reason": str(exc), "intensity": []}


def _first_waveform_row(waveform: dict[str, Any]) -> np.ndarray:
    rows = waveform.get("data") or []
    return np.asarray(rows[0] if rows else [], dtype=float)


def _moving_average(row: np.ndarray, window_size: int) -> np.ndarray:
    if row.size == 0:
        return row
    radius = max(1, int(window_size) // 2)
    padded = np.pad(row, (radius, radius), mode="edge")
    kernel = np.ones(radius * 2 + 1, dtype=float) / float(radius * 2 + 1)
    return np.convolve(padded, kernel, mode="valid")[: row.size]


def _review_filter_config(session: Any, event: dict[str, Any], sample_freq: float) -> dict[str, float]:
    run_filter = (getattr(session, "review_run_filters", None) or {}).get(str(event.get("run_id") or "")) or {}
    request_filter = run_filter or (((getattr(session, "last_detection", None) or {}).get("request") or {}).get("filter") or {})
    biomarker = str(event.get("biomarker") or (getattr(session, "last_detection", None) or {}).get("biomarker") or "Spike").lower()
    defaults = {"fp": 4.0, "fs": min(80.0, sample_freq * 0.45), "rp": 0.5, "rs": 93.0, "space": 0.5}
    if "hfo" in biomarker:
        defaults.update({"fp": min(80.0, sample_freq * 0.35), "fs": min(500.0, sample_freq * 0.45)})
    elif "spindle" in biomarker:
        defaults.update({"fp": 1.0, "fs": min(30.0, sample_freq * 0.45)})
    config = {**defaults, **{key: _float_or_default(value, defaults.get(key, 0.0)) for key, value in request_filter.items() if key in defaults}}
    nyquist = sample_freq / 2.0
    config["fp"] = max(0.1, min(float(config["fp"]), max(0.2, nyquist * 0.80)))
    config["fs"] = max(config["fp"] + 0.1, min(float(config["fs"]), max(config["fp"] + 0.2, nyquist * 0.95)))
    return config


def _review_frequency_range(session: Any, event: dict[str, Any], sample_freq: float, filter_config: dict[str, Any]) -> tuple[float, float]:
    nyquist = sample_freq / 2.0
    freq_min = float(filter_config.get("fp") or 1.0)
    freq_max = float(filter_config.get("fs") or nyquist)
    if freq_max <= freq_min:
        freq_min, freq_max = 1.0, nyquist
    return max(1.0, min(freq_min, nyquist - 1.0)), min(nyquist, max(freq_min + 1.0, freq_max))


def _percentile_limits(data: np.ndarray, lower: float = 5.0, upper: float = 99.5) -> tuple[float, float]:
    finite_values = np.asarray(data)[np.isfinite(data)]
    if finite_values.size == 0:
        return 0.0, 1.0
    lower_value = float(np.nanpercentile(finite_values, lower))
    upper_value = float(np.nanpercentile(finite_values, upper))
    if np.isclose(lower_value, upper_value):
        data_min = float(np.nanmin(finite_values))
        data_max = float(np.nanmax(finite_values))
        magnitude = max(abs(data_min), abs(data_max), 1.0)
        return data_min - magnitude * 0.1, data_max + magnitude * 0.1
    return lower_value, upper_value


def _normalize_display_matrix(matrix: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    denom = max(float(vmax) - float(vmin), np.finfo(float).eps)
    return np.clip((np.asarray(matrix, dtype=float) - float(vmin)) / denom, 0.0, 1.0)


def _downsample_matrix(matrix: np.ndarray, *, max_rows: int, max_cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if matrix.ndim != 2 or matrix.size == 0:
        return np.zeros((0, 0), dtype=float), np.asarray([], dtype=int), np.asarray([], dtype=int)
    row_idx = np.unique(np.linspace(0, matrix.shape[0] - 1, min(max_rows, matrix.shape[0]), dtype=int))
    col_idx = np.unique(np.linspace(0, matrix.shape[1] - 1, min(max_cols, matrix.shape[1]), dtype=int))
    return matrix[np.ix_(row_idx, col_idx)], row_idx, col_idx


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def event_feature_summary(session: Any, event_id: str, *, events: list[dict[str, Any]] | None = None, run_id: str | None = None) -> dict[str, Any]:
    event = find_event(session, event_id, events=events, run_id=run_id)
    duration = float(event.get("duration_seconds") or 0.0)
    features = dict(event.get("features") or {})
    summary = {"event_id": event_id, "channel": event.get("channel"), "duration_seconds": duration, "features": features, "spectrum": {"status": "not_computed", "reason": "Use event snippet waveform for renderer-side or future backend spectrum calculation."}}
    return json_safe({"status": "complete", "summary": summary})


def overlap_review(session: Any, *, events: list[dict[str, Any]] | None = None, tolerance_seconds: float = 0.0, run_id: str | None = None) -> dict[str, Any]:
    all_events = session_events(session, events, run_id=run_id)
    overlaps = []
    for i, left in enumerate(all_events):
        for right in all_events[i + 1 :]:
            if left.get("channel") == right.get("channel"):
                continue
            if float(left.get("start_seconds") or 0.0) <= float(right.get("end_seconds") or 0.0) + tolerance_seconds and float(right.get("start_seconds") or 0.0) <= float(left.get("end_seconds") or 0.0) + tolerance_seconds:
                overlaps.append({"left_event_id": left["event_id"], "right_event_id": right["event_id"], "channels": [left.get("channel"), right.get("channel")]})
    return json_safe({"status": "complete", "overlap_count": len(overlaps), "overlaps": overlaps})



def _consensus_support_threshold(strategy: str, run_count: int) -> int:
    normalized = str(strategy or "majority").lower()
    if normalized == "union":
        return 1
    if normalized == "intersection":
        return max(1, int(run_count))
    return max(1, int(np.ceil(float(run_count) / 2.0)))


def _consensus_source_events(session: Any, events_by_run: Any = None, run_ids: list[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    source = events_by_run if isinstance(events_by_run, dict) else getattr(session, "review_events_by_run", {}) or {}
    selected_ids = [str(run_id) for run_id in (run_ids or list(source.keys())) if str(run_id) in source]
    rows: dict[str, list[dict[str, Any]]] = {}
    for run_id in selected_ids:
        events = [dict(event, run_id=str(event.get("run_id") or run_id)) for event in (source.get(run_id) or []) if isinstance(event, dict)]
        if events:
            rows[run_id] = json_safe(events)
    return rows


def _normalized_timed_events(source: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    normalized: dict[str, list[dict[str, Any]]] = {}
    for run_id, events in source.items():
        rows: list[dict[str, Any]] = []
        for event in events:
            channel = event.get("channel")
            try:
                start = float(event.get("start_seconds"))
                end = float(event.get("end_seconds"))
            except (TypeError, ValueError):
                continue
            if not channel or not np.isfinite(start) or not np.isfinite(end):
                continue
            rows.append({**event, "run_id": run_id, "channel": str(channel), "start_seconds": start, "end_seconds": end})
        normalized[run_id] = rows
    return normalized


def _events_temporally_overlap(left: dict[str, Any], right: dict[str, Any], tolerance_seconds: float) -> bool:
    if left.get("channel") != right.get("channel"):
        return False
    left_start = float(left.get("start_seconds") or 0.0)
    left_end = float(left.get("end_seconds") or left_start)
    right_start = float(right.get("start_seconds") or 0.0)
    right_end = float(right.get("end_seconds") or right_start)
    return left_start <= right_end + tolerance_seconds and right_start <= left_end + tolerance_seconds


def compare_detector_runs(
    session: Any,
    *,
    events_by_run: Any = None,
    run_ids: list[str] | None = None,
    tolerance_seconds: float = 0.0,
    max_overlap_examples: int = 100,
) -> dict[str, Any]:
    source = _consensus_source_events(session, events_by_run=events_by_run, run_ids=run_ids)
    if len(source) < 2:
        raise PyHFOAgentError("insufficient_compare_runs", "Run comparison requires event rows from at least two source runs.")

    run_id_list = list(source.keys())
    normalized = _normalized_timed_events(source)
    per_run: list[dict[str, Any]] = []
    channel_stats: dict[str, dict[str, Any]] = {}
    for run_id in run_id_list:
        events = normalized.get(run_id, [])
        counts: dict[str, int] = {}
        for event in events:
            channel = str(event.get("channel"))
            counts[channel] = counts.get(channel, 0) + 1
            stats = channel_stats.setdefault(channel, {"channel": channel, "count_by_run": {}})
            stats["count_by_run"][run_id] = stats["count_by_run"].get(run_id, 0) + 1
        per_run.append({
            "run_id": run_id,
            "event_count": len(events),
            "channel_count": len(counts),
            "channel_counts": dict(sorted(counts.items())),
        })

    channel_comparison: list[dict[str, Any]] = []
    for channel, stats in channel_stats.items():
        count_by_run = {run_id: int(stats["count_by_run"].get(run_id, 0)) for run_id in run_id_list}
        nonzero_counts = [count for count in count_by_run.values() if count > 0]
        support_count = len(nonzero_counts)
        total_events = int(sum(count_by_run.values()))
        if support_count == len(run_id_list):
            stability = "present_all_runs"
        elif support_count >= 2:
            stability = "recurrent"
        else:
            stability = "run_specific"
        channel_comparison.append({
            "channel": channel,
            "count_by_run": count_by_run,
            "support_count": support_count,
            "total_events": total_events,
            "stability": stability,
            "parameter_sensitive": support_count < len(run_id_list) or len(set(nonzero_counts)) > 1,
        })
    channel_comparison.sort(key=lambda row: (-int(row["support_count"]), -int(row["total_events"]), str(row["channel"])))

    temporal_overlaps: list[dict[str, Any]] = []
    for left_index, left_run_id in enumerate(run_id_list):
        for right_run_id in run_id_list[left_index + 1:]:
            for left_event in normalized.get(left_run_id, []):
                for right_event in normalized.get(right_run_id, []):
                    if not _events_temporally_overlap(left_event, right_event, tolerance_seconds):
                        continue
                    overlap_start = max(float(left_event["start_seconds"]), float(right_event["start_seconds"]))
                    overlap_end = min(float(left_event["end_seconds"]), float(right_event["end_seconds"]))
                    temporal_overlaps.append({
                        "channel": left_event.get("channel"),
                        "left_run_id": left_run_id,
                        "left_event_id": left_event.get("event_id"),
                        "right_run_id": right_run_id,
                        "right_event_id": right_event.get("event_id"),
                        "overlap_start_seconds": overlap_start,
                        "overlap_end_seconds": overlap_end,
                        "overlap_duration_seconds": max(0.0, overlap_end - overlap_start),
                    })

    return json_safe({
        "status": "complete",
        "source_run_ids": run_id_list,
        "run_count": len(run_id_list),
        "per_run": per_run,
        "channel_comparison": channel_comparison,
        "stable_channels": [row for row in channel_comparison if row["stability"] in {"present_all_runs", "recurrent"}],
        "run_specific_channels": [row for row in channel_comparison if row["stability"] == "run_specific"],
        "temporal_overlap_count": len(temporal_overlaps),
        "temporal_overlaps": temporal_overlaps[:max(0, int(max_overlap_examples))],
        "limitations": [
            "Comparison uses available event rows and timing only; detector settings and waveform morphology still require review.",
            "Stable/recurrent channels are algorithmic signal candidates, not clinical labels or diagnosis.",
        ],
        "interpretation_policy": "Run comparison is signal-derived stability evidence for expert review; it is not a seizure-onset-zone or pathology determination.",
    })


def create_consensus_run(session: Any, *, events_by_run: Any = None, run_ids: list[str] | None = None, strategy: str = "majority", biomarker: str | None = None) -> dict[str, Any]:
    normalized_strategy = str(strategy or "majority").lower()
    if normalized_strategy not in {"union", "majority", "intersection"}:
        raise PyHFOAgentError("invalid_consensus_strategy", "strategy must be union, majority, or intersection.")
    source = _consensus_source_events(session, events_by_run=events_by_run, run_ids=run_ids)
    if len(source) < 2:
        raise PyHFOAgentError("insufficient_consensus_runs", "Consensus requires event rows from at least two source runs.")
    run_id_list = list(source.keys())
    threshold = _consensus_support_threshold(normalized_strategy, len(run_id_list))
    flat_rows = [event for events in _normalized_timed_events(source).values() for event in events]
    flat_rows.sort(key=lambda row: (str(row.get("channel")), float(row.get("start_seconds") or 0.0), float(row.get("end_seconds") or 0.0)))
    clusters: list[list[dict[str, Any]]] = []
    for event in flat_rows:
        if not clusters:
            clusters.append([event])
            continue
        latest = clusters[-1]
        latest_end = max(float(item.get("end_seconds") or 0.0) for item in latest)
        if latest[0].get("channel") != event.get("channel") or float(event.get("start_seconds") or 0.0) > latest_end:
            clusters.append([event])
        else:
            latest.append(event)

    consensus_events: list[dict[str, Any]] = []
    for cluster in clusters:
        source_run_ids = sorted({str(event.get("run_id")) for event in cluster if event.get("run_id")})
        if len(source_run_ids) < threshold:
            continue
        starts = [float(event.get("start_seconds") or 0.0) for event in cluster]
        ends = [float(event.get("end_seconds") or event.get("start_seconds") or 0.0) for event in cluster]
        start = min(starts)
        end = max(ends)
        idx = len(consensus_events)
        consensus_events.append({
            "event_id": f"consensus-{normalized_strategy}-evt-{idx + 1}",
            "run_id": "",
            "index": idx,
            "channel": cluster[0].get("channel"),
            "start_seconds": start,
            "end_seconds": end,
            "duration_seconds": max(0.0, end - start),
            "biomarker": biomarker or cluster[0].get("biomarker") or "HFO",
            "source": "backend_consensus",
            "features": {
                "consensus": True,
                "consensus_strategy": normalized_strategy,
                "support_count": len(source_run_ids),
                "support_threshold": threshold,
                "source_run_ids": source_run_ids,
                "source_event_ids": [event.get("event_id") for event in cluster if event.get("event_id")],
            },
        })
    created_at = utc_now_iso()
    consensus_run_id = f"consensus-{normalized_strategy}-{hashlib.sha1((created_at + ','.join(run_id_list)).encode('utf-8')).hexdigest()[:10]}"
    stamped_events = [{**event, "event_id": f"{consensus_run_id}-evt-{index + 1}", "run_id": consensus_run_id, "index": index} for index, event in enumerate(consensus_events)]
    run = {
        "runId": consensus_run_id,
        "run_id": consensus_run_id,
        "biomarker": biomarker or (stamped_events[0].get("biomarker") if stamped_events else "HFO"),
        "detector": f"Consensus {normalized_strategy.title()}",
        "events": len(stamped_events),
        "channels": len({event.get("channel") for event in stamped_events if event.get("channel")}),
        "status": "consensus",
        "classifier": f"from {len(source)} runs",
        "createdAt": created_at,
        "parameters": {
            "consensus_strategy": normalized_strategy,
            "consensus_source_run_ids": ",".join(run_id_list),
            "consensus_support_threshold": threshold,
            "source_event_count": sum(len(events) for events in source.values()),
            "backend_consensus": True,
        },
    }
    if hasattr(session, "review_events_by_run"):
        session.review_events_by_run = {**(getattr(session, "review_events_by_run", {}) or {}), consensus_run_id: stamped_events}
    if hasattr(session, "review_events"):
        session.review_events = stamped_events
    if hasattr(session, "review_active_run_id"):
        session.review_active_run_id = consensus_run_id
    if hasattr(session, "review_visible_run_ids"):
        existing = [run_id for run_id in (getattr(session, "review_visible_run_ids", []) or []) if run_id != consensus_run_id]
        session.review_visible_run_ids = [consensus_run_id, *existing]
    return json_safe({"status": "complete", "run": run, "events": stamped_events, "source_run_ids": run_id_list, "support_threshold": threshold, "strategy": normalized_strategy})

def export_review_state(session: Any) -> dict[str, Any]:
    return json_safe({
        "status": "complete",
        "cursor_event_id": getattr(session, "review_cursor_event_id", None),
        "active_run_id": getattr(session, "review_active_run_id", None),
        "accepted_run_id": getattr(session, "review_accepted_run_id", None),
        "visible_run_ids": getattr(session, "review_visible_run_ids", []) or [],
        "events_by_run": getattr(session, "review_events_by_run", {}) or {},
        "annotations": getattr(session, "review_annotations", {}) or {},
        "annotation_history": getattr(session, "annotation_audit_trail", []) or [],
        "events": session_events(session),
    })
