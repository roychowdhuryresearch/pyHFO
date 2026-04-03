from __future__ import annotations

import html
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def export_analysis_report(
    output_path,
    backend,
    *,
    biomarker_label: str = "Event",
    app_name: str = "PyBrain",
    snapshot_source_path: str | Path | None = None,
):
    html_path = _normalize_report_path(output_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    assets_dir = html_path.with_name(f"{html_path.stem}_files")
    assets_dir.mkdir(parents=True, exist_ok=True)

    session = getattr(backend, "analysis_session", None)
    selected_run = None
    accepted_run = session.get_accepted_run() if session is not None and hasattr(session, "get_accepted_run") else None
    active_run = session.get_active_run() if session is not None and hasattr(session, "get_active_run") else None
    selected_run = accepted_run or active_run
    if selected_run is None:
        raise ValueError("No accepted or active run is available to export.")

    recording_info = _build_recording_info(backend)
    runs_df = _safe_dataframe(getattr(backend, "get_run_summaries", lambda: [])())
    ranking_df = _safe_dataframe(getattr(backend, "get_channel_ranking", lambda _run_id=None: [])(selected_run.run_id))
    comparison_payload = getattr(backend, "compare_runs", lambda _run_ids=None: {})(None)
    comparison_df = _safe_dataframe(comparison_payload.get("pairwise_overlap", []) if isinstance(comparison_payload, dict) else [])
    events_df = _safe_dataframe(selected_run.event_features.to_df() if getattr(selected_run, "event_features", None) is not None and hasattr(selected_run.event_features, "to_df") else [])

    exported_at = datetime.now().astimezone()
    artifact_links = {}
    export_notes = []

    workbook_relpath = _export_workbook_if_available(backend, assets_dir, selected_run.run_id, export_notes)
    if workbook_relpath:
        artifact_links["Clinical workbook"] = workbook_relpath

    events_csv_path = assets_dir / "events.csv"
    if not events_df.empty:
        events_df.to_csv(events_csv_path, index=False)
        artifact_links["Event CSV"] = f"{assets_dir.name}/{events_csv_path.name}"
    else:
        export_notes.append("Event-level preview was empty, so no events.csv file was created.")

    waveform_relpath = _copy_snapshot_if_available(snapshot_source_path, assets_dir)
    if waveform_relpath:
        artifact_links["Waveform snapshot"] = waveform_relpath

    metadata = {
        "app_name": app_name,
        "biomarker_type": biomarker_label,
        "exported_at": exported_at.isoformat(),
        "recording": _json_safe(recording_info),
        "selected_run": _json_safe(_describe_run(selected_run)),
        "accepted_run": _json_safe(_describe_run(accepted_run)),
        "active_run": _json_safe(_describe_run(active_run)),
        "parameters": _json_safe(
            {
                "filter": _param_to_dict(getattr(selected_run, "param_filter", None)),
                "detector": _param_to_dict(getattr(selected_run, "param_detector", None)),
                "classifier": _param_to_dict(getattr(selected_run, "param_classifier", None)),
            }
        ),
        "artifacts": artifact_links,
        "summary": {
            "run_count": int(len(getattr(session, "runs", {}))) if session is not None else 0,
            "visible_run_count": int(len(getattr(session, "visible_run_ids", []))) if session is not None else 0,
            "top_channel": _json_safe(ranking_df.iloc[0].to_dict()) if not ranking_df.empty else None,
            "event_preview_count": int(len(events_df)),
        },
        "runs": _json_safe(runs_df.to_dict(orient="records")),
        "channel_ranking": _json_safe(ranking_df.to_dict(orient="records")),
        "pairwise_overlap": _json_safe(comparison_df.to_dict(orient="records")),
        "events_preview": _json_safe(events_df.head(50).to_dict(orient="records")),
        "notes": export_notes,
    }

    metadata_path = assets_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    artifact_links["Metadata JSON"] = f"{assets_dir.name}/{metadata_path.name}"

    html_content = _build_report_html(
        app_name=app_name,
        biomarker_label=biomarker_label,
        recording_info=recording_info,
        exported_at=exported_at,
        selected_run=selected_run,
        accepted_run=accepted_run,
        active_run=active_run,
        runs_df=runs_df,
        ranking_df=ranking_df,
        comparison_df=comparison_df,
        events_df=events_df,
        artifact_links=artifact_links,
        export_notes=export_notes,
        waveform_relpath=waveform_relpath,
    )
    html_path.write_text(html_content, encoding="utf-8")
    return html_path


def _normalize_report_path(output_path) -> Path:
    path = Path(output_path)
    if path.suffix.lower() not in {".html", ".htm"}:
        return path.with_suffix(".html")
    return path


def _safe_dataframe(payload) -> pd.DataFrame:
    if payload is None:
        return pd.DataFrame()
    if isinstance(payload, pd.DataFrame):
        return payload.copy()
    try:
        return pd.DataFrame(payload)
    except Exception:
        return pd.DataFrame()


def _build_recording_info(backend) -> dict:
    info = dict(getattr(backend, "edf_param", {}) or {})
    file_path = info.get("edf_fn") or getattr(backend, "current_recording_path", None)
    channel_names = np.array(getattr(backend, "channel_names", np.array([])))
    sample_freq = float(getattr(backend, "sample_freq", 0) or 0)
    sample_count = 0
    eeg_data = getattr(backend, "eeg_data", None)
    if eeg_data is not None and getattr(eeg_data, "shape", None):
        sample_count = int(eeg_data.shape[1]) if len(eeg_data.shape) > 1 else int(eeg_data.shape[0])

    info["recording_file"] = str(file_path) if file_path else ""
    info["channel_count"] = int(len(channel_names)) or int(info.get("nchan", 0) or 0)
    info["sample_count"] = sample_count
    info["duration_seconds"] = round(sample_count / sample_freq, 2) if sample_freq > 0 and sample_count > 0 else 0
    if "channels" in info:
        channels = list(np.array(info["channels"]).tolist())
        preview = ", ".join(str(ch) for ch in channels[:10])
        if len(channels) > 10:
            preview += ", ..."
        info["channels"] = preview
    return info


def _export_workbook_if_available(backend, assets_dir: Path, run_id: str, export_notes: list[str]) -> str | None:
    workbook_path = assets_dir / "clinical_summary.xlsx"
    try:
        if hasattr(backend, "export_clinical_summary"):
            backend.export_clinical_summary(str(workbook_path), run_id=run_id)
            return f"{assets_dir.name}/{workbook_path.name}"
        if hasattr(backend, "export_excel"):
            backend.export_excel(str(workbook_path))
            return f"{assets_dir.name}/{workbook_path.name}"
        export_notes.append("Workbook export is not available for the current biomarker mode.")
    except Exception as exc:  # pragma: no cover - defensive UI fallback
        export_notes.append(f"Workbook export failed: {exc}")
    return None


def _copy_snapshot_if_available(snapshot_source_path, assets_dir: Path) -> str | None:
    if not snapshot_source_path:
        return None
    source = Path(snapshot_source_path)
    if not source.exists():
        return None
    destination = assets_dir / "waveform_snapshot.png"
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return f"{assets_dir.name}/{destination.name}"


def _describe_run(run) -> dict | None:
    if run is None:
        return None
    summary = dict(getattr(run, "summary", {}) or {})
    return {
        "run_id": getattr(run, "run_id", ""),
        "detector_name": getattr(run, "detector_name", ""),
        "display_name": getattr(run, "display_name", ""),
        "created_at": getattr(run, "created_at", ""),
        "selected_channels": len(getattr(run, "selected_channels", []) or []),
        "classified": bool(getattr(run, "classified", False)),
        "summary": summary,
    }


def _param_to_dict(param) -> dict:
    if param is None:
        return {}
    if hasattr(param, "to_dict"):
        return dict(param.to_dict())
    if isinstance(param, dict):
        return dict(param)
    return {"value": str(param)}


def _flatten_mapping(mapping, prefix: str = "") -> list[tuple[str, str]]:
    rows = []
    for key, value in (mapping or {}).items():
        label = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(_flatten_mapping(value, label))
        else:
            rows.append((label, _format_value(value)))
    return rows


def _format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return str(value.item())
    if isinstance(value, np.ndarray):
        return _format_value(value.tolist())
    if isinstance(value, (list, tuple, set)):
        items = [str(item) for item in list(value)]
        preview = ", ".join(items[:8])
        if len(items) > 8:
            preview += ", ..."
        return preview
    return str(value)


def _json_safe(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _render_dataframe(df: pd.DataFrame, *, empty_message: str, limit: int | None = None) -> str:
    if df.empty:
        return f'<p class="empty">{html.escape(empty_message)}</p>'
    frame = df.head(limit).copy() if limit else df.copy()
    frame.columns = [str(column).replace("_", " ").title() for column in frame.columns]
    for column in frame.columns:
        frame[column] = frame[column].map(_format_value)
    return frame.to_html(index=False, classes="report-table", border=0, escape=True)


def _render_kv_table(rows: list[tuple[str, str]], *, empty_message: str) -> str:
    if not rows:
        return f'<p class="empty">{html.escape(empty_message)}</p>'
    table = pd.DataFrame(rows, columns=["Setting", "Value"])
    return _render_dataframe(table, empty_message=empty_message)


def _metric_card(label: str, value: str) -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        "</div>"
    )


def _build_report_html(
    *,
    app_name: str,
    biomarker_label: str,
    recording_info: dict,
    exported_at: datetime,
    selected_run,
    accepted_run,
    active_run,
    runs_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    events_df: pd.DataFrame,
    artifact_links: dict[str, str],
    export_notes: list[str],
    waveform_relpath: str | None,
) -> str:
    recording_name = Path(recording_info.get("recording_file") or "untitled").name or "untitled"
    selected_summary = dict(getattr(selected_run, "summary", {}) or {})
    top_channel = str(ranking_df.iloc[0]["channel_name"]) if not ranking_df.empty and "channel_name" in ranking_df.columns else "--"
    metric_cards = [
        _metric_card("Selected detector", str(getattr(selected_run, "detector_name", "--"))),
        _metric_card("Events", str(selected_summary.get("num_events", 0))),
        _metric_card("Channels", str(selected_summary.get("num_channels", 0))),
        _metric_card("Top channel", top_channel),
        _metric_card("Accepted run", str(getattr(accepted_run, "detector_name", "--")) if accepted_run is not None else "--"),
        _metric_card("Runs in session", str(len(runs_df))),
    ]

    artifact_items = "".join(
        f'<li><a href="{html.escape(relpath)}">{html.escape(label)}</a></li>'
        for label, relpath in artifact_links.items()
    ) or '<li>No auxiliary files were produced.</li>'
    notes_markup = "".join(f"<li>{html.escape(note)}</li>" for note in export_notes)
    notes_block = f"<ul>{notes_markup}</ul>" if notes_markup else '<p class="empty">No export notes.</p>'

    filter_rows = _flatten_mapping(_param_to_dict(getattr(selected_run, "param_filter", None)))
    detector_rows = _flatten_mapping(_param_to_dict(getattr(selected_run, "param_detector", None)))
    classifier_rows = _flatten_mapping(_param_to_dict(getattr(selected_run, "param_classifier", None)))

    waveform_block = ""
    if waveform_relpath:
        waveform_block = (
            '<section class="panel wide">'
            '<h2>Waveform Snapshot</h2>'
            '<p class="section-copy">This is the current waveform view at export time.</p>'
            f'<img class="waveform-image" src="{html.escape(waveform_relpath)}" alt="Current waveform snapshot">'
            "</section>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(app_name)} {html.escape(biomarker_label)} Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f1ea;
      --panel: #fffdf9;
      --ink: #24313a;
      --muted: #64707b;
      --line: #d9d0c2;
      --accent: #9a4d21;
      --accent-soft: #f3e6dc;
      --shadow: 0 14px 40px rgba(36, 49, 58, 0.08);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top right, rgba(154, 77, 33, 0.12), transparent 24rem),
        linear-gradient(180deg, #f9f5ef 0%, var(--bg) 100%);
      color: var(--ink);
      line-height: 1.5;
    }}
    .page {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(154, 77, 33, 0.92), rgba(111, 56, 32, 0.94));
      color: #fff8f2;
      border-radius: 24px;
      padding: 28px 30px;
      box-shadow: var(--shadow);
    }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 12px;
      opacity: 0.82;
      margin-bottom: 8px;
    }}
    h1 {{
      margin: 0;
      font-size: 34px;
      line-height: 1.15;
    }}
    .hero-copy {{
      margin: 12px 0 0;
      color: rgba(255, 248, 242, 0.88);
      max-width: 56rem;
    }}
    .meta-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 18px;
    }}
    .meta-pill {{
      background: rgba(255, 248, 242, 0.12);
      border: 1px solid rgba(255, 248, 242, 0.2);
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 13px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 14px;
      margin: 22px 0 0;
    }}
    .metric-card {{
      background: var(--panel);
      border: 1px solid rgba(255, 255, 255, 0.32);
      border-radius: 18px;
      padding: 16px 18px;
      color: var(--ink);
      box-shadow: var(--shadow);
    }}
    .metric-label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .metric-value {{
      font-size: 24px;
      font-weight: 700;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 18px;
      margin-top: 22px;
    }}
    .panel {{
      grid-column: span 6;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 20px;
      box-shadow: var(--shadow);
      min-width: 0;
    }}
    .panel.wide {{
      grid-column: 1 / -1;
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 18px;
    }}
    .section-copy {{
      margin: 0 0 14px;
      color: var(--muted);
      font-size: 14px;
    }}
    .report-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: white;
      border-radius: 14px;
      overflow: hidden;
    }}
    .report-table th,
    .report-table td {{
      padding: 10px 12px;
      border-bottom: 1px solid #ece4d7;
      text-align: left;
      vertical-align: top;
    }}
    .report-table th {{
      background: #f7f1e8;
      color: #3d4952;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .report-table tr:last-child td {{
      border-bottom: none;
    }}
    .artifact-list,
    ul {{
      margin: 0;
      padding-left: 18px;
    }}
    a {{
      color: var(--accent);
    }}
    .empty {{
      margin: 0;
      color: var(--muted);
      font-style: italic;
    }}
    .waveform-image {{
      width: 100%;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: white;
    }}
    @media (max-width: 860px) {{
      .panel {{
        grid-column: 1 / -1;
      }}
      .page {{
        padding: 18px 14px 36px;
      }}
      .hero {{
        padding: 22px 20px;
      }}
      h1 {{
        font-size: 28px;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="eyebrow">{html.escape(app_name)} export package</div>
      <h1>{html.escape(biomarker_label)} report for {html.escape(recording_name)}</h1>
      <p class="hero-copy">This report packages the currently selected analysis run with the key recording metadata, detector configuration, channel prioritization, detector agreement, and shareable analysis artifacts.</p>
      <div class="meta-row">
        <div class="meta-pill">Selected run: {html.escape(getattr(selected_run, "display_name", getattr(selected_run, "detector_name", "--")))}</div>
        <div class="meta-pill">Active run: {html.escape(getattr(active_run, "detector_name", "--")) if active_run is not None else "--"}</div>
        <div class="meta-pill">Accepted run: {html.escape(getattr(accepted_run, "detector_name", "--")) if accepted_run is not None else "--"}</div>
        <div class="meta-pill">Exported: {html.escape(exported_at.strftime("%Y-%m-%d %H:%M:%S %Z"))}</div>
      </div>
      <div class="metrics">
        {"".join(metric_cards)}
      </div>
    </section>

    <section class="grid">
      <section class="panel">
        <h2>Recording</h2>
        <p class="section-copy">Source file and acquisition context for this export.</p>
        {_render_kv_table(_flatten_mapping(recording_info), empty_message="Recording metadata is unavailable.")}
      </section>

      <section class="panel">
        <h2>Artifacts</h2>
        <p class="section-copy">These files travel with the HTML report so collaborators can inspect the same run in more detail.</p>
        <ul class="artifact-list">{artifact_items}</ul>
      </section>

      <section class="panel">
        <h2>Filter Parameters</h2>
        <p class="section-copy">Signal preparation settings used for the selected run.</p>
        {_render_kv_table(filter_rows, empty_message="No filter parameters were stored for this run.")}
      </section>

      <section class="panel">
        <h2>Detector Parameters</h2>
        <p class="section-copy">Detection settings associated with the selected run.</p>
        {_render_kv_table(detector_rows, empty_message="No detector parameters were stored for this run.")}
      </section>

      <section class="panel">
        <h2>Classifier Parameters</h2>
        <p class="section-copy">Classification configuration captured with the selected run.</p>
        {_render_kv_table(classifier_rows, empty_message="No classifier parameters were stored for this run.")}
      </section>

      <section class="panel">
        <h2>Export Notes</h2>
        <p class="section-copy">Warnings or skipped optional steps while assembling the report package.</p>
        {notes_block}
      </section>

      {waveform_block}

      <section class="panel wide">
        <h2>Run Inventory</h2>
        <p class="section-copy">All saved runs in the current session.</p>
        {_render_dataframe(runs_df, empty_message="No saved runs were available.", limit=20)}
      </section>

      <section class="panel wide">
        <h2>Channel Ranking</h2>
        <p class="section-copy">Priority channels for the selected export run.</p>
        {_render_dataframe(ranking_df, empty_message="No ranked channels were available.", limit=20)}
      </section>

      <section class="panel wide">
        <h2>Detector Agreement</h2>
        <p class="section-copy">Pairwise overlap between saved runs in the session.</p>
        {_render_dataframe(comparison_df, empty_message="At least two runs are required before detector agreement can be computed.", limit=20)}
      </section>

      <section class="panel wide">
        <h2>Event Preview</h2>
        <p class="section-copy">A preview of the selected run's event-level table for quick review.</p>
        {_render_dataframe(events_df, empty_message="No event-level rows were available for preview.", limit=50)}
      </section>
    </section>
  </main>
</body>
</html>
"""
