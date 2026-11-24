from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from pyhfo.agent_api.errors import PyHFOAgentError
from pyhfo.agent_api.pipeline import get_recording_info
from pyhfo.agent_api.serialization import json_safe
from pyhfo.utils.utils_io import load_mne_raw
from pyhfo.utils.utils_montage import _strip_known_prefixes


def channel_role(name: str) -> tuple[str, str]:
    raw = str(name or "").strip().upper()
    if not raw:
        return "exclude", "empty channel name"
    # Drop acquisition prefixes (POL/EEG/SEEG/ECOG) before structural checks so that
    # exporter-prefixed contacts like "POL PT3" are judged on their real name ("PT3").
    # Reuses the montage convention to avoid a divergent copy of the prefix rule.
    upper = _strip_known_prefixes(raw).strip()
    if not upper:
        return "exclude", "empty channel name"
    auxiliary_prefixes = (
        "EKG",
        "ECG",
        "EOG",
        "EMG",
        "RESP",
        "PLETH",
        "TRIG",
        "TRIGGER",
        "STIM",
        "DC",
        "AUX",
        "SYNC",
        "MARK",
    )
    if upper.startswith(auxiliary_prefixes) or any(token in upper for token in ["-EKG", "-ECG", "EKG-", "ECG-"]):
        return "exclude", "physiology/auxiliary channel, not brain signal"
    if upper.startswith("X"):
        return "uncertain", "ambiguous X-prefixed channel; exclude from default brain ROI unless user confirms it is neural"
    if "-" in upper:
        return "brain", "bipolar-looking neural contact pair"
    if re.match(r"^[A-Z]+[0-9]+$", upper):
        return "brain", "monopolar-looking neural contact"
    return "uncertain", "unrecognized channel naming; needs review before treating as brain signal"


def infer_brain_signal_roi_from_channels(channel_names: list[str]) -> dict[str, Any]:
    brain: list[str] = []
    excluded: list[dict[str, str]] = []
    uncertain: list[dict[str, str]] = []
    for channel in [str(item) for item in channel_names]:
        role, reason = channel_role(channel)
        if role == "brain":
            brain.append(channel)
        elif role == "exclude":
            excluded.append({"channel": channel, "reason": reason})
        else:
            uncertain.append({"channel": channel, "reason": reason})
    return json_safe(
        {
            "status": "complete",
            "policy": "Label-free ROI inference from EDF channel names only; does not read SOZ/resection/pathology labels.",
            "brain_signal_channels": brain,
            "brain_signal_channel_count": len(brain),
            "excluded_non_brain_channels": excluded,
            "excluded_non_brain_channel_count": len(excluded),
            "uncertain_channels": uncertain,
            "uncertain_channel_count": len(uncertain),
            "default_detection_exclusions": [item["channel"] for item in excluded] + [item["channel"] for item in uncertain],
            "warnings": [
                "ROI is inferred from channel names and should be confirmed for ambiguous labels.",
                "Do not treat this as clinical pathology localization.",
            ],
        }
    )


def infer_brain_signal_roi(input_path: str | Path) -> dict[str, Any]:
    info = get_recording_info(input_path)
    return infer_brain_signal_roi_from_channels([str(channel) for channel in info.get("channel_names") or []])


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _robust_modified_z(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return np.zeros_like(values, dtype=float)
    median = np.median(finite)
    mad = np.median(np.abs(finite - median))
    if not np.isfinite(mad) or mad <= 0:
        return np.zeros_like(values, dtype=float)
    return 0.6745 * (values - median) / mad


def _read_raw(input_path: str | Path):
    path = Path(str(input_path)).expanduser().resolve()
    if not path.exists():
        raise PyHFOAgentError("file_not_found", f"Input file does not exist: {path}")
    raw = load_mne_raw(str(path))
    return path, raw


def _analysis_window(raw, payload: dict[str, Any]) -> tuple[int, int, float, float]:
    sfreq = float(raw.info.get("sfreq") or 0.0)
    n_times = int(getattr(raw, "n_times", 0) or 0)
    start_seconds = max(0.0, float(payload.get("start_seconds") or 0.0))
    default_duration = 60.0
    max_duration = payload.get("max_duration_seconds", default_duration)
    if max_duration in (None, ""):
        max_duration = default_duration
    max_duration = max(0.001, float(max_duration))
    start = min(int(round(start_seconds * sfreq)), n_times) if sfreq and n_times else 0
    stop = min(n_times, start + int(round(max_duration * sfreq))) if sfreq and n_times else n_times
    if stop <= start and n_times:
        stop = min(n_times, start + 1)
    return start, stop, (start / sfreq if sfreq else 0.0), (stop / sfreq if sfreq else 0.0)


def _get_window_data(raw, picks: list[int], start: int, stop: int):
    try:
        return raw.get_data(picks=picks, start=start, stop=stop)
    except TypeError:
        data = raw.get_data(picks=picks)
        return data[:, start:stop] if data.ndim == 2 and stop else data


def compute_channel_qc(payload: dict[str, Any]) -> dict[str, Any]:
    path, raw = _read_raw(payload.get("input_path"))
    channel_names = [str(channel) for channel in raw.ch_names]
    roi = payload.get("roi") if isinstance(payload.get("roi"), dict) else infer_brain_signal_roi_from_channels(channel_names)
    exclude = set(str(ch) for ch in (payload.get("exclude_channels") or roi.get("default_detection_exclusions") or []))
    selected = [ch for ch in channel_names if ch not in exclude]
    picks = [channel_names.index(ch) for ch in selected]
    start, stop, start_seconds, stop_seconds = _analysis_window(raw, payload)
    data = _get_window_data(raw, picks, start, stop) if picks else np.empty((0, max(0, stop - start)))
    sfreq = float(raw.info.get("sfreq") or 0.0)

    if data.size:
        rms = np.sqrt(np.mean(np.square(data), axis=1))
        std = np.std(data, axis=1)
        ptp = np.ptp(data, axis=1)
        nan_fraction = np.mean(~np.isfinite(data), axis=1)
        flat_fraction = np.mean(np.abs(np.diff(data, axis=1)) <= 1e-15, axis=1) if data.shape[1] > 1 else np.ones(data.shape[0])
        log_rms_z = _robust_modified_z(np.log10(np.maximum(rms, np.finfo(float).tiny)))
        flat_channels = [selected[i] for i, value in enumerate(ptp) if _safe_float(value) is not None and float(value) <= 1e-15]
        outlier_threshold = float(payload.get("outlier_z_threshold") or 8.0)
        outlier_channels = [selected[i] for i, value in enumerate(log_rms_z) if np.isfinite(value) and float(value) >= outlier_threshold]
        per_channel = []
        for i, channel in enumerate(selected):
            per_channel.append(
                {
                    "channel": channel,
                    "rms": float(rms[i]),
                    "std": float(std[i]),
                    "peak_to_peak": float(ptp[i]),
                    "nan_fraction": float(nan_fraction[i]),
                    "flat_fraction": float(flat_fraction[i]),
                    "log_rms_modified_z": float(log_rms_z[i]) if np.isfinite(log_rms_z[i]) else None,
                }
            )
    else:
        flat_channels = []
        outlier_channels = []
        per_channel = []

    return json_safe(
        {
            "status": "complete",
            "input_path": str(path),
            "sample_freq": sfreq,
            "analysis_window": {
                "start_seconds": start_seconds,
                "stop_seconds": stop_seconds,
                "samples_analyzed": int(max(0, stop - start)),
                "bounded": stop < int(getattr(raw, "n_times", 0) or 0),
            },
            "roi_channel_count": len(selected),
            "excluded_channels": sorted(exclude),
            "flat_channels": flat_channels,
            "outlier_channels": outlier_channels,
            "per_channel": per_channel,
            "warnings": [
                "QC metrics are heuristic and should be visually reviewed before dropping channels.",
                "QC is computed on a bounded preview window by default; set max_duration_seconds explicitly for broader review.",
            ],
        }
    )


def compute_spectral_profile(payload: dict[str, Any]) -> dict[str, Any]:
    path, raw = _read_raw(payload.get("input_path"))
    channel_names = [str(channel) for channel in raw.ch_names]
    roi = payload.get("roi") if isinstance(payload.get("roi"), dict) else infer_brain_signal_roi_from_channels(channel_names)
    exclude = set(str(ch) for ch in (payload.get("exclude_channels") or roi.get("default_detection_exclusions") or []))
    selected = [ch for ch in channel_names if ch not in exclude]
    max_channels = int(payload.get("max_channels") or 32)
    picks = [channel_names.index(ch) for ch in selected[:max_channels]]
    start, stop, start_seconds, stop_seconds = _analysis_window(raw, payload)
    data = _get_window_data(raw, picks, start, stop) if picks else np.empty((0, max(0, stop - start)))
    sfreq = float(raw.info.get("sfreq") or 0.0)
    nyquist = sfreq / 2.0 if sfreq else 0.0
    if data.size and data.shape[1] > 1 and sfreq > 0:
        demeaned = data - np.mean(data, axis=1, keepdims=True)
        freqs = np.fft.rfftfreq(data.shape[1], d=1.0 / sfreq)
        psd = np.mean(np.abs(np.fft.rfft(demeaned, axis=1)) ** 2, axis=0)
        psd_curve = _compact_psd_curve(freqs, psd, max_points=int(payload.get("max_psd_points") or 96))
        total_power = float(np.sum(psd)) or 1.0
        bands = {}
        for label, low, high in [
            ("delta", 1, 4),
            ("theta", 4, 8),
            ("alpha", 8, 13),
            ("beta", 13, 30),
            ("gamma", 30, 80),
            ("ripple", 80, min(250, nyquist)),
        ]:
            mask = (freqs >= low) & (freqs < high)
            bands[label] = float(np.sum(psd[mask]) / total_power) if np.any(mask) else 0.0
        line_candidates = []
        for freq in [50.0, 60.0, 100.0, 120.0, 150.0, 180.0, 200.0, 240.0]:
            if freq >= nyquist:
                continue
            narrow = (freqs >= freq - 1.0) & (freqs <= freq + 1.0)
            shoulder = (freqs >= freq - 5.0) & (freqs <= freq + 5.0) & ~narrow
            if np.any(narrow) and np.any(shoulder):
                ratio = float((np.mean(psd[narrow]) + 1e-30) / (np.mean(psd[shoulder]) + 1e-30))
                if ratio >= float(payload.get("line_ratio_threshold") or 5.0):
                    line_candidates.append({"frequency_hz": freq, "local_power_ratio": ratio})
    else:
        bands = {}
        line_candidates = []
        psd_curve = []

    hfo_upper = min(500.0, 0.95 * nyquist)
    recommended_hfo_band = [80.0, hfo_upper] if hfo_upper > 80 else None
    return json_safe(
        {
            "status": "complete",
            "input_path": str(path),
            "sample_freq": sfreq,
            "nyquist_hz": nyquist,
            "analysis_window": {
                "start_seconds": start_seconds,
                "stop_seconds": stop_seconds,
                "samples_analyzed": int(max(0, stop - start)),
                "bounded": stop < int(getattr(raw, "n_times", 0) or 0),
            },
            "channels_analyzed": selected[:max_channels],
            "roi_channel_count": len(selected),
            "relative_band_power": bands,
            "psd_curve": psd_curve,
            "line_noise_candidates": line_candidates,
            "recommended_hfo_band": recommended_hfo_band,
            "warnings": [
                "Spectral profile is a compact FFT summary for planning, not a publication-grade PSD estimate.",
                "Spectral profile is computed on a bounded preview window by default; set max_duration_seconds explicitly for broader review.",
                "Confirm line-noise frequency before applying notch filters.",
            ],
        }
    )


def _payload_candidate_freqs(payload: dict[str, Any]) -> list[float]:
    raw = payload.get("candidate_frequencies_hz") or payload.get("candidate_freqs") or [50.0, 60.0]
    freqs: list[float] = []
    for item in raw if isinstance(raw, list) else [raw]:
        value = _safe_float(item)
        if value and value > 0 and value not in freqs:
            freqs.append(value)
    return freqs or [50.0, 60.0]


def detect_line_noise(payload: dict[str, Any]) -> dict[str, Any]:
    path, raw = _read_raw(payload.get("input_path"))
    channel_names = [str(channel) for channel in raw.ch_names]
    roi = payload.get("roi") if isinstance(payload.get("roi"), dict) else infer_brain_signal_roi_from_channels(channel_names)
    exclude = set(str(ch) for ch in (payload.get("exclude_channels") or roi.get("default_detection_exclusions") or []))
    selected = [ch for ch in channel_names if ch not in exclude]
    max_channels = int(payload.get("max_channels") or 32)
    picks = [channel_names.index(ch) for ch in selected[:max_channels]]
    start, stop, start_seconds, stop_seconds = _analysis_window(raw, payload)
    data = _get_window_data(raw, picks, start, stop) if picks else np.empty((0, max(0, stop - start)))
    sfreq = float(raw.info.get("sfreq") or 0.0)
    nyquist = sfreq / 2.0 if sfreq else 0.0
    bandwidth = max(0.05, float(payload.get("bandwidth_hz") or 1.0))
    shoulder_width = max(bandwidth * 2.5, float(payload.get("shoulder_width_hz") or 5.0))
    ratio_threshold = max(1.0, float(payload.get("ratio_threshold") or 5.0))
    max_harmonics = max(1, int(payload.get("max_harmonics") or 4))
    candidate_frequencies = _payload_candidate_freqs(payload)
    metadata_line_freq = _safe_float(raw.info.get("line_freq"))
    user_line_freq = _safe_float(payload.get("line_noise_notch_hz") or payload.get("site_line_freq_hz"))

    detected: list[dict[str, Any]] = []
    if data.size and data.shape[1] > 1 and sfreq > 0:
        demeaned = data - np.mean(data, axis=1, keepdims=True)
        freqs = np.fft.rfftfreq(data.shape[1], d=1.0 / sfreq)
        psd_by_channel = np.abs(np.fft.rfft(demeaned, axis=1)) ** 2
        for base_freq in candidate_frequencies:
            for harmonic in range(1, max_harmonics + 1):
                freq = base_freq * harmonic
                if freq <= 0 or freq >= nyquist:
                    continue
                narrow = (freqs >= freq - bandwidth) & (freqs <= freq + bandwidth)
                shoulder = (freqs >= freq - shoulder_width) & (freqs <= freq + shoulder_width) & ~narrow
                if not np.any(narrow) or not np.any(shoulder):
                    continue
                ratios = (np.mean(psd_by_channel[:, narrow], axis=1) + 1e-30) / (np.mean(psd_by_channel[:, shoulder], axis=1) + 1e-30)
                strong = [
                    {"channel": selected[i], "local_power_ratio": float(ratio)}
                    for i, ratio in enumerate(ratios)
                    if np.isfinite(ratio) and float(ratio) >= ratio_threshold
                ]
                mean_ratio = float(np.mean(ratios[np.isfinite(ratios)])) if np.any(np.isfinite(ratios)) else 0.0
                if strong:
                    detected.append(
                        {
                            "frequency_hz": float(freq),
                            "base_frequency_hz": float(base_freq),
                            "harmonic": harmonic,
                            "mean_power_ratio": mean_ratio,
                            "channel_count_above_threshold": len(strong),
                            "channels_above_threshold": strong,
                        }
                    )
    detected.sort(key=lambda item: (float(item["base_frequency_hz"]), int(item["harmonic"])))

    explicit_freq = user_line_freq or metadata_line_freq
    observed_base_freqs = {float(item["base_frequency_hz"]) for item in detected}
    warnings = [
        "Line-noise detection is a bounded FFT heuristic for preprocessing decisions, not a clinical interpretation.",
        "Do not apply a notch filter solely from geography; use header/user site frequency or ask for confirmation.",
    ]
    if explicit_freq and explicit_freq < nyquist:
        decision = {
            "should_notch": True,
            "frequency_hz": explicit_freq,
            "confidence": "high" if explicit_freq in observed_base_freqs else "medium",
            "requires_user_confirmation": False,
            "rationale": "Line frequency was explicitly available from user request or recording metadata.",
        }
    elif observed_base_freqs:
        decision = {
            "should_notch": False,
            "frequency_hz": sorted(observed_base_freqs)[0],
            "confidence": "review",
            "requires_user_confirmation": True,
            "rationale": "A candidate mains peak was detected, but no site/header line frequency was explicitly available.",
        }
    else:
        decision = {
            "should_notch": False,
            "frequency_hz": None,
            "confidence": "low",
            "requires_user_confirmation": False,
            "rationale": "No candidate 50/60 Hz mains peak exceeded the configured local power ratio threshold in the bounded preview.",
        }

    return json_safe(
        {
            "status": "complete",
            "input_path": str(path),
            "sample_freq": sfreq,
            "nyquist_hz": nyquist,
            "metadata_line_freq_hz": metadata_line_freq,
            "user_or_site_line_freq_hz": user_line_freq,
            "analysis_window": {
                "start_seconds": start_seconds,
                "stop_seconds": stop_seconds,
                "samples_analyzed": int(max(0, stop - start)),
                "bounded": stop < int(getattr(raw, "n_times", 0) or 0),
            },
            "channels_analyzed": selected[:max_channels],
            "excluded_channels": sorted(exclude),
            "candidate_frequencies_hz": candidate_frequencies,
            "ratio_threshold": ratio_threshold,
            "detected_frequencies": detected,
            "decision": decision,
            "recommended_notch_hz": decision["frequency_hz"] if decision["should_notch"] else None,
            "warnings": warnings,
        }
    )


def _compact_psd_curve(freqs: np.ndarray, psd: np.ndarray, *, max_points: int = 96) -> list[dict[str, float]]:
    if freqs.size == 0 or psd.size == 0:
        return []
    max_points = max(8, min(512, int(max_points or 96)))
    indices = np.linspace(0, freqs.size - 1, min(max_points, freqs.size), dtype=int)
    values = np.asarray(psd[indices], dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return []
    floor = max(float(np.min(finite[finite > 0])) if np.any(finite > 0) else 1e-30, 1e-30)
    db = 10.0 * np.log10(np.maximum(values, floor))
    finite_db = db[np.isfinite(db)]
    min_db = float(np.min(finite_db)) if finite_db.size else 0.0
    max_db = float(np.max(finite_db)) if finite_db.size else min_db + 1.0
    span = max(max_db - min_db, 1e-9)
    curve = []
    for index, value_db in zip(indices, db):
        if not np.isfinite(value_db):
            continue
        curve.append(
            {
                "frequency_hz": float(freqs[index]),
                "db": float(value_db),
                "relative_power": float((value_db - min_db) / span),
            }
        )
    return curve


def recommend_detector_plan(payload: dict[str, Any]) -> dict[str, Any]:
    info = payload.get("recording_info") if isinstance(payload.get("recording_info"), dict) else get_recording_info(payload.get("input_path"))
    channel_names = [str(ch) for ch in info.get("channel_names") or []]
    roi = payload.get("roi") if isinstance(payload.get("roi"), dict) else infer_brain_signal_roi_from_channels(channel_names)
    sfreq = float(info.get("sample_freq") or 0.0)
    duration = float(info.get("duration_seconds") or 0.0)
    nyquist = sfreq / 2.0 if sfreq else 0.0
    runs: list[dict[str, Any]] = []
    warnings: list[str] = []
    if duration < 10:
        warnings.append("Recording is shorter than 10 seconds; do not run HFO statistics, only metadata/QC.")
    elif duration < 60:
        warnings.append("Recording is short; treat detector output as smoke-test evidence only.")
    if roi.get("brain_signal_channel_count", 0) <= 0:
        warnings.append("No brain-signal ROI channels are available for detection.")
    upper = min(500.0, 0.95 * nyquist) if nyquist else 0.0
    if upper > 80 and duration >= 10 and roi.get("brain_signal_channel_count", 0) > 0:
        runs.append(
            {
                "biomarker": "HFO",
                "detector": "STE",
                "filter": {"fp": 80, "fs": round(upper, 3)},
                "n_jobs": 1,
                "exports": ["summary-json", "csv"],
                "reason": "Nyquist-safe conservative HFO screen on label-free brain ROI.",
            }
        )
    if duration >= 10 and roi.get("brain_signal_channel_count", 0) > 0:
        runs.append(
            {
                "biomarker": "Spike",
                "detector": "RMS/LL",
                # Emit a Nyquist-safe filter (like the HFO run) so the recommended Spike
                # screen doesn't fall back to the static fs=80 default and fail the
                # Nyquist guard on low-sample-rate recordings (e.g. <=160 Hz).
                "filter": {"fp": 4, "fs": round(min(80.0, 0.95 * nyquist), 3) if nyquist else 80.0},
                "n_jobs": 1,
                "exports": ["summary-json", "csv"],
                "reason": "Spike screen for candidate epileptiform activity on label-free brain ROI.",
            }
        )
    return json_safe(
        {
            "status": "complete",
            "sample_freq": sfreq,
            "duration_seconds": duration,
            "nyquist_hz": nyquist,
            "roi_channel_count": roi.get("brain_signal_channel_count"),
            "excluded_channels": roi.get("default_detection_exclusions", []),
            "recommended_runs": runs,
            "warnings": warnings,
            "interpretation_policy": "Detector outputs are algorithmic candidate evidence, not clinical pathology labels.",
        }
    )


def _read_events_csv(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(str(path)).expanduser().resolve()
    if not csv_path.exists():
        raise PyHFOAgentError("file_not_found", f"Events CSV does not exist: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _channel_column(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "channel_names"
    for key in ["channel_names", "channel_name", "channel", "channels"]:
        if key in rows[0]:
            return key
    raise PyHFOAgentError("invalid_artifact", "Could not find channel column in events CSV.")


def rank_candidate_channels(payload: dict[str, Any]) -> dict[str, Any]:
    duration_seconds = float(payload.get("duration_seconds") or 0.0)
    exclude = set(str(ch) for ch in payload.get("exclude_channels") or [])
    runs = payload.get("runs") or []
    if not isinstance(runs, list) or not runs:
        raise PyHFOAgentError("invalid_request", "runs must be a non-empty list of {biomarker, events_csv} objects.")
    counts_by_channel: dict[str, Counter] = defaultdict(Counter)
    artifacts: dict[str, str] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        biomarker = str(run.get("biomarker") or "event").lower()
        events_csv = run.get("events_csv") or run.get("csv")
        rows = _read_events_csv(events_csv)
        artifacts[biomarker] = str(Path(str(events_csv)).expanduser().resolve())
        if not rows:
            continue
        column = _channel_column(rows)
        for row in rows:
            channel = str(row.get(column) or "")
            if not channel or channel in exclude:
                continue
            counts_by_channel[channel][biomarker] += 1
    ranked = []
    for channel, counts in counts_by_channel.items():
        total = int(sum(counts.values()))
        item = {"channel": channel, "total_events": total}
        for biomarker, count in sorted(counts.items()):
            item[f"{biomarker}_count"] = int(count)
            if duration_seconds > 0:
                item[f"{biomarker}_per_minute"] = float(count / (duration_seconds / 60.0))
        if duration_seconds > 0:
            item["total_events_per_minute"] = float(total / (duration_seconds / 60.0))
        ranked.append(item)
    ranked.sort(key=lambda item: (-int(item["total_events"]), str(item["channel"])))
    return json_safe(
        {
            "status": "complete",
            "duration_seconds": duration_seconds,
            "excluded_channels": sorted(exclude),
            "artifact_paths": artifacts,
            "candidate_channel_count": len(ranked),
            "top_channels": ranked[: int(payload.get("top_n") or 20)],
            "interpretation_policy": "Rankings are signal-derived algorithmic candidates after ROI exclusions; they are not clinical labels.",
        }
    )
