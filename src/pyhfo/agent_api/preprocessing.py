from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from pyhfo.agent_api.errors import PyHFOAgentError
from pyhfo.agent_api.serialization import file_fingerprint, json_safe
from pyhfo.utils.utils_io import load_mne_raw


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _canonical_payload_sha256(value: Any) -> str:
    encoded = json.dumps(json_safe(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _parse_contact_name(name: str):
    import re

    cleaned = str(name).strip()
    if "-" in cleaned:
        return None
    match = re.match(r"^(.+?)(\d+)$", cleaned)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def auto_adjacent_bipolar_pairs(channel_names: list[str]) -> list[dict[str, str]]:
    groups: dict[str, list[tuple[int, str]]] = {}
    for channel in channel_names:
        parsed = _parse_contact_name(channel)
        if parsed is None:
            continue
        prefix, index = parsed
        groups.setdefault(prefix, []).append((index, channel))

    pairs: list[dict[str, str]] = []
    for _prefix, contacts in sorted(groups.items()):
        contacts = sorted(contacts)
        for (left_index, left_name), (right_index, right_name) in zip(contacts, contacts[1:]):
            if right_index == left_index + 1:
                pairs.append({"anode": left_name, "cathode": right_name, "name": f"{left_name}-{right_name}"})
    return pairs


def _normalize_bipolar_pairs(raw, payload: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    mode = str(payload.get("bipolar_montage") or "none").strip().lower()
    if mode in {"", "none", "off", "false"}:
        return [], "none"
    if mode == "auto":
        mode = "auto_adjacent"
    if mode == "auto_adjacent":
        return auto_adjacent_bipolar_pairs([str(channel) for channel in raw.ch_names]), mode
    if mode != "pairs":
        raise PyHFOAgentError("invalid_request", "bipolar_montage must be one of: none, auto_adjacent, pairs")

    normalized: list[dict[str, str]] = []
    for item in payload.get("bipolar_pairs") or []:
        if isinstance(item, dict):
            anode = str(item.get("anode") or item.get("ch_1") or item.get("from") or "").strip()
            cathode = str(item.get("cathode") or item.get("ch_2") or item.get("to") or "").strip()
            name = str(item.get("name") or item.get("derived_name") or f"{anode}-{cathode}").strip()
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            anode = str(item[0]).strip()
            cathode = str(item[1]).strip()
            name = str(item[2]).strip() if len(item) >= 3 and str(item[2]).strip() else f"{anode}-{cathode}"
        else:
            continue
        if anode and cathode:
            normalized.append({"anode": anode, "cathode": cathode, "name": name or f"{anode}-{cathode}"})
    return normalized, mode


def _apply_bipolar_montage(raw, payload: dict[str, Any]) -> dict[str, Any]:
    pairs, mode = _normalize_bipolar_pairs(raw, payload)
    drop_refs = _bool(payload.get("drop_original_after_bipolar"), False)
    if not pairs:
        return {
            "mode": mode,
            "pairs": [],
            "added_channels": [],
            "drop_original_after_bipolar": drop_refs,
            "warnings": (["No valid bipolar pairs were generated or provided."] if mode != "none" else []),
        }

    available = set(raw.ch_names)
    valid_pairs: list[dict[str, str]] = []
    warnings: list[str] = []
    for pair in pairs:
        if pair["anode"] not in available or pair["cathode"] not in available:
            warnings.append(f"Skipped bipolar pair {pair['anode']}-{pair['cathode']}: source channel missing after cleaning.")
            continue
        valid_pairs.append(pair)
    if not valid_pairs:
        return {
            "mode": mode,
            "pairs": [],
            "added_channels": [],
            "drop_original_after_bipolar": drop_refs,
            "warnings": warnings or ["No valid bipolar pairs remained after channel cleaning."],
        }

    import mne

    mne.set_bipolar_reference(
        raw,
        anode=[pair["anode"] for pair in valid_pairs],
        cathode=[pair["cathode"] for pair in valid_pairs],
        ch_name=[pair["name"] for pair in valid_pairs],
        drop_refs=drop_refs,
        copy=False,
        verbose="ERROR",
    )
    return {
        "mode": mode,
        "pairs": valid_pairs,
        "added_channels": [pair["name"] for pair in valid_pairs],
        "drop_original_after_bipolar": drop_refs,
        "warnings": warnings,
    }


def prepare_recording(payload: dict[str, Any]) -> dict[str, Any]:
    """Create a reproducible PyHFO-ready FIF/FIF.GZ copy.

    This is intentionally headless and JSON-first so generic agents can request
    explicit preprocessing without importing or controlling the PyQt UI.
    """

    input_path = Path(str(payload.get("input_path") or "")).expanduser().resolve()
    output_path = Path(str(payload.get("output_path") or "")).expanduser().resolve()
    if not input_path.exists():
        raise PyHFOAgentError("file_not_found", f"Input file does not exist: {input_path}")
    if not output_path.name.lower().endswith((".fif", ".fif.gz")):
        raise PyHFOAgentError("invalid_request", "output_path must end with .fif or .fif.gz")
    input_recording_fingerprint = file_fingerprint(input_path)["sha256"]
    parameters_sha256 = _canonical_payload_sha256(payload)

    raw = load_mne_raw(str(input_path))
    raw.load_data(verbose="ERROR")
    original_channels = [str(channel) for channel in raw.ch_names]
    original_sfreq = float(raw.info.get("sfreq") or 0.0)
    file_bad_channels = [str(channel) for channel in raw.info.get("bads", [])]
    extra_bad_channels = [str(channel) for channel in payload.get("extra_bad_channels") or [] if str(channel) in raw.ch_names]

    # Per-channel peak-to-peak (flat detection) and RMS (outlier detection), computed
    # in channel batches rather than one full raw.get_data() copy. load_data() above
    # already holds the recording in RAM once; a second full copy here used to DOUBLE
    # the peak RSS (~2x the recording) and — worse — stay alive through the notch/
    # bandpass/resample steps below, so the filter temporaries piled on top of 2x.
    # On a large recording that peak is what got the local backend SIGKILLed. Batching
    # caps the extra copy to ~64MB regardless of recording length; results are identical.
    n_times = int(getattr(raw, "n_times", 0) or 0)
    n_channels = len(raw.ch_names)
    peak_to_peak = np.zeros(n_channels, dtype=float)
    rms = np.zeros(n_channels, dtype=float)
    has_samples = n_times > 0 and n_channels > 0
    if has_samples:
        stats_batch = max(1, min(n_channels, (64 * 1024 * 1024) // max(1, n_times * 8)))
        for batch_start in range(0, n_channels, stats_batch):
            picks = list(range(batch_start, min(batch_start + stats_batch, n_channels)))
            segment = raw.get_data(picks=picks)
            peak_to_peak[batch_start:batch_start + len(picks)] = np.ptp(segment, axis=1)
            rms[batch_start:batch_start + len(picks)] = np.sqrt(np.mean(np.square(segment), axis=1))
            del segment
    flat_channels = [raw.ch_names[index] for index, value in enumerate(peak_to_peak) if has_samples and float(value) <= 1e-15]

    outlier_channels: list[str] = []
    if _bool(payload.get("mark_outlier_channels"), True) and has_samples:
        finite_positive = np.asarray([value for value in rms if np.isfinite(value) and value > 0], dtype=float)
        if finite_positive.size >= 3:
            log_rms = np.log10(np.maximum(rms, np.finfo(float).tiny))
            finite_log = log_rms[np.isfinite(log_rms)]
            median = float(np.median(finite_log))
            mad = float(np.median(np.abs(finite_log - median)))
            if mad > 0:
                modified_z = 0.6745 * (log_rms - median) / mad
                threshold = float(payload.get("outlier_z_threshold") or 8.0)
                outlier_channels = [raw.ch_names[index] for index, value in enumerate(modified_z) if np.isfinite(value) and value >= threshold]

    channels_to_drop: set[str] = set()
    if _bool(payload.get("drop_bad_channels"), True):
        channels_to_drop.update(file_bad_channels)
        channels_to_drop.update(extra_bad_channels)
    if _bool(payload.get("drop_flat_channels"), True):
        channels_to_drop.update(flat_channels)
    if _bool(payload.get("drop_outlier_channels"), False):
        channels_to_drop.update(outlier_channels)
    ordered_drop = [channel for channel in original_channels if channel in channels_to_drop]
    if ordered_drop:
        raw.drop_channels(ordered_drop)

    bipolar_report = _apply_bipolar_montage(raw, payload)

    notch_freqs = list(payload.get("notch_freqs") or [])
    if payload.get("line_noise_notch_hz") not in (None, ""):
        line_noise_hz = float(payload.get("line_noise_notch_hz"))
        if line_noise_hz not in [float(freq) for freq in notch_freqs]:
            notch_freqs.append(line_noise_hz)
    if notch_freqs:
        raw.notch_filter(freqs=[float(freq) for freq in notch_freqs], verbose="ERROR")

    bandpass = payload.get("bandpass") if isinstance(payload.get("bandpass"), dict) else {}
    if bandpass:
        l_freq = bandpass.get("l_freq", bandpass.get("low_freq"))
        h_freq = bandpass.get("h_freq", bandpass.get("high_freq"))
        raw.filter(
            l_freq=None if l_freq in (None, "") else float(l_freq),
            h_freq=None if h_freq in (None, "") else float(h_freq),
            verbose="ERROR",
        )

    if payload.get("resample_sfreq") not in (None, ""):
        raw.resample(float(payload.get("resample_sfreq")), verbose="ERROR")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(str(output_path), overwrite=True, verbose="ERROR")
    output_recording_fingerprint = file_fingerprint(output_path)["sha256"]
    # Prime the waveform raw-cache with the in-memory prepared recording so the
    # first /waveform/window after prepare reuses it instead of cold-loading the
    # just-written .fif -- the slow step that times out when a detection is pinning
    # the CPU. Best-effort: a cache hiccup must never fail an otherwise-good prepare.
    try:
        from pyhfo.agent_api.waveform import seed_raw_cache

        seed_raw_cache(output_path, raw)
    except Exception:
        pass
    return json_safe(
        {
            "status": "complete",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "input_recording_fingerprint": input_recording_fingerprint,
            "output_recording_fingerprint": output_recording_fingerprint,
            "parameters_sha256": parameters_sha256,
            "original_sfreq": original_sfreq,
            "prepared_sfreq": float(raw.info.get("sfreq") or 0.0),
            "original_n_channels": len(original_channels),
            "prepared_n_channels": len(raw.ch_names),
            "original_channels": original_channels,
            "prepared_channels": [str(channel) for channel in raw.ch_names],
            "file_bad_channels": file_bad_channels,
            "extra_bad_channels": extra_bad_channels,
            "flat_channels": flat_channels,
            "outlier_channels": outlier_channels,
            "dropped_channels": ordered_drop,
            "bipolar_montage": bipolar_report,
            "notch_freqs": notch_freqs,
            "line_noise_notch_hz": payload.get("line_noise_notch_hz"),
            "bandpass": bandpass,
            "resample_sfreq": payload.get("resample_sfreq"),
            "warnings": (["Outlier channels are heuristic candidates; review before aggressive exclusion."] if outlier_channels and not _bool(payload.get("drop_outlier_channels"), False) else []),
        }
    )
