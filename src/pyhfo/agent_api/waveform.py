from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from pyhfo.agent_api.errors import PyHFOAgentError
from pyhfo.agent_api.serialization import json_safe
from pyhfo.utils.utils_io import load_mne_raw

MAX_WAVEFORM_CHANNELS = 32
MAX_WINDOW_DURATION_SECONDS = 60.0
MAX_TARGET_SAMPLES_PER_CHANNEL = 5000
DEFAULT_TARGET_SAMPLES_PER_CHANNEL = 1200
MAX_SOURCE_SAMPLES_PER_CHANNEL = 250_000
MAX_OVERVIEW_BINS = 2000



def _safe_sosfiltfilt(sos: np.ndarray, data: np.ndarray) -> np.ndarray:
    try:
        return signal.sosfiltfilt(sos, data, axis=1)
    except ValueError:
        return signal.sosfilt(sos, data, axis=1)


def _safe_notch_filter(data: np.ndarray, sfreq: float, notch_hz: float = 60.0) -> np.ndarray:
    nyquist = sfreq / 2.0
    if notch_hz <= 0 or notch_hz >= nyquist:
        return data
    b, a = signal.iirnotch(w0=notch_hz, Q=30.0, fs=sfreq)
    try:
        return signal.filtfilt(b, a, data, axis=1)
    except ValueError:
        return signal.lfilter(b, a, data, axis=1)


def _safe_bandpass_filter(data: np.ndarray, sfreq: float, low_hz: float = 1.0, high_hz: float = 80.0) -> np.ndarray:
    nyquist = sfreq / 2.0
    low = max(0.001, float(low_hz))
    high = min(float(high_hz), nyquist * 0.95)
    if high <= low or data.shape[1] < 3:
        return data
    sos = signal.butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return _safe_sosfiltfilt(sos, data)


def _apply_waveform_filters(
    data: np.ndarray,
    sfreq: float,
    *,
    show_filtered: bool = False,
    filter_60hz: bool = False,
    filter_low: float | None = None,
    filter_high: float | None = None,
) -> tuple[np.ndarray, list[str]]:
    filters: list[str] = []
    filtered = np.asarray(data, dtype=float)
    if show_filtered:
        # An explicit display band (e.g. an HFO run's 80-500 Hz, a spindle run's
        # 11-16 Hz) overrides the default broadband clinical view. Without it the
        # fixed 1-80 Hz pass would HIDE an HFO entirely, so a per-run band is what
        # lets the waveform actually reveal the active biomarker's morphology.
        if filter_low is not None and filter_high is not None and float(filter_high) > float(filter_low) > 0:
            low = float(filter_low)
            high = float(filter_high)
            filtered = _safe_bandpass_filter(filtered, sfreq, low, high)
            filters.append(f"bandpass_{int(round(low))}_{int(round(high))}hz")
        else:
            filtered = _safe_bandpass_filter(filtered, sfreq)
            filters.append("bandpass_1_80hz")
    if filter_60hz:
        filtered = _safe_notch_filter(filtered, sfreq, 60.0)
        filters.append("notch_60hz")
    return filtered, filters

def _resolve_path(input_path: str | Path | None) -> Path:
    if not input_path:
        raise PyHFOAgentError("invalid_request", "input_path is required.")
    path = Path(str(input_path)).expanduser().resolve()
    if not path.exists():
        raise PyHFOAgentError("file_not_found", f"Input file does not exist: {path}")
    return path


# ---------------------------------------------------------------------------
# Raw-object cache.
#
# Every /waveform/window request used to re-open and re-parse the whole recording
# from disk via load_mne_raw(). Under rapid scrolling that meant dozens of full
# re-reads per gesture: the backend log filled with identical MNE "filename does
# not conform" warnings, each handler thread ran long, the accept loop starved
# for the GIL, and the tiny TCP listen backlog overflowed -> the renderer's
# fetch() was reset before any response ("request failed before response").
#
# We cache the loaded raw keyed by (path, mtime, size) so repeated windows reuse
# one handle, and re-load automatically when the file changes (e.g. re-prepared).
# A per-entry lock serializes the (now cheap) segment reads so concurrent scroll
# requests never race on a shared MNE object.
# ---------------------------------------------------------------------------
_RawCacheKey = tuple[str, "int | None", "int | None"]
_RAW_CACHE: "OrderedDict[_RawCacheKey, tuple[Any, threading.Lock]]" = OrderedDict()
_RAW_CACHE_LOCK = threading.Lock()
# Per-key load locks so a simultaneous cold burst for ONE file collapses to a
# single read (thundering-herd guard) while different files still load in
# parallel. The cache lock only guards the dicts, never the slow load itself.
_RAW_LOAD_LOCKS: "dict[_RawCacheKey, threading.Lock]" = {}
_RAW_CACHE_MAX_ENTRIES = 6

# ---------------------------------------------------------------------------
# Computed-window cache.
#
# Caching the raw handle stopped the per-request file re-parse, but every scroll
# step STILL re-read the segment from disk (preload=False) and re-ran the
# filters. Dragging back and forth over a long, high-sample-rate recording
# therefore recomputed identical windows over and over, keeping CPU/allocation
# churn high (~160ms/scroll on a 2 kHz, 32-channel, 30 s window) and adding to
# the memory pressure that can get the local backend SIGKILLed.
#
# We memoize the final (scaled, filtered, downsampled) window payload keyed by
# the full request signature AND file identity (path, mtime, size). A revisited
# window is then a dict lookup: no disk read, no filtfilt, no fresh allocation.
# Bounded LRU so the cache footprint stays flat regardless of session length;
# the (path, mtime, size) prefix auto-invalidates when a recording is re-prepared.
# ---------------------------------------------------------------------------
_WINDOW_CACHE: "OrderedDict[tuple, dict[str, Any]]" = OrderedDict()
_WINDOW_CACHE_LOCK = threading.Lock()
_WINDOW_CACHE_MAX_ENTRIES = 64


def _window_cache_key(
    path: Path,
    start: float,
    duration: float,
    channels: Any,
    target: int | None,
    units: str,
    show_filtered: bool,
    filter_60hz: bool,
    filter_low: float | None = None,
    filter_high: float | None = None,
) -> tuple:
    if channels in (None, "", []):
        ch_key: Any = None
    elif isinstance(channels, (list, tuple)):
        ch_key = tuple(str(c) for c in channels)
    else:
        ch_key = (str(channels),)
    return (
        *_raw_cache_key(path),
        round(float(start), 6),
        round(float(duration), 6),
        ch_key,
        int(target) if target else None,
        str(units).lower(),
        bool(show_filtered),
        bool(filter_60hz),
        round(float(filter_low), 3) if filter_low is not None else None,
        round(float(filter_high), 3) if filter_high is not None else None,
    )


def _raw_cache_key(path: Path) -> _RawCacheKey:
    try:
        stat = path.stat()
        return (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (str(path), None, None)


def _get_cached_raw(input_path: str | Path) -> tuple[Any, threading.Lock]:
    path = _resolve_path(input_path)
    key = _raw_cache_key(path)
    with _RAW_CACHE_LOCK:
        entry = _RAW_CACHE.get(key)
        if entry is not None:
            _RAW_CACHE.move_to_end(key)
            return entry
        load_lock = _RAW_LOAD_LOCKS.setdefault(key, threading.Lock())
    # Serialize cold loads of the same file: the first thread reads it, the rest
    # wait here and then hit the cache below instead of each re-reading the file.
    with load_lock:
        with _RAW_CACHE_LOCK:
            entry = _RAW_CACHE.get(key)
            if entry is not None:
                _RAW_CACHE.move_to_end(key)
                return entry
        raw = load_mne_raw(str(path))
        entry = (raw, threading.Lock())
        with _RAW_CACHE_LOCK:
            _RAW_CACHE[key] = entry
            _RAW_CACHE.move_to_end(key)
            while len(_RAW_CACHE) > _RAW_CACHE_MAX_ENTRIES:
                _RAW_CACHE.popitem(last=False)
            _RAW_LOAD_LOCKS.pop(key, None)
        return entry


def seed_raw_cache(input_path: str | Path, raw: Any) -> None:
    """Pre-populate the raw cache with an already-loaded MNE raw object.

    prepare_recording() holds the freshly-filtered recording in memory the moment
    it writes the prepared .fif. Without this, the very first /waveform/window for
    that file cold-loads it again via load_mne_raw() -- and when a detection is
    saturating the single-process backend's CPU, that cold parse is the slow step
    that blows the renderer's 20s waveform timeout (the "request exceeded 20000ms"
    error). Seeding lets the first window reuse the in-memory object (a fast in-RAM
    slice) instead. The (path, mtime, size) key auto-invalidates on re-prepare.

    Best-effort: never raise into the caller -- cache priming must not fail prepare.
    """
    if raw is None:
        return
    try:
        path = Path(str(input_path)).expanduser().resolve()
    except (OSError, ValueError):
        return
    key = _raw_cache_key(path)
    entry = (raw, threading.Lock())
    with _RAW_CACHE_LOCK:
        _RAW_CACHE[key] = entry
        _RAW_CACHE.move_to_end(key)
        while len(_RAW_CACHE) > _RAW_CACHE_MAX_ENTRIES:
            _RAW_CACHE.popitem(last=False)
        _RAW_LOAD_LOCKS.pop(key, None)


def _read_raw(input_path: str | Path):
    raw, _read_lock = _get_cached_raw(input_path)
    return raw


def _channel_indices(raw, channels: Any = None) -> tuple[list[int], list[str]]:
    names = [str(channel) for channel in raw.info.get("ch_names", [])]
    if channels in (None, "", []):
        if len(names) > MAX_WAVEFORM_CHANNELS:
            raise PyHFOAgentError("request_too_large", f"Explicit channel selection is required when a recording has more than {MAX_WAVEFORM_CHANNELS} channels.")
        return list(range(len(names))), names
    requested = [str(ch) for ch in (channels if isinstance(channels, (list, tuple)) else [channels])]
    missing = [ch for ch in requested if ch not in names]
    if missing:
        raise PyHFOAgentError(
            "channel_not_found",
            f"Requested channels are not present: {missing}",
            details={"missing_channels": missing, "available_channels": names},
        )
    if len(requested) > MAX_WAVEFORM_CHANNELS:
        raise PyHFOAgentError("request_too_large", f"At most {MAX_WAVEFORM_CHANNELS} channels may be requested at once.")
    return [names.index(ch) for ch in requested], requested


def _downsample(data: np.ndarray, target_samples_per_channel: int | None) -> tuple[np.ndarray, int]:
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    n_samples = int(data.shape[1]) if data.ndim == 2 else 0
    target = int(target_samples_per_channel or DEFAULT_TARGET_SAMPLES_PER_CHANNEL)
    if target <= 0:
        raise PyHFOAgentError("invalid_request", "target_samples_per_channel must be positive.")
    if n_samples <= target:
        return data, 1
    stride = int(np.ceil(n_samples / target))
    return data[:, ::stride], stride


def waveform_channels(input_path: str | Path) -> dict[str, Any]:
    raw = _read_raw(input_path)
    sfreq = float(raw.info.get("sfreq") or 0.0)
    channels = [str(channel) for channel in raw.info.get("ch_names", [])]
    return json_safe(
        {
            "status": "complete",
            "input_path": str(_resolve_path(input_path)),
            "sample_freq": sfreq,
            "duration_seconds": float(getattr(raw, "n_times", 0) / sfreq) if sfreq else 0.0,
            "channels": [
                {"name": name, "index": idx, "type": str(raw.get_channel_types(picks=[idx])[0]) if hasattr(raw, "get_channel_types") else "unknown"}
                for idx, name in enumerate(channels)
            ],
        }
    )


def waveform_window(
    input_path: str | Path,
    *,
    start_seconds: float = 0.0,
    duration_seconds: float = 10.0,
    channels: Any = None,
    target_samples_per_channel: int | None = 1200,
    units: str = "uV",
    show_filtered: bool = False,
    filter_60hz: bool = False,
    filter_low: float | None = None,
    filter_high: float | None = None,
) -> dict[str, Any]:
    path = _resolve_path(input_path)
    cache_key = _window_cache_key(path, start_seconds, duration_seconds, channels, target_samples_per_channel, units, show_filtered, filter_60hz, filter_low, filter_high)
    with _WINDOW_CACHE_LOCK:
        cached = _WINDOW_CACHE.get(cache_key)
        if cached is not None:
            _WINDOW_CACHE.move_to_end(cache_key)
            return cached
    raw, read_lock = _get_cached_raw(path)
    sfreq = float(raw.info.get("sfreq") or 0.0)
    if sfreq <= 0:
        raise PyHFOAgentError("invalid_recording", "Recording has invalid sample frequency.")
    start = float(start_seconds)
    duration = float(duration_seconds)
    if start < 0 or duration <= 0:
        raise PyHFOAgentError("invalid_time_window", "start_seconds must be >= 0 and duration_seconds must be > 0.")
    if duration > MAX_WINDOW_DURATION_SECONDS:
        raise PyHFOAgentError("request_too_large", f"duration_seconds must be <= {MAX_WINDOW_DURATION_SECONDS}.")
    n_times = int(getattr(raw, "n_times", 0) or 0)
    first = int(round(start * sfreq))
    last = min(n_times, int(round((start + duration) * sfreq)))
    if (last - first) > MAX_SOURCE_SAMPLES_PER_CHANNEL:
        raise PyHFOAgentError("request_too_large", f"Requested source window must be <= {MAX_SOURCE_SAMPLES_PER_CHANNEL} samples per channel.")
    if first >= n_times or last <= first:
        raise PyHFOAgentError(
            "invalid_time_window",
            "Requested waveform window is outside the recording.",
            details={"start_seconds": start, "duration_seconds": duration, "recording_duration_seconds": n_times / sfreq},
        )
    picks, selected_channels = _channel_indices(raw, channels)
    with read_lock:
        data = np.asarray(raw.get_data(picks=picks, start=first, stop=last), dtype=float)
    data, filters_applied = _apply_waveform_filters(data, sfreq, show_filtered=bool(show_filtered), filter_60hz=bool(filter_60hz), filter_low=filter_low, filter_high=filter_high)
    scale = 1e6 if units.lower() in {"uv", "microvolt", "microvolts"} else 1.0
    data = data * scale
    if target_samples_per_channel and int(target_samples_per_channel) > MAX_TARGET_SAMPLES_PER_CHANNEL:
        raise PyHFOAgentError("request_too_large", f"target_samples_per_channel must be <= {MAX_TARGET_SAMPLES_PER_CHANNEL}.")
    data, stride = _downsample(data, target_samples_per_channel)
    sample_indices = np.arange(first, last, stride, dtype=int)[: data.shape[1]]
    times = sample_indices / sfreq
    result = json_safe(
        {
            "status": "complete",
            "input_path": str(path),
            "start_seconds": start,
            "duration_seconds": float((last - first) / sfreq),
            "sample_freq": sfreq,
            "source_sample_range": [first, last],
            "downsample_stride": stride,
            "target_samples_per_channel": target_samples_per_channel or DEFAULT_TARGET_SAMPLES_PER_CHANNEL,
            "returned_samples_per_channel": int(data.shape[1]),
            "units": "uV" if scale != 1.0 else units,
            "source": "filtered" if (show_filtered or filter_60hz) else "raw",
            "filters_applied": filters_applied,
            "filter_request": {"show_filtered": bool(show_filtered), "filter_60hz": bool(filter_60hz)},
            "channels": selected_channels,
            "times_seconds": times.tolist(),
            "data": data.tolist(),
        }
    )
    with _WINDOW_CACHE_LOCK:
        _WINDOW_CACHE[cache_key] = result
        _WINDOW_CACHE.move_to_end(cache_key)
        while len(_WINDOW_CACHE) > _WINDOW_CACHE_MAX_ENTRIES:
            _WINDOW_CACHE.popitem(last=False)
    return result


def waveform_overview(
    input_path: str | Path,
    *,
    channels: Any = None,
    bins: int = 512,
    units: str = "uV",
) -> dict[str, Any]:
    raw, read_lock = _get_cached_raw(input_path)
    sfreq = float(raw.info.get("sfreq") or 0.0)
    if sfreq <= 0:
        raise PyHFOAgentError("invalid_recording", "Recording has invalid sample frequency.")
    if int(bins) <= 0:
        raise PyHFOAgentError("invalid_request", "bins must be positive.")
    if int(bins) > MAX_OVERVIEW_BINS:
        raise PyHFOAgentError("request_too_large", f"bins must be <= {MAX_OVERVIEW_BINS}.")
    picks, selected_channels = _channel_indices(raw, channels)
    with read_lock:
        data = np.asarray(raw.get_data(picks=picks), dtype=float)
    scale = 1e6 if units.lower() in {"uv", "microvolt", "microvolts"} else 1.0
    data *= scale
    n = data.shape[1]
    bin_count = min(int(bins), max(1, n))
    edges = np.linspace(0, n, bin_count + 1, dtype=int)
    mins: list[list[float]] = []
    maxs: list[list[float]] = []
    rms: list[list[float]] = []
    for row in data:
        row_min = []
        row_max = []
        row_rms = []
        for left, right in zip(edges[:-1], edges[1:]):
            segment = row[left:max(right, left + 1)]
            row_min.append(float(np.min(segment)))
            row_max.append(float(np.max(segment)))
            row_rms.append(float(np.sqrt(np.mean(segment**2))))
        mins.append(row_min)
        maxs.append(row_max)
        rms.append(row_rms)
    return json_safe(
        {
            "status": "complete",
            "input_path": str(_resolve_path(input_path)),
            "sample_freq": sfreq,
            "duration_seconds": float(n / sfreq),
            "units": "uV" if scale != 1.0 else units,
            "channels": selected_channels,
            "bins": bin_count,
            "bin_edges_seconds": (edges / sfreq).tolist(),
            "min": mins,
            "max": maxs,
            "rms": rms,
        }
    )
