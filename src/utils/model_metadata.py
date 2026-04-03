from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


DEFAULT_PREPROCESSING_BY_MODEL = {
    "artifact": {
        "freq_range_hz": [10, 500],
        "fs": 2000,
        "image_size": 224,
        "random_shift_ms": 50,
        "selected_freq_range_hz": [10, 220],
        "selected_window_size_ms": 214.28571429,
        "time_range_ms": [0, 1000],
    },
    "spike": {
        "freq_range_hz": [10, 500],
        "fs": 2000,
        "image_size": 224,
        "random_shift_ms": 50,
        "selected_freq_range_hz": [10, 220],
        "selected_window_size_ms": 214.28571429,
        "time_range_ms": [0, 1000],
    },
    "ehfo": {
        "freq_range_hz": [10, 500],
        "fs": 2000,
        "image_size": 224,
        "random_shift_ms": 0,
        "selected_freq_range_hz": [10, 500],
        "selected_window_size_ms": 500,
        "time_range_ms": [0, 1000],
    },
}

_ALIAS_MAP = {
    "freq_range": "freq_range_hz",
    "time_range": "time_range_ms",
    "crop_freq": "selected_freq_range_hz",
    "crop_time": "selected_window_size_ms",
}

_REQUIRED_KEYS = {
    "freq_range_hz",
    "fs",
    "image_size",
    "selected_freq_range_hz",
    "selected_window_size_ms",
    "time_range_ms",
}


def resolve_preprocessing_metadata(config: Any, fallback: Mapping[str, Any] | None = None) -> dict[str, Any]:
    fallback_dict = _normalize_preprocessing_dict(fallback) if fallback is not None else None
    config_dict = _config_to_dict(config)

    for candidate in _iter_candidates(config_dict):
        normalized = _normalize_preprocessing_dict(candidate, fallback=fallback_dict)
        if _has_required_keys(normalized):
            return normalized

    if fallback_dict is not None:
        return fallback_dict

    raise ValueError("Model config is missing preprocessing metadata.")


def _config_to_dict(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    return dict(vars(config))


def _iter_candidates(config_dict: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates: list[Mapping[str, Any]] = []
    nested = config_dict.get("preprocessing")
    if isinstance(nested, Mapping):
        candidates.append(nested)
    candidates.append(config_dict)
    return candidates


def _normalize_preprocessing_dict(
    data: Mapping[str, Any] | None, fallback: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    normalized = deepcopy(dict(fallback)) if fallback is not None else {}
    if data is None:
        return normalized

    for key, value in data.items():
        normalized[_ALIAS_MAP.get(key, key)] = deepcopy(value)

    if "random_shift_ms" not in normalized:
        normalized["random_shift_ms"] = 0

    return normalized


def _has_required_keys(data: Mapping[str, Any]) -> bool:
    return _REQUIRED_KEYS.issubset(data.keys())
