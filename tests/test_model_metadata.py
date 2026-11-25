from types import SimpleNamespace

from src.utils.model_metadata import DEFAULT_PREPROCESSING_BY_MODEL, resolve_preprocessing_metadata


def test_resolve_preprocessing_metadata_prefers_nested_preprocessing_block():
    config = {
        "preprocessing": {
            "freq_range_hz": [20, 300],
            "fs": 1000,
            "image_size": 128,
            "random_shift_ms": 25,
            "selected_freq_range_hz": [40, 200],
            "selected_window_size_ms": 150,
            "time_range_ms": [0, 500],
        }
    }

    resolved = resolve_preprocessing_metadata(config, fallback=DEFAULT_PREPROCESSING_BY_MODEL["artifact"])

    assert resolved["fs"] == 1000
    assert resolved["selected_window_size_ms"] == 150
    assert resolved["random_shift_ms"] == 25


def test_resolve_preprocessing_metadata_supports_legacy_field_names():
    config = SimpleNamespace(
        freq_range=[10, 500],
        fs=2000,
        image_size=224,
        crop_freq=[80, 224],
        crop_time=100,
        time_range=[0, 1000],
    )

    resolved = resolve_preprocessing_metadata(config)

    assert resolved["freq_range_hz"] == [10, 500]
    assert resolved["selected_freq_range_hz"] == [80, 224]
    assert resolved["selected_window_size_ms"] == 100
    assert resolved["time_range_ms"] == [0, 1000]
    assert resolved["random_shift_ms"] == 0


def test_resolve_preprocessing_metadata_falls_back_when_config_is_missing_metadata():
    resolved = resolve_preprocessing_metadata({}, fallback=DEFAULT_PREPROCESSING_BY_MODEL["ehfo"])

    assert resolved == DEFAULT_PREPROCESSING_BY_MODEL["ehfo"]
