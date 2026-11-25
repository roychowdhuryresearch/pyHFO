"""Regression guard for the prepare -> waveform cache seed.

Background: after prepare_recording() writes a prepared .fif, the renderer's very
first /waveform/window for that file used to cold-load it again via load_mne_raw().
When a detection is saturating the single-process backend, that cold parse is the
slow step that blows the renderer's 20s waveform timeout ("request exceeded 20000ms
... was aborted"). prepare_recording() now primes the raw-cache with the in-memory
prepared object via waveform.seed_raw_cache(), so the first window reuses it.

These assert the *mechanism* (no second file read after seeding / after prepare),
which is deterministic, rather than the timing-dependent timeout it prevents.
"""
from __future__ import annotations

import pytest

from pyhfo.agent_api import preprocessing as preprocessing_module
from pyhfo.agent_api import waveform as waveform_module
from pyhfo.agent_api.waveform import seed_raw_cache, waveform_window

CHANNELS = ["B10Ref", "A2Ref"]


@pytest.fixture(autouse=True)
def _clear_caches():
    def reset():
        for name in ("_RAW_CACHE", "_RAW_LOAD_LOCKS", "_WINDOW_CACHE"):
            container = getattr(waveform_module, name, None)
            if container is not None:
                container.clear()

    reset()
    yield
    reset()


def test_seeded_window_does_not_cold_load(tiny_fif_path, tiny_mne_raw, monkeypatch):
    # Prime the cache with the already-loaded raw, then make any disk read explode:
    # a cache hit must serve the window without ever touching load_mne_raw().
    seed_raw_cache(tiny_fif_path, tiny_mne_raw)

    def _boom(*_args, **_kwargs):
        raise AssertionError("waveform_window cold-loaded the file despite a seeded cache")

    monkeypatch.setattr(waveform_module, "load_mne_raw", _boom)

    result = waveform_window(
        str(tiny_fif_path),
        start_seconds=0.0,
        duration_seconds=1.0,
        channels=CHANNELS,
        target_samples_per_channel=50,
    )
    assert result["status"] == "complete"
    assert result["channels"] == CHANNELS
    assert len(result["data"]) == len(CHANNELS)


def test_seed_is_best_effort_on_bad_input():
    # Cache priming must never raise into prepare_recording, even on junk input.
    seed_raw_cache("", None)
    seed_raw_cache(None, None)  # type: ignore[arg-type]
    assert len(waveform_module._RAW_CACHE) == 0


def test_prepare_recording_seeds_the_cache(tmp_path, tiny_fif_path, monkeypatch):
    # End-to-end: after a real prepare, the first window must be served from cache
    # (no load_mne_raw) -- the exact path the renderer hits right after preparing.
    output_path = tmp_path / "prepared_raw.fif"
    preprocessing_module.prepare_recording(
        {
            "input_path": str(tiny_fif_path),
            "output_path": str(output_path),
            # Keep it a pure copy so channel names survive for the assertion below.
            "drop_bad_channels": False,
            "drop_flat_channels": False,
        }
    )
    assert output_path.exists()

    def _boom(*_args, **_kwargs):
        raise AssertionError("first window after prepare cold-loaded instead of using the seeded cache")

    monkeypatch.setattr(waveform_module, "load_mne_raw", _boom)
    result = waveform_window(
        str(output_path),
        start_seconds=0.0,
        duration_seconds=1.0,
        channels=CHANNELS,
        target_samples_per_channel=50,
    )
    assert result["status"] == "complete"
    assert result["channels"] == CHANNELS
