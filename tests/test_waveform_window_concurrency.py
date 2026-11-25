"""Regression guards for the waveform scroll-storm fix.

Background: scrolling a long, filtered recording used to crash the renderer with
`waveform:window ... Network error: fetch failed`. Three compounding causes, each
guarded below so a revert turns red in CI instead of in the user's session:

  1. /waveform/window re-read & re-parsed the ENTIRE file (load_mne_raw) on every
     request -> a scroll gesture flooded the backend with dozens of full reads
     (the repeated MNE "filename does not conform" warnings) and multi-second
     latency. Now cached, keyed by (path, mtime, size), with a per-key load lock
     so a simultaneous cold burst collapses to a single read.
  2. ThreadingHTTPServer's default listen backlog of 5 overflowed under the burst
     -> connections were reset/timed out. Raised to 128.
  3. (frontend, not covered here) updatePlot now single-flights navigation.

These assert the *mechanisms* deterministically rather than the timing-dependent
connection timeout, which would be flaky in CI. The real-recording before/after
(131/250 undici requests failing on old code, 0 on new) was verified manually.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.parse
import urllib.request

import pytest

from pyhfo.agent_api import waveform as waveform_module
from pyhfo.agent_api.server import build_server
from pyhfo.agent_api.sidecar import PyHFOSidecarSession
from pyhfo.agent_api.waveform import waveform_window

CHANNELS = ["B10Ref", "A2Ref"]


@pytest.fixture(autouse=True)
def _clear_raw_cache():
    """Each test starts and ends with a cold cache so reads are deterministic.

    Defensive getattr so that if the cache is ever removed the assertion-level
    guards below fail with a meaningful message instead of an AttributeError here.
    """
    def reset():
        for name in ("_RAW_CACHE", "_RAW_LOAD_LOCKS", "_WINDOW_CACHE"):
            container = getattr(waveform_module, name, None)
            if container is not None:
                container.clear()

    reset()
    yield
    reset()


@pytest.fixture
def counting_loader(monkeypatch):
    """Wrap the real load_mne_raw so tests can assert how often the file is read."""
    real_loader = waveform_module.load_mne_raw
    calls = {"n": 0}
    count_lock = threading.Lock()

    def wrapper(path):
        with count_lock:
            calls["n"] += 1
        return real_loader(path)

    monkeypatch.setattr(waveform_module, "load_mne_raw", wrapper)
    return calls


def test_repeated_window_requests_read_the_file_once(tiny_fif_path, counting_loader):
    # A scroll across the recording is many windows for the SAME file: it must
    # parse the file once, not once per request (the warning-flood regression).
    for step in range(20):
        result = waveform_window(
            tiny_fif_path,
            start_seconds=step * 0.05,
            duration_seconds=0.5,
            channels=CHANNELS,
            target_samples_per_channel=50,
        )
        assert result["status"] == "complete"
    assert counting_loader["n"] == 1


def test_concurrent_cold_requests_collapse_to_single_read(tiny_fif_path, counting_loader):
    # Thundering-herd guard: a simultaneous burst of cold-cache requests (e.g. the
    # coarse+full initial pass, or a fast drag before the first load lands) must
    # collapse to ONE read, not one read per concurrent request.
    n = 32
    barrier = threading.Barrier(n)
    samples: list[int] = []
    errors: list[str] = []

    def worker(i: int):
        try:
            barrier.wait(timeout=10)
            result = waveform_window(
                tiny_fif_path,
                start_seconds=(i % 10) * 0.05,
                duration_seconds=0.5,
                channels=CHANNELS,
                target_samples_per_channel=50,
            )
            samples.append(result["returned_samples_per_channel"])
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(samples) == n
    assert counting_loader["n"] == 1


def test_cache_invalidates_when_file_changes(tiny_fif_path, tiny_mne_raw, counting_loader):
    # The cache key is (path, mtime, size): re-preparing/overwriting the file must
    # force a fresh read so stale signal is never served.
    waveform_window(tiny_fif_path, duration_seconds=0.5, channels=["B10Ref"], target_samples_per_channel=20)
    assert counting_loader["n"] == 1

    time.sleep(0.01)
    tiny_mne_raw.save(tiny_fif_path, overwrite=True, verbose=False)  # new mtime

    waveform_window(tiny_fif_path, duration_seconds=0.5, channels=["B10Ref"], target_samples_per_channel=20)
    assert counting_loader["n"] == 2


def test_local_http_server_uses_large_listen_backlog():
    # Default ThreadingHTTPServer backlog is 5; scroll bursts overflowed it and the
    # kernel reset the renderer's fetch(). Keep it large.
    server = build_server("127.0.0.1", 0)
    try:
        assert server.request_queue_size >= 128
        assert server.daemon_threads is True
    finally:
        server.server_close()


def test_concurrent_http_window_requests_all_succeed(tiny_fif_path):
    # End-to-end through the real threaded HTTP server: a simultaneous burst of
    # filtered /waveform/window requests must all return cleanly (no reset, no
    # race on the shared cached raw via the per-entry read lock).
    server = build_server("127.0.0.1", 0, PyHFOSidecarSession(session_id="concurrency"))
    host, port = server.server_address[0], int(server.server_address[1])
    serve_thread = threading.Thread(target=server.serve_forever, daemon=True)
    serve_thread.start()
    try:
        base = f"http://{host}:{port}"
        n = 40
        barrier = threading.Barrier(n)
        statuses: list[str] = []
        errors: list[str] = []

        def fire(i: int):
            query = urllib.parse.urlencode({
                "input_path": str(tiny_fif_path),
                "start_seconds": f"{(i % 10) * 0.05:.3f}",
                "duration_seconds": "0.5",
                "channels": ",".join(CHANNELS),
                "target_samples_per_channel": "50",
                "show_filtered": "true",
                "filter_60hz": "true",
            })
            try:
                barrier.wait(timeout=10)
                with urllib.request.urlopen(f"{base}/waveform/window?{query}", timeout=20) as response:
                    body = json.loads(response.read())
                statuses.append(body.get("status"))
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(repr(exc))

        threads = [threading.Thread(target=fire, args=(i,)) for i in range(n)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == [], errors[:3]
        assert statuses == ["complete"] * n
    finally:
        server.shutdown()
        server.server_close()
        serve_thread.join(timeout=5)
