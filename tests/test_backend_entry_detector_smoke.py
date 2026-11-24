from __future__ import annotations

import runpy
from importlib.util import find_spec
from pathlib import Path

import pytest


PYHFO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ENTRY = PYHFO_ROOT / "scripts" / "pyhfo_backend_entry.py"


@pytest.mark.skipif(find_spec("yasa") is None, reason="requires pyhfo[spindle]")
def test_backend_self_test_runs_yasa_detection_algorithm():
    entry = runpy.run_path(str(BACKEND_ENTRY))

    result = entry["_run_yasa_detector_smoke"]()

    assert result["status"] == "passed"
    assert result["algorithm"] == "spindles_detect"
    assert result["events"] >= 0
