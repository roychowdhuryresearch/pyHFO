from __future__ import annotations

import builtins
import runpy
import sys
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(find_spec("yasa") is None, reason="requires pyhfo[spindle]")


PYHFO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_HOOK = PYHFO_ROOT / "scripts" / "pyinstaller_headless_runtime.py"


def _reset_yasa_modules():
    for module_name in list(sys.modules):
        if module_name == "yasa" or module_name.startswith("yasa."):
            sys.modules.pop(module_name)


def test_frozen_runtime_hook_avoids_eager_matplotlib_import(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ModuleNotFoundError("matplotlib intentionally excluded", name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    _reset_yasa_modules()

    runpy.run_path(str(RUNTIME_HOOK))
    import yasa

    assert callable(yasa.spindles_detect)
    assert sys.modules["yasa.plotting"].__all__ == ["plot_hypnogram"]
    assert sys.modules["yasa.staging"].__all__ == []


def test_frozen_runtime_hook_preserves_real_yasa_detection():
    _reset_yasa_modules()
    runpy.run_path(str(RUNTIME_HOOK))
    import yasa

    rng = np.random.default_rng(7)
    signal = rng.normal(0.0, 8.0, 3_000)
    result = yasa.spindles_detect(signal, sf=100.0, ch_names=["C3"], verbose=False)

    assert result is None or callable(result.summary)
