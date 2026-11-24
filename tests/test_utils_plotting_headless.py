from __future__ import annotations

import builtins
import importlib
import sys

import numpy as np


def _block_gui_plotting_imports(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("matplotlib intentionally unavailable")
        if name == "PyQt5" or name.startswith("PyQt5.") or name == "pyqtgraph":
            raise ImportError("GUI plotting dependency intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_time_frequency_import_and_compute_do_not_require_matplotlib(monkeypatch):
    _block_gui_plotting_imports(monkeypatch)
    sys.modules.pop("pyhfo.utils.utils_plotting", None)

    plotting = importlib.import_module("pyhfo.utils.utils_plotting")
    signal = np.sin(np.linspace(0.0, 8.0 * np.pi, 512))
    spectrum = np.asarray(
        plotting.calculate_time_frequency(signal, 1000.0, freq_min=10.0, freq_max=400.0)
    )

    assert spectrum.ndim == 2
    assert spectrum.size > 0


def test_plot_feature_keeps_headless_jpeg_export_without_matplotlib(monkeypatch, tmp_path):
    _block_gui_plotting_imports(monkeypatch)
    sys.modules.pop("pyhfo.utils.utils_plotting", None)
    plotting = importlib.import_module("pyhfo.utils.utils_plotting")

    plotting.plot_feature(
        folder=tmp_path,
        feature=np.linspace(0.0, 1.0, 32 * 32).reshape(1, 32, 32),
        start=100,
        end=140,
        data=np.sin(np.linspace(0.0, 4.0 * np.pi, 256)),
        data_filtered=np.cos(np.linspace(0.0, 4.0 * np.pi, 256)),
        channel_name="A1",
        biomarker_start=96,
        biomarker_end=132,
    )

    output = tmp_path / "A1_100_140.jpg"
    assert output.exists()
    assert output.stat().st_size > 0
