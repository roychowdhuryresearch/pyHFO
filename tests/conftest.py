import os
import sys
from pathlib import Path

import mne
import numpy as np
import pytest
from PyQt5 import QtWidgets


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def tiny_mne_raw():
    sfreq = 200.0
    samples = int(sfreq * 2)
    times = np.arange(samples) / sfreq
    info = mne.create_info(["B10Ref", "A2Ref", "A1Ref"], sfreq, ch_types="eeg")
    data = np.vstack(
        [
            np.sin(2 * np.pi * 8 * times),
            np.cos(2 * np.pi * 12 * times) * 0.8,
            np.sin(2 * np.pi * 16 * times) * 0.5,
        ]
    ) * 1e-6
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.info["line_freq"] = 60
    return raw


@pytest.fixture
def tiny_fif_path(tmp_path, tiny_mne_raw):
    path = tmp_path / "tiny_raw.fif"
    tiny_mne_raw.save(path, overwrite=True, verbose=False)
    return path


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app
