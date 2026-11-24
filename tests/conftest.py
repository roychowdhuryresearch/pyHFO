import os
import sys
from importlib.util import find_spec
from pathlib import Path

import mne
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
# Some tests exercise the repository's release scripts. Prefer the installed
# distribution when testing a wheel, and support a dependency-only source checkout.
sys.path.append(str(ROOT))
source = ROOT / "src"
spec = find_spec("pyhfo")
if spec is None or spec.origin is None:
    sys.path.insert(0, str(source))
if Path(find_spec("pyhfo").origin).parent == source / "pyhfo":
    paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    os.environ["PYTHONPATH"] = os.pathsep.join([str(source), *(p for p in paths if p)])


@pytest.fixture(autouse=True)
def pyhfo_home(tmp_path, monkeypatch):
    monkeypatch.setenv("PYHFO_HOME", str(tmp_path / "pyhfo-home"))


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
