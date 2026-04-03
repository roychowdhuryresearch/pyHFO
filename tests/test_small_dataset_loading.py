import numpy as np

import src.hfo_app as hfo_module
import src.spindle_app as spindle_module
from src.hfo_app import HFO_App
from src.param.param_detector import ParamDetector, ParamSTE, ParamYASA
from src.param.param_filter import ParamFilter
from src.spindle_app import SpindleApp


def test_hfo_app_loads_tiny_mne_recording(tiny_fif_path):
    app = HFO_App()

    app.load_edf(str(tiny_fif_path))

    assert app.sample_freq == 200
    assert app.eeg_data.shape == (3, 400)
    assert app.channel_names.tolist() == ["A1Ref", "A2Ref", "B10Ref"]
    assert app.edf_param["edf_fn"] == str(tiny_fif_path)
    np.testing.assert_allclose(app.eeg_data_un60, app.eeg_data)


def test_spindle_app_loads_tiny_mne_recording(tiny_fif_path):
    app = SpindleApp()

    app.load_edf(str(tiny_fif_path))

    assert app.sample_freq == 200
    assert app.eeg_data.shape == (3, 400)
    assert app.channel_names.tolist() == ["A1Ref", "A2Ref", "B10Ref"]
    assert app.edf_param["edf_fn"] == str(tiny_fif_path)
    np.testing.assert_allclose(app.eeg_data_un60, app.eeg_data)


def test_hfo_app_set_n_jobs_rebuilds_current_detector(tiny_fif_path, monkeypatch):
    rebuilt = []

    def fake_set_ste_detector(args):
        rebuilt.append(args.n_jobs)
        return {"detector": "ste", "n_jobs": args.n_jobs}

    monkeypatch.setattr(hfo_module, "set_STE_detector", fake_set_ste_detector)

    app = HFO_App()
    app.load_edf(str(tiny_fif_path))
    app.param_filter = ParamFilter(fp=80, fs=90, sample_freq=200)
    app.set_detector(ParamDetector(ParamSTE(sample_freq=200, n_jobs=4), detector_type="STE"))

    app.set_n_jobs(2)

    assert app.n_jobs == 2
    assert app.param_detector.detector_param.n_jobs == 2
    assert app.detector == {"detector": "ste", "n_jobs": 2}
    assert rebuilt == [4, 2]


def test_spindle_app_set_n_jobs_rebuilds_current_detector(tiny_fif_path, monkeypatch):
    rebuilt = []

    def fake_set_yasa_detector(args):
        rebuilt.append(args.n_jobs)
        return {"detector": "yasa", "n_jobs": args.n_jobs}

    monkeypatch.setattr(spindle_module, "set_YASA_detector", fake_set_yasa_detector)

    app = SpindleApp()
    app.load_edf(str(tiny_fif_path))
    app.set_detector(ParamDetector(ParamYASA(sample_freq=200, n_jobs=4), detector_type="YASA"))

    app.set_n_jobs(3)

    assert app.n_jobs == 3
    assert app.param_detector.detector_param.n_jobs == 3
    assert app.detector == {"detector": "yasa", "n_jobs": 3}
    assert rebuilt == [4, 3]
