"""Regression test for the in-repo HFODetector.compute_rms window==1 fix.

Upstream `compute_rms` crashes when the RMS window rounds to 1 (low/non-integer sample rates,
e.g. sub-sourcesinkumf003 @ 499.7Hz). The in-repo patch (src/pyhfo/utils/hfodetector_patch.py) corrects it.
"""
import numpy as np

from pyhfo.utils.hfodetector_patch import apply_hfodetector_patch

apply_hfodetector_patch()

from HFODetector import mni, ste  # noqa: E402  (import after patch is applied)


def test_ste_rms_window_collapses_to_one_no_crash():
    # sfreq 499 + 3ms window -> round(0.003*499) == 1 (the upstream crash case).
    data = np.random.RandomState(0).randn(32934).astype(float)
    rms = ste.compute_rms(data, 499, 0.003, detector="RMS")
    assert rms.shape == data.shape
    assert np.all(np.isfinite(rms))


def test_mni_low_sfreq_window_one_no_crash():
    data = np.random.RandomState(1).randn(2000).astype(float)
    rms = mni.compute_rms(data, 300, detector="MNI")  # round(0.002*300) == 1
    assert rms.shape == data.shape
    assert np.all(np.isfinite(rms))


def test_normal_sfreq_uses_original_path_unchanged():
    # sfreq 1024 + 3ms -> window 3, shift 1 -> takes the original (unchanged) branch.
    data = np.random.RandomState(2).randn(5000).astype(float)
    rms = ste.compute_rms(data, 1024, 0.003, detector="RMS")
    assert rms.shape == data.shape
    assert np.all(np.isfinite(rms))
