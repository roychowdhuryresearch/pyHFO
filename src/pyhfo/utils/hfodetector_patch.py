"""In-repo fix for an upstream HFODetector bug (user-approved route: library inline-fix, 2026-06-05).

`HFODetector.utils.compute_rms` crashes when the RMS window collapses to 1 sample:
    rms_window = round(window_size * sample_freq)        # STE 3ms @ sfreq<=~500Hz  -> 1
    rms_window = round(0.002 * sample_freq)              # MNI      @ sfreq<=~750Hz  -> 1
    ...
    rms[:int(-np.floor(rms_window/2))] = temp[int(np.floor(rms_window/2)):]
With rms_window==1, floor(0.5)==0, so the LHS slice `rms[:0]` has shape (0,) while the RHS is (N,):
    ValueError: could not broadcast input array from shape (N,) into shape (0,)
Real recordings with non-integer/low sample rates (e.g. sub-sourcesinkumf003 @ 499.7Hz) hit this.

Fix: when the latency shift is 0 (window==1), copy `temp` directly — a 1-sample RMS window has no
lfilter group delay to compensate, so this is the mathematically correct behaviour, not a workaround.
The window>=3 path is byte-for-byte the original. We monkeypatch the function on every HFODetector
module that bound it via `from .utils import *` (ste/mni/hil) plus `utils` itself. Idempotent.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _patched_compute_rms(filtered, sample_freq, window_size=6 * 1e-3, detector=None):
    from HFODetector import utils as _u

    filtered = _u.validate_type(filtered, "filtered", np.ndarray)
    if filtered.ndim != 1:
        raise ValueError("filtered must be a 1D array")
    sample_freq = _u.validate_param(sample_freq, "sample_freq", (int, float))
    window_size = _u.validate_param(window_size, "window_size", (int, float))
    if detector is not None:
        detector = _u.validate_type(detector, "detector", str)
        if detector not in ["RMS", "MNI"]:
            raise ValueError("detector must be RMS or MNI")

    if detector == "RMS":
        rms_window = round(window_size * sample_freq)
    elif detector == "MNI":
        rms_window = round(0.002 * sample_freq)

    if rms_window % 2 == 0:
        rms_window += 1

    temp = np.square(filtered)
    temp = _u.lfilter(np.ones(rms_window), 1, temp, axis=0) / rms_window
    rms = np.zeros(len(temp))
    shift = int(np.floor(rms_window / 2))
    if shift > 0:
        rms[:-shift] = temp[shift:]   # original group-delay compensation (window >= 3) — unchanged
    else:
        rms[:] = temp[:]              # FIX: window == 1 has no latency to compensate -> copy directly
    rms = np.sqrt(rms)
    return rms


def apply_hfodetector_patch() -> bool:
    """Patch compute_rms on HFODetector.{utils,ste,mni,hil}. Safe to call repeatedly."""
    try:
        from HFODetector import utils as _u
    except Exception as exc:  # HFODetector not importable — nothing to patch.
        logger.warning("hfodetector_patch: HFODetector not importable: %s", exc)
        return False
    if getattr(_u, "_ieeg_compute_rms_patched", False):
        return True
    modules = [_u]
    for name in ("ste", "mni", "hil"):
        try:
            modules.append(__import__(f"HFODetector.{name}", fromlist=[name]))
        except Exception:
            pass
    patched = 0
    for mod in modules:
        if hasattr(mod, "compute_rms"):
            mod.compute_rms = _patched_compute_rms
            patched += 1
    _u._ieeg_compute_rms_patched = True
    logger.info("hfodetector_patch: applied compute_rms window==1 fix to %d module(s)", patched)
    return True
