"""Regression tests for round-3 bug-hunt fixes (pyhfo signal core)."""

import pytest

from pyhfo.agent_api.pipeline import _filter_param_for
from pyhfo.agent_api.errors import PyHFOAgentError


# --- Bug #8: empty/absent HFO filter must be Nyquist-safe on common rates ---

def test_hfo_empty_filter_is_nyquist_safe_at_1000hz():
    # Nyquist=500; the old static default fs=500 raised invalid_filter. Now clamped.
    param = _filter_param_for("HFO", 1000.0, {})
    assert float(param.fs) < 500.0
    assert float(param.fp) == 80


def test_spike_empty_filter_is_nyquist_safe_at_low_rate():
    # Nyquist=64; the old static default fs=80 >= 64 raised. Now clamped below Nyquist.
    param = _filter_param_for("Spike", 128.0, {})
    assert float(param.fs) < 64.0


# --- Bug #6: pass_band/stop_band aliases are honored (were dead code) ---

def test_pass_band_stop_band_aliases_are_honored():
    param = _filter_param_for("HFO", 2000.0, {"pass_band": 100, "stop_band": 300})
    assert float(param.fp) == 100
    assert float(param.fs) == 300


def test_fp_fs_take_precedence_over_aliases():
    param = _filter_param_for("HFO", 2000.0, {"fp": 80, "fs": 250, "pass_band": 999})
    assert float(param.fp) == 80
    assert float(param.fs) == 250


def test_explicit_filter_still_validated_against_nyquist():
    # An explicit over-Nyquist band still raises (the guard is intact).
    with pytest.raises(PyHFOAgentError):
        _filter_param_for("HFO", 1000.0, {"fp": 80, "fs": 600})
