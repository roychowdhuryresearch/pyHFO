"""
HuggingFace config for the BASED severity scoring model.

Stores:
  - CBRAMODPreferenceLearner architecture hyperparameters
  - Score lookup table (raw_knots, cal_knots) so the entire model ships
    as a single HF directory with no external JSON dependency
  - Preprocessing constants needed by SeverityScorer
"""
from typing import List, Optional
from transformers import PretrainedConfig


class BasedSeverityConfig(PretrainedConfig):
    model_type = "based_severity"

    def __init__(
        self,
        # ── architecture ──────────────────────────────────────────────────
        seq_len: int = 15,
        fs: int = 200,
        d_model: int = 200,
        out_dim: int = 200,
        nhead: int = 8,
        dim_feedforward: int = 800,
        # ── score lookup table ────────────────────────────────────────────
        # Piecewise-linear map: raw score → calibrated score.
        # Both lists must have equal length (n_knots).
        lookup_raw_knots: Optional[List[float]] = None,
        lookup_cal_knots: Optional[List[float]] = None,
        # ── preprocessing constants ───────────────────────────────────────
        input_scale: float = 1e-2,   # raw µV are divided by this before the model
        low_freq: float = 0.3,       # bandpass low cutoff (Hz)
        high_freq: float = 75.0,     # bandpass high cutoff (Hz)
        notch_freq: float = 60.0,    # notch filter frequency (Hz)
        **kwargs,
    ):
        self.seq_len        = seq_len
        self.fs             = fs
        self.d_model        = d_model
        self.out_dim        = out_dim
        self.nhead          = nhead
        self.dim_feedforward = dim_feedforward

        self.lookup_raw_knots = lookup_raw_knots or []
        self.lookup_cal_knots = lookup_cal_knots or []

        self.input_scale = input_scale
        self.low_freq    = low_freq
        self.high_freq   = high_freq
        self.notch_freq  = notch_freq

        super().__init__(**kwargs)
