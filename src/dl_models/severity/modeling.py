"""
HuggingFace PreTrainedModel wrapper for the BASED severity scoring model.

Wraps CBRAMODPreferenceLearner and applies the score lookup table so that
forward() returns calibrated scores ready for downstream use.
"""
import numpy as np
import torch
from transformers import PreTrainedModel

from .configuration import BasedSeverityConfig
from .cbramod_preference_learner import CBRAMODPreferenceLearner


class BasedSeverityModel(PreTrainedModel):
    config_class = BasedSeverityConfig
    _tied_weights_keys = []

    # transformers 5.x renamed _tied_weights_keys to all_tied_weights_keys (dict)
    @property
    def all_tied_weights_keys(self):
        return {}

    def __init__(self, config: BasedSeverityConfig):
        super().__init__(config)
        self.model = CBRAMODPreferenceLearner(
            seq_len        = config.seq_len,
            d_model        = config.d_model,
            out_dim        = config.out_dim,
            nhead          = config.nhead,
            dim_feedforward = config.dim_feedforward,
            fs             = config.fs,
        )
        # Build lookup arrays from config once at init time
        self._lookup_raw = np.array(config.lookup_raw_knots, dtype=np.float64)
        self._lookup_cal = np.array(config.lookup_cal_knots, dtype=np.float64)
        self._has_lookup = len(self._lookup_raw) > 0

    # ── lookup helper ──────────────────────────────────────────────────────

    def _apply_lookup(self, raw_scores: np.ndarray) -> np.ndarray:
        """Piecewise-linear calibration: raw score → calibrated score."""
        return np.interp(raw_scores, self._lookup_raw, self._lookup_cal)

    # ── forward ────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 19, time_samples) — pre-scaled waveform (raw_uV / 100)
        Returns:
            calibrated_scores: (batch,) float32 tensor in the calibrated range
        """
        # Cast input to match stored weight dtype (fp16 on GPU, fp32 on CPU)
        x = x.to(dtype=next(self.model.parameters()).dtype)
        raw = self.model(x)          # (batch,) in [0, 5], raw uncalibrated
        raw = raw.float()            # always return float32 scores

        if not self._has_lookup:
            return raw

        # Apply lookup on CPU numpy, return as tensor on same device as input
        raw_np = raw.detach().cpu().numpy().astype(np.float64)
        cal_np = self._apply_lookup(raw_np).astype(np.float32)
        return torch.from_numpy(cal_np).to(raw.device)
