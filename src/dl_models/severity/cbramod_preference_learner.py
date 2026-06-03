"""
CBRAMODPreferenceLearner: CBraMod backbone + mu_head scoring network.
Adapted from the BASED training repo
(biot/model_implementations/cbramod_wrapper.py).
"""
import torch
import torch.nn as nn

from .cbramod_backbone import CBraMod


class CBRAMODPreferenceLearner(nn.Module):
    """
    Preference-scoring wrapper around CBraMod.

    Input : (batch, 19, time_samples) in scaled µV units (raw_uV / 100)
    Output: (batch,) scores in [0, 5]  — raw, uncalibrated

    Internally pads/trims the time axis to 3000 samples (15 s × 200 Hz),
    then reshapes to (batch, 19, 15, 200) before passing to CBraMod.
    """

    # Architecture constants matching the trained checkpoint
    N_CHANNELS   = 19
    SEQ_LEN      = 15   # time patches fed to CBraMod
    PATCH_SIZE   = 200  # samples per patch
    FLAT_DIM     = N_CHANNELS * SEQ_LEN * PATCH_SIZE  # 57 000

    def __init__(self, seq_len: int = 15, d_model: int = 200, out_dim: int = 200,
                 nhead: int = 8, dim_feedforward: int = 800, fs: int = 200):
        super().__init__()
        self.seq_len = seq_len
        self.fs      = fs
        self.expected_samples = seq_len * fs  # 3000

        # CBraMod backbone — uses seq_len=30 internally to match pretrained weights
        self.backbone = CBraMod(
            in_dim=200, out_dim=out_dim, d_model=d_model,
            dim_feedforward=dim_feedforward,
            seq_len=30,   # original CBraMod uses 30 time patches
            n_layer=12, nhead=nhead,
        )
        # Remove the final projection head (identity during fine-tuning)
        self.backbone.proj_out = nn.Sequential()

        # Scoring head trained on BASED pairwise preference data
        self.mu_head = nn.Sequential(
            nn.Linear(self.FLAT_DIM, self.SEQ_LEN * self.PATCH_SIZE),  # 57000 → 3000
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(self.SEQ_LEN * self.PATCH_SIZE, self.PATCH_SIZE),  # 3000 → 200
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(self.PATCH_SIZE, 1),  # 200 → 1
        )

    def _reshape_for_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 19, T) → (B, 19, 15, 200), padding/trimming T to 3000."""
        bsz, ch, t = x.shape
        target = self.SEQ_LEN * self.PATCH_SIZE  # 3000
        if t == target:
            trimmed = x
        elif t > target:
            start   = (t - target) // 2
            trimmed = x[:, :, start:start + target]
        else:
            trimmed = torch.nn.functional.pad(x, (0, target - t))
        return trimmed.view(bsz, ch, self.SEQ_LEN, self.PATCH_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 19, time_samples) — scaled µV (raw_uV / 100)
        Returns:
            scores: (batch,) in [0, 5]
        """
        x    = self._reshape_for_backbone(x)          # (B, 19, 15, 200)
        feats = self.backbone(x)                       # (B, 19, 15, 200)

        bz, ch, s, d = feats.shape
        flat = feats.contiguous().view(bz, ch * s * d)  # (B, 57000)

        logits = self.mu_head(flat)                    # (B, 1)
        scores = torch.sigmoid(logits) * 5             # (B, 1) in [0, 5]

        scores = scores.squeeze(-1)
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)
        return scores                                  # (B,)
