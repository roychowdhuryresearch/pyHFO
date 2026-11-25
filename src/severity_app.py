"""
SeverityScorer: end-to-end severity scoring for scalp EEG recordings.

Pipeline
--------
1. Receive raw EEG (µV, any channel set, any sample rate) from the app
2. Preprocess: resample → bandpass → notch → pick TUEG-19 → average reference
3. Segment into consecutive 15-second windows
4. Scale by 1/100  (model was trained on µV / 100)
5. Batch inference through BasedSeverityModel → calibrated scores per segment
6. Return per-segment DataFrame + summary stats

Usage
-----
    scorer = SeverityScorer(model_dir)          # load once
    result = scorer.run(eeg_data, channel_names, sample_freq)
    # result.scores_df  — DataFrame with segment_idx, start_sec, end_sec, score
    # result.mean_score, result.peak_score, result.peak_time_sec
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.dl_models.severity import BasedSeverityModel

# ── TUEG 19-channel definitions ──────────────────────────────────────────────

TUEG_19 = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1',  'O2',  'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'FZ',  'CZ',  'PZ',
]

_REMOVE = ['-REF', '-ref', 'EEG ', 'eeg ', 'POL ', 'pol ', '-Ref', ' ']


def _clean_ch(name: str) -> str:
    """Normalise a channel name to bare uppercase label, e.g. 'EEG Fp1-Ref' → 'FP1'."""
    s = name.upper()
    for tok in [r.upper() for r in _REMOVE]:
        s = s.replace(tok, '')
    return s.strip()


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class SeverityResult:
    scores_df:      pd.DataFrame   # columns: segment_idx, start_sec, end_sec, score
    mean_score:     float
    peak_score:     float
    peak_time_sec:  float          # start time of the peak segment


# ── main class ────────────────────────────────────────────────────────────────

class SeverityScorer:
    """
    Loads BasedSeverityModel once and scores EEG recordings on demand.

    Parameters
    ----------
    model_dir : str
        Path to a directory produced by convert_checkpoint.py containing
        config.json and model.safetensors.
    device : str
        Torch device string.  Defaults to 'cuda' when available, else 'cpu'.
    batch_size : int
        Number of 15-second segments per inference batch.
    """

    TARGET_FS    = 200          # Hz expected by the model
    SEG_SAMPLES  = 15 * TARGET_FS  # 3000 samples per segment
    INPUT_SCALE  = 1e-2         # µV → model input units

    def __init__(self, model_dir: str, device: Optional[str] = None,
                 batch_size: int = 32):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device     = torch.device(device)
        self.batch_size = batch_size
        print(f"SeverityScorer: using device '{device}'")

        self.model = BasedSeverityModel.from_pretrained(model_dir)
        self.model.to(self.device).eval()

        cfg = self.model.config
        self._low_freq   = cfg.low_freq
        self._high_freq  = cfg.high_freq
        self._notch_freq = cfg.notch_freq

        # Calibration output range — used to rescale scores to exactly [0, 5]
        knots = getattr(cfg, "lookup_cal_knots", None) or []
        if knots:
            self._cal_min = float(min(knots))
            self._cal_max = float(max(knots))
        else:
            self._cal_min = 0.0
            self._cal_max = 5.0
        print(f"SeverityScorer: calibration range [{self._cal_min:.4f}, {self._cal_max:.4f}]")

    # ── public API ────────────────────────────────────────────────────────────

    def run(self, eeg_data: np.ndarray, channel_names: np.ndarray,
            sample_freq: int, progress_callback=None) -> SeverityResult:
        """
        Score a recording.

        Parameters
        ----------
        eeg_data : (n_channels, n_samples) float array in µV
        channel_names : array of channel name strings (matching eeg_data rows)
        sample_freq : sampling frequency of eeg_data (Hz)

        Returns
        -------
        SeverityResult
        """
        data = self._preprocess(eeg_data, channel_names, sample_freq)
        # data: (19, n_samples_at_200Hz)  float32 µV

        segments = self._segment(data)
        # segments: (n_segs, 19, 3000)

        if len(segments) == 0:
            empty = pd.DataFrame(columns=['segment_idx', 'start_sec', 'end_sec', 'score'])
            return SeverityResult(empty, float('nan'), float('nan'), float('nan'))

        scores = self._infer(segments, progress_callback=progress_callback)
        # scores: (n_segs,)

        n = len(scores)
        seg_dur = self.SEG_SAMPLES / self.TARGET_FS   # 15 s
        starts  = np.arange(n) * seg_dur
        ends    = starts + seg_dur

        df = pd.DataFrame({
            'segment_idx': np.arange(n),
            'start_sec':   starts,
            'end_sec':     ends,
            'score':       scores,
        })

        peak_idx  = int(np.argmax(scores))
        return SeverityResult(
            scores_df     = df,
            mean_score    = float(np.mean(scores)),
            peak_score    = float(scores[peak_idx]),
            peak_time_sec = float(starts[peak_idx]),
        )

    def export_csv(self, result: SeverityResult, path: str):
        result.scores_df.to_csv(path, index=False)

    @staticmethod
    def export_csv_static(result: SeverityResult, path: str):
        result.scores_df.to_csv(path, index=False)

    # ── preprocessing ─────────────────────────────────────────────────────────

    def _preprocess(self, eeg_data: np.ndarray, channel_names: np.ndarray,
                    sample_freq: int) -> np.ndarray:
        """
        Replicate the exact MNE-based preprocessing pipeline used to create
        the training NPZ files:
          bandpass (MNE FIR) → notch (MNE FIR) → resample → rename channels
          → pick TUEG-19 → average reference → µV

        Using MNE rather than scipy is essential — the FIR filter produces
        different values from Butterworth IIR, and value-level agreement with
        the reference scores requires the same filter.
        """
        import mne

        # eeg_data is in µV; MNE works in SI (Volts) internally
        data_V = eeg_data.astype(np.float64) / 1e6

        ch_names = [str(c) for c in channel_names]
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=float(sample_freq),
            ch_types=['eeg'] * len(ch_names),
        )
        raw = mne.io.RawArray(data_V, info, verbose=False)

        # 1) bandpass — same as NPZ creation pipeline
        raw.filter(l_freq=self._low_freq, h_freq=self._high_freq, verbose=False)

        # 2) notch
        raw.notch_filter(freqs=[self._notch_freq], verbose=False)

        # 3) resample if needed
        if sample_freq != self.TARGET_FS:
            raw.resample(self.TARGET_FS, verbose=False)

        # 4) rename channels to TUEG bare labels
        rename_dict = {}
        for ch in raw.ch_names:
            cleaned = _clean_ch(ch)
            if cleaned in set(TUEG_19):
                rename_dict[ch] = cleaned
        if rename_dict:
            raw.rename_channels(rename_dict)

        # 5) pick available TUEG-19 channels
        available = [ch for ch in TUEG_19 if ch in raw.ch_names]
        raw.pick(available)
        raw.reorder_channels(available)

        # 6) average reference across the selected channels
        raw.set_eeg_reference(ref_channels='average', verbose=False)

        data_uV = raw.get_data(units='uV')  # (n_available, n_samples)

        # 7) pad any missing TUEG-19 channels with zeros
        if len(available) < 19:
            full = np.zeros((19, data_uV.shape[1]), dtype=np.float32)
            for row, ch in enumerate(TUEG_19):
                if ch in available:
                    full[row] = data_uV[available.index(ch)]
            return full

        return data_uV.astype(np.float32)

    # ── segmentation ──────────────────────────────────────────────────────────

    def _segment(self, data: np.ndarray) -> np.ndarray:
        """Split (19, T) into (n_segs, 19, SEG_SAMPLES) — drop the last partial window."""
        n_segs = data.shape[1] // self.SEG_SAMPLES
        if n_segs == 0:
            return np.empty((0, 19, self.SEG_SAMPLES), dtype=np.float32)
        trimmed = data[:, :n_segs * self.SEG_SAMPLES]
        return trimmed.reshape(19, n_segs, self.SEG_SAMPLES).transpose(1, 0, 2)

    # ── inference ─────────────────────────────────────────────────────────────

    def _infer(self, segments: np.ndarray, progress_callback=None) -> np.ndarray:
        """
        segments: (n_segs, 19, 3000) float32 µV
        returns:  (n_segs,) float32 calibrated scores
        """
        import sys
        from tqdm import tqdm

        n      = len(segments)
        scores = np.empty(n, dtype=np.float32)

        bar = tqdm(total=n, desc="Severity scoring", unit="seg",
                   file=sys.stdout, ascii=True, dynamic_ncols=True)
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                batch_np = segments[start:start + self.batch_size]
                batch_t  = torch.from_numpy(batch_np * self.INPUT_SCALE).to(self.device)
                out      = self.model(batch_t)
                batch_len = len(out)
                scores[start:start + batch_len] = out.cpu().numpy()
                bar.update(batch_len)
                if progress_callback is not None:
                    pct = min(int((start + batch_len) / n * 100), 99)
                    progress_callback.emit(pct)
        bar.close()

        # Rescale to [0, 5] based on calibration table's achievable range
        cal_range = self._cal_max - self._cal_min
        if cal_range > 0:
            scores = (scores - self._cal_min) / cal_range * 5.0

        # Reverse: model output 0=severe, 5=benign → app convention 0=benign, 5=severe
        return np.clip(5.0 - scores, 0.0, 5.0)
