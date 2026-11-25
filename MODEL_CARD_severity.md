# PyHFO EEG severity model (draft)

The UCLA Roychowdhury lab packages this BASED severity scorer for research on conventional scalp EEG. The model produces a continuous, calibrated score from 0 to 5 for each complete 15-second segment. PyHFO also reports recording-level mean and peak scores.

## Architecture

A CBraMod encoder extracts spatiotemporal features from the scalp channels. The BASED preference learner maps those features to a raw segment score. A piecewise-linear lookup table stored in `config.json` calibrates the output. The accompanying `model.safetensors` contains the trained weights.

## Input contract

Use the TUEG-19 scalp montage in the order below.

```text
FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 FZ CZ PZ
```

PyHFO bandpass filters at 0.3–75 Hz, applies a 60 Hz notch, resamples to 200 Hz and computes an average reference across the available montage channels. It zero-fills missing channels and rejects recordings with no matching channels. The network processes 15 s segments (19 × 3,000 samples) with amplitudes in µV scaled by 0.01. Performance with missing channels requires separate validation.

## Training data and attribution

TUH EEG Corpus — confirm data-use terms before publishing.

- CBraMod backbone — Wang J, Zhao S, Luo Z, Zhou Y, Jiang H, Li S, Li T, Pan G. *CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding.* ICLR 2025 ([OpenReview](https://openreview.net/forum?id=NPNUHgHF2w), [arXiv:2412.07236](https://arxiv.org/abs/2412.07236)). Upstream code: https://github.com/wjq-learning/CBraMod, MIT License, Copyright (c) 2025 Jiquan Wang. Keep this copyright and permission notice with redistributed CBraMod-derived weights.
- BASED preference-learning model and calibration — [owners to supply authors, upstream URL, citation, weight provenance and redistribution terms].
- [Owners to document training cohort, splits, evaluation metrics, and population-specific limitations before release.]

## Intended use and limits

Use these scores for offline research on scalp EEG. A qualified reviewer should inspect the recording and signal quality before interpreting a score. Clinical diagnostic accuracy and applicability to intracranial EEG have not been established. This research software is not a medical device.
