# EEG/iEEG Agent Signal Workflow Plan

## Target behavior

The agent is a generic ReAct agent with EEG/iEEG skills. For EDF-only input it must not read clinical sidecar labels by default. It must understand the signal first, select a real brain-signal ROI, run appropriate QC/signal-processing/detection tools, and interpret outputs as algorithmic research evidence requiring expert review.

## Evidence layers

1. **Raw signal metadata**: file format, sample rate, duration, channel names, high/low-pass, line frequency.
2. **Label-free signal understanding / QC**: brain ROI, auxiliary/uncertain channels, flat/outlier/noisy channels, spectral profile, line-noise candidates.
3. **Algorithmic detections**: spike/HFO/spindle events, detector/filter/classifier settings, event counts/rates, artifacts, run stability.
4. **Clinical labels**: SOZ, resection, pathology labels, doctor annotations, ground truth. These are explicit opt-in only.

## Required skills

### 1. EEG signal understanding
- Infer modality/montage from channel names and metadata.
- Select brain-signal ROI.
- Exclude EKG/ECG/EOG/EMG/AUX/STIM/TRIG and uncertain labels such as `X*` by default.
- Report ROI assumptions and uncertainty.

### 2. Preprocessing decision
- Use metadata + ROI + QC to choose conservative preprocessing.
- Keep broad preprocessing filters separate from detector-specific filters.
- Do not notch 50/60 Hz unless line frequency is known or requested.
- Do not re-bipolarize already-bipolar channels.

### 3. Algorithm/detector/classifier selection
- Validate sampling rate and Nyquist before HFO bands.
- Select HFO detector/parameters and spike detector/parameters for the request.
- Distinguish smoke tests from research runs.
- Record all detector/filter/classifier settings.

### 4. Result interpretation
- Read structured detector/QC outputs.
- Rank channels by spike/HFO burden and event rates within brain ROI.
- Separate high spike burden, high HFO burden, overlap/co-localized candidates, artifact suspects, and unstable candidates.
- Never present algorithmic candidates as clinical pathological channels.

### 5. Iterative rerun optimization
- Compare detector runs and parameter sweeps.
- Rerun after excluding artifact/auxiliary channels.
- Identify stable vs parameter-sensitive candidate channels.
- Recommend next run based on observed warnings/counts, not guesses.

## Tool roadmap

### Phase 1: implemented/near-term
- `infer_brain_signal_roi`
- `compute_channel_qc`
- `compute_spectral_profile`
- `detect_line_noise`
- `recommend_detector_plan`
- `rank_candidate_channels`

### Phase 2
- `compute_event_rates`
- `compute_spike_hfo_overlap`
- `compare_detector_runs`

### Phase 3
- `parameter_sweep`
- artifact/classifier summaries
- event-snippet export
- doctor review queue

## Expected answer for EDF-only “pathological channels”

1. State that clinical pathological labels are unavailable from EDF alone and no labels were loaded.
2. Show signal ROI: included brain channels, excluded aux/uncertain channels, reasons.
3. Run or propose QC + spike/HFO detection on ROI.
4. Report candidate channels by signal evidence: spike count/rate, HFO count/rate, overlap, warnings.
5. State that results are candidate pathology-related/epileptiform activity for expert review, not definitive clinical pathology.
6. Offer explicit label loading only if the user wants SOZ/resection/ground-truth comparison.

## Implementation invariants

- EDF-only requests never load `channels.tsv` labels automatically.
- “Pathological/abnormal” words alone do not grant label access.
- ROI selection happens before detection or candidate ranking.
- EKG/aux/uncertain channels are excluded from default candidate rankings.
- Every detector result includes settings, ROI/exclusions, artifact paths, and limitations.
