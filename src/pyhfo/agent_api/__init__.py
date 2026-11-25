from pyhfo.agent_api.errors import PyHFOAgentError
from pyhfo.agent_api.pipeline import classify_existing_run, create_batch, get_recording_info, run_batch, run_detection, run_severity_scoring
from pyhfo.agent_api.preprocessing import prepare_recording
from pyhfo.agent_api.signal_analysis import (
    compute_channel_qc,
    compute_spectral_profile,
    detect_line_noise,
    infer_brain_signal_roi,
    infer_brain_signal_roi_from_channels,
    rank_candidate_channels,
    recommend_detector_plan,
)
from pyhfo.agent_api.schemas import BatchResult, DetectionRequest, DetectionResult, RecordingInfo
from pyhfo.agent_api.parameters import detector_defaults, recommend_parameters, sweep_parameters, validate_parameters
from pyhfo.agent_api.waveform import waveform_channels, waveform_overview, waveform_window
from pyhfo.agent_api.sidecar import (
    PyHFOSidecarSession,
    dispatch_action,
    get_capability_manifest,
    load_sidecar_session,
    save_sidecar_session,
)

__all__ = [
    "BatchResult",
    "DetectionRequest",
    "DetectionResult",
    "PyHFOAgentError",
    "PyHFOSidecarSession",
    "RecordingInfo",
    "classify_existing_run",
    "create_batch",
    "dispatch_action",
    "detector_defaults",
    "get_capability_manifest",
    "load_sidecar_session",
    "get_recording_info",
    "infer_brain_signal_roi",
    "infer_brain_signal_roi_from_channels",
    "compute_channel_qc",
    "compute_spectral_profile",
    "detect_line_noise",
    "recommend_detector_plan",
    "rank_candidate_channels",
    "prepare_recording",
    "run_batch",
    "run_detection",
    "run_severity_scoring",
    "recommend_parameters",
    "sweep_parameters",
    "validate_parameters",
    "waveform_channels",
    "waveform_overview",
    "waveform_window",
    "save_sidecar_session",
]
