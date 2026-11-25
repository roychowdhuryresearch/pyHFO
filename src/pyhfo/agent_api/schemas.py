from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pyhfo import __version__
from pyhfo.agent_api.errors import PyHFOAgentError


SUPPORTED_BIOMARKERS = {"HFO", "Spindle", "Spike"}
DETECTORS_BY_BIOMARKER = {
    "HFO": {"STE", "MNI", "HIL", "RMS", "LineLength", "LL", "HFO RMS", "HFO LL"},
    "Spindle": {"YASA", "LSM", "Kramer LSM", "A7", "MOLLE", "FASST", "Spindle RMS"},
    "Spike": {"RMS/LL"},
}
DEFAULT_DETECTOR_BY_BIOMARKER = {
    "HFO": "STE",
    "Spindle": "YASA",
    "Spike": "RMS/LL",
}


def normalize_biomarker(value: str | None) -> str:
    raw = (value or "HFO").strip().lower()
    aliases = {
        "hfo": "HFO",
        "spindle": "Spindle",
        "spindles": "Spindle",
        "spike": "Spike",
        "spikes": "Spike",
    }
    try:
        return aliases[raw]
    except KeyError as exc:
        raise PyHFOAgentError(
            "invalid_biomarker",
            f"Unsupported biomarker: {value}",
            details={"supported": sorted(SUPPORTED_BIOMARKERS)},
        ) from exc


def normalize_detector(biomarker: str, value: str | None) -> str:
    canonical_biomarker = normalize_biomarker(biomarker)
    raw = (value or DEFAULT_DETECTOR_BY_BIOMARKER[canonical_biomarker]).strip().lower().replace("_", " ").replace("-", " ")
    # Spike runs a single combined RMS/LL detector; accept the facet spellings a clinician may name
    # ("RMS", "LL", "line length" / "line-length") and the combined forms, all resolving to that one
    # detector (the detector string is otherwise ignored for Spike downstream — see pipeline._detector_param_for).
    if canonical_biomarker == "Spike" and raw in {"rms/ll", "rms ll", "rmsll", "spike rms/ll", "rms", "ll", "linelength", "line length"}:
        return "RMS/LL"
    aliases = {
        "ste": "STE",
        "mni": "MNI",
        "hil": "HIL",
        "rms": "RMS",
        "hfo rms": "HFO RMS",
        "linelength": "LineLength",
        "line length": "LineLength",
        "ll": "LL",
        "hfo ll": "HFO LL",
        "yasa": "YASA",
        "lsm": "LSM",
        "kramer lsm": "Kramer LSM",
        "a7": "A7",
        "molle": "MOLLE",
        "fasst": "FASST",
        "spindle rms": "Spindle RMS",
        "rms/ll": "RMS/LL",
        "rmsll": "RMS/LL",
        "spike rms/ll": "RMS/LL",
    }
    detector = aliases.get(raw)
    if detector is None or detector not in DETECTORS_BY_BIOMARKER[canonical_biomarker]:
        raise PyHFOAgentError(
            "invalid_detector",
            f"Unsupported {canonical_biomarker} detector: {value}",
            details={"supported": sorted(DETECTORS_BY_BIOMARKER[canonical_biomarker])},
        )
    return detector


def _dict_or_empty(value: Any, field_name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PyHFOAgentError("invalid_request", f"{field_name} must be an object.")
    return dict(value)


def _list_or_default(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    if not isinstance(value, (list, tuple)):
        raise PyHFOAgentError("invalid_request", "exports must be a list of strings.")
    return [str(item) for item in value]


def _string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, (list, tuple)):
        raise PyHFOAgentError("invalid_request", f"{field_name} must be a list of strings.")
    return [str(item) for item in value if str(item)]


def _session_format(value: Any) -> str:
    raw = str(value or "pybrain").strip().lower().lstrip(".").replace("-", "_")
    aliases = {
        "pybrain": "pybrain",
        "pybrain_session": "pybrain",
        "npz": "npz",
        "legacy_npz": "npz",
    }
    if raw not in aliases:
        raise PyHFOAgentError("invalid_request", "session_format must be 'pybrain' or 'npz'.")
    return aliases[raw]


@dataclass
class RecordingInfo:
    input_path: str
    format: str
    sample_freq: float
    duration_seconds: float
    channel_names: list[str]
    recording_channel_names: list[str]
    line_freq: float | None = None
    highpass: float | None = None
    lowpass: float | None = None
    n_channels: int = 0
    eeg_type: str | None = None  # "scalp" or "ieeg", from infer_eeg_type(channel_names)
    recording_fingerprint: str | None = None
    software_version: str = __version__

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DetectionRequest:
    input_path: str
    output_dir: str
    biomarker: str = "HFO"
    detector: str = "STE"
    filter: dict[str, Any] = field(default_factory=dict)
    detector_params: dict[str, Any] = field(default_factory=dict)
    classifier: dict[str, Any] = field(default_factory=dict)
    n_jobs: int = 4
    exports: list[str] = field(default_factory=lambda: ["summary_json"])
    session_format: str = "pybrain"
    include_channels: list[str] = field(default_factory=list)
    exclude_channels: list[str] = field(default_factory=list)
    fail_on_error: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DetectionRequest":
        if not isinstance(payload, dict):
            raise PyHFOAgentError("invalid_request", "Detection request must be an object.")
        input_path = payload.get("input_path") or payload.get("input")
        output_dir = payload.get("output_dir") or payload.get("output")
        if not input_path:
            raise PyHFOAgentError("invalid_request", "input_path is required.")
        if not output_dir:
            raise PyHFOAgentError("invalid_request", "output_dir is required.")

        biomarker = normalize_biomarker(payload.get("biomarker"))
        detector = normalize_detector(biomarker, payload.get("detector"))
        try:
            n_jobs = int(payload.get("n_jobs", 4))
        except (TypeError, ValueError) as exc:
            raise PyHFOAgentError("invalid_request", "n_jobs must be an integer.") from exc
        if n_jobs < 1:
            raise PyHFOAgentError("invalid_request", "n_jobs must be positive.")

        return cls(
            input_path=str(input_path),
            output_dir=str(output_dir),
            biomarker=biomarker,
            detector=detector,
            filter=_dict_or_empty(payload.get("filter"), "filter"),
            detector_params=_dict_or_empty(payload.get("detector_params"), "detector_params"),
            classifier=_dict_or_empty(payload.get("classifier"), "classifier"),
            n_jobs=n_jobs,
            exports=_list_or_default(payload.get("exports"), ["summary_json"]),
            session_format=_session_format(payload.get("session_format") or payload.get("sessionFormat")),
            include_channels=_string_list(payload.get("include_channels") or payload.get("include"), "include_channels"),
            exclude_channels=_string_list(payload.get("exclude_channels") or payload.get("exclude"), "exclude_channels"),
            fail_on_error=bool(payload.get("fail_on_error", True)),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def resolved_input_path(self) -> Path:
        return Path(self.input_path).expanduser().resolve()

    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()


@dataclass
class DetectionResult:
    status: str
    biomarker: str
    detector: str
    input_path: str
    output_dir: str
    recording_fingerprint: str | None = None
    parameters_sha256: str | None = None
    num_events: int = 0
    num_channels: int = 0
    active_run_id: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)
    run_summaries: list[dict[str, Any]] = field(default_factory=list)
    decision_summary: dict[str, Any] = field(default_factory=dict)
    classifier_metadata: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    software_version: str = __version__

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SeverityScoringResult:
    status: str
    input_path: str
    eeg_type: str | None = None
    recording_fingerprint: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    model_sha256: str | None = None
    parameters_sha256: str | None = None
    num_segments: int = 0
    mean_score: float | None = None
    peak_score: float | None = None
    peak_time_sec: float | None = None
    # Artifact-excluded companions to the three above: severity navigation ranks on these so a
    # movement burst cannot be offered as the recording's most severe moment.
    num_clean_segments: int = 0
    clean_mean_score: float | None = None
    clean_peak_score: float | None = None
    clean_peak_time_sec: float | None = None
    artifact_threshold_uv: float | None = None
    artifact_quality: dict[str, Any] = field(default_factory=dict)
    scores: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    software_version: str = __version__

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BatchResult:
    status: str
    manifest_path: str
    summary: dict[str, int]
    cases: list[dict[str, Any]]
    errors: list[dict[str, Any]] = field(default_factory=list)
    software_version: str = __version__

    def to_dict(self) -> dict:
        return asdict(self)
