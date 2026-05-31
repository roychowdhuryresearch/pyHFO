import json

import pytest

from src.agent_api import PyHFOAgentError
from src.agent_api.schemas import DetectionRequest, normalize_biomarker, normalize_detector


def test_detection_request_normalizes_aliases_and_serializes():
    request = DetectionRequest.from_dict(
        {
            "input": "input.edf",
            "output": "out",
            "biomarker": "hfo",
            "detector": "line length",
            "exports": "excel",
            "n_jobs": "2",
        }
    )

    assert request.biomarker == "HFO"
    assert request.detector == "LineLength"
    assert request.exports == ["excel"]
    assert request.n_jobs == 2
    json.dumps(request.to_dict())


def test_detection_request_rejects_bad_biomarker():
    with pytest.raises(PyHFOAgentError) as exc_info:
        DetectionRequest.from_dict({"input": "a.edf", "output": "out", "biomarker": "bad"})

    assert exc_info.value.code == "invalid_biomarker"


def test_detection_request_rejects_bad_detector_for_biomarker():
    with pytest.raises(PyHFOAgentError) as exc_info:
        DetectionRequest.from_dict(
            {
                "input": "a.edf",
                "output": "out",
                "biomarker": "Spike",
                "detector": "STE",
            }
        )

    assert exc_info.value.code == "invalid_detector"


def test_normalizers_accept_supported_values():
    assert normalize_biomarker("spindles") == "Spindle"
    assert normalize_detector("Spindle", "kramer lsm") == "Kramer LSM"
    assert normalize_detector("Spike", "rmsll") == "RMS/LL"
