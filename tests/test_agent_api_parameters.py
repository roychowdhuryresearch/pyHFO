from __future__ import annotations

from pyhfo.agent_api.parameters import GOAL_PROFILES, detector_defaults, recommend_parameters, sweep_parameters, validate_parameters


def test_recommend_parameters_supports_versioned_goal_profiles():
    for goal in ["qc", "exploratory", "specificity", "comparison", "batch", "reproduction", "smoke", "full"]:
        plan = recommend_parameters({"goal_profile": goal, "biomarker": "HFO", "sample_freq": 1000.0})
        assert plan["parameters"]["goal_profile"]["name"] == goal
        assert plan["validation"]["valid"] is True
        assert plan["ui_card"]["validation_status"] == "valid"
        assert plan["diff"]
    assert GOAL_PROFILES["qc"]["version"]


def test_validate_parameters_reports_nyquist_errors():
    result = validate_parameters({"sample_freq": 200.0, "biomarker": "HFO", "filter": {"fp": 80, "fs": 120}})

    assert result["valid"] is False
    assert result["errors"][0]["code"] in {"nyquist_violation", "invalid_filter_band"}


def test_sweep_parameters_omits_unsafe_candidates_and_has_diff():
    result = sweep_parameters({"goal_profile": "exploratory", "biomarker": "Spike", "sample_freq": 500.0})

    assert result["status"] == "complete"
    assert result["base_plan"]["diff"]
    assert result["candidates"]
    assert all(item["validation"]["valid"] for item in result["candidates"])


def test_detector_defaults_are_hydrated_from_pyhfo_param_objects():
    ste = detector_defaults({"biomarker": "HFO", "detector": "STE", "sample_freq": 1000.0, "n_jobs": 3})

    assert ste["status"] == "complete"
    assert ste["source"] == "pyhfo_param_detector_defaults"
    assert ste["detector_params"]["sample_freq"] == 1000.0
    assert ste["detector_params"]["n_jobs"] == 3
    assert ste["ui_values"]["rms_window"] == "0.003"
    assert ste["ui_values"]["epoch_length"] == "600"
    assert ste["ui_values"]["min_oscillation"] == "6"

    mni = detector_defaults({"biomarker": "HFO", "detector": "MNI"})
    assert mni["sample_freq"] == 200.0
    assert mni["ui_values"]["threshold_percentage"] == "99.9999"
    assert mni["ui_values"]["chf_percentage"] == "0.95"

    spindle = detector_defaults({"biomarker": "Spindle", "detector": "YASA"})
    assert spindle["ui_values"]["freq_sp_low"] == "12"
    assert spindle["ui_values"]["freq_sp_high"] == "15"

    lsm = detector_defaults({"biomarker": "Spindle", "detector": "LSM"})
    assert lsm["ui_values"]["preset"] == "AllAges (Kwon 2023)"
    assert lsm["ui_values"]["preset_path"].endswith("all_ages_kwon_2023.json")
    assert lsm["ui_values"]["lsm_model_parameters_enabled"] == "false"
    assert lsm["ui_values"]["lsm_window_duration"] == "0.5"
    assert lsm["ui_values"]["lsm_theta_index"] == "7"
    assert lsm["ui_values"]["lsm_nine_15_index"] == "12,14"
    assert lsm["ui_values"]["lsm_transition_11"] == "0.9913189105383765"
    original_lsm = detector_defaults({"biomarker": "Spindle", "detector": "LSM", "preset": "Original (Kramer 2021)"})
    assert original_lsm["ui_values"]["preset"] == "Original (Kramer 2021)"
    assert original_lsm["ui_values"]["preset_path"].endswith("original_kramer_2021.json")
    assert original_lsm["ui_values"]["lsm_transition_11"] == "0.9953317206148224"

    spike = detector_defaults({"biomarker": "Spike", "detector": "RMS/LL"})
    assert spike["ui_values"]["rms_window"] == "0.02"
    assert spike["ui_values"]["ll_window"] == "0.01"


def test_parameter_planning_accepts_saved_session_sample_frequency_aliases():
    recording_info = {"sample_frequency_hz": 1000.0, "channel_names": ["A1"]}

    plan = recommend_parameters({"goal_profile": "smoke", "biomarker": "HFO", "recording_info": recording_info})
    defaults = detector_defaults({"biomarker": "HFO", "detector": "STE", "recording_info": recording_info})

    assert plan["parameters"]["filter"]["fs"] == 475.0
    assert defaults["sample_freq"] == 1000.0
