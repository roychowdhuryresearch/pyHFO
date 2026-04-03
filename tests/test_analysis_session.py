import numpy as np

from src.hfo_feature import HFO_Feature
from src.param.param_detector import ParamDetector, ParamMNI, ParamSTE
from src.param.param_filter import ParamFilter
from src.utils.analysis_session import AnalysisSession, DetectionRun
from src.utils.session_store import load_session_checkpoint, save_session_checkpoint


def _build_hfo_run(detector_name, intervals, channel_names):
    if detector_name == "STE":
        detector_param = ParamSTE(2000).to_dict()
    else:
        detector_param = ParamMNI(2000).to_dict()
    return DetectionRun.create(
        biomarker_type="HFO",
        detector_name=detector_name,
        selected_channels=np.array(channel_names),
        param_filter=ParamFilter(),
        param_detector=ParamDetector.from_dict({"detector_type": detector_name, "detector_param": detector_param}),
        event_features=HFO_Feature(np.array(channel_names), np.array(intervals), np.array([]), detector_name, 2000),
        detector_output=np.array([intervals], dtype=object),
        classified=False,
    )


def test_analysis_session_tracks_multiple_runs_and_active_run():
    session = AnalysisSession("HFO")
    first_run = _build_hfo_run("STE", [[10, 20], [30, 40]], ["A1", "A2"])
    second_run = _build_hfo_run("MNI", [[50, 70]], ["A1"])

    session.add_run(first_run)
    session.add_run(second_run)

    assert session.active_run_id == second_run.run_id
    assert session.get_active_run().detector_name == "MNI"
    assert session.get_active_run().summary["num_events"] == 1


def test_session_store_round_trip_preserves_multiple_runs(tmp_path):
    session = AnalysisSession("HFO")
    session.add_run(_build_hfo_run("STE", [[10, 20], [30, 40]], ["A1", "A2"]))
    session.add_run(_build_hfo_run("MNI", [[50, 70]], ["A1"]))
    session.accept_run(session.active_run_id)
    session.set_run_visible(session.active_run_id, False)

    checkpoint = {
        "app_state_version": 3,
        "biomarker_type": "HFO",
        "sample_freq": 2000,
        "channel_names": np.array(["A1", "A2"]),
        "analysis_session": session.to_dict(),
    }
    path = tmp_path / "case.pybrain"
    save_session_checkpoint(path, checkpoint)

    loaded = load_session_checkpoint(path)
    restored = AnalysisSession.from_dict(loaded["analysis_session"])

    assert len(restored.runs) == 2
    assert restored.get_active_run().detector_name == "MNI"
    assert restored.get_accepted_run().detector_name == "MNI"
    assert restored.get_active_run().summary["top_channels"][0]["channel_name"] == "A1"
    assert restored.visible_run_ids == [run_id for run_id in restored.visible_run_ids]


def test_visible_runs_default_to_newest_run_and_can_be_toggled():
    session = AnalysisSession("HFO")
    first = _build_hfo_run("STE", [[10, 20], [30, 40]], ["A1", "A2"])
    second = _build_hfo_run("MNI", [[50, 70]], ["A1"])
    session.add_run(first)
    session.add_run(second)

    assert session.is_run_visible(first.run_id)
    assert session.is_run_visible(second.run_id)

    session.set_run_visible(first.run_id, False)

    assert not session.is_run_visible(first.run_id)
    assert session.get_visible_runs()[0].run_id == second.run_id


def test_activate_run_restores_hidden_run_visibility():
    session = AnalysisSession("HFO")
    first = _build_hfo_run("STE", [[10, 20], [30, 40]], ["A1", "A2"])
    second = _build_hfo_run("MNI", [[50, 70]], ["A1"])
    session.add_run(first)
    session.add_run(second)

    session.set_run_visible(first.run_id, False)
    session.activate_run(first.run_id)

    assert session.active_run_id == first.run_id
    assert session.is_run_visible(first.run_id)
    assert first.run_id in [run.run_id for run in session.get_visible_runs()]


def test_channel_ranking_and_run_comparison_support_decision_workflow():
    session = AnalysisSession("HFO")
    first = _build_hfo_run("STE", [[10, 20], [30, 40]], ["A1", "A2"])
    second = _build_hfo_run("MNI", [[10, 20], [80, 100]], ["A1", "A1"])
    session.add_run(first)
    session.add_run(second)
    session.accept_run(first.run_id)

    ranking = session.get_channel_ranking(first.run_id)
    comparison = session.compare_runs([first.run_id, second.run_id])

    assert ranking[0]["channel_name"] == "A1"
    assert ranking[0]["total_events"] == 1
    assert comparison["pairwise_overlap"][0]["overlap_events"] == 1
    assert comparison["pairwise_overlap"][0]["left_only"] == 1
    assert comparison["pairwise_overlap"][0]["right_only"] == 1


def test_decision_summary_prefers_accepted_run_context():
    session = AnalysisSession("HFO")
    first = _build_hfo_run("STE", [[10, 20], [30, 40]], ["A1", "A2"])
    second = _build_hfo_run("MNI", [[10, 20], [80, 100]], ["A1", "A1"])
    session.add_run(first)
    session.add_run(second)
    session.accept_run(first.run_id)

    accepted = session.get_accepted_run()
    ranking = session.get_channel_ranking(accepted.run_id)

    assert accepted.detector_name == "STE"
    assert ranking[0]["channel_name"] in {"A1", "A2"}
    assert session.accepted_run_id == first.run_id
