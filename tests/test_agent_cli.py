import json
import subprocess
import sys
from pathlib import Path

from pyhfo import agent_cli


def test_agent_cli_help_runs():
    completed = subprocess.run(
        [sys.executable, "-m", "pyhfo.agent_cli", "--help"],
        check=False,
        text=True,
        capture_output=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert completed.returncode == 0
    assert "Agent-friendly PyHFO CLI" in completed.stdout


def test_info_command_outputs_json(monkeypatch, capsys):
    monkeypatch.setattr(
        agent_cli,
        "get_recording_info",
        lambda path: {"input_path": path, "sample_freq": 200.0},
    )

    code = agent_cli.main(["info", "--input", "sample.fif", "--json"])

    assert code == 0
    assert json.loads(capsys.readouterr().out)["sample_freq"] == 200.0


def test_line_noise_command_maps_arguments(monkeypatch, capsys):
    captured = {}

    def fake_detect_line_noise(request):
        captured.update(request)
        return {"status": "complete", "decision": {"should_notch": True, "frequency_hz": 60.0}}

    monkeypatch.setattr(agent_cli, "detect_line_noise", fake_detect_line_noise)

    code = agent_cli.main(
        [
            "line-noise",
            "--input",
            "sample.fif",
            "--exclude-channel",
            "EKG1",
            "--candidate-frequency",
            "60",
            "--site-line-freq-hz",
            "60",
            "--ratio-threshold",
            "4",
        ]
    )

    assert code == 0
    assert captured["input_path"] == "sample.fif"
    assert captured["exclude_channels"] == ["EKG1"]
    assert captured["candidate_frequencies_hz"] == [60.0]
    assert captured["site_line_freq_hz"] == 60.0
    assert captured["ratio_threshold"] == 4.0
    assert json.loads(capsys.readouterr().out)["decision"]["frequency_hz"] == 60.0


def test_detect_command_maps_arguments(monkeypatch, capsys, tmp_path):
    captured = {}

    def fake_run_detection(request):
        captured.update(request)
        return {"status": "complete", "artifacts": {}}

    monkeypatch.setattr(agent_cli, "run_detection", fake_run_detection)

    code = agent_cli.main(
        [
            "detect",
            "--input",
            "sample.edf",
            "--output",
            str(tmp_path),
            "--biomarker",
            "HFO",
            "--detector",
            "STE",
            "--n-jobs",
            "2",
            "--filter",
            "fp=80",
            "--filter",
            "fp=90",
            "--detector-param",
            "min_osc=6",
            "--export",
            "excel",
        ]
    )

    assert code == 0
    assert captured["input_path"] == "sample.edf"
    assert captured["output_dir"] == str(tmp_path)
    assert captured["filter"] == {"fp": 90}
    assert captured["detector_params"] == {"min_osc": 6}
    assert captured["exports"] == ["excel"]
    assert captured["include_channels"] == []
    assert captured["exclude_channels"] == []
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


def test_classify_command_maps_arguments(monkeypatch, capsys, tmp_path):
    captured = {}

    def fake_classify_existing_run(request):
        captured.update(request)
        return {"status": "complete", "classifier_metadata": {"classification_mode": "active_run_session"}}

    monkeypatch.setattr(agent_cli, "classify_existing_run", fake_classify_existing_run)

    code = agent_cli.main(
        [
            "classify",
            "--session",
            "runs/case.pybrain",
            "--output",
            str(tmp_path / "classified"),
            "--biomarker",
            "HFO",
            "--run-id",
            "run-1",
            "--classifier-param",
            "enabled=true",
            "--classifier-param",
            "model_type=default_cpu",
            "--classifier-param",
            "batch_size=8",
            "--classifier-param",
            "steps=artifact",
            "--classifier-param",
            "steps=spike",
            "--export",
            "csv",
            "--export",
            "session",
            "--session-format",
            "pybrain",
        ]
    )

    assert code == 0
    assert captured["session_path"] == "runs/case.pybrain"
    assert captured["output_dir"] == str(tmp_path / "classified")
    assert captured["biomarker"] == "HFO"
    assert captured["run_id"] == "run-1"
    assert captured["classifier"] == {"enabled": True, "model_type": "default_cpu", "batch_size": 8, "steps": ["artifact", "spike"]}
    assert captured["exports"] == ["csv", "session"]
    assert captured["session_format"] == "pybrain"
    assert json.loads(capsys.readouterr().out)["classifier_metadata"]["classification_mode"] == "active_run_session"


def test_quick_detect_builds_plan_then_runs_detection(monkeypatch, capsys, tmp_path):
    captured = {}

    monkeypatch.setattr(agent_cli, "get_recording_info", lambda path: {"input_path": path, "sample_freq": 512.0})

    def fake_recommend_parameters(request):
        captured["plan_request"] = request
        return {
            "status": "complete",
            "parameters": {
                "biomarker": "HFO",
                "detector": "STE",
                "filter": {"fp": 80, "fs": 200},
                "n_jobs": 1,
                "include_channels": ["A1"],
                "exclude_channels": ["EKG1"],
            },
            "validation": {"valid": True},
        }

    def fake_run_detection(request):
        captured["detection_request"] = request
        return {"status": "complete", "artifacts": {"summary_json": "summary.json"}}

    def fake_validate_parameters(request):
        captured["final_validation_request"] = request
        return {"valid": True, "normalized": {"filter": request["filter"]}}

    monkeypatch.setattr(agent_cli, "recommend_parameters", fake_recommend_parameters)
    monkeypatch.setattr(agent_cli, "run_detection", fake_run_detection)
    monkeypatch.setattr(agent_cli, "validate_parameters", fake_validate_parameters)

    code = agent_cli.main(
        [
            "quick-detect",
            "--input",
            "case.edf",
            "--output",
            str(tmp_path / "quick"),
            "--biomarker",
            "HFO",
            "--detector",
            "STE",
            "--goal-profile",
            "smoke",
            "--filter",
            "fs=180",
            "--classifier-param",
            "enabled=true",
            "--classifier-param",
            "steps=artifact",
            "--export",
            "summary-json",
        ]
    )

    assert code == 0
    assert captured["plan_request"]["recording_info"]["sample_freq"] == 512.0
    assert captured["plan_request"]["goal_profile"] == "smoke"
    assert captured["detection_request"]["input_path"] == "case.edf"
    assert captured["detection_request"]["output_dir"] == str(tmp_path / "quick")
    assert captured["detection_request"]["filter"] == {"fp": 80, "fs": 180}
    assert captured["detection_request"]["classifier"] == {"enabled": True, "steps": ["artifact"]}
    assert captured["detection_request"]["exports"] == ["summary-json"]
    assert captured["final_validation_request"]["filter"] == {"fp": 80, "fs": 180}
    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "quick_detection"
    assert output["final_validation"]["valid"] is True
    assert output["detection"]["status"] == "complete"


def test_parameters_commands_dispatch_to_parameter_api(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(agent_cli, "detector_defaults", lambda request: calls.append(("defaults", request)) or {"status": "complete", "tool": "defaults"})
    monkeypatch.setattr(agent_cli, "recommend_parameters", lambda request: calls.append(("recommend", request)) or {"status": "complete", "tool": "recommend"})
    monkeypatch.setattr(agent_cli, "validate_parameters", lambda request: calls.append(("validate", request)) or {"status": "valid", "tool": "validate"})
    monkeypatch.setattr(agent_cli, "sweep_parameters", lambda request: calls.append(("sweep", request)) or {"status": "complete", "tool": "sweep"})

    assert agent_cli.main(["parameters", "defaults", "--biomarker", "HFO", "--detector", "STE", "--sample-freq", "1000", "--n-jobs", "2"]) == 0
    assert agent_cli.main(["parameters", "recommend", "--goal-profile", "smoke", "--biomarker", "HFO", "--detector", "STE", "--recording-info-json", '{"sample_freq":512}', "--previous-json", '{"n_jobs":4}']) == 0
    assert agent_cli.main(["parameters", "validate", "--biomarker", "HFO", "--detector", "STE", "--sample-freq", "200", "--filter", "fp=80", "--filter", "fs=90", "--require-channels"]) == 0
    assert agent_cli.main(["parameters", "sweep", "--goal-profile", "exploratory", "--biomarker", "Spike", "--detector", "RMS/LL", "--sample-freq", "500"]) == 0

    capsys.readouterr()
    assert calls[0] == ("defaults", {"biomarker": "HFO", "detector": "STE", "sample_freq": 1000.0, "n_jobs": 2})
    assert calls[1] == (
        "recommend",
        {
            "biomarker": "HFO",
            "detector": "STE",
            "goal_profile": "smoke",
            "recording_info": {"sample_freq": 512},
            "previous_parameters": {"n_jobs": 4},
        },
    )
    assert calls[2] == (
        "validate",
        {"biomarker": "HFO", "detector": "STE", "sample_freq": 200.0, "filter": {"fp": 80, "fs": 90}, "require_channels": True},
    )
    assert calls[3] == ("sweep", {"biomarker": "Spike", "detector": "RMS/LL", "sample_freq": 500.0, "goal_profile": "exploratory"})


def test_detect_command_invalid_detector_exits_2(capsys, tmp_path):
    input_path = tmp_path / "sample.edf"
    input_path.write_text("not real", encoding="utf-8")

    code = agent_cli.main(
        [
            "detect",
            "--input",
            str(input_path),
            "--output",
            str(tmp_path / "out"),
            "--biomarker",
            "Spike",
            "--detector",
            "STE",
        ]
    )

    assert code == 2
    stderr = json.loads(capsys.readouterr().err)
    assert stderr["error"]["code"] == "invalid_detector"


def test_detect_command_runtime_failure_exits_1(monkeypatch, capsys, tmp_path):
    def fake_run_detection(_request):
        raise RuntimeError("boom")

    monkeypatch.setattr(agent_cli, "run_detection", fake_run_detection)

    code = agent_cli.main(
        [
            "detect",
            "--input",
            "sample.edf",
            "--output",
            str(tmp_path),
        ]
    )

    assert code == 1
    stderr = json.loads(capsys.readouterr().err)
    assert stderr["error"]["code"] == "RuntimeError"


def test_batch_create_writes_manifest(capsys, tmp_path):
    input_dir = tmp_path / "recordings"
    input_dir.mkdir()
    (input_dir / "case.edf").write_text("x", encoding="utf-8")

    code = agent_cli.main(
        [
            "batch-create",
            "--input",
            str(input_dir),
            "--output",
            str(tmp_path / "batch"),
        ]
    )

    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert Path(output["manifest_path"]).exists()
    assert output["summary"]["total"] == 1


def test_manifest_command_outputs_sidecar_contract(capsys):
    code = agent_cli.main(["manifest", "--callable-only"])

    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["schema_version"] == "pyhfo-sidecar.v1"
    names = {action["name"] for action in output["actions"]}
    assert "detector_defaults" in names
    assert "recommend_parameters" in names
    assert "validate_parameters" in names
    assert "sweep_parameters" in names
    assert all(
        action["status"] == "callable"
        for action in output["actions"]
        if action["name"] not in {"load_bids_channel_annotations", "compare_candidate_channels_to_bids_labels"}
    )
    labels = next(action for action in output["actions"] if action["name"] == "load_bids_channel_annotations")
    assert labels["status"] == "disabled_by_default"
    comparison = next(action for action in output["actions"] if action["name"] == "compare_candidate_channels_to_bids_labels")
    assert comparison["status"] == "disabled_by_default"


def test_prepare_command_maps_arguments(monkeypatch, capsys, tmp_path):
    captured = {}

    def fake_prepare(request):
        captured.update(request)
        return {"status": "complete", "output_path": request["output_path"]}

    monkeypatch.setattr(agent_cli, "prepare_recording", fake_prepare)

    code = agent_cli.main(
        [
            "prepare",
            "--input",
            "sample.edf",
            "--output",
            str(tmp_path / "clean.fif"),
            "--extra-bad-channel",
            "A1",
            "--bipolar-montage",
            "auto_adjacent",
            "--line-noise-notch-hz",
            "60",
            "--bandpass-json",
            '{"l_freq":1,"h_freq":300}',
        ]
    )

    assert code == 0
    assert captured["input_path"] == "sample.edf"
    assert captured["extra_bad_channels"] == ["A1"]
    assert captured["bipolar_montage"] == "auto_adjacent"
    assert captured["line_noise_notch_hz"] == 60
    assert captured["bandpass"] == {"l_freq": 1, "h_freq": 300}
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


def test_action_command_dispatches_sidecar_action(monkeypatch, capsys):
    class FakeSession:
        def dispatch(self, name, args):
            return {"status": "complete", "action": name, "result": args}

    saved = []
    monkeypatch.setattr(agent_cli, "load_sidecar_session", lambda session_id: FakeSession())
    monkeypatch.setattr(agent_cli, "save_sidecar_session", lambda session: saved.append(session))

    code = agent_cli.main(["action", "--name", "get_state", "--args-json", "{}"])

    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["action"] == "get_state"
    assert len(saved) == 1


def test_state_and_action_commands_persist_session(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("PYHFO_AGENT_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setattr(
        "pyhfo.agent_api.sidecar.get_recording_info",
        lambda path: {"input_path": path, "sample_freq": 512.0, "channel_names": ["A1-A2"]},
    )

    assert agent_cli.main(["action", "--session-id", "case-1", "--name", "load_recording", "--args-json", '{"input_path":"sample.fif"}']) == 0
    capsys.readouterr()

    assert agent_cli.main(["state", "--session-id", "case-1"]) == 0
    state = json.loads(capsys.readouterr().out)
    assert state["loaded_recording_path"] == "sample.fif"


def test_signal_analysis_commands_map_arguments(monkeypatch, capsys, tmp_path):
    calls = []
    monkeypatch.setattr(agent_cli, "infer_brain_signal_roi", lambda path: calls.append(("roi", path)) or {"status": "complete", "tool": "roi"})
    monkeypatch.setattr(agent_cli, "compute_channel_qc", lambda request: calls.append(("qc", request)) or {"status": "complete", "tool": "qc"})
    monkeypatch.setattr(agent_cli, "compute_spectral_profile", lambda request: calls.append(("spectral", request)) or {"status": "complete", "tool": "spectral"})
    monkeypatch.setattr(agent_cli, "recommend_detector_plan", lambda request: calls.append(("plan", request)) or {"status": "complete", "tool": "plan"})
    monkeypatch.setattr(agent_cli, "rank_candidate_channels", lambda request: calls.append(("rank", request)) or {"status": "complete", "tool": "rank"})

    assert agent_cli.main(["roi", "--input", "case.edf"]) == 0
    assert agent_cli.main(["qc", "--input", "case.edf", "--exclude-channel", "EKG1-EKG2", "--outlier-z-threshold", "7"]) == 0
    assert agent_cli.main(["spectral", "--input", "case.edf", "--exclude-channel", "X1-X2", "--max-channels", "8"]) == 0
    assert agent_cli.main(["detector-plan", "--input", "case.edf"]) == 0
    assert agent_cli.main(["rank-candidates", "--run", "HFO=/tmp/hfo.csv", "--duration-seconds", "300", "--exclude-channel", "EKG1-EKG2", "--top-n", "5"]) == 0

    # Drain emitted JSON so pytest does not keep large captured output around.
    capsys.readouterr()
    assert calls[0] == ("roi", "case.edf")
    assert calls[1] == ("qc", {"input_path": "case.edf", "exclude_channels": ["EKG1-EKG2"], "outlier_z_threshold": 7.0, "max_duration_seconds": 60.0})
    assert calls[2] == ("spectral", {"input_path": "case.edf", "exclude_channels": ["X1-X2"], "max_channels": 8, "max_duration_seconds": 60.0})
    assert calls[3] == ("plan", {"input_path": "case.edf"})
    assert calls[4] == (
        "rank",
        {"runs": [{"biomarker": "HFO", "events_csv": "/tmp/hfo.csv"}], "duration_seconds": 300.0, "exclude_channels": ["EKG1-EKG2"], "top_n": 5},
    )
