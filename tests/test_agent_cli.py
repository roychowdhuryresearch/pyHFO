import json
import subprocess
import sys
from pathlib import Path

from src import agent_cli


def test_agent_cli_help_runs():
    completed = subprocess.run(
        [sys.executable, "-m", "src.agent_cli", "--help"],
        check=False,
        text=True,
        capture_output=True,
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
            "--detector-param",
            "min_osc=6",
            "--export",
            "excel",
        ]
    )

    assert code == 0
    assert captured["input_path"] == "sample.edf"
    assert captured["output_dir"] == str(tmp_path)
    assert captured["filter"] == {"fp": 80}
    assert captured["detector_params"] == {"min_osc": 6}
    assert captured["exports"] == ["excel"]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


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
