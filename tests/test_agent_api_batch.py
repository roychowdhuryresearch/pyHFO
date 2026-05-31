from pathlib import Path

from src.agent_api import create_batch, run_batch


def test_create_batch_discovers_supported_files(tmp_path):
    input_dir = tmp_path / "recordings"
    input_dir.mkdir()
    (input_dir / "a.edf").write_text("a", encoding="utf-8")
    (input_dir / "b.fif.gz").write_text("b", encoding="utf-8")
    (input_dir / "ignore.txt").write_text("x", encoding="utf-8")

    result = create_batch(input_dir, tmp_path / "batch")

    assert Path(result["manifest_path"]).exists()
    assert result["summary"]["total"] == 2
    assert {case["case_id"] for case in result["cases"]} == {"a", "b"}


def test_run_batch_updates_case_statuses(monkeypatch, tmp_path):
    input_dir = tmp_path / "recordings"
    input_dir.mkdir()
    (input_dir / "a.edf").write_text("a", encoding="utf-8")
    (input_dir / "b.edf").write_text("b", encoding="utf-8")
    batch = create_batch(input_dir, tmp_path / "batch")

    def fake_run_detection(request):
        output_dir = Path(request["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "status": "complete",
            "input_path": request["input_path"],
            "output_dir": str(output_dir),
            "artifacts": {"summary_json": str(output_dir / "summary.json")},
        }

    monkeypatch.setattr("src.agent_api.pipeline.run_detection", fake_run_detection)

    result = run_batch(
        {
            "manifest_path": batch["manifest_path"],
            "detection": {"biomarker": "HFO", "detector": "STE"},
        }
    )

    assert result["status"] == "complete"
    assert result["summary"]["complete"] == 2
    assert all(case["status"] == "complete" for case in result["cases"])
