import json

from src.hfo_app import HFO_App
from src.utils.batch_project import create_batch_project, load_batch_project


def test_batch_project_discovers_supported_recordings_and_tracks_resume_state(tmp_path):
    input_dir = tmp_path / "input"
    nested_dir = input_dir / "nested"
    nested_dir.mkdir(parents=True)
    for path in (
        input_dir / "case_a.edf",
        input_dir / "case_b.vhdr",
        nested_dir / "case_c.fif",
        nested_dir / "case_d.fif.gz",
        input_dir / "ignore.txt",
    ):
        path.write_text("placeholder")

    project = create_batch_project(input_dir, tmp_path / "output")

    assert [case.case_id for case in project.cases] == ["case_a", "case_b", "case_c", "case_d"]
    assert project.progress_summary()["pending"] == 4

    next_case = project.next_case()
    project.mark_running(next_case.case_id)
    project.mark_failed(next_case.case_id, "detector failed")
    assert project.next_case(retry_failed=True).case_id == next_case.case_id

    project.mark_complete(next_case.case_id, {"run_count": 2})
    manifest_path = project.save()
    restored = load_batch_project(manifest_path)

    assert restored.get_case(next_case.case_id).status == "complete"
    assert restored.get_case(next_case.case_id).result["run_count"] == 2
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["format"] == "pybrain-batch-project"


def test_backend_load_database_returns_batch_project(tmp_path):
    recording = tmp_path / "case.edf"
    recording.write_text("placeholder")

    project = HFO_App().load_database(tmp_path, tmp_path / "batch")

    assert project.progress_summary()["total"] == 1
    assert project.cases[0].input_path.endswith("case.edf")
