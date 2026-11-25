import json
import os
import subprocess
import sys
from pathlib import Path

from pyhfo import __version__
from pyhfo.agent_api.pipeline import get_recording_info
from pyhfo.agent_api.sessions import create_project, export_session, open_project
from pyhfo.agent_api.sidecar import PyHFOSidecarSession, load_sidecar_session, save_sidecar_session
from pyhfo.agent_api.sidecar_web import AGENT_UI_COMMANDS_REQUIRING_CONFIRMATION
from pyhfo.agent_mcp.server import PyHFOMCPServer
from pyhfo.utils.analysis_session import DetectionRun
from pyhfo.utils.batch_project import create_batch_project


def test_persistent_state_uses_pyhfo_home(tmp_path, monkeypatch):
    home = tmp_path / "pyhfo-home"
    monkeypatch.setenv("PYHFO_HOME", str(home))
    for name in ("SESSION", "PROJECT", "EXPORT"):
        monkeypatch.delenv(f"PYHFO_AGENT_{name}_DIR", raising=False)
    session = PyHFOSidecarSession(session_id="home-test")
    session.action_log.append({"action": "inspect"})
    assert save_sidecar_session(session) == home / "sessions" / "home-test.json"
    assert load_sidecar_session("home-test").action_log == session.action_log
    assert Path(create_project({"project_id": "case"})["project_path"]) == home / "projects/case/project.json"
    exported = export_session({"output_path": "test.json", "session_state": {"subject": "synthetic"}})
    assert Path(exported["artifact_path"]) == home / "exports/test.json"
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    assert Path(create_batch_project(recordings).output_dir).parent == home / "exports"


def test_existing_desktop_project_at_legacy_location_remains_readable(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PYHFO_HOME", raising=False)
    monkeypatch.delenv("PYHFO_AGENT_PROJECT_DIR", raising=False)
    project = tmp_path / ".pyhfo_agent_projects" / "case" / "project.json"
    project.parent.mkdir(parents=True)
    project.write_text(json.dumps({"project_id": "case"}), encoding="utf-8")
    assert open_project(project)["state"]["project_id"] == "case"
    assert Path(create_project({"project_id": "new"})["project_path"]).is_relative_to(tmp_path / ".pyhfo")


def test_package_version_and_ui_catalog_on_public_surfaces(tiny_fif_path):
    assert PyHFOMCPServer().initialize_result()["serverInfo"]["version"] == __version__
    assert get_recording_info(tiny_fif_path)["software_version"] == __version__
    assert "detection.run" in AGENT_UI_COMMANDS_REQUIRING_CONFIRMATION
    run = DetectionRun.create("HFO", "STE", ["A1"]).to_dict()
    assert run["software_version"] == __version__
    assert DetectionRun.from_dict(run).software_version == __version__
    del run["software_version"]
    assert DetectionRun.from_dict(run).software_version is None


def test_analysis_caches_default_to_home_and_honor_host_overrides(tmp_path):
    home = tmp_path / "home"
    env = {**os.environ, "PYHFO_HOME": str(home)}
    for name in ("MPLCONFIGDIR", "NUMBA_CACHE_DIR"):
        env.pop(name, None)
    probe = ("import os, json, pyhfo; "
             "print(json.dumps([os.environ['MPLCONFIGDIR'], os.environ['NUMBA_CACHE_DIR']]))")

    def read_cache_paths():
        out = subprocess.check_output([sys.executable, "-c", probe], env=env, text=True)
        return json.loads(out)

    assert read_cache_paths() == [str(home / "cache/matplotlib"), str(home / "cache/numba")]
    env.update(MPLCONFIGDIR=str(tmp_path / "plots"), NUMBA_CACHE_DIR=str(tmp_path / "jit"))
    assert read_cache_paths() == [env["MPLCONFIGDIR"], env["NUMBA_CACHE_DIR"]]
