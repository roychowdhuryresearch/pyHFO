import subprocess
import sys
from pathlib import Path


def test_classifier_import_does_not_load_torch_model_module():
    script = """
import importlib
import sys
importlib.import_module("pyhfo.classifer")
assert "pyhfo.model" not in sys.modules
assert "torch" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stderr or result.stdout
