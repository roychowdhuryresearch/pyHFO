import builtins
import json
import sys
import types
from pathlib import Path

import pytest

from pyhfo.agent_api.pipeline import _classifier_metadata, _load_default_classifier_if_requested
from pyhfo.classifer import Classifier
from pyhfo.param.param_classifier import ParamClassifier
from pyhfo.spindle_app import SpindleApp
from pyhfo.utils.onnx_inference import discover_local_onnx_sources


def _preprocessing_spec():
    return {
        "preprocessing": {
            "freq_range_hz": [10, 500],
            "fs": 2000,
            "image_size": 224,
            "random_shift_ms": 0,
            "selected_freq_range_hz": [10, 220],
            "selected_window_size_ms": 214.28571429,
            "time_range_ms": [0, 1000],
        },
        "input": {"name": "input", "channels": 1},
        "output": {"name": "prob", "threshold": 0.5},
    }


def _write_loader_ready_model(directory: Path) -> Path:
    directory.mkdir(parents=True)
    onnx_path = directory / "model.onnx"
    onnx_path.write_bytes(b"tiny-onnx-placeholder")
    (directory / "model.spec.json").write_text(json.dumps(_preprocessing_spec()), encoding="utf-8")
    return onnx_path


def test_resolve_onnx_returns_model_onnx_for_loader_ready_directory(tmp_path):
    model_dir = tmp_path / "model_a"
    _write_loader_ready_model(model_dir)

    resolved = Classifier._resolve_onnx(str(model_dir))

    assert resolved == str(model_dir / "model.onnx")


def test_real_generated_onnx_layout_is_loader_ready_when_present():
    generated_root = Path(__file__).resolve().parents[1] / "ckpt_onnx"
    model_dirs = [generated_root / name for name in ("model_a", "model_s", "model_e")]
    if not all((model_dir / "model.onnx").exists() and (model_dir / "model.spec.json").exists() for model_dir in model_dirs):
        pytest.skip("No generated ckpt_onnx loader-ready layout is present in this checkout.")

    resolved = [Classifier._resolve_onnx(str(model_dir)) for model_dir in model_dirs]

    assert resolved == [str(model_dir / "model.onnx") for model_dir in model_dirs]


def test_discover_default_onnx_sources_requires_a_complete_bundle(tmp_path):
    for tag in ("a", "s"):
        (tmp_path / f"model_{tag}.onnx").write_bytes(b"onnx")
        (tmp_path / f"model_{tag}.spec.json").write_text("{}", encoding="utf-8")
    assert discover_local_onnx_sources(tmp_path) == {}

    (tmp_path / "model_e.onnx").write_bytes(b"onnx")
    (tmp_path / "model_e.spec.json").write_text("{}", encoding="utf-8")
    sources = discover_local_onnx_sources(tmp_path)
    assert set(sources) == {"a", "s", "e"}
    assert sources["a"].endswith("model_a.onnx")


def test_onnx_cpu_model_type_uses_bundled_loader_and_hydrates_payload():
    class FakeApp:
        def __init__(self):
            self.called = 0
            self.param_classifier = None

        def set_default_onnx_classifier(self):
            self.called += 1
            self.param_classifier = ParamClassifier(
                artifact_onnx="/bundle/model_a.onnx",
                spike_onnx="/bundle/model_s.onnx",
                ehfo_onnx="/bundle/model_e.onnx",
                model_type="onnx_cpu",
            )

    app = FakeApp()
    payload = {"model_type": "onnx_cpu", "batch_size": 16}
    assert _load_default_classifier_if_requested(app, payload) is True
    assert app.called == 1
    assert payload["model_type"] == "onnx_cpu"
    assert payload["artifact_onnx"] == "/bundle/model_a.onnx"
    assert payload["spike_onnx"] == "/bundle/model_s.onnx"
    assert payload["ehfo_onnx"] == "/bundle/model_e.onnx"
    assert app.param_classifier.batch_size == 16


def test_spindle_default_cpu_classifier_uses_onnx_without_ehfo(monkeypatch):
    sources = {
        "a": "/bundle/model_a.onnx",
        "s": "/bundle/model_s.onnx",
        "e": "/bundle/model_e.onnx",
    }
    monkeypatch.setattr("pyhfo.spindle_app.discover_local_onnx_sources", lambda: sources)
    monkeypatch.setattr(SpindleApp, "set_classifier", lambda self, param: None)
    app = SpindleApp()

    app.set_default_cpu_classifier()

    assert app.param_classifier.model_type == "onnx_cpu"
    assert app.param_classifier.artifact_onnx == sources["a"]
    assert app.param_classifier.spike_onnx == sources["s"]
    assert app.param_classifier.ehfo_onnx is None
    assert app.param_classifier.use_ehfo is False


def test_resolve_onnx_downloads_model_and_sidecar_for_huggingface_source(monkeypatch, tmp_path):
    calls = []

    def fake_hf_hub_download(repo_id, filename, *, cache_dir):
        calls.append((repo_id, filename))
        path = tmp_path / repo_id.replace("/", "_") / filename
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"downloaded")
        return str(path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fake_hf_hub_download))

    resolved = Classifier._resolve_onnx("org/hfo-artifact-onnx")

    assert resolved.endswith("model.onnx")
    assert calls == [
        ("org/hfo-artifact-onnx", "model.onnx"),
        ("org/hfo-artifact-onnx", "model.spec.json"),
    ]


def test_explicit_onnx_sources_skip_default_classifier_loader():
    class FakeApp:
        def __init__(self):
            self.default_cpu_called = 0

        def set_default_cpu_classifier(self):
            self.default_cpu_called += 1

    payload = {
        "model_type": "default_cpu",
        "artifact_onnx": "/models/artifact/model.onnx",
        "spike_onnx": "/models/spike/model.onnx",
        "ehfo_onnx": "/models/ehfo/model.onnx",
    }

    used_default = _load_default_classifier_if_requested(FakeApp(), payload)

    assert used_default is False


def test_classifier_metadata_reports_explicit_onnx_keys():
    payload = {
        "enabled": True,
        "device": "cpu",
        "batch_size": 8,
        "model_type": "default_cpu",
        "source_preference": "local",
        "artifact_onnx": "/models/artifact/model.onnx",
        "spike_onnx": "/models/spike/model.onnx",
        "ehfo_onnx": "/models/ehfo/model.onnx",
        "artifact_card": "org/hfo-artifact-onnx",
        "spike_card": "org/hfo-spkhfo-onnx",
        "ehfo_card": "org/hfo-ehfo-onnx",
        "steps": ["artifact"],
    }

    metadata = _classifier_metadata(payload, "HFO", executed_steps=["artifact"])

    assert metadata["artifact_onnx"] == "/models/artifact/model.onnx"
    assert metadata["spike_onnx"] == "/models/spike/model.onnx"
    assert metadata["ehfo_onnx"] == "/models/ehfo/model.onnx"
    assert metadata["artifact_card"] == "org/hfo-artifact-onnx"
    assert metadata["executed_steps"] == ["artifact"]


def test_onnx_classifier_loading_does_not_import_torch_stack(monkeypatch, tmp_path):
    model_dir = tmp_path / "model_a"
    _write_loader_ready_model(model_dir)
    blocked_imports = {
        "torch",
        "torchvision",
        "transformers",
        "pyhfo.model",
        "pyhfo.dl_models",
        "pyhfo.utils.utils_inference",
    }
    loaded_paths = []

    class FakeOnnxBiomarkerModel:
        def __init__(self, onnx_path):
            loaded_paths.append(onnx_path)
            spec_path = Path(onnx_path).with_name("model.spec.json")
            self.preprocessing = json.loads(spec_path.read_text(encoding="utf-8"))["preprocessing"]

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in blocked_imports:
            raise AssertionError(f"ONNX loading unexpectedly imported {name}")
        return real_import(name, globals, locals, fromlist, level)

    for name in blocked_imports:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr("pyhfo.classifer.OnnxBiomarkerModel", FakeOnnxBiomarkerModel)

    classifier = Classifier(ParamClassifier(artifact_onnx=str(model_dir)))
    classifier.update_model_a_onnx(ParamClassifier(artifact_onnx=str(model_dir)))

    assert loaded_paths == [str(model_dir / "model.onnx")]
