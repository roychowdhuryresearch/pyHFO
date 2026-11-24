import hashlib
import sys
import types
from importlib.util import find_spec
from pathlib import Path

import pytest

from pyhfo.classifer import Classifier
from pyhfo.hfo_app import HFO_App
from pyhfo.spindle_app import SpindleApp
from pyhfo.utils.default_onnx_models import (
    DEFAULT_HF_ONNX_MODELS,
    DEFAULT_HF_ONNX_MODEL_TAGS,
    HFOnnxModelSource,
    HFOnnxModelError,
    hf_onnx_uri,
    parse_hf_onnx_uri,
    resolve_hf_onnx_uri,
)
from pyhfo.utils.onnx_inference import discover_default_onnx_sources, discover_local_onnx_sources
from scripts import pyhfo_backend_entry


EXPECTED = {
    "a": {
        "repo_id": "roychowdhuryresearch/HFO-artifact",
        "revision": "7f95298b60deff4aae8b1ddfe64acda32a1c1fa7",
        "model_sha256": "51100e66fc8d77dd61d22c8c622f83650e17e13937894a40e0b378c2c07c8585",
        "spec_sha256": "f0250f37dca76653ff8fac63475746b23316b4f42343c2a7a2f4771649c705cd",
    },
    "s": {
        "repo_id": "roychowdhuryresearch/HFO-spkHFO",
        "revision": "de371a79569e81921e0174a4e440df2383dfdb8b",
        "model_sha256": "fdb3b333184e2f8073a097d345bbb2d7f68ea97fc9b16e648bac5e7c76ab8ca2",
        "spec_sha256": "51d528132388caf93629ae01695a5e9f15f4560c086e9eb7868ceadaba3aaff0",
    },
    "e": {
        "repo_id": "roychowdhuryresearch/HFO-eHFO",
        "revision": "2e59449dd81f588bf1a49bde647cd0e6d1db0a8c",
        "model_sha256": "aa33ee136a5862b398f9a20bc5cf8634f7f501e8f907eb4997e1beaf1dca7fcf",
        "spec_sha256": "b95fd045b5baf3fc4f33b06b73ebf79a191fd40704ae132054c25a196f1d0528",
    },
}


def test_default_hf_onnx_registry_pins_exact_sources():
    assert DEFAULT_HF_ONNX_MODEL_TAGS == ("a", "s", "e")
    assert set(DEFAULT_HF_ONNX_MODELS) == {"a", "s", "e"}
    for tag, expected in EXPECTED.items():
        model = DEFAULT_HF_ONNX_MODELS[tag]
        assert model.repo_id == expected["repo_id"]
        assert model.revision == expected["revision"]
        assert model.model_sha256 == expected["model_sha256"]
        assert model.spec_sha256 == expected["spec_sha256"]
        assert model.filename == "model.onnx"
        assert model.spec_filename == "model.spec.json"


def test_hf_uri_round_trip_preserves_model_identity():
    source = DEFAULT_HF_ONNX_MODELS["a"]

    parsed = parse_hf_onnx_uri(hf_onnx_uri(source))

    assert parsed == source


@pytest.mark.parametrize(
    ("revision", "model_sha256"),
    [
        ("main", "0" * 64),
        ("g" * 40, "0" * 64),
        ("0" * 40, "not-a-sha256"),
    ],
)
def test_hf_registry_rejects_mutable_or_malformed_pins(revision, model_sha256):
    with pytest.raises(HFOnnxModelError):
        HFOnnxModelSource(
            tag="a",
            repo_id="org/model",
            revision=revision,
            model_sha256=model_sha256,
            spec_sha256="0" * 64,
        )


def test_default_discovery_falls_back_to_hf_registry_after_incomplete_local_override(tmp_path):
    (tmp_path / "model_a.onnx").write_bytes(b"local-a")
    (tmp_path / "model_a.spec.json").write_text("{}", encoding="utf-8")

    sources = discover_default_onnx_sources(tmp_path)

    assert set(sources) == {"a", "s", "e"}
    assert sources["a"].startswith("hf://roychowdhuryresearch/HFO-artifact@")


def test_default_discovery_prefers_complete_local_override(tmp_path):
    for tag in ("a", "s", "e"):
        (tmp_path / f"model_{tag}.onnx").write_bytes(b"onnx")
        (tmp_path / f"model_{tag}.spec.json").write_text("{}", encoding="utf-8")

    sources = discover_default_onnx_sources(tmp_path)

    assert sources["a"].endswith("model_a.onnx")
    assert not sources["a"].startswith("hf://")


def test_default_discovery_does_not_implicitly_use_checkout_models(monkeypatch):
    monkeypatch.delenv("PYHFO_ONNX_DIR", raising=False)

    assert discover_local_onnx_sources() == {}
    assert discover_default_onnx_sources()["a"].startswith("hf://")


def test_hf_resolver_downloads_model_and_spec_with_revision_and_offline_flag(monkeypatch, tmp_path):
    calls = []
    contents = {
        "model.onnx": b"model-bytes",
        "model.spec.json": b'{"preprocessing": {}}',
    }
    model = DEFAULT_HF_ONNX_MODELS["a"].with_hashes(
        model_sha256=hashlib.sha256(contents["model.onnx"]).hexdigest(),
        spec_sha256=hashlib.sha256(contents["model.spec.json"]).hexdigest(),
    )

    def fake_hf_hub_download(repo_id, filename, *, revision=None, local_files_only=False, cache_dir):
        assert cache_dir == tmp_path / "home/cache/huggingface/hub"
        calls.append((repo_id, filename, revision, local_files_only))
        path = tmp_path / filename
        path.write_bytes(contents[filename])
        return str(path)

    for key in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PYHFO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fake_hf_hub_download))

    resolved = resolve_hf_onnx_uri(hf_onnx_uri(model))

    assert resolved == str(tmp_path / "model.onnx")
    assert calls == [
        (model.repo_id, "model.onnx", model.revision, True),
        (model.repo_id, "model.spec.json", model.revision, True),
    ]


def test_hf_resolver_fails_before_model_construction_on_hash_mismatch(monkeypatch, tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"wrong-model")
    spec_path = tmp_path / "model.spec.json"
    spec_path.write_bytes(b"{}")
    model = DEFAULT_HF_ONNX_MODELS["a"].with_hashes(
        model_sha256="0" * 64,
        spec_sha256=hashlib.sha256(b"{}").hexdigest(),
    )

    def fake_hf_hub_download(repo_id, filename, *, revision=None, local_files_only=False, cache_dir):
        return str(path if filename == "model.onnx" else spec_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fake_hf_hub_download))

    with pytest.raises(HFOnnxModelError, match="SHA256 mismatch.*model.onnx"):
        Classifier._resolve_onnx(hf_onnx_uri(model))


def test_default_cpu_classifier_uses_hf_onnx_when_local_override_absent(monkeypatch):
    monkeypatch.setattr("pyhfo.hfo_app.discover_local_onnx_sources", lambda: {})
    monkeypatch.setattr(HFO_App, "set_classifier", lambda self, param: None)
    app = HFO_App()

    app.set_default_cpu_classifier()

    assert app.param_classifier.model_type == "onnx_cpu"
    assert app.param_classifier.source_preference == "huggingface"
    assert app.param_classifier.artifact_onnx.startswith("hf://roychowdhuryresearch/HFO-artifact@")
    assert app.param_classifier.spike_onnx.startswith("hf://roychowdhuryresearch/HFO-spkHFO@")
    assert app.param_classifier.ehfo_onnx.startswith("hf://roychowdhuryresearch/HFO-eHFO@")


def test_spindle_default_cpu_classifier_uses_only_artifact_and_spike_hf_onnx(monkeypatch):
    monkeypatch.setattr("pyhfo.spindle_app.discover_local_onnx_sources", lambda: {})
    monkeypatch.setattr(SpindleApp, "set_classifier", lambda self, param: None)
    app = SpindleApp()

    app.set_default_cpu_classifier()

    assert app.param_classifier.model_type == "onnx_cpu"
    assert app.param_classifier.source_preference == "huggingface"
    assert app.param_classifier.artifact_onnx.startswith("hf://roychowdhuryresearch/HFO-artifact@")
    assert app.param_classifier.spike_onnx.startswith("hf://roychowdhuryresearch/HFO-spkHFO@")
    assert app.param_classifier.ehfo_onnx is None
    assert app.param_classifier.use_ehfo is False


@pytest.mark.skipif(find_spec("yasa") is None, reason="frozen desktop self-test includes YASA")
def test_backend_self_test_validates_registry_without_download(monkeypatch, capsys):
    def fail_download(*args, **kwargs):
        raise AssertionError("self-test must not download models by default")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fail_download))

    assert pyhfo_backend_entry._self_test() == 0

    payload = capsys.readouterr().out
    assert '"status": "passed"' in payload
    assert '"onnx_registry"' in payload
