import logging

from pyhfo.hfo_app import HFO_App
from pyhfo.classifer import _silence_hf_public_download_notice
from pyhfo.param.param_classifier import ParamClassifier
from pyhfo.utils.default_onnx_models import DEFAULT_HF_ONNX_MODELS, hf_onnx_uri
from pyhfo.utils.model_metadata import describe_classifier_sources


def test_param_classifier_round_trip_preserves_source_preference():
    classifier_param = ParamClassifier(
        artifact_path="/tmp/model_a.tar",
        artifact_card="roychowdhuryresearch/HFO-artifact",
        model_type="default_cpu",
        source_preference="huggingface",
    )

    restored = ParamClassifier.from_dict(classifier_param.to_dict())

    assert restored.source_preference == "huggingface"
    assert restored.artifact_card == "roychowdhuryresearch/HFO-artifact"


def test_default_classifier_presets_prefer_hugging_face_onnx_without_local_override(monkeypatch):
    monkeypatch.setattr("pyhfo.hfo_app.discover_local_onnx_sources", lambda: {})
    monkeypatch.setattr(HFO_App, "set_classifier", lambda self, param: None)
    app = HFO_App()

    app.set_default_cpu_classifier()

    assert app.param_classifier.model_type == "onnx_cpu"
    assert app.param_classifier.source_preference == "huggingface"
    assert app.param_classifier.artifact_onnx.startswith("hf://roychowdhuryresearch/HFO-artifact@")
    assert app.param_classifier.artifact_card is None


def test_default_cpu_classifier_prefers_complete_onnx_bundle(monkeypatch):
    sources = {
        "a": "/bundle/model_a.onnx",
        "s": "/bundle/model_s.onnx",
        "e": "/bundle/model_e.onnx",
    }
    monkeypatch.setattr("pyhfo.hfo_app.discover_local_onnx_sources", lambda: sources)
    monkeypatch.setattr(HFO_App, "set_classifier", lambda self, param: None)
    app = HFO_App()

    app.set_default_cpu_classifier()

    assert app.param_classifier.model_type == "onnx_cpu"
    assert app.param_classifier.source_preference == "local"
    assert app.param_classifier.artifact_onnx == sources["a"]
    assert app.param_classifier.spike_onnx == sources["s"]
    assert app.param_classifier.ehfo_onnx == sources["e"]


def test_classifier_source_description_records_local_hash_and_hf_card(tmp_path):
    model_path = tmp_path / "model.tar"
    model_path.write_bytes(b"model-weights")
    classifier_param = ParamClassifier(
        artifact_path=str(model_path),
        artifact_card="roychowdhuryresearch/HFO-artifact",
        source_preference="local",
        use_spike=False,
        use_ehfo=False,
    )

    description = describe_classifier_sources(classifier_param)

    artifact = description["models"]["artifact"]
    assert artifact["preferred_source_kind"] == "local"
    assert artifact["local_exists"] is True
    assert len(artifact["local_sha256"]) == 64
    assert artifact["huggingface_card"] == "roychowdhuryresearch/HFO-artifact"


def test_classifier_source_description_records_local_onnx_hash(tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx-model")
    classifier_param = ParamClassifier(
        artifact_onnx=str(model_path),
        use_spike=False,
        use_ehfo=False,
        model_type="onnx_cpu",
    )

    artifact = describe_classifier_sources(classifier_param)["models"]["artifact"]

    assert artifact["preferred_source_kind"] == "onnx_local"
    assert artifact["preferred_source"] == str(model_path)
    assert artifact["onnx_local_exists"] is True
    assert len(artifact["onnx_sha256"]) == 64


def test_classifier_source_description_records_pinned_hf_onnx_identity():
    source = DEFAULT_HF_ONNX_MODELS["a"]
    classifier_param = ParamClassifier(
        artifact_onnx=hf_onnx_uri(source),
        use_spike=False,
        use_ehfo=False,
        model_type="onnx_cpu",
        source_preference="huggingface",
    )

    artifact = describe_classifier_sources(classifier_param)["models"]["artifact"]

    assert artifact["preferred_source_kind"] == "onnx_huggingface"
    assert artifact["onnx_repo_id"] == source.repo_id
    assert artifact["onnx_revision"] == source.revision
    assert artifact["onnx_expected_sha256"] == source.model_sha256
    assert artifact["onnx_spec_sha256"] == source.spec_sha256


def test_silence_hf_public_download_notice_restores_logger_level():
    logger = logging.getLogger("huggingface_hub.utils._http")
    previous_level = logger.level
    logger.setLevel(logging.WARNING)

    with _silence_hf_public_download_notice():
        assert logger.level == logging.ERROR

    assert logger.level == logging.WARNING
    logger.setLevel(previous_level)


def test_default_gpu_uses_pinned_hub_weights_without_checkout(tmp_path, monkeypatch):
    from pyhfo.utils import torch_models

    monkeypatch.setattr(HFO_App, "set_classifier", lambda self, param: None)
    app = HFO_App()
    app.set_default_gpu_classifier()
    calls = []

    def fetch(repo, revision, hashes):
        calls.append((repo, revision, hashes))
        return {"config.json": tmp_path / "config.json", "model.safetensors": tmp_path / "model.safetensors"}

    monkeypatch.setattr(torch_models, "download_verified_files", fetch)
    for tag, prefix in (("a", "artifact"), ("s", "spike"), ("e", "ehfo")):
        card = getattr(app.param_classifier, f"{prefix}_card")
        assert getattr(app.param_classifier, f"{prefix}_path") is None
        assert torch_models.resolve_classifier_card(card) == str(tmp_path)
        repo, revision, hashes = calls[-1]
        assert repo == DEFAULT_HF_ONNX_MODELS[tag].repo_id
        assert revision == DEFAULT_HF_ONNX_MODELS[tag].revision
        assert set(hashes) == {"config.json", "model.safetensors"}
        assert all(len(bytes.fromhex(digest)) == 32 for digest in hashes.values())


def test_legacy_archives_resolve_both_model_module_names(monkeypatch):
    import io
    import sys
    from types import ModuleType
    from pyhfo.utils.legacy_checkpoint import Unpickler

    model = ModuleType("pyhfo.model")
    model.NeuralCNN = object()
    monkeypatch.setitem(sys.modules, "pyhfo.model", model)
    for module_name in ("src.model", "src.model_pyhfo_process"):
        archive = io.BytesIO(f"c{module_name}\nNeuralCNN\n.".encode())
        assert Unpickler(archive).load() is model.NeuralCNN
