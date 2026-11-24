"""P0-4 parity: Electron 'Run classifier' with default model_type and null paths
must load the bundled default models, mirroring PyQt's Hugging Face CPU/GPU buttons
(set_default_cpu_classifier / set_default_gpu_classifier). Previously the agent
pipeline built an all-null ParamClassifier and loaded no model at all."""

from pyhfo.agent_api.pipeline import _run_classifier, _load_default_classifier_if_requested


class _FakeParam:
    def __init__(self):
        self.artifact_path = None
        self.spike_path = None
        self.ehfo_path = None
        self.artifact_card = None
        self.spike_card = None
        self.ehfo_card = None
        self.batch_size = 32
        self.device = "cpu"


class _FakeApp:
    def __init__(self):
        self.param_classifier = None
        self.default_cpu_called = 0
        self.default_gpu_called = 0
        self.set_classifier_called = 0
        self.classified = []

    def set_default_cpu_classifier(self):
        self.default_cpu_called += 1
        param = _FakeParam()
        param.artifact_path = "/ckpt/model_a.tar"
        param.spike_path = "/ckpt/model_s.tar"
        param.ehfo_path = "/ckpt/model_e.tar"
        param.artifact_card = "roychowdhuryresearch/HFO-artifact"
        self.param_classifier = param

    def set_default_gpu_classifier(self):
        self.default_gpu_called += 1
        self.set_default_cpu_classifier()
        self.param_classifier.device = "cuda:0"

    def set_classifier(self, param):
        self.set_classifier_called += 1
        self.param_classifier = param

    def classify_artifacts(self, ignore_region=None, threshold=0.5):
        self.classified.append("artifact")

    def classify_spikes(self):
        self.classified.append("spike")

    def classify_ehfos(self):
        self.classified.append("ehfo")


def _payload(**over):
    base = {
        "enabled": True,
        "artifact_path": None, "spike_path": None, "ehfo_path": None,
        "artifact_card": None, "spike_card": None, "ehfo_card": None,
        "use_spike": True, "use_ehfo": True,
        "device": "cpu", "batch_size": 32,
        "model_type": "default_cpu", "source_preference": "auto",
        "steps": ["artifact", "spike", "ehfo"],
    }
    base.update(over)
    return base


def test_default_cpu_classifier_loads_bundled_models_when_paths_null():
    app = _FakeApp()
    payload = _payload()
    meta = _run_classifier(app, "HFO", payload, [])
    assert app.default_cpu_called == 1
    # used the bundled-default loader, not the all-null ParamClassifier
    assert app.set_classifier_called == 0
    assert app.classified == ["artifact", "spike", "ehfo"]
    # metadata + payload now reflect the bundled defaults instead of null UI values
    assert meta["artifact_path"] == "/ckpt/model_a.tar"
    assert payload["artifact_path"] == "/ckpt/model_a.tar"
    assert payload["artifact_card"] == "roychowdhuryresearch/HFO-artifact"


def test_default_gpu_model_type_uses_gpu_loader():
    app = _FakeApp()
    _run_classifier(app, "HFO", _payload(model_type="default_gpu", device="cuda:0"), [])
    assert app.default_gpu_called == 1
    assert app.default_cpu_called == 1  # gpu loader delegates to cpu helper in this fake


def test_explicit_local_model_skips_default_loader():
    app = _FakeApp()
    assert _load_default_classifier_if_requested(app, _payload(artifact_path="/custom/model.tar")) is False
    _run_classifier(app, "HFO", _payload(artifact_path="/custom/model.tar"), [])
    assert app.default_cpu_called == 0
    assert app.set_classifier_called == 1  # explicit path -> ParamClassifier flow


def test_explicit_onnx_model_skips_default_loader_and_is_reported():
    app = _FakeApp()
    payload = _payload(
        artifact_onnx="/onnx/model_a.onnx",
        spike_onnx="/onnx/model_s.onnx",
        ehfo_onnx="/onnx/model_e.onnx",
    )
    assert _load_default_classifier_if_requested(app, payload) is False

    meta = _run_classifier(app, "HFO", payload, [])

    assert app.default_cpu_called == 0
    assert app.set_classifier_called == 1
    assert app.param_classifier.artifact_onnx == "/onnx/model_a.onnx"
    assert meta["artifact_onnx"] == "/onnx/model_a.onnx"
    assert meta["spike_onnx"] == "/onnx/model_s.onnx"
    assert meta["ehfo_onnx"] == "/onnx/model_e.onnx"


def test_batch_size_override_applied_to_default_param():
    app = _FakeApp()
    _run_classifier(app, "HFO", _payload(batch_size=8), [])
    assert app.param_classifier.batch_size == 8


def test_missing_default_loader_falls_back_without_error():
    class _NoDefaultApp:
        def __init__(self):
            self.param_classifier = None
            self.set_classifier_called = 0

        def set_classifier(self, param):
            self.set_classifier_called += 1
            self.param_classifier = param

        def classify_artifacts(self, *a, **k):
            pass

        def classify_spikes(self):
            pass

    app = _NoDefaultApp()
    _run_classifier(app, "HFO", _payload(steps=["artifact", "spike"]), [])
    assert app.set_classifier_called == 1


def test_failing_classify_step_is_isolated_and_warned():
    """A model whose preprocessing is out of range (e.g. an eHFO model whose
    selected_freq_range_hz exceeds its freq_range_hz -> PreProcessing.check_bound)
    must not abort the whole run: the other steps still execute and the failure
    surfaces as a per-step warning instead of a top-level crash."""

    class _BoundErrorApp(_FakeApp):
        def classify_ehfos(self):
            raise AssertionError(
                "Index out of bound on selected_freq_range_hz_high: index -233.0 is outside [0, 224]."
            )

    app = _BoundErrorApp()
    warnings: list[str] = []
    meta = _run_classifier(app, "HFO", _payload(), warnings)
    # artifact + spike still ran; ehfo was skipped, not aborted
    assert app.classified == ["artifact", "spike"]
    assert meta["executed_steps"] == ["artifact", "spike"]
    # the failure is visible and names the offending step + cause
    assert any("ehfo" in w and "Index out of bound" in w for w in warnings), warnings
