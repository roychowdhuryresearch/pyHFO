import numpy as np
from contextlib import contextmanager
import logging

from pyhfo.param.param_classifier import ParamClassifier
from pyhfo.preprocessing import PreProcessing
from pyhfo.utils.default_onnx_models import HFOnnxModelError, resolve_hf_onnx_uri
from pyhfo.utils.model_registry import configure_hub, hub_cache_dir
from pyhfo.utils.torch_models import resolve_classifier_card
from pyhfo.utils.model_metadata import DEFAULT_PREPROCESSING_BY_MODEL, resolve_preprocessing_metadata
from pyhfo.utils.onnx_inference import OnnxBiomarkerModel


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Install pyhfo[gpu] for PyTorch classifiers. "
            "Use artifact_onnx/spike_onnx/ehfo_onnx for torch-free inference."
        ) from exc
    return torch


def _torch_inference_helpers():
    from pyhfo.utils.utils_inference import inference, load_ckpt, load_preprocessing_param

    return inference, load_ckpt, load_preprocessing_param


def _load_card(card):
    configure_hub()
    try:
        from pyhfo.dl_models.modeling_neuralcnn import NeuralCNNForImageClassification
    except ImportError as exc:
        raise ImportError("Install pyhfo[gpu] for PyTorch classifiers.") from exc

    with _silence_hf_public_download_notice():
        return NeuralCNNForImageClassification.from_pretrained(resolve_classifier_card(card), cache_dir=hub_cache_dir())


@contextmanager
def _silence_hf_public_download_notice():
    logger = logging.getLogger("huggingface_hub.utils._http")
    previous_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous_level)


class Classifier():
    def __init__(self, param:ParamClassifier):
        self.device = param.device
        self.batch_size = param.batch_size
        self.model_type = param.model_type
        self.use_spike = param.use_spike
        self.use_ehfo = param.use_ehfo
        self.load_func = None
        self.model_a = None
        self.model_s = None
        self.model_e = None

    def update_model_s(self, param:ParamClassifier):
        torch = _torch()
        _, load_ckpt, _ = _torch_inference_helpers()
        self.model_type = param.model_type
        self.spike_path = param.spike_path
        self.param_spike_preprocessing, self.model_s = load_ckpt(param.spike_path)
        #print("spike model param", self.param_spike.to_dict())
        if self.model_type == "default_cpu":
            param.device = "cpu"
        elif self.model_type == "default_gpu":
            param.device = "cuda:0"
        else:
            raise ValueError("Model type not supported!")
        self.device = param.device if torch.cuda.is_available() else "cpu"
        self.model_s = self.model_s.to(self.device)
        self.preprocessing_spike = PreProcessing.from_param(self.param_spike_preprocessing)

    def update_model_a(self, param:ParamClassifier):
        torch = _torch()
        _, load_ckpt, _ = _torch_inference_helpers()
        self.model_type = param.model_type
        self.artifact_path = param.artifact_path
        self.param_artifact_preprocessing, model = load_ckpt(param.artifact_path)
        if "default" in self.model_type:
            model.channel_selection = True
            model.in_channels = 1
        if self.model_type == "default_cpu":
            param.device = "cpu"
        elif self.model_type == "default_gpu":
            param.device = "cuda:0"
        else:
            raise ValueError("Model type not supported!")
        self.device = param.device if torch.cuda.is_available() else "cpu"
        self.model_a = model.to(self.device)
        self.preprocessing_artifact = PreProcessing.from_param(self.param_artifact_preprocessing)

    def update_model_e(self, param:ParamClassifier):
        torch = _torch()
        self.model_type = param.model_type
        self.ehfo_path = param.ehfo_path
        from pyhfo.utils.legacy_checkpoint import load_legacy_checkpoint

        model_load = load_legacy_checkpoint(self.ehfo_path)
        self.model_e = model_load["model"]

        if self.model_type == "default_cpu":
            param.device = "cpu"
        elif self.model_type == "default_gpu":
            param.device = "cuda:0"
        else:
            raise ValueError("Model type not supported!")
        self.device = param.device if torch.cuda.is_available() else "cpu"
        self.model_e = self.model_e.to(self.device)
        self.preprocessing_ehfo = PreProcessing.from_dict(model_load["preprocessing"])

    def update_model_spkhfo(self, param:ParamClassifier):
        torch = _torch()
        _, _, load_preprocessing_param = _torch_inference_helpers()
        self.model_type = param.model_type
        self.spike_card = param.spike_card

        model = _load_card(param.spike_card)
        preprocessing_param_dict = resolve_preprocessing_metadata(
            model.config, fallback=DEFAULT_PREPROCESSING_BY_MODEL["spike"]
        )
        self.param_spike_preprocessing = load_preprocessing_param(preprocessing_param_dict)
        if self.model_type == "default_cpu":
            param.device = "cpu"
        elif self.model_type == "default_gpu":
            param.device = "cuda:0"
        else:
            raise ValueError("Model type not supported!")
        self.device = param.device if torch.cuda.is_available() else "cpu"
        self.model_s = model.to(self.device)
        self.preprocessing_spike = PreProcessing.from_param(self.param_spike_preprocessing)

    def update_model_artifact(self, param:ParamClassifier):
        torch = _torch()
        _, _, load_preprocessing_param = _torch_inference_helpers()
        self.model_type = param.model_type
        self.artifact_card = param.artifact_card

        model = _load_card(param.artifact_card)
        preprocessing_param_dict = resolve_preprocessing_metadata(
            model.config, fallback=DEFAULT_PREPROCESSING_BY_MODEL["artifact"]
        )
        self.param_artifact_preprocessing = load_preprocessing_param(preprocessing_param_dict)
        if "default" in self.model_type:
            model.channel_selection = True
            model.input_channels = 1
        if self.model_type == "default_cpu":
            param.device = "cpu"
        elif self.model_type == "default_gpu":
            param.device = "cuda:0"
        else:
            raise ValueError("Model type not supported!")
        self.device = param.device if torch.cuda.is_available() else "cpu"
        self.model_a = model.to(self.device)
        self.preprocessing_artifact = PreProcessing.from_param(self.param_artifact_preprocessing)

    def update_model_ehfo(self, param:ParamClassifier):
        torch = _torch()
        self.model_type = param.model_type
        self.ehfo_card = param.ehfo_card

        model = _load_card(param.ehfo_card)
        preprocessing_param_dict = resolve_preprocessing_metadata(
            model.config, fallback=DEFAULT_PREPROCESSING_BY_MODEL["ehfo"]
        )

        if self.model_type == "default_cpu":
            param.device = "cpu"
        elif self.model_type == "default_gpu":
            param.device = "cuda:0"
        else:
            raise ValueError("Model type not supported!")
        self.device = param.device if torch.cuda.is_available() else "cpu"
        self.model_e = model.to(self.device)
        self.preprocessing_ehfo = PreProcessing.from_dict(preprocessing_param_dict)

    # ---- ONNX (torch-free) model loaders ----------------------------------
    @staticmethod
    def _resolve_onnx(source):
        """Resolve an onnx source to a local .onnx path.

        ``source`` may be a local .onnx file, a directory containing
        ``model.onnx``, or a HuggingFace repo id hosting ``model.onnx`` +
        ``model.spec.json`` (downloaded + cached, public repo needs no token).
        """
        import os

        if source and os.path.exists(source):
            return os.path.join(source, "model.onnx") if os.path.isdir(source) else source

        if isinstance(source, str) and source.startswith("hf://"):
            return resolve_hf_onnx_uri(source)

        configure_hub()
        from huggingface_hub import hf_hub_download

        onnx_path = hf_hub_download(source, "model.onnx", cache_dir=hub_cache_dir())
        try:
            hf_hub_download(source, "model.spec.json", cache_dir=hub_cache_dir())  # sibling spec, cached next to the onnx
        except Exception as exc:
            raise HFOnnxModelError(
                f"ONNX Hugging Face source {source!r} is missing model.spec.json. "
                "Provide a loader-ready local directory, pinned hf:// source, or a repo with model.onnx and model.spec.json."
            ) from exc
        return onnx_path

    def _load_onnx_model(self, source):
        model = OnnxBiomarkerModel(self._resolve_onnx(source))
        if not model.preprocessing:
            raise ValueError(f"ONNX model at {source!r} is missing its model.spec.json preprocessing block.")
        return model, PreProcessing.from_dict(model.preprocessing)

    def update_model_a_onnx(self, param: ParamClassifier):
        self.model_a, self.preprocessing_artifact = self._load_onnx_model(param.artifact_onnx)

    def update_model_s_onnx(self, param: ParamClassifier):
        self.model_s, self.preprocessing_spike = self._load_onnx_model(param.spike_onnx)

    def update_model_e_onnx(self, param: ParamClassifier):
        self.model_e, self.preprocessing_ehfo = self._load_onnx_model(param.ehfo_onnx)

    def _predict(self, model, features, threshold=0.5):
        """Route to the torch-free ONNX path or the torch ``inference`` path."""
        if isinstance(model, OnnxBiomarkerModel):
            return model.inference(features, self.batch_size, threshold=threshold)
        inference, _, _ = _torch_inference_helpers()
        return inference(model, features, self.device, self.batch_size, threshold=threshold)

    def artifact_detection(self, biomarker_features, ignore_region, threshold=0.5):
        if self.model_a is None:
            raise ValueError("Please load artifact model first!")
        return self._classify_artifacts(self.model_a, biomarker_features, ignore_region, threshold=threshold)

    def spike_detection(self, biomarker_features):
        if self.model_s is None:
            raise ValueError("Please load spike model first!")
        return self._classify_spikes(self.model_s, biomarker_features)

    def ehfo_detection(self, biomarker_features):
        if self.model_e is None:
            raise ValueError("Please load eHFO model first!")
        return self._classify_ehfos(self.model_e, biomarker_features)

    def _classify_artifacts(self, model, biomarker_feature, ignore_region, threshold=0.5):
        features = self.preprocessing_artifact.process_biomarker_feature(biomarker_feature)
        artifact_predictions = np.zeros(features.shape[0]) -1
        starts = biomarker_feature.starts
        ends = biomarker_feature.ends
        keep_index = np.where(np.logical_and(starts > ignore_region[0], ends < ignore_region[1]) == True)[0]
        features = features[keep_index]
        if len(features) != 0:
            predictions = self._predict(model, features, threshold=threshold)
            artifact_predictions[keep_index] = predictions
        biomarker_feature.update_artifact_pred(artifact_predictions)
        return biomarker_feature

    def _classify_spikes(self, model, biomarker_feature):
        if len(biomarker_feature.artifact_predictions) == 0:
            raise ValueError("Please run artifact classifier first!")
        features = self.preprocessing_spike.process_biomarker_feature(biomarker_feature)
        spike_predictions = np.zeros(features.shape[0]) -1
        keep_index = np.where(biomarker_feature.artifact_predictions > 0)[0]
        features = features[keep_index]
        if len(features) != 0:
            predictions = self._predict(model, features)
            spike_predictions[keep_index] = predictions
        biomarker_feature.update_spike_pred(spike_predictions)
        return biomarker_feature

    def _classify_ehfos(self, model, biomarker_feature):
        if len(biomarker_feature.artifact_predictions) == 0:
            raise ValueError("Please run artifact classifier first!")
        features = self.preprocessing_ehfo.process_biomarker_feature(biomarker_feature)
        features = features[:, 0, :, :]
        features = np.expand_dims(features, axis=1)  # New shape: (batch, 1, height, width)
        features = np.repeat(features, repeats=3, axis=1)  # New shape: (batch, 3, height, w
        ehfo_predictions = np.zeros(features.shape[0]) -1
        keep_index = np.where(biomarker_feature.artifact_predictions > 0)[0]
        features = features[keep_index]
        if len(features) != 0:
            predictions = self._predict(model, features)
            ehfo_predictions[keep_index] = predictions
        biomarker_feature.update_ehfo_pred(ehfo_predictions)
        return biomarker_feature
