"""Torch-free ONNX inference for the biomarker classifiers (Path B).

This is the runtime counterpart to ``utils_inference.inference`` but with **no
torch dependency** — it runs the exported ``.onnx`` models with onnxruntime so a
packaged build can drop torch/torchvision/transformers (~440MB).

It is deliberately **model-independent**: the runner reads the expected input
channel count from the ONNX graph itself and adapts (slices the feature stack to
that many channels, normalizes per channel exactly like ``_prepare_batch``).
Image H/W is dynamic in the exported graph, so the per-model preprocessing crop
(artifact/spike ~96x96, eHFO 224x224) is handled with no special-casing.

Verified end-to-end on 583 real STE-detected events (SM_B_ave.edf): label output
is identical to the torch path for all three models (0 mismatches across
artifact/spike/eHFO). See scripts/export_onnx.py for the converter.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from pyhfo.utils.default_onnx_models import default_hf_onnx_sources


DEFAULT_ONNX_MODEL_TAGS = ("a", "s", "e")


def _onnx_source_from_root(root: str | os.PathLike[str], tag: str) -> str | None:
    """Return a loader-ready source for one exported classifier model."""
    root_path = Path(root).expanduser()
    direct_model = root_path / f"model_{tag}.onnx"
    direct_spec = root_path / f"model_{tag}.spec.json"
    if direct_model.is_file() and direct_spec.is_file():
        return str(direct_model.resolve())

    model_dir = root_path / f"model_{tag}"
    if (model_dir / "model.onnx").is_file() and (model_dir / "model.spec.json").is_file():
        return str(model_dir.resolve())
    return None


def discover_local_onnx_sources(root: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """Discover the complete artifact/spike/eHFO ONNX bundle.

    ``PYHFO_ONNX_DIR`` is an explicit local override. A partial bundle is
    rejected so the default classifier never mixes local and hosted models.
    """
    candidate_roots = []
    if root:
        candidate_roots.append(Path(root))
    else:
        env_root = (os.environ.get("PYHFO_ONNX_DIR") or "").strip()
        if env_root:
            candidate_roots.append(Path(env_root))

    for candidate_root in candidate_roots:
        sources = {
            tag: _onnx_source_from_root(candidate_root, tag)
            for tag in DEFAULT_ONNX_MODEL_TAGS
        }
        if all(sources.values()):
            return {tag: str(source) for tag, source in sources.items()}
    return {}


def discover_default_onnx_sources(root: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """Return default ONNX sources, preferring a complete local override.

    Local packaged/source bundles still win when complete. If no complete bundle
    is present, return the pinned Hugging Face ONNX registry sources.
    """
    return discover_local_onnx_sources(root) or default_hf_onnx_sources()


def _normalize_img(channel: np.ndarray) -> np.ndarray:
    """Per-image min-max scale to 0..255 — numpy port of ``utils_inference.normalize_img``."""
    n = channel.shape[0]
    flat = channel.reshape(n, -1).astype(np.float64)
    lo = flat.min(axis=1, keepdims=True)
    hi = flat.max(axis=1, keepdims=True)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    return (255.0 * (flat - lo) / span).reshape(channel.shape)


def _prepare_batch(x: np.ndarray) -> np.ndarray:
    """Numpy port of ``utils_inference._prepare_batch``: normalize each channel (up to 3)."""
    x = x.astype(np.float64).copy()
    for c in range(min(x.shape[1], 3)):
        x[:, c] = _normalize_img(x[:, c])
    return x


class OnnxBiomarkerModel:
    """Drop-in, torch-free replacement for a loaded classifier model.

    ``inference(x)`` mirrors ``utils_inference.inference(model, x, ...)`` and
    returns a boolean array of ``prob > threshold``.
    """

    def __init__(self, onnx_path, threshold: float | None = None, spec_path=None):
        import json
        import os

        import onnxruntime as ort  # lazy: keep the module importable without onnxruntime

        onnx_path = str(onnx_path)
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        graph_input = self.session.get_inputs()[0]
        self.input_name = graph_input.name
        # Channel dim is fixed in the graph (e.g. [batch, 1, H, W]); H/W are dynamic.
        self.in_channels = (
            graph_input.shape[1]
            if len(graph_input.shape) >= 2 and isinstance(graph_input.shape[1], int)
            else None
        )

        # Sidecar spec (model_<tag>.spec.json): carries the feature preprocessing
        # params and output threshold. Lets the runtime stay model-independent.
        if spec_path is None:
            spec_path = onnx_path[:-5] + ".spec.json" if onnx_path.endswith(".onnx") else onnx_path + ".spec.json"
        self.spec = {}
        if os.path.exists(spec_path):
            with open(spec_path) as fh:
                self.spec = json.load(fh)
        self.preprocessing = self.spec.get("preprocessing")
        spec_threshold = (self.spec.get("output") or {}).get("threshold")
        self.threshold = 0.5 if threshold is None and spec_threshold is None else (
            threshold if threshold is not None else spec_threshold
        )

    def predict_proba(self, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
        x = np.asarray(x)
        if x.shape[0] == 0:
            return np.empty((0, 1), dtype=np.float32)
        # The feature stack may carry more channels than the model consumes
        # (e.g. artifact model takes only channel 0 of a 2-channel feature).
        if self.in_channels is not None and x.shape[1] > self.in_channels:
            x = x[:, : self.in_channels]
        x = _prepare_batch(x).astype(np.float32)
        outputs = []
        for start in range(0, x.shape[0], batch_size):
            chunk = x[start : start + batch_size]
            outputs.append(self.session.run(None, {self.input_name: chunk})[0])
        return np.concatenate(outputs, axis=0)

    def inference(self, x: np.ndarray, batch_size: int = 256, threshold: float | None = None) -> np.ndarray:
        thresh = self.threshold if threshold is None else threshold
        return np.squeeze(self.predict_proba(x, batch_size) > thresh)
