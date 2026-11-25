"""Pinned Hugging Face sources for optional PyTorch inference."""

import os
from pathlib import Path

from pyhfo.utils.default_onnx_models import DEFAULT_HF_ONNX_MODELS
from pyhfo.utils.model_registry import download_verified_files

# The three official cards host converted PyTorch weights in safetensors format
# alongside the ONNX export at the same pinned commit.
_GPU_HASHES = {
    "a": ("a43c4086f20c6e6d6a22c3ec79f08792689c60be7b50e72975643820bcc7c653",
          "931167170e6affd61941b4e1758334cca5a11dc1c520840fd6c112aa85dd7db6"),
    "s": ("38b50d68af1434c17e25299da2ce6b6144b3e7cf20af3147995c5ea265ce5ed8",
          "55165831f1b2d085bb6d54d5685ac0256d8ec10a4387620b66be7dca02c6f001"),
    "e": ("a65a5b47891dfe90bbe972fe85f016f3da2e2c249cc55a35cc98cd5388f9adfe",
          "4e10044454df5043c044f889542431a8897c66344dc95db203a4898ed8bb6a4f"),
}

SEVERITY_REPO = "roychowdhuryresearch/EEG-severity"
# Fill all three pins together after the owners upload and clear the model.
SEVERITY_REVISION = ""
SEVERITY_CONFIG_SHA256 = ""
SEVERITY_MODEL_SHA256 = ""


def resolve_classifier_card(card: str) -> str:
    """Resolve official cards to verified snapshots, preserving custom sources."""
    for tag, source in DEFAULT_HF_ONNX_MODELS.items():
        if card == source.repo_id:
            config_sha, model_sha = _GPU_HASHES[tag]
            files = download_verified_files(card, source.revision, {
                "config.json": config_sha, "model.safetensors": model_sha,
            })
            return str(files["config.json"].parent)
    return card


def severity_model_dir(model_dir=None) -> str:
    """Resolve explicit local weights first, then the pinned hosted release."""
    from pyhfo.agent_api.errors import PyHFOAgentError

    override = model_dir or os.environ.get("PYHFO_SEVERITY_MODEL_DIR")
    if override:
        root = Path(override).expanduser().resolve()
        for name in ("config.json", "model.safetensors"):
            path = root / name
            if not path.is_file() or path.stat().st_size == 0:
                raise PyHFOAgentError("model_not_found", f"Missing severity model file: {path}")
            with path.open("rb") as handle:
                if handle.readline(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
                    raise PyHFOAgentError("model_not_found", f"{path} is a Git LFS pointer. Supply the actual model files.")
        return str(root)

    repo = os.environ.get("PYHFO_SEVERITY_REPO") or SEVERITY_REPO
    if not (SEVERITY_REVISION and SEVERITY_CONFIG_SHA256 and SEVERITY_MODEL_SHA256):
        raise PyHFOAgentError(
            "severity_model_unpublished",
            f"PyHFO does not yet have a pinned severity model release for {repo}. "
            "Pass --model-dir or set PYHFO_SEVERITY_MODEL_DIR to a directory containing "
            "config.json and model.safetensors.",
        )
    files = download_verified_files(repo, SEVERITY_REVISION, {
        "config.json": SEVERITY_CONFIG_SHA256, "model.safetensors": SEVERITY_MODEL_SHA256,
    })
    return str(files["config.json"].parent)
