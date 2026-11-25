"""Pinned default Hugging Face ONNX classifier sources."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import parse_qs, urlencode

from pyhfo.utils.model_registry import ModelDownloadError, download_verified_files


class HFOnnxModelError(RuntimeError):
    """Raised when a pinned Hugging Face ONNX model cannot be resolved safely."""


@dataclass(frozen=True)
class HFOnnxModelSource:
    tag: str
    repo_id: str
    revision: str
    model_sha256: str
    spec_sha256: str
    filename: str = "model.onnx"
    spec_filename: str = "model.spec.json"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", self.tag):
            raise HFOnnxModelError(f"Invalid HF ONNX model tag: {self.tag!r}")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", self.repo_id):
            raise HFOnnxModelError(f"Invalid Hugging Face model repository id: {self.repo_id!r}")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise HFOnnxModelError(
                f"HF ONNX revision must be an immutable 40-character commit SHA: {self.revision!r}"
            )
        for label, digest in (
            ("model_sha256", self.model_sha256),
            ("spec_sha256", self.spec_sha256),
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise HFOnnxModelError(f"{label} must be a 64-character lowercase SHA256 digest.")
        for label, filename in (
            ("filename", self.filename),
            ("spec_filename", self.spec_filename),
        ):
            if not filename or Path(filename).name != filename:
                raise HFOnnxModelError(f"{label} must be a repository-root filename: {filename!r}")

    def with_hashes(self, *, model_sha256: str, spec_sha256: str) -> "HFOnnxModelSource":
        return replace(self, model_sha256=model_sha256, spec_sha256=spec_sha256)


DEFAULT_HF_ONNX_MODEL_TAGS = ("a", "s", "e")

DEFAULT_HF_ONNX_MODELS: dict[str, HFOnnxModelSource] = {
    "a": HFOnnxModelSource(
        tag="a",
        repo_id="roychowdhuryresearch/HFO-artifact",
        revision="7f95298b60deff4aae8b1ddfe64acda32a1c1fa7",
        model_sha256="51100e66fc8d77dd61d22c8c622f83650e17e13937894a40e0b378c2c07c8585",
        spec_sha256="f0250f37dca76653ff8fac63475746b23316b4f42343c2a7a2f4771649c705cd",
    ),
    "s": HFOnnxModelSource(
        tag="s",
        repo_id="roychowdhuryresearch/HFO-spkHFO",
        revision="de371a79569e81921e0174a4e440df2383dfdb8b",
        model_sha256="fdb3b333184e2f8073a097d345bbb2d7f68ea97fc9b16e648bac5e7c76ab8ca2",
        spec_sha256="51d528132388caf93629ae01695a5e9f15f4560c086e9eb7868ceadaba3aaff0",
    ),
    "e": HFOnnxModelSource(
        tag="e",
        repo_id="roychowdhuryresearch/HFO-eHFO",
        revision="2e59449dd81f588bf1a49bde647cd0e6d1db0a8c",
        model_sha256="aa33ee136a5862b398f9a20bc5cf8634f7f501e8f907eb4997e1beaf1dca7fcf",
        spec_sha256="b95fd045b5baf3fc4f33b06b73ebf79a191fd40704ae132054c25a196f1d0528",
    ),
}


def get_default_hf_onnx_model(tag: str) -> HFOnnxModelSource:
    try:
        return DEFAULT_HF_ONNX_MODELS[tag]
    except KeyError as exc:
        raise KeyError(f"Unknown default ONNX model tag: {tag!r}") from exc


def default_hf_onnx_sources(tags: tuple[str, ...] = DEFAULT_HF_ONNX_MODEL_TAGS) -> dict[str, str]:
    return {tag: hf_onnx_uri(get_default_hf_onnx_model(tag)) for tag in tags}


def hf_onnx_uri(source: HFOnnxModelSource) -> str:
    query = urlencode(
        {
            "tag": source.tag,
            "filename": source.filename,
            "spec_filename": source.spec_filename,
            "model_sha256": source.model_sha256,
            "spec_sha256": source.spec_sha256,
        }
    )
    return f"hf://{source.repo_id}@{source.revision}?{query}"


def parse_hf_onnx_uri(uri: str) -> HFOnnxModelSource:
    if not isinstance(uri, str) or not uri.startswith("hf://"):
        raise HFOnnxModelError(f"Unsupported HF ONNX source URI: {uri!r}")
    body = uri[len("hf://") :]
    location, _, raw_query = body.partition("?")
    repo_id, separator, revision = location.rpartition("@")
    if not separator or not repo_id or not revision:
        raise HFOnnxModelError(
            f"Invalid HF ONNX source URI {uri!r}. Expected hf://<repo>@<revision>?<checksums>."
        )
    query = {key: values[-1] for key, values in parse_qs(raw_query).items() if values}
    return HFOnnxModelSource(
        tag=query.get("tag", ""),
        repo_id=repo_id,
        revision=revision,
        filename=query.get("filename", "model.onnx"),
        spec_filename=query.get("spec_filename", "model.spec.json"),
        model_sha256=query.get("model_sha256", ""),
        spec_sha256=query.get("spec_sha256", ""),
    )


def resolve_hf_onnx_uri(uri: str, *, local_files_only: bool | None = None) -> str:
    source = parse_hf_onnx_uri(uri)
    try:
        paths = download_verified_files(
            source.repo_id,
            source.revision,
            {source.filename: source.model_sha256, source.spec_filename: source.spec_sha256},
            local_files_only=local_files_only,
        )
    except ModelDownloadError as exc:
        raise HFOnnxModelError(
            f"{exc} You can also set PYHFO_ONNX_DIR to a complete local artifact/spike/eHFO bundle."
        ) from exc
    return str(paths[source.filename])
