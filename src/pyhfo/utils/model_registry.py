"""Local resolution and integrity checks shared by the published model presets."""

import hashlib
import os
import re
from pathlib import Path

from pyhfo.paths import pyhfo_home


class ModelDownloadError(RuntimeError):
    pass


def hub_cache_dir() -> Path:
    for variable in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(variable):
            return Path(os.environ[variable]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    return pyhfo_home() / "cache" / "huggingface" / "hub"


def hub_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}


def configure_hub():
    # Hugging Face reads these flags on its first import, including via Transformers.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def download_verified_files(repo_id, revision, hashes, *, local_files_only=None):
    """Return snapshot paths only after verifying every requested file."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ModelDownloadError(f"Expected an immutable commit SHA for {repo_id}: {revision!r}")
    for name, expected in hashes.items():
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ModelDownloadError(f"Invalid SHA256 pin for {repo_id}/{name}")

    configure_hub()
    from huggingface_hub import hf_hub_download

    offline = hub_offline() if local_files_only is None else local_files_only
    paths = {}
    for name, expected in hashes.items():
        try:
            path = Path(hf_hub_download(
                repo_id, name, revision=revision, cache_dir=hub_cache_dir(), local_files_only=offline
            ))
        except Exception as exc:
            raise ModelDownloadError(
                f"Could not load {repo_id}@{revision}/{name} from {hub_cache_dir()}. "
                "For first use, enable network access, unset HF_HUB_OFFLINE, and retry. "
                "For offline use, copy the populated Hugging Face hub cache and set HF_HUB_CACHE to it."
            ) from exc
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ModelDownloadError(
                f"SHA256 mismatch for {name} in {repo_id}@{revision}: expected {expected}, got "
                f"{digest.hexdigest()}. Remove the corrupt file from the cache and retry."
            )
        paths[name] = path
    return paths
