"""Upload a cleared severity model bundle and print pins for torch_models.py."""

import argparse
import hashlib
import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="Directory containing the real config.json and model.safetensors.")
    parser.add_argument("--repo", default=os.getenv("PYHFO_SEVERITY_REPO") or "roychowdhuryresearch/EEG-severity",
                        help="Existing Hugging Face model repository. Authenticate with hf auth login or HF_TOKEN.")
    args = parser.parse_args()
    operations = []
    hashes = {}
    for name in ("config.json", "model.safetensors"):
        path = args.model_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            parser.error(f"Missing or empty file: {path}")
        with path.open("rb") as handle:
            if handle.readline(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
                parser.error(f"{path} is a Git LFS pointer. Supply the actual file.")
            handle.seek(0)
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
        operations.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=path))

    commit = HfApi().create_commit(repo_id=args.repo, repo_type="model", operations=operations,
                                   commit_message="Publish PyHFO severity model weights and configuration")
    print(f'SEVERITY_REVISION = "{commit.oid}"')
    print(f'SEVERITY_CONFIG_SHA256 = "{hashes["config.json"]}"')
    print(f'SEVERITY_MODEL_SHA256 = "{hashes["model.safetensors"]}"')


if __name__ == "__main__":
    main()
