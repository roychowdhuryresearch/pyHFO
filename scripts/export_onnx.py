"""Export the PyHFO biomarker classifiers (model_a/e/s) from torch .tar to ONNX.

Path B (small app + HF-hosted models): the shipped app runs inference with
onnxruntime instead of torch, dropping ~440MB of native deps (torch/torchvision/
transformers). This script does the one-time, dev-time conversion. The resulting
`.onnx` + `.spec.json` are what get uploaded to the (public) HF Hub repo.

Why this is safe: the models are plain ResNet18-based image CNNs (input
(N, C, 224, 224), Sigmoid output). Verified that onnxruntime reproduces the torch
output to float32 noise (max|diff| ~1e-7) in a clean torch-free env.

Usage:
    python scripts/export_onnx.py --ckpt-dir /path/to/legacy-ckpt --out ckpt_onnx \
        --check --hub-layout

The legacy torch checkpoints (model_{a,e,s}.tar) are NOT part of this repository;
the published weights ship through the pinned Hugging Face model cards. Obtain the
.tar checkpoints separately from the maintainers and pass their directory explicitly.

The torch/onnx/onnxscript deps are EXPORT-TIME ONLY; the runtime needs only
onnxruntime + numpy + huggingface_hub.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pyhfo.utils.legacy_checkpoint import load_legacy_checkpoint


def _jsonable(obj):
    """Coerce numpy scalars/arrays in the preprocessing dict to plain JSON types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _first_conv_in_channels(model: torch.nn.Module) -> int:
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            return module.in_channels
    return 1


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_one(tag: str, ckpt_dir: str, out_dir: str, image_size: int, check: bool, hub_layout: bool) -> dict:
    blob = load_legacy_checkpoint(Path(ckpt_dir) / f"model_{tag}.tar")
    model = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.eval().float()

    channels = _first_conv_in_channels(model)
    preprocessing = blob.get("preprocessing") if isinstance(blob, dict) else None

    dummy = torch.randn(2, channels, image_size, image_size, dtype=torch.float32)
    with torch.inference_mode():
        torch_out = model(dummy)
        torch_out = torch_out if isinstance(torch_out, torch.Tensor) else torch_out[0]

    onnx_path = os.path.join(out_dir, f"model_{tag}.onnx")
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["prob"],
        # batch AND spatial dims are dynamic: the per-model preprocessing crop
        # produces different image sizes (artifact/spike ~96x96, ehfo 224x224),
        # and ResNet's adaptive pooling handles any size — so don't hardcode H/W.
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"}, "prob": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # legacy exporter; the dynamo path needs onnxscript and adds nothing here
    )

    # Self-describing spec hosted alongside the .onnx (model-independent runtime:
    # the generic onnxruntime runner reads this instead of importing a model class).
    spec = {
        "format": "onnx",
        "opset": 17,
        "input": {
            "name": "input",
            "in_channels": channels,
            "dynamic_axes": ["batch", "height", "width"],
            "dtype": "float32",
            "normalize": "per_channel_minmax_255",
        },
        "output": {"name": "prob", "activation": "sigmoid", "threshold": 0.5},
        "preprocessing": preprocessing,
    }
    spec_path = os.path.join(out_dir, f"model_{tag}.spec.json")
    with open(spec_path, "w") as fh:
        json.dump(spec, fh, indent=2, default=_jsonable)

    result = {
        "tag": tag,
        "class": type(model).__name__,
        "channels": channels,
        "out_shape": tuple(torch_out.shape),
        "onnx_bytes": os.path.getsize(onnx_path),
        "onnx_sha256": _sha256(onnx_path),
        "onnx_file": os.path.basename(onnx_path),
        "spec_file": os.path.basename(spec_path),
    }

    if hub_layout:
        hub_dir = os.path.join(out_dir, f"model_{tag}")
        os.makedirs(hub_dir, exist_ok=True)
        shutil.copy2(onnx_path, os.path.join(hub_dir, "model.onnx"))
        shutil.copy2(spec_path, os.path.join(hub_dir, "model.spec.json"))
        result["hub_dir"] = hub_dir

    if check:
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            onnx_out = session.run(["prob"], {"input": dummy.numpy()})[0]
            result["max_abs_diff"] = float(np.abs(onnx_out - torch_out.detach().numpy()).max())
        except ImportError:
            result["check_skipped"] = "onnxruntime not installed"

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt-dir",
        required=True,
        help="directory holding model_{a,e,s}.tar; the legacy torch checkpoints are NOT in this "
             "repository (weights are fetched from the pinned Hugging Face model cards) and must "
             "be obtained separately from the maintainers",
    )
    parser.add_argument("--out", default="ckpt_onnx", help="output dir for .onnx + .spec.json")
    parser.add_argument("--models", default="a,e,s", help="comma-separated model tags")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--check", action="store_true", help="verify onnxruntime parity vs torch")
    parser.add_argument(
        "--hub-layout",
        action="store_true",
        help="also write out/model_<tag>/model.onnx + model.spec.json for HF repos or local directory sources",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    failed = False
    exported = []
    for tag in (t.strip() for t in args.models.split(",") if t.strip()):
        info = export_one(tag, args.ckpt_dir, args.out, args.image_size, args.check, args.hub_layout)
        exported.append(info)
        diff = info.get("max_abs_diff")
        if diff is not None:
            status = f" max|diff|={diff:.2e} {'OK' if diff < 1e-4 else 'FAIL'}"
        elif info.get("check_skipped"):
            status = f" (check skipped: {info['check_skipped']})"
        else:
            status = ""
        if diff is not None and diff >= 1e-4:
            failed = True
        if args.check and info.get("check_skipped"):
            failed = True
        hub = f" hub_dir={info['hub_dir']}" if info.get("hub_dir") else ""
        print(f"model_{info['tag']}: {info['class']} ch={info['channels']} "
              f"out={info['out_shape']} bytes={info['onnx_bytes']}{status}{hub}")

    manifest = {
        "format": "pyhfo-onnx-bundle",
        "opset": 17,
        "models": [
            {
                key: info.get(key)
                for key in (
                    "tag",
                    "class",
                    "channels",
                    "out_shape",
                    "onnx_bytes",
                    "onnx_sha256",
                    "onnx_file",
                    "spec_file",
                    "max_abs_diff",
                )
                if info.get(key) is not None
            }
            for info in exported
        ],
    }
    with open(os.path.join(args.out, "onnx-manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)

    print("EXPORT FAILED — parity check exceeded tolerance" if failed else "EXPORT OK")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
