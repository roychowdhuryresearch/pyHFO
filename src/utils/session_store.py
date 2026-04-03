import json
import os
import shutil
from pathlib import Path

import numpy as np

from src.utils.utils_io import dump_to_npz


SESSION_FORMAT_VERSION = 1


def _sanitize_key(name):
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(name))


def _encode_value(value, arrays_dir, key_path, array_index):
    if isinstance(value, np.ndarray):
        array_name = f"{array_index:04d}_{_sanitize_key('_'.join(key_path))}.npy"
        array_index += 1
        target = arrays_dir / array_name
        np.save(target, value, allow_pickle=True)
        return {"__array__": array_name}, array_index
    if isinstance(value, np.generic):
        return value.item(), array_index
    if isinstance(value, dict):
        encoded = {}
        for key, item in value.items():
            encoded_item, array_index = _encode_value(item, arrays_dir, key_path + [str(key)], array_index)
            encoded[key] = encoded_item
        return encoded, array_index
    if isinstance(value, (list, tuple)):
        encoded_list = []
        for idx, item in enumerate(value):
            encoded_item, array_index = _encode_value(item, arrays_dir, key_path + [str(idx)], array_index)
            encoded_list.append(encoded_item)
        return encoded_list, array_index
    if isinstance(value, Path):
        return str(value), array_index
    return value, array_index


def _decode_value(value, arrays_dir):
    if isinstance(value, dict) and "__array__" in value:
        return np.load(arrays_dir / value["__array__"], allow_pickle=True)
    if isinstance(value, dict):
        return {key: _decode_value(item, arrays_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_value(item, arrays_dir) for item in value]
    return value


def save_session_checkpoint(path, checkpoint):
    path = Path(path)
    if path.suffix == ".npz":
        dump_to_npz(checkpoint, path)
        return

    arrays_dir = Path(f"{path}.data")
    if path.exists():
        path.unlink()
    if arrays_dir.exists():
        shutil.rmtree(arrays_dir)
    arrays_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format": "pybrain-session",
        "version": SESSION_FORMAT_VERSION,
    }
    encoded, _ = _encode_value(checkpoint, arrays_dir, ["root"], 0)
    manifest["checkpoint"] = encoded

    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_session_checkpoint(path):
    path = Path(path)
    if path.suffix == ".npz":
        return np.load(path, allow_pickle=True)

    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    arrays_dir = Path(f"{path}.data")
    return _decode_value(manifest["checkpoint"], arrays_dir)


def is_session_path(path):
    path = Path(path)
    return path.suffix in {".pybrain", ".npz"}
