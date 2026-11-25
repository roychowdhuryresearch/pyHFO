"""Frozen entry point for the Electron PyHFO analysis backend."""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "src"
if not getattr(sys, "frozen", False) and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _restore_persistent_matplotlib_cache() -> None:
    configured = (os.getenv("PYHFO_MPLCONFIGDIR") or "").strip()
    if configured:
        os.environ["MPLCONFIGDIR"] = configured


def _install_parent_death_guard() -> None:
    expected = os.environ.get("IEEG_PARENT_PID")
    if not expected:
        return
    try:
        expected_pid = int(expected)
    except (TypeError, ValueError):
        return

    def watch() -> None:
        while True:
            time.sleep(float(os.environ.get("IEEG_PARENT_GUARD_INTERVAL", "2") or "2"))
            try:
                if os.getppid() != expected_pid or os.getppid() == 1:
                    os._exit(0)
            except Exception:
                os._exit(0)

    threading.Thread(target=watch, name="parent-death-guard", daemon=True).start()


def _run_yasa_detector_smoke() -> dict[str, object]:
    import numpy as np

    from pyhfo.utils.utils_detector import get_yasa_import_error, has_yasa, yasa

    if not has_yasa() or yasa is None:
        raise RuntimeError(f"YASA detector import failed: {get_yasa_import_error()}")
    signal = np.random.default_rng(7).normal(0.0, 8.0, 3_000)
    result = yasa.spindles_detect(signal, sf=100.0, ch_names=["C3"], verbose=False)
    events = 0 if result is None else len(result.summary())
    return {"status": "passed", "algorithm": "spindles_detect", "events": int(events)}


def _self_test(download_models: bool = False) -> int:
    from pyhfo.agent_api.sidecar import get_capability_manifest
    from pyhfo.classifer import Classifier
    from pyhfo.utils.default_onnx_models import (
        DEFAULT_HF_ONNX_MODELS,
        DEFAULT_HF_ONNX_MODEL_TAGS,
        hf_onnx_uri,
        parse_hf_onnx_uri,
    )

    missing = sorted(set(DEFAULT_HF_ONNX_MODEL_TAGS).difference(DEFAULT_HF_ONNX_MODELS))
    if missing:
        raise RuntimeError(f"Default ONNX registry is incomplete: {', '.join(missing)}")
    registry = {
        tag: {
            "repo_id": model.repo_id,
            "revision": model.revision,
            "uri": hf_onnx_uri(model),
        }
        for tag, model in DEFAULT_HF_ONNX_MODELS.items()
    }
    for model in DEFAULT_HF_ONNX_MODELS.values():
        parse_hf_onnx_uri(hf_onnx_uri(model))
    yasa_detection = _run_yasa_detector_smoke()
    loaded = None
    if download_models:
        from pyhfo.hfo_app import HFO_App

        app = HFO_App()
        app.set_default_onnx_classifier()
        if app.classifier is None:
            raise RuntimeError("Default ONNX classifier was not initialized")
        loaded = {
            "artifact": app.classifier.model_a is not None,
            "spike": app.classifier.model_s is not None,
            "ehfo": app.classifier.model_e is not None,
        }
        if not all(loaded.values()):
            raise RuntimeError(f"Default ONNX classifier did not load completely: {loaded}")
    print(
        json.dumps(
            {
                "status": "passed",
                "service": "pyhfo-backend",
                "frozen": bool(getattr(sys, "frozen", False)),
                "classifier_import": Classifier.__name__,
                "onnx_registry": registry,
                "onnx_models": loaded,
                "yasa_detection": yasa_detection,
                "capabilities": get_capability_manifest(),
            }
        ),
        flush=True,
    )
    return 0


# Marker that makes this executable run one pipeline job and exit, instead of serving. A frozen app
# has no interpreter to hand `-m pyhfo.agent_api.pipeline_worker`, so the way to keep pipelines in a
# separate process is for the app to be able to BE that process.
PIPELINE_WORKER_FLAG = "--pipeline-worker"


def main() -> int:
    _restore_persistent_matplotlib_cache()
    _install_parent_death_guard()
    if PIPELINE_WORKER_FLAG in sys.argv[1:]:
        # Dispatch before anything else: this process is a short-lived worker, not a server.
        from pyhfo.agent_api.pipeline_worker import main as worker_main

        index = sys.argv.index(PIPELINE_WORKER_FLAG)
        return worker_main(sys.argv[index + 1:])
    if "--self-test-download-models" in sys.argv[1:]:
        return _self_test(download_models=True)
    if "--self-test" in sys.argv[1:]:
        return _self_test()
    from pyhfo.agent_api.server import main as server_main

    return server_main()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
