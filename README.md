# PyHFO

PyHFO is the UCLA Roychowdhury group's headless EEG analysis engine. It reads EDF, BrainVision and FIF recordings, detects high-frequency oscillations (HFOs), spikes and spindles, and exports event tables and review sessions. Detector outputs are candidates for review.

To review those candidates with an AI agent, use [BrainLens](https://github.com/zyp5511/BrainLens), AI co-review for EEG built on this engine. The clinician and any MCP agent (Claude Code, Codex, Cursor, Antigravity) read the same recording together, and BrainLens runs PyHFO for detection and classification.

## Install

Use Python 3.10–3.13 in a virtual environment.

```sh
python -m pip install pyhfo
pyhfo --version
```

The base install includes classical HFO detection and CPU ONNX classification. Install the extras for the analyses you use.

| Extra | Install command | Enables |
| --- | --- | --- |
| spindle | `python -m pip install 'pyhfo[spindle]'` | YASA spindle detection. Bundled Kramer LSM presets and threshold detectors work with the base install. |
| severity | `python -m pip install 'pyhfo[severity]'` | Segment-level severity scoring with compatible local weights |
| gpu | `python -m pip install 'pyhfo[gpu]'` | PyTorch classifier presets and loading legacy checkpoints |
| dev | `python -m pip install 'pyhfo[dev]'` | Testing, EDF fixture generation, ONNX graph tooling and distribution builds |

## Analyze a recording

Inspect the sample rate and channels, then run detection. This example uses high-rate data suitable for HFO analysis.

```sh
pyhfo info --input recording.edf
pyhfo detect --input recording.edf --output results --biomarker HFO \
  --detector RMS --n-jobs 1 --export csv --export summary-json
```

The CLI writes a JSON result to stdout. The result includes the candidate count, recording fingerprint and paths to the artifacts. Use `pyhfo parameters recommend --help` to plan a detector configuration for another recording.

To classify the HFO candidates during detection, enable the classifier. PyHFO selects
the `default_cpu` preset and loads the pinned ONNX models on first use. The JSON
result records the selected preset in `classifier_metadata.model_preset` and the
loaded model type in `classifier_metadata.model_type`.

```sh
pyhfo detect --input recording.edf --output classified-results --biomarker HFO \
  --detector RMS --classifier-param enabled=true --export csv --export summary-json
```

`quick-detect` accepts the same flag. In MCP and HTTP detection requests,
pass `"classifier": {"enabled": true}` in the JSON payload.

## Connect an agent or BrainLens

[BrainLens](https://github.com/zyp5511/BrainLens) installs this distribution in its Python environment and calls the public API directly. Other agents can use the same API or the MCP server below.

```python
from pyhfo import __version__
from pyhfo.agent_api import get_recording_info, run_detection

info = get_recording_info("recording.edf")
result = run_detection({
    "input_path": "recording.edf",
    "output_dir": "results",
    "biomarker": "HFO",
    "detector": "RMS",
    "n_jobs": 1,
    "exports": ["csv", "summary-json"],
})
```

An MCP client can start `pyhfo-mcp` from the same environment. Put the absolute path to that executable in the client's configuration if the virtual environment is outside its search path.

```json
{"mcpServers": {"pyhfo": {"command": "pyhfo-mcp", "args": ["--transport", "stdio"]}}}
```

For HTTP clients, start the loopback server and inspect its action manifest.

```sh
pyhfo-server --host 127.0.0.1 --port 8765
# In a second terminal
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/manifest
```

## Desktop app and agent runtime

The Electron desktop app and the agent runtime built on this engine live in
[roychowdhuryresearch/ieeg-agent](https://github.com/roychowdhuryresearch/ieeg-agent).
[BrainLens](https://github.com/zyp5511/BrainLens) (AI co-review for EEG) is the clinical review workbench built on this engine.

## State and models

PyHFO keeps its sessions, projects, implicit exports and model cache under `$PYHFO_HOME` (default `~/.pyhfo`). Explicit output paths take precedence. To continue using an existing session directory, set `PYHFO_AGENT_SESSION_DIR` to its absolute path. Projects and exports also accept `PYHFO_AGENT_PROJECT_DIR` and `PYHFO_AGENT_EXPORT_DIR`.

Detection runs without model downloads. On first classification the CPU preset fetches the artifact, spkHFO and eHFO ONNX models from pinned Hugging Face revisions and checks the SHA-256 digest of each model and its preprocessing specification. The GPU preset uses the PyTorch weights on the corresponding cards. See the [artifact](https://huggingface.co/roychowdhuryresearch/HFO-artifact), [spkHFO](https://huggingface.co/roychowdhuryresearch/HFO-spkHFO) and [eHFO](https://huggingface.co/roychowdhuryresearch/HFO-eHFO) model cards.

For offline classification, run classification once while connected using the same cache, then set `HF_HUB_OFFLINE=1`. Set `HF_HUB_CACHE` to reuse an existing Hugging Face hub cache, or `HF_HOME` to change its parent directory. Alternatively, point `PYHFO_ONNX_DIR` to a complete local bundle containing `model_a/`, `model_s/` and `model_e/`, each with `model.onnx` and `model.spec.json`.

The severity registry reserves `roychowdhuryresearch/EEG-severity`. Until the owners publish its revision and digests, supply a local directory containing `config.json` and real `model.safetensors` weights.

```sh
pyhfo severity --input scalp.edf --model-dir /path/to/severity-model
```

`PYHFO_SEVERITY_MODEL_DIR` sets the same local override. `PYHFO_SEVERITY_REPO` selects a registry mirror containing the pinned release. The scorer uses a 19-channel TUEG montage and 15-second segments. See the [draft model card](https://github.com/roychowdhuryresearch/pyHFO/blob/main/MODEL_CARD_severity.md).

## Citation

Zhang et al. introduced PyHFO in the *Journal of Neural Engineering* (2024). Ding et al. describe the deep learning classification workflow in PyHFO 2.0 (2025). Include the software acknowledgment in paragraph 4 of the license in publications using PyHFO. [PyHFO](https://doi.org/10.1088/1741-2552/ad4916) · [PyHFO 2.0](https://doi.org/10.1088/1741-2552/ae10e0)

## License

The UCLA Academic Software License governs educational and academic research use. The license includes the required acknowledgment and commercial licensing contacts. [LICENSE.txt](https://github.com/roychowdhuryresearch/pyHFO/blob/main/LICENSE.txt)
