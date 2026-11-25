# Checkpoint to ONNX Runbook

This runbook converts PyHFO torch classifier checkpoints into ONNX artifacts that
the torch-free classifier path can load with `onnxruntime`.

## Scope

The converter is for the three HFO classifier checkpoints:

- `model_a.tar` -> artifact classifier
- `model_s.tar` -> spk-HFO classifier
- `model_e.tar` -> eHFO classifier

Each checkpoint must be a torch `.tar` payload containing:

- `model`: the trained PyTorch model
- `preprocessing`: the preprocessing dictionary used to build event features

The runtime ONNX loader requires both files for each model:

- `model_*.onnx`: the exported graph
- `model_*.spec.json`: preprocessing, input contract, output threshold

## Export Environment

Run conversion from the repository root in an environment that has the export
dependencies. Torch is needed only for this conversion step.

```bash
python -m pip install -r requirements.txt
python -m pip install onnx onnxruntime
```

`onnxruntime` is only required for `--check`, but keep it installed so the same
environment can verify the exported graph.

## Convert Checkpoints

The script expects the default checkpoint names in the checkpoint directory:
`model_a.tar`, `model_s.tar`, and `model_e.tar`.

The legacy torch checkpoints are **not** part of this repository: the published
weights ship through the pinned Hugging Face model cards, and no checkpoint directory is
tracked. Obtain the `.tar` checkpoints separately from the maintainers and
pass their directory explicitly with `--ckpt-dir` (the argument is required).

```bash
python scripts/export_onnx.py \
  --ckpt-dir /path/to/legacy-checkpoints \
  --out ckpt_onnx \
  --models a,e,s \
  --check \
  --hub-layout
```

Expected direct outputs:

```text
ckpt_onnx/model_a.onnx
ckpt_onnx/model_a.spec.json
ckpt_onnx/model_s.onnx
ckpt_onnx/model_s.spec.json
ckpt_onnx/model_e.onnx
ckpt_onnx/model_e.spec.json
ckpt_onnx/onnx-manifest.json
```

With `--hub-layout`, the script also writes loader-ready directories:

```text
ckpt_onnx/model_a/model.onnx
ckpt_onnx/model_a/model.spec.json
ckpt_onnx/model_s/model.onnx
ckpt_onnx/model_s/model.spec.json
ckpt_onnx/model_e/model.onnx
ckpt_onnx/model_e/model.spec.json
```

Use the direct files for explicit local `.onnx` paths. Use the loader-ready
directories for local directory sources or one Hugging Face repo per model.

## Validate Export

`--check` runs the exported graph with `onnxruntime` against the torch model on
the export dummy batch. Treat `EXPORT OK` and `max|diff| < 1e-4` as the minimum
acceptance bar. The manifest records each graph's SHA-256 digest, size, input
channels, output shape, and measured parity error.

Run the focused regression tests after conversion-related code changes:

```bash
pytest -q \
  tests/test_onnx_inference.py \
  tests/test_classifier_default_loading.py \
  tests/test_classifier_torch_free_import.py
```

For release signoff, also run one real recording through the torch classifier
and the ONNX classifier, then compare the artifact/spike/eHFO predictions. The
previous reference check used `SM_B_ave.edf` with 583 STE events and produced
zero label mismatches across all three models.

For a repeatable comparison on any saved detection session:

```bash
python scripts/verify_onnx_parity.py \
  --session /path/to/session.pybrain \
  --out /tmp/onnx-parity \
  --biomarker HFO \
  --checkpoint-dir /path/to/legacy-checkpoints \
  --onnx-dir ckpt_onnx
```

The command writes `onnx-parity-report.json` and exits non-zero if any event's
artifact/spike/eHFO labels differ.

## Runtime Model Delivery

The production CPU classifier is Hugging Face first. Each model is pinned to a
full Hub commit and both downloaded files are checked against the SHA-256
values in the runtime registry before ONNX Runtime opens the graph.

| Classifier | Repository | Pinned revision |
| --- | --- | --- |
| artifact | `roychowdhuryresearch/HFO-artifact` | `7f95298b60deff4aae8b1ddfe64acda32a1c1fa7` |
| spike | `roychowdhuryresearch/HFO-spkHFO` | `de371a79569e81921e0174a4e440df2383dfdb8b` |
| eHFO | `roychowdhuryresearch/HFO-eHFO` | `2e59449dd81f588bf1a49bde647cd0e6d1db0a8c` |

The first classification downloads `model.onnx` and `model.spec.json` from the
required repositories. Hugging Face stores them in its versioned cache, and
subsequent classifications reuse those local files. Public downloads do not
require a token.

Set `HF_HOME` before starting PyHFO to choose the cache location. After one
successful download, set `HF_HUB_OFFLINE=1` to require cache-only operation.
An offline first run fails clearly because ONNX Runtime needs a local cached
file even though the distribution itself remains model-free.

## Use Local ONNX Artifacts

Set `PYHFO_ONNX_DIR=/absolute/path/to/ckpt_onnx` for an explicitly managed local
or air-gapped bundle. A checkout-local `ckpt_onnx` directory is not selected
implicitly. When no complete override exists, the preset resolves the pinned
Hugging Face ONNX registry. It does not fall back to a Torch model.

For direct file paths, pass the exported `.onnx` files. The runtime finds the
same-stem `.spec.json` sidecar automatically.

```python
from pyhfo.param.param_classifier import ParamClassifier

param = ParamClassifier(
    artifact_onnx="ckpt_onnx/model_a.onnx",
    spike_onnx="ckpt_onnx/model_s.onnx",
    ehfo_onnx="ckpt_onnx/model_e.onnx",
    use_spike=True,
    use_ehfo=True,
    device="cpu",
    batch_size=32,
    model_type="default_cpu",
    source_preference="local",
)
```

For loader-ready directories, pass the directory path:

```python
param = ParamClassifier(
    artifact_onnx="ckpt_onnx/model_a",
    spike_onnx="ckpt_onnx/model_s",
    ehfo_onnx="ckpt_onnx/model_e",
)
```

## Publish to Hugging Face

Create one public model repo per classifier and upload the two files from the
matching `--hub-layout` directory:

```text
model.onnx
model.spec.json
```

The production repositories are the three entries in the runtime table above.
For a custom repository, configure a loader-ready repo containing both files:

```python
param = ParamClassifier(
    artifact_onnx="org/hfo-artifact-onnx",
    spike_onnx="org/hfo-spkhfo-onnx",
    ehfo_onnx="org/hfo-ehfo-onnx",
)
```

Default production sources use a pinned `hf://` URI that carries the repo,
full revision, filenames, and SHA-256 digests. Explicit custom repo ids remain
supported, but callers are responsible for their reproducibility.

## Runtime Dependencies

The ONNX inference path needs:

- `numpy`
- `onnxruntime`
- `huggingface_hub` only when loading from Hugging Face

Torch, torchvision, and transformers are export-time dependencies only for the
production ONNX path. A runtime build that uses `artifact_onnx`, `spike_onnx`,
and `ehfo_onnx` does not import `pyhfo.model` or `pyhfo.dl_models`.

## Electron Packaging

The Electron packaging flow lives in the
[roychowdhuryresearch/ieeg-agent](https://github.com/roychowdhuryresearch/ieeg-agent)
monorepo, not in this repository. There, `npm run package:mac-smoke` freezes the torch-free
backend and assembles
the application without `Contents/Resources/onnx`. The packaged Electron main
process points `HF_HOME` at a persistent writable directory under Electron
`userData/models/huggingface`. Package smoke tests reject an embedded ONNX
bundle and validate backend readiness without forcing a network download. The
separate `npm run qa:hf-onnx-runtime` release gate uses a clean cache to load
all three models online, then repeats the same frozen-backend self-test with
`HF_HUB_OFFLINE=1`.

## Common Failures

- Missing `model.spec.json`: the ONNX graph may load, but classifier setup fails
  because preprocessing cannot be reconstructed.
- Wrong HF layout: the runtime looks for exactly `model.onnx` and
  `model.spec.json` in each repo.
- Offline cache miss: allow one online classification or prefetch the pinned
  files before enabling `HF_HUB_OFFLINE=1`.
- SHA-256 mismatch: remove the corrupt cache entry and download the pinned
  revision again; the runtime will not load unverified bytes.
- Passing an ONNX source with `model_type="default_cpu"` is supported; explicit
  `*_onnx` sources take precedence over the default torch/HF presets.
- An eHFO preprocessing crop outside the model spectrogram range is a bad spec
  or checkpoint/spec mismatch, not an ONNX runtime problem.
