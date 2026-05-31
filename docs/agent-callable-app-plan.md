# PyHFO Agent-Callable App Plan

## Objective

Turn PyHFO into an app that agents can call reliably without driving the PyQt UI. The work should land in four phases:

1. Headless Python API.
2. Agent-friendly CLI.
3. MCP server.
4. MCP App UI.

The desktop GUI remains supported. The agent path should reuse the existing backend and avoid duplicating detection algorithms.

## Current Codebase Facts

- GUI startup is in `main.py:20` to `main.py:27`; it creates `QApplication` and `MainWindow`.
- HFO already has a reusable backend. `HFO_App.load_raw()` is in `src/hfo_app.py:108` to `src/hfo_app.py:118`, and `HFO_App.load_edf()` is in `src/hfo_app.py:122` to `src/hfo_app.py:124`.
- HFO filtering and detection are backend methods: `set_filter_parameter()` / `filter_eeg_data()` in `src/hfo_app.py:312` to `src/hfo_app.py:345`, and `set_detector()` / `detect_biomarker()` in `src/hfo_app.py:391` to `src/hfo_app.py:433`.
- Export hooks already exist in `src/hfo_app.py:655` to `src/hfo_app.py:664`.
- Batch discovery and manifest creation already exist in `src/utils/batch_project.py:25` to `src/utils/batch_project.py:30` and `src/utils/batch_project.py:195` to `src/utils/batch_project.py:217`.
- Existing opt-in backend smoke tests already run a real STE detection and verify Excel/session exports in `tests/test_backend_smoke.py`.

## Design Principles

- Keep the GUI path and agent path separate at the boundary; share backend logic underneath.
- Make every agent-facing input/output JSON-serializable.
- Put large artifacts on disk and return paths/metadata, not raw EEG arrays.
- Use structured errors with stable error codes.
- Keep classifier downloads and GPU usage explicit.
- Add MCP/App dependencies as optional agent dependencies unless the project explicitly decides they are core runtime dependencies.

## Phase 1: Headless Python API

### Goal

Expose PyHFO's core workflows as GUI-free Python functions:

- inspect recording metadata
- run detection
- optionally run classification
- export Excel/session/summary JSON
- create and run resumable batch projects

### Planned Files

- `src/agent_api/__init__.py`
- `src/agent_api/schemas.py`
- `src/agent_api/pipeline.py`
- `src/agent_api/serialization.py`
- `src/agent_api/errors.py`
- `tests/test_agent_api_schemas.py`
- `tests/test_agent_api_recording_info.py`
- `tests/test_agent_api_detection.py`
- `tests/test_agent_api_batch.py`

### Public API Shape

```python
from src.agent_api import get_recording_info, run_detection, run_batch

info = get_recording_info("sample.edf")

result = run_detection({
    "input_path": "sample.edf",
    "output_dir": "out/sample",
    "biomarker": "HFO",
    "detector": "STE",
    "filter": {"fp": 80, "fs": 500},
    "n_jobs": 4,
    "exports": ["excel", "session", "summary_json"],
})
```

### Core Schemas

`RecordingInfo`:

- `input_path`
- `format`
- `sample_freq`
- `duration_seconds`
- `channel_names`
- `recording_channel_names`
- `line_freq`

`DetectionRequest`:

- `input_path`
- `output_dir`
- `biomarker`
- `detector`
- `filter`
- `detector_params`
- `classifier`
- `n_jobs`
- `exports`

`DetectionResult`:

- `status`
- `biomarker`
- `detector`
- `num_events`
- `num_channels`
- `active_run_id`
- `artifacts`
- `warnings`
- `timing`
- `error`

### Implementation Steps

1. Build schema/dataclass layer with dict conversion and validation.
2. Add backend factory: `HFO -> HFO_App`, `Spindle -> SpindleApp`, `Spike -> SpikeApp`.
3. Add detector/filter/classifier parameter builders from dict input to existing `Param*` classes.
4. Implement `get_recording_info()` using existing MNE loading helpers.
5. Implement `run_detection()` as a fixed pipeline: load -> configure filter -> filter -> configure detector -> detect -> optional classify -> export -> summary.
6. Implement `run_batch()` using existing `BatchProject` status transitions.
7. Add structured error handling; failed runs return error metadata and never masquerade as success.
8. Add docs/examples after tests prove the API.

### Acceptance Criteria

- `import src.agent_api` does not create or require `QApplication`.
- HFO + STE can run end to end through `run_detection()`.
- Result objects can be serialized with `json.dumps()`.
- Artifacts are written inside the requested output directory and returned as absolute paths.
- Invalid input returns stable structured errors.
- Existing GUI tests still pass.

### Testing

Unit tests:

```bash
pytest tests/test_agent_api_schemas.py
pytest tests/test_agent_api_recording_info.py
```

Integration tests:

```bash
pytest tests/test_agent_api_detection.py
pytest tests/test_agent_api_batch.py
```

Regression tests:

```bash
pytest tests/test_batch_project.py tests/test_threshold_detectors.py tests/test_spike_detector.py
```

Real-data smoke:

```bash
PYHFO_RUN_SMOKE=1 pytest tests/test_backend_smoke.py
```

## Phase 2: Agent-Friendly CLI

### Goal

Let agents call PyHFO with shell commands and receive structured JSON.

### Planned Files

- `src/agent_cli.py`
- `tests/test_agent_cli.py`
- README section for agent/CLI usage
- optional console entry point in packaging config

### CLI Commands

```bash
python -m src.agent_cli info --input sample.edf --json
python -m src.agent_cli detect --input sample.edf --output out/sample --biomarker HFO --detector STE --export excel --export summary-json
python -m src.agent_cli batch-create --input recordings --output out/batch
python -m src.agent_cli batch-run --manifest out/batch/batch_project.json --biomarker HFO --detector STE
```

### CLI Rules

- JSON results go to stdout.
- logs/progress/warnings go to stderr.
- success exits `0`.
- argument/schema errors exit `2`.
- runtime detection errors exit `1`.
- batch failures are summarized per case; `--fail-fast` controls process-level failure behavior.

### Implementation Steps

1. Build argparse command tree.
2. Map CLI args to Phase 1 schemas.
3. Return Phase 1 results as JSON.
4. Add deterministic exit code mapping.
5. Add examples to README.
6. Add optional packaging entry point only after `python -m src.agent_cli` works.

### Acceptance Criteria

- Agent can run a single command and get parseable JSON.
- CLI does not import `src.ui` and does not create `QApplication`.
- CLI and Phase 1 API use the same schema.
- Output artifacts and errors are visible in JSON.

### Testing

Unit/subprocess tests:

```bash
pytest tests/test_agent_cli.py
```

Specific coverage:

- `python -m src.agent_cli --help` exits `0`.
- `info --json` returns valid JSON.
- `detect` maps args to `run_detection()`; monkeypatch for fast tests.
- invalid detector exits `2`.
- simulated runtime failure exits `1`.
- `batch-create` writes a manifest.

Manual smoke:

```bash
python -m src.agent_cli info --input SM_B_ave.edf --json
python -m src.agent_cli detect --input SM_B_ave.edf --output artifacts/agent-smoke --biomarker HFO --detector STE --export excel --export summary-json
```

## Phase 3: MCP Server

### Goal

Expose PyHFO as standard MCP tools so MCP-compatible agent hosts can call it directly.

Official implementation references to verify before coding:

- `https://modelcontextprotocol.io/docs/learn/server-concepts`
- `https://modelcontextprotocol.io/specification/2025-06-18/basic/transports`
- `https://modelcontextprotocol.io/docs/sdk`

### Planned Files

- `src/agent_mcp/__init__.py`
- `src/agent_mcp/server.py`
- `src/agent_mcp/tools.py`
- `src/agent_mcp/resources.py`
- `tests/test_agent_mcp_tools.py`
- `tests/test_agent_mcp_protocol.py`
- optional `requirements-agent.txt` or `requirements-mcp.txt`

### Tools

- `pyhfo_get_recording_info`
  - input: `input_path`
  - output: `RecordingInfo`
- `pyhfo_detect_events`
  - input: `DetectionRequest`
  - output: `DetectionResult`
- `pyhfo_create_batch_project`
  - input: `input_dir`, `output_dir`, `recursive`
  - output: manifest path and case summary
- `pyhfo_run_batch`
  - input: manifest path, detection config, retry flags
  - output: batch summary and per-case artifact paths
- `pyhfo_export_session_summary`
  - input: session path or detection result path
  - output: summary JSON/resource URI

### Resources

- `pyhfo://runs/{run_id}/summary`
- `pyhfo://runs/{run_id}/events`
- `pyhfo://batches/{batch_id}/manifest`

### Implementation Steps

1. Add MCP server startup with stdio transport first.
2. Register tools whose schemas mirror Phase 1 schemas.
3. Route tool calls to Phase 1 API.
4. Map Phase 1 structured errors into MCP tool errors.
5. Add resource readers for summary/events/manifest artifacts.
6. Add docs with local MCP client configuration examples.

### Acceptance Criteria

- `python -m src.agent_mcp.server --transport stdio` starts.
- `tools/list` includes all PyHFO tools.
- `pyhfo_get_recording_info` returns metadata.
- `pyhfo_detect_events` creates the same artifacts as the CLI path.
- bad inputs return tool errors, not server crashes.
- MCP tests run without GUI.

### Testing

Unit tests:

```bash
pytest tests/test_agent_mcp_tools.py
```

Protocol tests:

```bash
pytest tests/test_agent_mcp_protocol.py
```

Combined regression:

```bash
pytest tests/test_agent_mcp_tools.py tests/test_agent_mcp_protocol.py tests/test_agent_api_detection.py tests/test_agent_cli.py
```

Manual smoke:

```bash
python -m src.agent_mcp.server --transport stdio
```

Then use an MCP inspector/client to:

- list tools
- call `pyhfo_get_recording_info`
- call `pyhfo_detect_events`

## Phase 4: MCP App UI

### Goal

Add lightweight MCP App UI resources for hosts that support interactive components. This is not a PyQt replacement; it is an agent-host panel for results and batch progress.

### Planned Files

- `src/agent_mcp/app_resources.py`
- `src/agent_mcp/ui/index.html`
- `src/agent_mcp/ui/app.js`
- `src/agent_mcp/ui/styles.css`
- `tests/test_agent_mcp_app_resources.py`

### UI Surfaces

Detection result panel:

- input file
- detector/filter parameters
- total event count
- top channels
- artifact links/paths
- error/warning state

Batch panel:

- total/pending/running/complete/failed counts
- failed case snippets
- retry/resume action

Export panel:

- Excel/session/report paths
- summary JSON preview
- follow-up export actions

### Implementation Steps

1. Register UI resources in the MCP server.
2. Build static HTML/CSS/JS panel with no external script dependency by default.
3. Wire UI actions through MCP tools only.
4. Add component metadata, permissions, and CSP.
5. Keep JSON-only MCP tools usable when the host does not support Apps.

### Acceptance Criteria

- Detection results can render in the MCP App panel.
- Batch status can render and refresh.
- UI buttons call MCP tools with inspectable arguments.
- UI payloads do not include raw EEG arrays.
- Phase 3 JSON tools remain usable without UI support.

### Testing

Resource tests:

```bash
pytest tests/test_agent_mcp_app_resources.py
```

Coverage:

- resources are registered
- MIME types are correct
- CSP/permissions are present
- sample detection result payload renders required fields
- large EEG data is not embedded in component payload

Optional browser tests if a browser runner is added:

```bash
pytest tests/test_agent_mcp_app_resources.py
```

with Playwright-backed checks for:

- result panel
- batch panel
- empty/error states
- mock `callServerTool` actions

Manual smoke:

1. Start PyHFO MCP server.
2. Connect from a host that supports MCP Apps.
3. Call `pyhfo_detect_events`.
4. Verify result panel shows event counts, artifacts, warnings, and errors.

## Overall Verification Matrix

Agent path tests:

```bash
pytest tests/test_agent_api_schemas.py \
       tests/test_agent_api_recording_info.py \
       tests/test_agent_api_detection.py \
       tests/test_agent_api_batch.py \
       tests/test_agent_cli.py \
       tests/test_agent_mcp_tools.py \
       tests/test_agent_mcp_protocol.py \
       tests/test_agent_mcp_app_resources.py
```

Backend regression:

```bash
pytest tests/test_batch_project.py \
       tests/test_analysis_session.py \
       tests/test_threshold_detectors.py \
       tests/test_spike_detector.py \
       tests/test_utils_io.py
```

GUI regression:

```bash
pytest tests/test_app_smoke.py \
       tests/test_main_window_toolbar.py \
       tests/test_subwindow_workflows.py \
       tests/test_ui_layout_regression.py
```

Real-data smoke:

```bash
PYHFO_RUN_SMOKE=1 pytest tests/test_backend_smoke.py
```

## Milestones

1. Phase 1 PR: headless API and tests only.
2. Phase 2 PR: CLI and README examples.
3. Phase 3 PR: MCP server, protocol tests, optional dependency file.
4. Phase 4 PR: MCP App UI resources and UI/resource tests.

## Risks

- Existing backend state may have hidden GUI-era ordering assumptions.
  - Mitigation: Phase 1 pipeline fixes the order and tests load/filter/detect/export.
- Real EEG smoke tests are slow.
  - Mitigation: daily tests use synthetic/monkeypatch paths; real data stays opt-in.
- MCP/App SDKs can change.
  - Mitigation: isolate MCP code under `src/agent_mcp/` and verify official docs before implementation.
- Classifier models can trigger downloads or hardware differences.
  - Mitigation: classifier is opt-in and records model metadata.
- Batch outputs can be large.
  - Mitigation: return artifact paths and summaries, not raw data.

## Definition of Done

The full 1-to-4 goal is complete when:

- `src.agent_api` can run without GUI and has tests.
- `python -m src.agent_cli detect ...` produces JSON and artifacts.
- `python -m src.agent_mcp.server --transport stdio` works with an MCP client.
- MCP App resources render detection and batch results in a supporting host.
- JSON-only MCP tools remain usable without UI support.
- Agent tests, backend regressions, GUI regressions, and opt-in real-data smoke results are recorded before release.
