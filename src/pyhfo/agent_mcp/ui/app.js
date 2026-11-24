(function () {
  const state = {
    lastResult: null,
  };

  function text(id, value) {
    const element = document.getElementById(id);
    if (element) {
      element.textContent = value == null || value === "" ? "-" : String(value);
    }
  }

  function formatPath(value) {
    if (!value) {
      return "-";
    }
    return String(value);
  }

  function renderArtifacts(artifacts) {
    const list = document.getElementById("artifact-list");
    if (!list) {
      return;
    }
    list.innerHTML = "";
    const entries = Object.entries(artifacts || {});
    text("artifacts-value", entries.length);
    if (!entries.length) {
      const item = document.createElement("li");
      item.textContent = "No artifacts yet.";
      list.appendChild(item);
      return;
    }
    for (const [name, path] of entries) {
      const item = document.createElement("li");
      const label = document.createElement("span");
      label.className = "artifact-name";
      label.textContent = name.replace(/_/g, " ");
      const value = document.createElement("span");
      value.className = "artifact-path";
      value.textContent = formatPath(path);
      item.append(label, value);
      list.appendChild(item);
    }
  }

  function renderBatch(summary) {
    const grid = document.getElementById("batch-grid");
    if (!grid) {
      return;
    }
    grid.innerHTML = "";
    const entries = Object.entries(summary || {});
    if (!entries.length) {
      const empty = document.createElement("p");
      empty.textContent = "No batch status.";
      grid.appendChild(empty);
      return;
    }
    for (const [name, value] of entries) {
      const metric = document.createElement("div");
      metric.className = "metric";
      const label = document.createElement("span");
      label.className = "metric-label";
      label.textContent = name;
      const number = document.createElement("strong");
      number.textContent = value;
      metric.append(label, number);
      grid.appendChild(metric);
    }
  }

  function renderError(error) {
    const panel = document.getElementById("error-panel");
    const value = document.getElementById("error-value");
    if (!panel || !value) {
      return;
    }
    if (!error) {
      panel.hidden = true;
      value.textContent = "";
      return;
    }
    panel.hidden = false;
    value.textContent = JSON.stringify(error, null, 2);
  }

  function render(result) {
    state.lastResult = result || {};
    const data = state.lastResult.structuredContent || state.lastResult.data || state.lastResult;
    const error = data.error || (data.status === "failed" ? data.error : null);

    text("status-value", data.status || "Complete");
    text("events-value", data.num_events || 0);
    text("channels-value", data.num_channels || 0);
    text("subtitle", data.output_dir || data.manifest_path || "PyHFO agent result");
    text("input-value", data.input_path || "-");
    text("biomarker-value", data.biomarker || "-");
    text("detector-value", data.detector || "-");
    text("run-value", data.active_run_id || "-");

    renderArtifacts(data.artifacts);
    renderBatch(data.summary);
    renderError(error);
  }

  async function callServerTool(name, args) {
    if (window.mcpApp && typeof window.mcpApp.callServerTool === "function") {
      return window.mcpApp.callServerTool({ name, arguments: args });
    }
    if (window.app && typeof window.app.callServerTool === "function") {
      return window.app.callServerTool({ name, arguments: args });
    }
    window.parent.postMessage({
      jsonrpc: "2.0",
      method: "tools/call",
      params: { name, arguments: args },
    }, "*");
    return null;
  }

  async function refreshBatch() {
    const current = state.lastResult || {};
    const data = current.structuredContent || current.data || current;
    if (!data.manifest_path) {
      return;
    }
    const response = await callServerTool("pyhfo_run_batch", {
      manifest_path: data.manifest_path,
      detection: {},
    });
    if (response) {
      render(response);
    }
  }

  window.addEventListener("message", function (event) {
    const message = event.data || {};
    if (message.method === "tool/result" || message.method === "ui/tool_result") {
      render(message.params || message.result || {});
    }
  });

  document.getElementById("refresh-button").addEventListener("click", refreshBatch);

  if (window.__PYHFO_RESULT__) {
    render(window.__PYHFO_RESULT__);
  } else {
    render({});
  }

  window.pyhfoResultsApp = {
    render,
    callServerTool,
  };
}());
