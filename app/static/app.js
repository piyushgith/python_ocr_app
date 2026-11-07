(() => {
  const JOBS_URL = "/api/v1/ocr/jobs";
  const POLL_INTERVAL_MS = 1000;

  const $ = (id) => document.getElementById(id);
  const el = {
    dropzone: $("dropzone"), fileInput: $("file-input"), fileInfo: $("file-info"),
    infoName: $("info-name"), infoSize: $("info-size"), infoType: $("info-type"),
    engine: $("engine"), mode: $("mode"), outputFormat: $("output-format"),
    extract: $("extract"), reset: $("reset"),
    status: $("status"), statusText: $("status-text"), progress: $("progress"),
    error: $("error"), errorCode: $("error-code"), errorMessage: $("error-message"),
    results: $("results"), warnings: $("warnings"), summary: $("result-summary"),
    pager: $("pager"), prev: $("prev"), next: $("next"), pageLabel: $("page-label"),
    textView: $("text-view"), jsonView: $("json-view"), pagesView: $("pages-view"),
    downloadTxt: $("download-txt"), downloadJson: $("download-json"),
  };

  const state = { file: null, result: null, jobId: null, pageIndex: 0, tab: "text", running: false };

  function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
  }

  function safeBaseName(filename) {
    const base = filename.replace(/\.[^.]*$/, "");
    const cleaned = base.replace(/[^\p{L}\p{N}._-]+/gu, "_").replace(/^[._]+|_+$/g, "").slice(0, 80);
    return cleaned || "document";
  }

  function selectedLanguages() {
    return [...document.querySelectorAll('input[name="language"]:checked')].map((box) => box.value);
  }

  function setFile(file) {
    clearResults();
    state.file = file;
    if (!file) {
      el.fileInfo.hidden = true;
      el.extract.disabled = true;
      return;
    }
    el.infoName.textContent = file.name;
    el.infoSize.textContent = formatSize(file.size);
    el.infoType.textContent = `${file.type || "unknown"} (reported by browser; verified by the server on upload)`;
    el.fileInfo.hidden = false;
    el.extract.disabled = false;
  }

  function clearResults() {
    state.result = null;
    state.jobId = null;
    state.pageIndex = 0;
    el.results.hidden = true;
    el.error.hidden = true;
    el.status.hidden = true;
  }

  function showError(code, message) {
    el.status.hidden = true;
    el.results.hidden = true;
    el.errorCode.textContent = code;
    el.errorMessage.textContent = message;
    el.error.hidden = false;
  }

  function setStatus(text, progress) {
    el.error.hidden = true;
    el.status.hidden = false;
    el.statusText.textContent = text;
    el.progress.hidden = progress == null;
    if (progress != null) el.progress.value = progress;
  }

  async function readError(response) {
    try {
      const body = await response.json();
      return body.error || { code: `HTTP_${response.status}`, message: response.statusText };
    } catch {
      return { code: `HTTP_${response.status}`, message: response.statusText || "Request failed" };
    }
  }

  async function extract() {
    if (!state.file || state.running) return;
    const languages = selectedLanguages();
    clearResults();
    state.running = true;
    el.extract.disabled = true;

    const form = new FormData();
    form.append("file", state.file);
    form.append("engine", el.engine.value);
    form.append("mode", el.mode.value);
    languages.forEach((language) => form.append("languages", language));

    try {
      setStatus("Uploading…");
      const response = await fetch(JOBS_URL, { method: "POST", body: form });
      if (!response.ok) {
        const { code, message } = await readError(response);
        showError(code, message);
        return;
      }
      const { job_id: jobId } = await response.json();
      state.jobId = jobId;
      await poll(jobId);
    } catch (error) {
      showError("NETWORK_ERROR", "Could not reach the server. Is it running?");
    } finally {
      state.running = false;
      el.extract.disabled = !state.file;
    }
  }

  async function poll(jobId) {
    for (;;) {
      const response = await fetch(`${JOBS_URL}/${jobId}`);
      if (!response.ok) {
        const { code, message } = await readError(response);
        showError(code, message);
        return;
      }
      const job = await response.json();
      if (job.status === "completed") {
        state.result = job.result;
        renderResults();
        return;
      }
      if (job.status === "failed") {
        showError(job.error.code, job.error.message);
        return;
      }
      if (job.status === "queued") {
        setStatus("Queued…");
      } else if (job.total_pages) {
        setStatus(`Processing page ${Math.min(job.pages_done + 1, job.total_pages)} of ${job.total_pages}…`,
          job.pages_done / job.total_pages);
      } else {
        setStatus("Processing…");
      }
      await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
    }
  }

  function renderLines(container, text) {
    container.replaceChildren();
    for (const line of text.split("\n")) {
      const div = document.createElement("div");
      div.className = "line";
      div.dir = "auto";
      div.textContent = line;
      container.appendChild(div);
    }
  }

  function cell(tag, text, className) {
    const node = document.createElement(tag);
    node.textContent = text;
    if (className) node.className = className;
    return node;
  }

  function renderResults() {
    const { document: doc, extraction, pages, warnings } = state.result;
    state.pageIndex = 0;
    el.status.hidden = true;
    el.error.hidden = true;
    el.results.hidden = false;

    el.warnings.replaceChildren();
    el.warnings.hidden = warnings.length === 0;
    if (warnings.length) {
      el.warnings.append(cell("strong", "Warnings"));
      const list = document.createElement("ul");
      warnings.forEach((warning) => list.appendChild(cell("li", warning)));
      el.warnings.appendChild(list);
    }

    const engine = extraction.engine || "none (native text only)";
    const languages = extraction.detected_languages
      ? `detected: ${extraction.detected_languages.join(", ") || "unknown"}`
      : extraction.requested_languages.join(", ");
    el.summary.textContent =
      `${doc.content_type} · ${doc.page_count} page(s) · mode: ${extraction.mode} · engine: ${engine} · ` +
      `languages: ${languages}`;

    el.jsonView.textContent = JSON.stringify(state.result, null, 2);
    renderPagesTab();
    selectTab(el.outputFormat.value === "json" ? "json" : "text");
    renderPage();
  }

  function renderPage() {
    const pages = state.result.pages;
    const page = pages[state.pageIndex];
    el.pageLabel.textContent = `${page.page_number} / ${pages.length}`;
    el.prev.disabled = state.pageIndex === 0;
    el.next.disabled = state.pageIndex === pages.length - 1;
    renderLines(el.textView, page.text);
  }

  function renderPagesTab() {
    el.pagesView.replaceChildren();
    for (const page of state.result.pages) {
      const heading = document.createElement("h3");
      heading.textContent = `Page ${page.page_number} — ${page.method}`;
      el.pagesView.appendChild(heading);

      if (page.blocks.length) {
        const table = document.createElement("table");
        const head = table.createTHead().insertRow();
        ["Text", "Confidence", "Box (x0, y0, x1, y1)"].forEach((label) => head.appendChild(cell("th", label)));
        const body = table.createTBody();
        for (const block of page.blocks) {
          const row = body.insertRow();
          const textCell = cell("td", block.text, "cell-text");
          textCell.dir = "auto";
          row.appendChild(textCell);
          row.appendChild(cell("td", block.confidence == null ? "—" : block.confidence.toFixed(3), "num"));
          row.appendChild(cell("td", block.bbox ? block.bbox.map((v) => v.toFixed(3)).join(", ") : "—", "num"));
        }
        const wrap = document.createElement("div");
        wrap.className = "table-wrap";
        wrap.appendChild(table);
        el.pagesView.appendChild(wrap);
      } else {
        el.pagesView.appendChild(cell("p", "No text blocks on this page.", "muted"));
      }

      for (const docTable of page.tables) {
        el.pagesView.appendChild(cell("h4", `Table (body element ${docTable.position})`));
        const table = document.createElement("table");
        const body = table.createTBody();
        docTable.rows.forEach((rowValues) => {
          const row = body.insertRow();
          rowValues.forEach((value) => {
            const td = cell("td", value, "cell-text");
            td.dir = "auto";
            row.appendChild(td);
          });
        });
        const wrap = document.createElement("div");
        wrap.className = "table-wrap";
        wrap.appendChild(table);
        el.pagesView.appendChild(wrap);
      }

      if (page.native_text) {
        el.pagesView.appendChild(cell("h4", "Embedded text layer (kept alongside OCR)"));
        const nativeBox = document.createElement("div");
        nativeBox.className = "text-view";
        renderLines(nativeBox, page.native_text);
        el.pagesView.appendChild(nativeBox);
      }
    }
  }

  function selectTab(tab) {
    state.tab = tab;
    document.querySelectorAll(".tabs button").forEach((button) => {
      button.setAttribute("aria-selected", String(button.dataset.tab === tab));
    });
    ["text", "json", "pages"].forEach((name) => {
      $(`tab-${name}`).hidden = name !== tab;
    });
    el.pager.style.visibility = tab === "text" ? "visible" : "hidden";
  }

  function saveBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function downloadText() {
    if (!state.jobId) return;
    const response = await fetch(`${JOBS_URL}/${state.jobId}/text`);
    if (!response.ok) {
      const { code, message } = await readError(response);
      showError(code, message);
      return;
    }
    saveBlob(await response.blob(), `${safeBaseName(state.file.name)}.txt`);
  }

  function downloadJson() {
    if (!state.result) return;
    const blob = new Blob([JSON.stringify(state.result, null, 2)], { type: "application/json" });
    saveBlob(blob, `${safeBaseName(state.file.name)}.json`);
  }

  function resetAll() {
    el.fileInput.value = "";
    setFile(null);
  }

  fetch("/health")
    .then((response) => response.json())
    .then((health) => { $("max-size").textContent = `${health.limits.max_upload_mb} MB`; })
    .catch(() => {});

  el.fileInput.addEventListener("change",() => setFile(el.fileInput.files[0] || null));
  el.dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      el.fileInput.click();
    }
  });
  ["dragenter", "dragover"].forEach((name) =>
    el.dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      el.dropzone.classList.add("dragover");
    }));
  ["dragleave", "drop"].forEach((name) =>
    el.dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      el.dropzone.classList.remove("dragover");
    }));
  el.dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];
    if (file) setFile(file);
  });

  el.extract.addEventListener("click", extract);
  el.reset.addEventListener("click", resetAll);
  el.prev.addEventListener("click", () => { state.pageIndex -= 1; renderPage(); });
  el.next.addEventListener("click", () => { state.pageIndex += 1; renderPage(); });
  document.querySelectorAll(".tabs button").forEach((button) =>
    button.addEventListener("click", () => selectTab(button.dataset.tab)));
  el.downloadTxt.addEventListener("click", downloadText);
  el.downloadJson.addEventListener("click", downloadJson);
})();
