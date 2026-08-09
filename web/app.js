(() => {
  const $ = (id) => document.getElementById(id);

  const apiBaseInput = $("apiBase");
  const healthStatus = $("healthStatus");
  const healthDetail = $("healthDetail");
  const resultMsg = $("resultMsg");
  const resultMeta = $("resultMeta");
  const player = $("player");
  const downloadLink = $("downloadLink");

  const modelBanner = $("modelBanner");
  const modelPhase = $("modelPhase");
  const modelMessage = $("modelMessage");
  const modelProgressBar = $("modelProgressBar");
  const modelProgressText = $("modelProgressText");
  const btnEnsure = $("btnEnsure");
  const synthButtons = [$("btnQuick"), $("btnFile"), $("btnLong")];

  let objectUrl = null;
  let modelReady = false;
  let pollTimer = null;

  function apiBase() {
    return (apiBaseInput.value || "").replace(/\/+$/, "");
  }

  function setBusy(button, busy) {
    if (!button) return;
    button.disabled = busy;
    button.dataset.label ||= button.textContent;
    button.textContent = busy ? "Working…" : button.dataset.label;
  }

  function setSynthEnabled(enabled) {
    for (const btn of synthButtons) {
      if (btn) btn.disabled = !enabled;
    }
  }

  function clearResult() {
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    }
    player.removeAttribute("src");
    player.classList.add("hidden");
    downloadLink.classList.add("hidden");
    downloadLink.removeAttribute("href");
    resultMeta.classList.add("hidden");
    resultMeta.textContent = "";
  }

  function showError(message) {
    clearResult();
    resultMsg.textContent = message;
    resultMsg.className = "status err";
  }

  function showSuccess(message, metaLines, blob, filename) {
    resultMsg.textContent = message;
    resultMsg.className = "status ok";

    if (metaLines && metaLines.length) {
      resultMeta.textContent = metaLines.join("\n");
      resultMeta.classList.remove("hidden");
    }

    if (blob) {
      objectUrl = URL.createObjectURL(blob);
      player.src = objectUrl;
      player.classList.remove("hidden");
      downloadLink.href = objectUrl;
      downloadLink.download = filename || "speech.bin";
      downloadLink.classList.remove("hidden");
      downloadLink.textContent = `Download ${filename || "audio"}`;
    }
  }

  function metaFromHeaders(headers) {
    const keys = [
      "X-Chunk-Count",
      "X-Generation-Time-Ms",
      "X-Model",
      "X-Engine",
      "X-Sample-Rate",
      "X-Language",
      "X-Speaker",
      "Content-Type",
      "Content-Disposition",
    ];
    const lines = [];
    for (const key of keys) {
      const value = headers.get(key);
      if (value) lines.push(`${key}: ${value}`);
    }
    return lines;
  }

  function filenameFromDisposition(disposition, fallback) {
    if (!disposition) return fallback;
    const star = /filename\*=UTF-8''([^;]+)/i.exec(disposition);
    if (star) {
      try {
        return decodeURIComponent(star[1].trim());
      } catch {
        /* fall through */
      }
    }
    const plain = /filename="?([^";]+)"?/i.exec(disposition);
    return plain ? plain[1].trim() : fallback;
  }

  async function readError(response) {
    const text = await response.text();
    try {
      const data = JSON.parse(text);
      if (typeof data.detail === "string") return data.detail;
      if (data.detail != null) return JSON.stringify(data.detail, null, 2);
    } catch {
      /* not JSON */
    }
    return text || `${response.status} ${response.statusText}`;
  }

  function renderModelStatus(body) {
    if (!body) return;
    modelBanner.hidden = false;
    const phase = body.phase || "unknown";
    const ready = !!body.ready;
    modelReady = ready;
    modelPhase.textContent = ready
      ? `ready · ${body.model || "?"}`
      : `${phase}${body.progress_pct != null ? ` · ${body.progress_pct}%` : ""}`;
    modelPhase.className =
      "status " + (ready ? "ok" : phase === "error" ? "err" : "warn");
    modelMessage.textContent = body.message || body.error || "";
    const track = modelProgressBar.parentElement;
    if (body.progress_pct != null && !ready) {
      track.classList.remove("indeterminate");
      modelProgressBar.style.width = `${Math.max(0, Math.min(100, body.progress_pct))}%`;
      modelProgressText.textContent =
        body.bytes_downloaded != null
          ? `${body.bytes_downloaded} bytes` +
            (body.bytes_total != null ? ` / ${body.bytes_total}` : "")
          : "";
    } else if (!ready && phase !== "error") {
      track.classList.add("indeterminate");
      modelProgressBar.style.width = "35%";
      modelProgressText.textContent =
        body.bytes_downloaded != null ? `${body.bytes_downloaded} bytes so far` : "Working…";
    } else {
      track.classList.remove("indeterminate");
      modelProgressBar.style.width = ready ? "100%" : "0%";
      modelProgressText.textContent = ready ? "" : body.error || "";
    }
    btnEnsure.classList.toggle("hidden", !(phase === "error" || (!ready && phase === "idle")));
    setSynthEnabled(ready);
  }

  async function fetchModelStatus() {
    try {
      const res = await fetch(`${apiBase()}/model/status`);
      if (!res.ok) throw new Error(`status ${res.status}`);
      const body = await res.json();
      renderModelStatus(body);
      return body;
    } catch (err) {
      modelBanner.hidden = false;
      modelPhase.textContent = "api unreachable";
      modelPhase.className = "status err";
      modelMessage.textContent = err.message || String(err);
      setSynthEnabled(false);
      return null;
    }
  }

  function schedulePoll() {
    if (pollTimer) clearInterval(pollTimer);
    const tick = async () => {
      const body = await fetchModelStatus();
      if (body && body.ready) {
        clearInterval(pollTimer);
        pollTimer = setInterval(fetchModelStatus, 10000);
      }
    };
    tick();
    pollTimer = setInterval(tick, 1500);
  }

  async function checkHealth() {
    healthStatus.textContent = "checking…";
    healthStatus.className = "status";
    healthDetail.classList.add("hidden");
    try {
      const res = await fetch(`${apiBase()}/health`);
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        healthStatus.textContent = `error ${res.status}`;
        healthStatus.className = "status err";
        return;
      }
      const ready = body.ready !== false && body.status === "ok";
      const phase = body.model_phase || body.status || res.status;
      healthStatus.textContent = ready
        ? `ok · ${body.engine || "?"} · ${body.model || "?"}`
        : `starting · ${phase}`;
      healthStatus.className = ready ? "status ok" : "status warn";
      healthDetail.textContent = JSON.stringify(body, null, 2);
      healthDetail.classList.remove("hidden");
    } catch (err) {
      healthStatus.textContent = `unreachable: ${err.message}`;
      healthStatus.className = "status err";
    }
  }

  async function postAudio(path, init, button, fallbackName) {
    clearResult();
    resultMsg.textContent = "Requesting synthesis…";
    resultMsg.className = "status";
    setBusy(button, true);
    try {
      const res = await fetch(`${apiBase()}${path}`, init);
      if (!res.ok) {
        showError(await readError(res));
        return;
      }
      const blob = await res.blob();
      const disposition = res.headers.get("Content-Disposition");
      const filename = filenameFromDisposition(disposition, fallbackName);
      const meta = metaFromHeaders(res.headers);
      showSuccess(`Done (${(blob.size / 1024).toFixed(1)} KB)`, meta, blob, filename);
    } catch (err) {
      showError(`Request failed: ${err.message}`);
    } finally {
      setBusy(button, false);
    }
  }

  // Tabs
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      const name = tab.dataset.tab;
      document.querySelectorAll(".tab").forEach((t) => {
        const on = t === tab;
        t.classList.toggle("active", on);
        t.setAttribute("aria-selected", on ? "true" : "false");
      });
      document.querySelectorAll(".tab-panel").forEach((panel) => {
        const on = panel.id === `panel-${name}`;
        panel.classList.toggle("active", on);
        panel.hidden = !on;
      });
    });
  });

  $("btnHealth").addEventListener("click", checkHealth);

  btnEnsure.addEventListener("click", async () => {
    setBusy(btnEnsure, true);
    try {
      const res = await fetch(`${apiBase()}/model/ensure`, { method: "POST" });
      const body = await res.json().catch(() => ({}));
      if (res.status === 409) {
        showError(body.error || body.message || body.detail || "Ensure conflict");
      }
      renderModelStatus(body.phase ? body : await fetchModelStatus());
      schedulePoll();
    } catch (err) {
      showError(`Ensure failed: ${err.message}`);
    } finally {
      setBusy(btnEnsure, false);
    }
  });

  $("btnQuick").addEventListener("click", () => {
    const text = $("quickText").value.trim();
    if (!text) {
      showError("Text is required.");
      return;
    }
    const format = $("quickFormat").value;
    postAudio(
      "/tts",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          language: $("quickLanguage").value,
          speaker: $("quickSpeaker").value.trim() || "default",
          speed: Number($("quickSpeed").value) || 1,
          format,
        }),
      },
      $("btnQuick"),
      `speech.${format}`,
    );
  });

  $("btnFile").addEventListener("click", () => {
    const file = $("fileInput").files?.[0];
    if (!file) {
      showError("Choose a text file first.");
      return;
    }
    const format = $("fileFormat").value;
    const form = new FormData();
    form.append("file", file);
    form.append("language", $("fileLanguage").value);
    form.append("speaker", $("fileSpeaker").value.trim() || "default");
    form.append("speed", String(Number($("fileSpeed").value) || 1));
    form.append("format", format);
    postAudio("/tts-file", { method: "POST", body: form }, $("btnFile"), `from_file.${format}`);
  });

  $("btnLong").addEventListener("click", () => {
    const text = $("longText").value.trim();
    const file = $("longFile").files?.[0];
    if (!text && !file) {
      showError("Provide text or a file for long-form TTS.");
      return;
    }

    if (file) {
      const form = new FormData();
      if (text) form.append("text", text);
      form.append("file", file);
      form.append("language", $("longLanguage").value);
      form.append("speaker", $("longSpeaker").value.trim() || "default");
      form.append("speed", String(Number($("longSpeed").value) || 1));
      postAudio("/tts-long", { method: "POST", body: form }, $("btnLong"), "article.mp3");
      return;
    }

    postAudio(
      "/tts-long",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          language: $("longLanguage").value,
          speaker: $("longSpeaker").value.trim() || "default",
          speed: Number($("longSpeed").value) || 1,
        }),
      },
      $("btnLong"),
      "article.mp3",
    );
  });

  // Persist API base across reloads
  const stored = localStorage.getItem("ttsApiBase");
  if (stored) apiBaseInput.value = stored;
  apiBaseInput.addEventListener("change", () => {
    localStorage.setItem("ttsApiBase", apiBaseInput.value.trim());
  });

  // Disable synth until first successful status shows ready
  setSynthEnabled(false);
  checkHealth();
  schedulePoll();
})();
