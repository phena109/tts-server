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

  const synthProgress = $("synthProgress");
  const synthElapsed = $("synthElapsed");
  const synthStage = $("synthStage");

  // CosyVoice CPU jobs regularly exceed 30–90s; keep the browser waiting longer.
  const SYNTH_TIMEOUT_MS = {
    quick: 10 * 60 * 1000,
    file: 15 * 60 * 1000,
    long: 30 * 60 * 1000,
  };

  const SYNTH_STAGES = [
    { afterSec: 0, text: "Queued — sending request to the API…" },
    { afterSec: 2, text: "Server accepted the job — preparing CosyVoice…" },
    { afterSec: 8, text: "Running language model (token generation)…" },
    { afterSec: 25, text: "Still generating speech tokens — normal on CPU…" },
    { afterSec: 45, text: "Flow / vocoder may be running — hang tight…" },
    { afterSec: 90, text: "Long run on CPU — often 1–3 minutes for short lines…" },
    { afterSec: 180, text: "Still working — long-form or cold start can take several minutes…" },
  ];

  let objectUrl = null;
  let modelReady = false;
  let pollTimer = null;
  let synthActive = false;
  let synthTimer = null;
  let synthStartedAt = 0;
  let lastGoodModelBody = null;

  function apiBase() {
    return (apiBaseInput.value || "").replace(/\/+$/, "");
  }

  function formatDuration(ms) {
    const totalSec = Math.floor(ms / 1000);
    const m = Math.floor(totalSec / 60);
    const s = totalSec % 60;
    if (m <= 0) return `${s}s`;
    return `${m}m ${String(s).padStart(2, "0")}s`;
  }

  function stageForElapsed(sec) {
    let text = SYNTH_STAGES[0].text;
    for (const stage of SYNTH_STAGES) {
      if (sec >= stage.afterSec) text = stage.text;
    }
    return text;
  }

  function setBusy(button, busy, busyLabel) {
    if (!button) return;
    button.disabled = busy;
    button.dataset.label ||= button.textContent;
    button.textContent = busy ? busyLabel || "Working…" : button.dataset.label;
  }

  function setSynthEnabled(enabled) {
    // Keep the active synth button disabled via setBusy; others follow model ready.
    if (synthActive && !enabled) return;
    for (const btn of synthButtons) {
      if (btn && !btn.dataset.synthLocked) btn.disabled = !enabled;
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

  function stopSynthProgress() {
    synthActive = false;
    if (synthTimer) {
      clearInterval(synthTimer);
      synthTimer = null;
    }
    if (synthProgress) synthProgress.classList.add("hidden");
    for (const btn of synthButtons) {
      if (btn) delete btn.dataset.synthLocked;
    }
  }

  function startSynthProgress(kind) {
    synthActive = true;
    synthStartedAt = Date.now();
    if (synthProgress) synthProgress.classList.remove("hidden");
    const tick = () => {
      const ms = Date.now() - synthStartedAt;
      const sec = Math.floor(ms / 1000);
      if (synthElapsed) {
        const limit = SYNTH_TIMEOUT_MS[kind] || SYNTH_TIMEOUT_MS.quick;
        synthElapsed.textContent = `Elapsed ${formatDuration(ms)} · timeout ${formatDuration(limit)}`;
      }
      if (synthStage) synthStage.textContent = stageForElapsed(sec);
      resultMsg.textContent = `Synthesizing… ${formatDuration(ms)}`;
      resultMsg.className = "status busy";
    };
    tick();
    synthTimer = setInterval(tick, 500);
  }

  function showError(message) {
    stopSynthProgress();
    clearResult();
    resultMsg.textContent = message;
    resultMsg.className = "status err";
  }

  function showSuccess(message, metaLines, blob, filename) {
    stopSynthProgress();
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
    lastGoodModelBody = body;
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
    if (!synthActive) setSynthEnabled(ready);
  }

  function renderBusyModelBanner(reason) {
    modelBanner.hidden = false;
    modelPhase.textContent = "busy · synthesizing";
    modelPhase.className = "status warn";
    modelMessage.textContent =
      reason ||
      "API is occupied with CosyVoice (CPU). Status polling pauses until the job finishes.";
    const track = modelProgressBar.parentElement;
    track.classList.add("indeterminate");
    modelProgressBar.style.width = "35%";
    modelProgressText.textContent = "Do not close this tab — job is still running in the browser.";
    btnEnsure.classList.add("hidden");
  }

  async function fetchModelStatus() {
    // During synthesis the API event loop is blocked — treat that as busy, not down.
    if (synthActive) {
      renderBusyModelBanner();
      return lastGoodModelBody;
    }
    try {
      const controller = new AbortController();
      const t = setTimeout(() => controller.abort(), 8000);
      const res = await fetch(`${apiBase()}/model/status`, { signal: controller.signal });
      clearTimeout(t);
      if (!res.ok) throw new Error(`status ${res.status}`);
      const body = await res.json();
      renderModelStatus(body);
      return body;
    } catch (err) {
      if (synthActive) {
        renderBusyModelBanner(err.message);
        return lastGoodModelBody;
      }
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
      if (synthActive) {
        renderBusyModelBanner();
        return;
      }
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
    if (synthActive) {
      healthStatus.textContent = "busy · synthesizing (probe skipped)";
      healthStatus.className = "status warn";
      return;
    }
    try {
      const controller = new AbortController();
      const t = setTimeout(() => controller.abort(), 8000);
      const res = await fetch(`${apiBase()}/health`, { signal: controller.signal });
      clearTimeout(t);
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
      if (synthActive) {
        healthStatus.textContent = "busy · synthesizing";
        healthStatus.className = "status warn";
        return;
      }
      healthStatus.textContent = `unreachable: ${err.message}`;
      healthStatus.className = "status err";
    }
  }

  async function postAudio(path, init, button, fallbackName, kind) {
    clearResult();
    stopSynthProgress();
    resultMsg.textContent = "Starting synthesis…";
    resultMsg.className = "status busy";
    startSynthProgress(kind || "quick");
    if (button) button.dataset.synthLocked = "1";
    setBusy(button, true, "Synthesizing…");
    setSynthEnabled(false);

    const timeoutMs = SYNTH_TIMEOUT_MS[kind] || SYNTH_TIMEOUT_MS.quick;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    const fetchInit = { ...init, signal: controller.signal };

    try {
      const res = await fetch(`${apiBase()}${path}`, fetchInit);
      if (!res.ok) {
        showError(await readError(res));
        return;
      }
      const blob = await res.blob();
      const disposition = res.headers.get("Content-Disposition");
      const filename = filenameFromDisposition(disposition, fallbackName);
      const meta = metaFromHeaders(res.headers);
      const elapsed = formatDuration(Date.now() - synthStartedAt);
      meta.unshift(`Client-Elapsed: ${elapsed}`);
      showSuccess(
        `Done in ${elapsed} · ${(blob.size / 1024).toFixed(1)} KB`,
        meta,
        blob,
        filename,
      );
      // Refresh health after a successful job
      checkHealth();
      fetchModelStatus();
    } catch (err) {
      const name = err && err.name;
      if (name === "AbortError") {
        showError(
          `Timed out after ${formatDuration(timeoutMs)}. CosyVoice on CPU can be slow — try shorter text, or check podman logs. The server may still be working; wait a bit before retrying.`,
        );
      } else {
        showError(
          `Request failed: ${err.message || String(err)}. If the server is synthesizing, wait for it to finish before retrying.`,
        );
      }
    } finally {
      clearTimeout(timeoutId);
      stopSynthProgress();
      setBusy(button, false);
      setSynthEnabled(modelReady);
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
      "quick",
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
    postAudio("/tts-file", { method: "POST", body: form }, $("btnFile"), `from_file.${format}`, "file");
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
      postAudio("/tts-long", { method: "POST", body: form }, $("btnLong"), "article.mp3", "long");
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
      "long",
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
