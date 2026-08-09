# Model download progress (API + UI)

**Date:** 2026-08-10  
**Status:** Approved  
**Approach:** In-memory progress manager (single process)

## Problem

Model weights are multi‑GB and download on first boot. Today:

1. The UI binds early (`UI_PORT`), but `entrypoint.sh` **blocks** on `python -m app.bootstrap ensure-model` before uvicorn starts.
2. During download, the API port is dead — health checks look “unreachable.”
3. `snapshot_download` runs with no process-wide status; logs are easy to miss and not structured for the UI.
4. There is no external trigger to resume/retry download after a failure without restarting the container.

Operators and the web UI need **detectable, pollable progress** and a way to **trigger ensure** without relying on silence or a full restart.

## Goals

- **Visible progress** via API and web UI.
- **API binds early** — uvicorn starts before (and while) the model downloads.
- **Auto-download on boot** when the model is missing/incomplete (unless `SKIP_MODEL_DOWNLOAD`).
- **On-demand trigger** — `POST /model/ensure` for resume/retry from API or UI.
- **Transparent readiness** — `/health` reflects model phase; TTS returns 503 until the engine is ready.
- Keep existing completeness / resume behavior for interrupted HF downloads.

## Non-goals

- SSE/WebSocket push streams.
- Multi-worker shared download state (`WORKERS>1`); document that in-process state requires **workers=1** (already the default).
- Changing which files count as a complete CosyVoice3 tree.
- Authentication on ensure/status (same open local API as today).

## Architecture

```
entrypoint.sh
  ├─ UI server (unchanged, early bind)
  └─ uvicorn app.main:app          # no blocking ensure-model
        lifespan
          ├─ ModelDownloadState (thread-safe)
          ├─ ModelManager
          ├─ if incomplete → background thread: download → verify → load engine
          └─ status/health routes always live
```

### Components

| Piece | Role |
|--------|------|
| `ModelDownloadState` | Thread-safe live snapshot: phase, message, bytes, error, timestamps |
| `ModelManager` | Completeness checks + download; reports progress into state |
| `GET /model/status` | Pollable full snapshot |
| `POST /model/ensure` | Idempotent start/resume/retry |
| `GET /health` | Existing fields + model phase / model readiness hints |
| Web UI | Banner, poll, disable TTS until ready, Retry button |

### Phases

`idle` → `checking` → `downloading` → `verifying` → `loading_engine` → `ready`  
or any active phase → `error`

| Phase | Meaning |
|--------|---------|
| `idle` | No job; model not ready (should be rare after boot logic) |
| `checking` | Completeness / interrupt detection |
| `downloading` | HF / ModelScope transfer in progress |
| `verifying` | Post-download completeness check |
| `loading_engine` | Weights OK; engine load in progress |
| `ready` | Weights complete and engine loaded; TTS available |
| `error` | Last ensure failed; message/error set; can retry via POST |

## Status model

JSON snapshot used by `GET /model/status` (and partially mirrored on health):

| Field | Type | Meaning |
|--------|------|---------|
| `phase` | string | One of the phases above |
| `ready` | bool | `true` only when weights complete **and** engine loaded |
| `model` | string | Resolved model id |
| `path` | string | Local model directory |
| `message` | string | Human-readable status line |
| `bytes_downloaded` | int \| null | Best-effort bytes so far |
| `bytes_total` | int \| null | Best-effort expected total |
| `progress_pct` | float \| null | 0–100 when total known |
| `files_done` | int \| null | Optional file-level progress |
| `files_total` | int \| null | Optional |
| `error` | string \| null | Set when `phase=error` |
| `started_at` | string \| null | ISO timestamp of current job |
| `updated_at` | string | ISO timestamp of last state change |
| `download_source` | string | `huggingface` \| `modelscope` |

### Progress sources (best-effort stack)

1. Hugging Face / ModelScope progress hooks or tqdm callbacks when available.
2. Periodic scan of model dir + `*.incomplete` sizes for byte estimates when hooks are weak.
3. Existing `_describe_missing` output during `verifying` / error messages.

Status reads must never block on the download thread (lock-free snapshot or short lock copy).

## API

### `GET /model/status`

- Always **200** with the snapshot above.
- Safe to poll every 1–2s from the UI.

### `POST /model/ensure`

Idempotent ensure/start/resume/retry:

| Condition | Response |
|-----------|----------|
| Already `downloading` / `verifying` / `loading_engine` | **200**, current status, `already_running: true` |
| Already `ready` and model complete | **200**, current status, `already_ready: true` |
| `idle` / `error` / incomplete on disk | Start background job → **202** + status |
| `SKIP_MODEL_DOWNLOAD=true` and model missing/incomplete | **409** with clear detail |

Response body includes the status snapshot plus optional flags `already_running` / `already_ready`.

Concurrency: **one download at a time** (mutex / running flag). Double POST is safe.

### `GET /health`

Keep existing fields (`status`, `engine`, `model`, `ready`). Add:

- `model_phase` — current phase string
- `model_ready` — weights+engine ready (same as status `ready` for TTS)

Semantics:

- `ready` remains “can serve TTS” (engine ready), not merely “process is up.”
- While downloading/loading: `status: "starting"`, `ready: false`, `model_phase` set so clients are not stuck with “unreachable.”

### TTS routes

When engine not ready: **503** with a body that mentions `/model/status` (and optionally `Retry-After`). Do not hang requests waiting for download.

## Lifecycle

### Entrypoint

1. Start UI first (unchanged).
2. **Do not** block on `ensure-model` by default.
3. Start uvicorn immediately.
4. Keep `python -m app.bootstrap ensure-model` as a **CLI** for ops/manual use.
5. Optional escape hatch: `ENSURE_MODEL_IN_ENTRYPOINT=true` restores old blocking behavior for special offline/debug setups (default **false**).

### App lifespan

1. Init settings, directories, logging.
2. Create `ModelDownloadState` + `ModelManager`; attach to `app.state` **before** engine exists so status routes work.
3. If model present and complete → load engine → `phase=ready`.
4. Else if not `SKIP_MODEL_DOWNLOAD` → start background ensure (download → verify → load engine).
5. Else → leave not-ready / error path; API still serves status/health.
6. On shutdown: do not corrupt partial downloads; thread may be daemon or joined with a short timeout.

Auto-download on boot **and** on-demand `POST /model/ensure` share the same worker entrypoint.

### Engine load after download

After successful verify, set `loading_engine`, call existing `create_engine` + `load`, wire `TTSService`, then `ready`. Engine load failure sets `phase=error` with message (weights may still be complete on disk).

## Web UI

- Poll `GET /model/status` ~every 1.5s while not ready; slow or stop when ready.
- Banner (Connection section or page top):
  - phase + `message`
  - determinate progress bar when `progress_pct` is set; otherwise indeterminate (+ bytes if known)
  - on error: message + **Retry download** → `POST /model/ensure`
- Health display uses phase (e.g. `downloading · 42%`) instead of only ok/unreachable.
- Disable synthesize buttons while `!ready`.

## Logging

- INFO on every phase transition.
- Throttled INFO progress during download (e.g. every ~5–10% or every ~30s) so `podman logs` is useful without pure tqdm spam.
- ERROR with exception context on failure.

## Error handling

| Case | Behavior |
|------|----------|
| Network / HF failure | `phase=error`, `error` set, logs have traceback; retry via POST or restart |
| Download “succeeds” but incomplete | Existing incomplete detection → error with missing-file list |
| Engine load fails after good weights | `phase=error` after `loading_engine`; not silent half-ready |
| Restart with partial weights | Boot `checking` → not complete → auto resume download |
| `SKIP_MODEL_DOWNLOAD` + missing | No auto job; ensure → 409; TTS 503 |

## Testing

### Unit

- `ModelDownloadState`: transitions, concurrent snapshot safety, progress fields.
- `POST /model/ensure` idempotency (already running / already ready / start).
- `GET /model/status` always returns a valid shape pre-engine.
- Existing completeness tests retained; progress callback can be mocked.
- TTS when not ready → 503.

### Integration-style (no multi‑GB download)

- Monkeypatch download to sleep + fake progress; assert status evolves.
- `SKIP_MODEL_DOWNLOAD` + missing → ensure 409; health not ready.

### Manual smoke

- Fresh volume: API answers immediately; phases move; UI banner tracks; TTS works after ready.
- Kill mid-download, restart: resumes; never stuck falsely `ready`.
- Force error, `POST /model/ensure`: retries.

## Documentation

- README: early-bind API, new endpoints, UI banner, remove implication that first boot blocks the API until download finishes.
- Note `WORKERS=1` for in-process download state.

## Implementation sketch (for planning)

Likely touch points:

- `app/services/model_manager.py` — progress reporting hooks
- New: `app/services/model_download_state.py` (or similar)
- `app/main.py` — non-blocking lifespan, background ensure + engine load
- `app/api/routes_health.py` + new `routes_model.py` (or extend health package)
- `app/models/schemas.py` — status schemas
- `entrypoint.sh` — stop default blocking ensure
- `web/index.html`, `web/app.js`, `web/styles.css` — banner + poll + retry
- `tests/` — state, routes, 503 behavior
- `README.md`

## Success criteria

1. On first boot with empty model volume, `curl localhost:27755/health` and `/model/status` work within seconds of container start (not after multi‑GB download).
2. Status phase progresses through download → ready with non-empty `message` and best-effort progress fields.
3. UI shows a visible banner and disables TTS until ready; Retry works after error.
4. `POST /model/ensure` is safe to call repeatedly.
5. Incomplete downloads are never reported as `ready`; existing completeness rules still apply.
6. Existing unit tests pass; new tests cover state and ensure idempotency.
