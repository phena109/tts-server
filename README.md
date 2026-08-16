# CosyVoice TTS Server (Podman / Apple Silicon)

Production-ready **text-to-speech API** wrapping [FunAudioLLM CosyVoice](https://github.com/FunAudioLLM/CosyVoice) behind a stable FastAPI surface, plus a barebones **web UI** on a second port.

Default weights are the Cantonese fine-tune [ASLP-lab/Cosyvoice2-Yue-ZoengJyutGaai](https://huggingface.co/ASLP-lab/Cosyvoice2-Yue-ZoengJyutGaai) (WenetSpeech-Yue / ZoengJyutGaai).

| Target | Value |
|--------|--------|
| Runtime | **Podman** (not Docker) |
| Host | macOS Sequoia · Apple Silicon (M4) |
| Execution | **CPU** in `linux/arm64` container (Podman machine VM) |
| Python | 3.10 |
| Default model | `ASLP-lab/Cosyvoice2-Yue-ZoengJyutGaai` |
| API port | **27755** |
| Web UI port | **27756** |

Models are **never baked into the image**. They download on first run into a mounted volume and survive image rebuilds.

The inference backend is pluggable (`TTS_ENGINE`). REST clients and the bundled web UI both talk to the same FastAPI surface, so MeloTTS / Fish Speech / Kokoro can be added later without client changes.

---

## Architecture

```
Browser UI (:27756)          API clients
        │                         │
        └──────────┬──────────────┘
                   ▼
            FastAPI  (app/api)  :27755
                   │
                   ▼
            TTSService  (chunk → synthesize → merge → encode)
                   │
                   ▼
            TTSEngine protocol  (app/engines/base.py)
                   │
                   ├── cosyvoice  (default)  → CosyVoice2/3 AutoModel
                   ├── melo       (future)
                   ├── fish       (future)
                   └── kokoro     (future)
```

### Package layout

```
app/
  api/           # HTTP routes
  config/        # Environment-driven settings
  engines/       # Pluggable TTS backends
    cosyvoice/
  models/        # Pydantic schemas
  services/      # Model manager, audio, orchestration
  utils/         # Chunking, logging, timing
  main.py        # App factory + lifespan
  bootstrap.py   # ensure-model CLI for entrypoint
  cli.py         # Batch TTS CLI (no HTTP; long jobs)
  ui_server.py   # Static web UI (port UI_PORT)
web/             # Browser UI (vanilla HTML/JS)
```

---

## Quick start (Podman on M4)

### Prerequisites

- [Podman](https://podman.io/) 4+ with a running machine  
  ```bash
  podman machine init
  podman machine start
  ```
- Optional: `podman-compose` or `podman compose` (Compose v2 plugin)
- Disk: ~5–10 GB free for image + model weights

### 1. Create volumes

Models and cache stay as named volumes. **Generations** go to host `./output`; **speaker refs** go to host `./input` (both bind-mounted into the container).

```bash
mkdir -p output input/speakers
podman volume create cosyvoice-tts-models
podman volume create cosyvoice-tts-cache
```

### 2. Build

```bash
podman build --platform=linux/arm64 -t cosyvoice-tts:latest .
```

### 3. Run

```bash
podman run -d \
  --name cosyvoice-tts \
  --platform=linux/arm64 \
  -p 27755:27755 \
  -p 27756:27756 \
  -e MODEL_NAME=ASLP-lab/Cosyvoice2-Yue-ZoengJyutGaai \
  -e COSYVOICE_MODEL=ASLP-lab/Cosyvoice2-Yue-ZoengJyutGaai \
  -e DEFAULT_LANGUAGE=yue \
  -e LOG_LEVEL=INFO \
  -e MAX_CHARS_PER_CHUNK=200 \
  -v cosyvoice-tts-models:/models \
  -v "$PWD/output:/output" \
  -v "$PWD/input:/input" \
  -v cosyvoice-tts-cache:/cache \
  cosyvoice-tts:latest
```

Saved audio appears under **`./output`** on the host. Custom speakers go under **`./input/speakers/<name>.wav`** (same directory as `compose.yaml`).

On first boot the API binds **immediately**; the model downloads in-process in the background (several GB into `/models`). Poll `GET /model/status` or watch the web UI banner while it runs. Subsequent starts skip the download and load the engine at startup.

| Service | URL |
|---------|-----|
| API | http://localhost:27755 |
| Web UI | http://localhost:27756 |
| Health | http://localhost:27755/health |
| Model status | http://localhost:27755/model/status |

### 4. Logs / stop

```bash
podman logs -f cosyvoice-tts
podman stop cosyvoice-tts
podman start cosyvoice-tts
podman rm -f cosyvoice-tts
```

### Batch mode (no server — recommended for long CPU jobs)

The HTTP API + web UI are fine for short lines, but on Apple Silicon CPU a longer
phrase can take **many minutes**. Browsers, reverse proxies, and healthchecks all
fight that. For “just run until done,” skip the server and use the **batch CLI**
inside the same image:

```bash
# One-shot: load model → synthesize → write ./output → exit (no timeout)
make batch TEXT='你好，我係測試。'

# Longer text → MP3, forced chunking
make batch TEXT='第一段……第二段……' FORMAT=mp3 LONG=1 OUT=article.mp3

# From a file on the host (./input is bind-mounted)
echo '你好，我係測試。' > input/sample.txt
make batch-file FILE=input/sample.txt FORMAT=wav

# Or raw podman (same volumes as the server)
podman run --rm \
  --platform=linux/arm64 \
  --memory=11g --shm-size=1g \
  -v cosyvoice-tts-models:/models \
  -v "$PWD/output:/output" \
  -v "$PWD/input:/input" \
  -v cosyvoice-tts-cache:/cache \
  --entrypoint python \
  cosyvoice-tts:latest \
  -m app.cli tts --text '你好' --format wav
```

What this does differently from the server:

| | Server (`make run` / compose) | Batch (`make batch`) |
|--|--|--|
| Process | uvicorn + UI, stays up | single Python process, exits when done |
| Timeouts | client / UI / healthcheck | **none** — runs until synthesis finishes or crashes |
| Model ensure | background while API is up | **blocking** before TTS |
| Output | `./output/tts_*.wav` + HTTP body | prints path; always under `./output` |
| Healthcheck | can kill long jobs indirectly | disabled (entrypoint overridden) |

Interactive shell in the image: `make shell`, then `python -m app.cli tts --help`.

### Compose

```bash
# Build + start
podman compose up -d --build

# Tail logs
podman compose logs -f

# Stop (volumes retained)
podman compose down
```

After compose is up: API on **:27755**, UI on **:27756**.

---

## Web UI

Base URL: `http://localhost:27756`

Barebones static page (no build step) covering the same use cases as the API:

| UI section | API endpoint |
|------------|--------------|
| Quick TTS | `POST /tts` |
| From file | `POST /tts-file` |
| Long form | `POST /tts-long` |
| Connection / health | `GET /health` |
| Model status banner | `GET /model/status` |
| Retry download | `POST /model/ensure` |

**How it runs**

- Files live under `web/` (`index.html`, `styles.css`, `app.js`).
- Container entrypoint starts `python -m app.ui_server` on `UI_PORT` (default `27756`) when `UI_ENABLED=true`.
- The page calls the API at `PORT` (default `27755`). The API base URL is editable in the UI and stored in `localStorage`.
- CORS is already open on the API (`allow_origins=["*"]`), so the browser UI on `:27756` can call `:27755`.

**Local UI only** (no model / no API process):

```bash
python -m app.ui_server --host 127.0.0.1 --port 27756 --root web
# open http://127.0.0.1:27756
```

**Disable UI** in the container: `-e UI_ENABLED=false`.

## API

Base URL: `http://localhost:27755`

### `GET /health`

```bash
curl -s http://localhost:27755/health
```

When the engine is ready:

```json
{
  "status": "ok",
  "engine": "cosyvoice",
  "model": "ASLP-lab/Cosyvoice2-Yue-ZoengJyutGaai",
  "ready": true,
  "model_phase": "ready",
  "model_ready": true
}
```

While the model is still downloading or loading, `status` is `"starting"`, `ready` is `false`, and `model_phase` reflects the current phase (for example `"downloading"`). TTS routes return **503** until ready — they do not block waiting for the download.

### `GET /model/status`

Pollable snapshot of download / readiness state (always **200**). Safe to call every 1–2s.

```bash
curl -s http://localhost:27755/model/status | jq
```

| Field | Meaning |
|-------|---------|
| `phase` | `idle`, `checking`, `downloading`, `verifying`, `loading_engine`, `ready`, or `error` |
| `ready` | `true` only when weights are complete **and** the engine is loaded |
| `model` / `path` | Resolved model id and local directory |
| `message` | Human-readable status line |
| `bytes_downloaded` / `bytes_total` | Best-effort byte progress (may be null) |
| `progress_pct` | 0–100 when total is known |
| `files_done` / `files_total` | Optional file-level progress |
| `error` | Set when `phase` is `error` |
| `started_at` / `updated_at` | ISO timestamps |
| `download_source` | `huggingface` or `modelscope` |

### `POST /model/ensure`

Idempotent start / resume / retry of model download + engine load. Same worker path as the automatic boot ensure.

```bash
curl -s -X POST http://localhost:27755/model/ensure | jq
```

| Condition | HTTP status | Notes |
|-----------|-------------|--------|
| Already downloading / verifying / loading | **200** | `already_running: true` |
| Already ready | **200** | `already_ready: true` |
| Started a new background job | **202** | Body is the current status snapshot |
| `SKIP_MODEL_DOWNLOAD` and model missing | **409** | Conflict; no download started |

### `POST /tts`

JSON body → audio bytes (`wav` or `mp3`).

```bash
curl -X POST http://localhost:27755/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，我係測試。",
    "language": "yue",
    "speaker": "default",
    "speed": 1.0,
    "format": "wav"
  }' \
  --output speech.wav
```

**Response headers** (structured metrics):

| Header | Meaning |
|--------|---------|
| `X-Chunk-Count` | Number of text chunks synthesized |
| `X-Generation-Time-Ms` | Wall-clock generation time |
| `X-Model` | Model id used |
| `X-Engine` | Engine name (`cosyvoice`, …) |
| `X-Sample-Rate` | Audio sample rate |
| `X-Language` / `X-Speaker` | Resolved request params |

### `POST /tts-file`

Upload a UTF-8 text file; return audio.

```bash
echo '你好，我係測試。今日天氣好好。' > sample.txt

curl -X POST http://localhost:27755/tts-file \
  -F "file=@sample.txt" \
  -F "language=yue" \
  -F "speaker=default" \
  -F "speed=1.0" \
  -F "format=wav" \
  --output from_file.wav
```

### `POST /tts-long`

Long articles: automatic chunking → per-chunk synthesis → merge → **single MP3**.

```bash
curl -X POST http://localhost:27755/tts-long \
  -H "Content-Type: application/json" \
  -d '{
    "text": "第一段……\n\n第二段……",
    "language": "yue",
    "speed": 1.0
  }' \
  --output article.mp3
```

Or multipart:

```bash
curl -X POST http://localhost:27755/tts-long \
  -F "file=@article.txt" \
  -F "language=zh" \
  --output article.mp3
```

---

## Configuration

All settings are environment variables (see also `.env.example`).

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_ENGINE` | `cosyvoice` | Backend registry key |
| `MODEL_NAME` | `ASLP-lab/Cosyvoice2-Yue-ZoengJyutGaai` | HF / ModelScope id |
| `COSYVOICE_MODEL` | _(empty)_ | Overrides `MODEL_NAME` when set |
| `MODEL_LOCAL_NAME` | `Cosyvoice2-Yue-ZoengJyutGaai` | Subfolder under `/models` |
| `MODEL_DIR` | `/models` | Model volume |
| `OUTPUT_DIR` | `/output` | Saved generations (host: `./output`) |
| `INPUT_DIR` | `/input` | Speaker refs (host: `./input/speakers/<name>.wav`) |
| `CACHE_DIR` | `/cache` | HF / torch caches |
| `OUTPUT_FORMAT` | `wav` | Default for `/tts` |
| `MAX_CHARS_PER_CHUNK` | `200` | Long-form chunk budget |
| `DEFAULT_LANGUAGE` | `yue` | Fallback language (Cantonese) |
| `DEFAULT_SPEAKER` | `default` | Bundled zero-shot prompt voice |
| `LOG_LEVEL` | `INFO` | Structured JSON logs |
| `DEVICE` | `cpu` | Inference device |
| `DOWNLOAD_SOURCE` | `huggingface` | `huggingface` or `modelscope` |
| `HF_TOKEN` | _(empty)_ | Optional Hugging Face token |
| `SKIP_MODEL_DOWNLOAD` | `false` | Do not auto-download; leave not-ready / error if missing |
| `ENSURE_MODEL_IN_ENTRYPOINT` | `false` | If `true`, block in entrypoint until download finishes (legacy); default defers ensure to the API process |
| `WORKERS` | `1` | Uvicorn workers. **Keep at 1** — download/progress state is in-process and not shared across workers |
| `HOST` / `PORT` | `0.0.0.0` / `27755` | API bind address |
| `UI_PORT` | `27756` | Web UI bind port |
| `UI_ENABLED` | `true` | Start static UI server in entrypoint |
| `UI_ROOT` | `/app/web` | Directory of static UI files |
| `DEFAULT_PROMPT_PATH` | CosyVoice `zero_shot_prompt.wav` | Bundled default speaker |

### Speakers

- `speaker=default` uses the **bundled** CosyVoice zero-shot prompt wav (good out-of-box experience).
- Custom speakers: place a short clean reference on the host at `./input/speakers/myvoice.wav` (bind-mounted to `/input/speakers/myvoice.wav` in the container) and call with `"speaker": "myvoice"`.

### Languages

Default model is **Cantonese-first** (`yue`). The engine uses CosyVoice2-style instruct (`用粤语说这句话`) and optional OpenCC simplified→traditional pre-processing for Yue.

CosyVoice3 instruct mapping (if you switch back to a v3 checkpoint) includes: `zh`, `yue`, `en`, `ja`, `ko`, `de`, `es`, `fr`, `it`, `ru`, plus Chinese dialects (`sc`, `sh`, …). See `app/engines/cosyvoice/engine.py`.

---

## Model management

By default the model is **not** downloaded in the entrypoint. UI and API bind first; ensure runs in the API process (FastAPI lifespan background thread) so `GET /model/status` and the UI banner work during multi-GB downloads.

Boot sequence (`entrypoint.sh` + app lifespan):

1. Start the static web UI on `UI_PORT` (unless `UI_ENABLED=false`).
2. Start Uvicorn on `PORT` immediately (no blocking ensure unless `ENSURE_MODEL_IN_ENTRYPOINT=true`).
3. In-process bootstrap checks `/models/<MODEL_LOCAL_NAME>` for known weight files.
4. If present → load CosyVoice `AutoModel` → `phase=ready`.
5. If missing and `SKIP_MODEL_DOWNLOAD` is false → background download (`huggingface_hub.snapshot_download` or ModelScope) → verify → load engine.
6. Clients poll `GET /model/status` or use `POST /model/ensure` to resume / retry after an error.

Manual / ops CLI (same ensure logic): `python -m app.bootstrap ensure-model`.

Change models without rebuilding (example: CosyVoice3 multipurpose):

```bash
podman run ... \
  -e COSYVOICE_MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  -e MODEL_LOCAL_NAME=Fun-CosyVoice3-0.5B \
  ...
```

Default Cantonese model:

```bash
podman run ... \
  -e MODEL_NAME=ASLP-lab/Cosyvoice2-Yue-ZoengJyutGaai \
  -e MODEL_LOCAL_NAME=Cosyvoice2-Yue-ZoengJyutGaai \
  -e DEFAULT_LANGUAGE=yue \
  ...
```

---

## Text chunking

`app/utils/chunking.py`:

1. Prefer **paragraphs** (blank lines)
2. Then **sentences** (CJK + Latin terminators)
3. Soft-pack under `MAX_CHARS_PER_CHUNK`
4. Avoid splits inside quotes / parentheses when possible
5. Last resort: whitespace / punctuation hard-split

---

## Logging

Stdout is **structured JSON** per line:

```json
{
  "ts": "2026-08-08T12:00:00+00:00",
  "level": "INFO",
  "logger": "app.services.tts_service",
  "message": "TTS job completed",
  "model": "ASLP-lab/Cosyvoice2-Yue-ZoengJyutGaai",
  "chunk_count": 3,
  "generation_time_ms": 4521.3,
  "format": "mp3"
}
```

Errors include exception text under `error`.

---

## Podman command cheat sheet

```bash
# Volumes (output + input are host bind mounts, not named volumes)
mkdir -p output input/speakers
podman volume create cosyvoice-tts-models
podman volume create cosyvoice-tts-cache
podman volume ls
podman volume inspect cosyvoice-tts-models

# Build / run
podman build --platform=linux/arm64 -t cosyvoice-tts:latest .
podman run -d --name cosyvoice-tts --platform=linux/arm64 -p 27755:27755 -p 27756:27756 \
  -v cosyvoice-tts-models:/models \
  -v "$PWD/output:/output" \
  -v "$PWD/input:/input" \
  -v cosyvoice-tts-cache:/cache \
  cosyvoice-tts:latest

# Lifecycle
podman ps
podman logs -f cosyvoice-tts
podman stop cosyvoice-tts
podman start cosyvoice-tts
podman rm -f cosyvoice-tts

# Compose
podman compose up -d --build
podman compose ps
podman compose logs -f tts
podman compose down
```

---

## Local development (without full CosyVoice)

Lightweight units (chunking / audio) run on the host:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pytest fastapi pydantic pydantic-settings
pytest tests/test_chunking.py tests/test_audio_service.py -q
```

Serve the web UI locally (points at `http://localhost:27755` by default):

```bash
python -m app.ui_server --host 127.0.0.1 --port 27756 --root web
```

Or via Makefile: `make ui`.

Full inference requires the container (or a native CosyVoice install with `COSYVOICE_REPO` set).

---

## Adding another engine later

1. Implement `TTSEngine` in `app/engines/<name>/engine.py`.
2. Register in `app/engines/registry.py`:
   ```python
   register_engine("melo", MeloTTSEngine.from_settings)
   ```
3. Run with `TTS_ENGINE=melo`.

No API or client changes required.

---

## Notes on Apple Silicon & GPU

- Podman on macOS runs a **Linux VM**. The container is `linux/arm64`, not macOS native.
- **PyTorch MPS (Metal)** is not available inside that VM today.
- This image uses **CPU** PyTorch wheels — correct and practical for M4 at moderate load.
- If GPU-from-container becomes viable later, swap the engine device / base image without changing the REST contract.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| First start download in progress | API is up; poll `GET /model/status` or the UI banner; TTS returns 503 until `ready` |
| Download failed / stuck in error | `POST /model/ensure` (or UI **Retry download**); check `podman logs -f` |
| `Model missing` + skip download | Unset `SKIP_MODEL_DOWNLOAD` or pre-populate `/models` |
| Status always idle / no progress with multi-worker | Keep `WORKERS=1` (in-process download state is not shared) |
| `default prompt audio missing` | Image must include CosyVoice `asset/zero_shot_prompt.wav`; rebuild |
| OOM during load or first TTS | CosyVoice CPU needs ~**5–8 GB** resident after load. Raise **both** Podman machine RAM and compose `mem_limit`. On a 16 GB Mac: `podman machine stop && podman machine set --memory 12288 && podman machine start` (leave headroom for macOS). Compose default is `mem_limit: 11g`. The Podman VM often has **no swap** — add a 4 GiB swapfile inside the machine if longer jobs still die: `podman machine ssh` then `sudo dd if=/dev/zero of=/var/swapfile bs=1M count=4096 && sudo chmod 600 /var/swapfile && sudo mkswap /var/swapfile && sudo swapon /var/swapfile`. Confirm real OOM with `podman inspect cosyvoice-tts --format '{{.State.OOMKilled}}'` — if `false` but the process dies mid-TTS, see SIGILL row |
| Quick TTS dies / empty reply / `Illegal instruction` / looks like “always OOM” | Usually **SIGILL**, not OOM: Apple Silicon Podman torch bugs (bias-less Linear, **3D+/4D matmul** in CosyVoice flow attention, mel/fbank). Logs show `Fatal Python error: Illegal instruction` under `transformer/attention.py` / `token2wav`. Image workarounds: `install_safe_matmul`, Qwen2 eager attention, linear biases, mel patches. Rebuild: `podman compose up -d --build --force-recreate` |
| Longer TTS “dies” / UI hangs many minutes then fails | Prefer **batch mode** (no server): `make batch TEXT='…'` or `make batch-file FILE=input/….txt`. That path has no HTTP/UI timeout. If you stay on the server: keep `COSYVOICE_STREAM=true` (default). Expect **slow** CPU RTF (~10–40×): ~1–3 min for a short line, ~10+ min for ~100 chars. Optional tighter hops: `COSYVOICE_TOKEN_HOP_LEN=15` |
| `no frontend is available` / wetext ModelScope auth errors | App clones wetext FSTs via ModelScope **git** into `/cache/wetext` and patches `snapshot_download`. Needs `modelscope.cn` once; assets persist on `tts-cache` |
| Download finished then UI shows `api unreachable` / `NetworkError` | After download the engine loads (~1–2 min). UI should show `loading_engine`, not unreachable. If the banner says `api unreachable` and the container restarts every few minutes, check OOM: `podman inspect cosyvoice-tts --format '{{.State.OOMKilled}} {{.RestartCount}}'`. Raise VM memory, then `podman compose up -d --build --force-recreate` |
| HF rate limits | Set `HF_TOKEN` or `DOWNLOAD_SOURCE=modelscope` |
| mp3 fails | Ensure `ffmpeg` is in the image (installed by Dockerfile) |
| UI page loads but health is unreachable | API may still be starting/`loading_engine` — poll `GET /model/status`. Confirm ports `-p 27755:27755` and `-p 27756:27756`. If the container restarts every few seconds, see OOM row |
| UI not listening | Check `UI_ENABLED=true`, `UI_ROOT` exists (`/app/web` in image), port map `27756` |
| Browser CORS errors | API enables open CORS by default; verify you are calling the API host/port shown in the UI |

---

## License

Application code in this repository is provided for your use. CosyVoice and model weights remain under their respective upstream licenses (see FunAudioLLM / Hugging Face model cards).
