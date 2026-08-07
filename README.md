# CosyVoice 3 TTS Server (Podman / Apple Silicon)

Production-ready **text-to-speech API** wrapping [FunAudioLLM CosyVoice 3](https://github.com/FunAudioLLM/CosyVoice) behind a stable FastAPI surface.

| Target | Value |
|--------|--------|
| Runtime | **Podman** (not Docker) |
| Host | macOS Sequoia · Apple Silicon (M4) |
| Execution | **CPU** in `linux/arm64` container (Podman machine VM) |
| Python | 3.10 |
| Default model | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` |

Models are **never baked into the image**. They download on first run into a mounted volume and survive image rebuilds.

The inference backend is pluggable (`TTS_ENGINE`). Clients talk only to the REST API, so MeloTTS / Fish Speech / Kokoro can be added later without client changes.

---

## Architecture

```
Client
  │
  ▼
FastAPI  (app/api)
  │
  ▼
TTSService  (chunk → synthesize → merge → encode)
  │
  ▼
TTSEngine protocol  (app/engines/base.py)
  │
  ├── cosyvoice  (default)  → FunAudioLLM AutoModel
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

```bash
podman volume create cosyvoice-tts-models
podman volume create cosyvoice-tts-output
podman volume create cosyvoice-tts-input
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
  -p 8000:8000 \
  -e MODEL_NAME=FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  -e COSYVOICE_MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  -e DEFAULT_LANGUAGE=yue \
  -e LOG_LEVEL=INFO \
  -e MAX_CHARS_PER_CHUNK=200 \
  -v cosyvoice-tts-models:/models \
  -v cosyvoice-tts-output:/output \
  -v cosyvoice-tts-input:/input \
  -v cosyvoice-tts-cache:/cache \
  cosyvoice-tts:latest
```

First boot downloads the model into `/models` (several GB). Subsequent starts skip the download.

### 4. Logs / stop

```bash
podman logs -f cosyvoice-tts
podman stop cosyvoice-tts
podman start cosyvoice-tts
podman rm -f cosyvoice-tts
```

### Compose

```bash
# Build + start
podman compose up -d --build

# Tail logs
podman compose logs -f

# Stop (volumes retained)
podman compose down
```

---

## API

Base URL: `http://localhost:8000`

### `GET /health`

```bash
curl -s http://localhost:8000/health
```

```json
{
  "status": "ok",
  "engine": "cosyvoice",
  "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
  "ready": true
}
```

### `POST /tts`

JSON body → audio bytes (`wav` or `mp3`).

```bash
curl -X POST http://localhost:8000/tts \
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

curl -X POST http://localhost:8000/tts-file \
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
curl -X POST http://localhost:8000/tts-long \
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
curl -X POST http://localhost:8000/tts-long \
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
| `MODEL_NAME` | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | HF / ModelScope id |
| `COSYVOICE_MODEL` | _(empty)_ | Overrides `MODEL_NAME` when set |
| `MODEL_LOCAL_NAME` | `Fun-CosyVoice3-0.5B` | Subfolder under `/models` |
| `MODEL_DIR` | `/models` | Model volume |
| `OUTPUT_DIR` | `/output` | Saved generations |
| `INPUT_DIR` | `/input` | Optional speaker refs: `/input/speakers/<name>.wav` |
| `CACHE_DIR` | `/cache` | HF / torch caches |
| `OUTPUT_FORMAT` | `wav` | Default for `/tts` |
| `MAX_CHARS_PER_CHUNK` | `200` | Long-form chunk budget |
| `DEFAULT_LANGUAGE` | `zh` | Fallback language |
| `DEFAULT_SPEAKER` | `default` | Bundled zero-shot prompt voice |
| `LOG_LEVEL` | `INFO` | Structured JSON logs |
| `DEVICE` | `cpu` | Inference device |
| `DOWNLOAD_SOURCE` | `huggingface` | `huggingface` or `modelscope` |
| `HF_TOKEN` | _(empty)_ | Optional Hugging Face token |
| `SKIP_MODEL_DOWNLOAD` | `false` | Fail if model missing instead of downloading |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Bind address |
| `DEFAULT_PROMPT_PATH` | CosyVoice `zero_shot_prompt.wav` | Bundled default speaker |

### Speakers

- `speaker=default` uses the **bundled** CosyVoice zero-shot prompt wav (good out-of-box experience).
- Custom speakers: place `myvoice.wav` at `/input/speakers/myvoice.wav` and call with `"speaker": "myvoice"`.

### Languages

CosyVoice 3 instruct mapping includes: `zh`, `yue` (Cantonese), `en`, `ja`, `ko`, `de`, `es`, `fr`, `it`, `ru`, plus several Chinese dialects (`sc`, `sh`, …). See `app/engines/cosyvoice/engine.py`.

---

## Model management

On startup (`entrypoint.sh` → `python -m app.bootstrap ensure-model`):

1. Check `/models/<MODEL_LOCAL_NAME>` for known weight files.
2. If present → **skip download**.
3. If missing → `huggingface_hub.snapshot_download` (or ModelScope) into the volume.
4. Load CosyVoice `AutoModel` and start Uvicorn.

Change models without rebuilding:

```bash
podman run ... \
  -e COSYVOICE_MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  -e MODEL_LOCAL_NAME=Fun-CosyVoice3-0.5B \
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
  "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
  "chunk_count": 3,
  "generation_time_ms": 4521.3,
  "format": "mp3"
}
```

Errors include exception text under `error`.

---

## Podman command cheat sheet

```bash
# Volumes
podman volume create cosyvoice-tts-models
podman volume create cosyvoice-tts-output
podman volume create cosyvoice-tts-input
podman volume create cosyvoice-tts-cache
podman volume ls
podman volume inspect cosyvoice-tts-models

# Build / run
podman build --platform=linux/arm64 -t cosyvoice-tts:latest .
podman run -d --name cosyvoice-tts --platform=linux/arm64 -p 8000:8000 \
  -v cosyvoice-tts-models:/models \
  -v cosyvoice-tts-output:/output \
  -v cosyvoice-tts-input:/input \
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
| First start very slow | Model download + load; watch `podman logs -f` |
| `Model missing` + skip download | Unset `SKIP_MODEL_DOWNLOAD` or pre-populate `/models` |
| `default prompt audio missing` | Image must include CosyVoice `asset/zero_shot_prompt.wav`; rebuild |
| OOM during load | Raise Podman machine memory; compose sets `mem_limit: 8g` |
| HF rate limits | Set `HF_TOKEN` or `DOWNLOAD_SOURCE=modelscope` |
| mp3 fails | Ensure `ffmpeg` is in the image (installed by Dockerfile) |

---

## License

Application code in this repository is provided for your use. CosyVoice and model weights remain under their respective upstream licenses (see FunAudioLLM / Hugging Face model cards).
