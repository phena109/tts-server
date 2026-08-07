#!/usr/bin/env bash
# Container entrypoint: prepare volumes, ensure model, start FastAPI.
set -euo pipefail

log() {
  # Simple structured-ish line for early boot (Python logger takes over later)
  printf '{"ts":"%s","level":"INFO","logger":"entrypoint","message":"%s"}\n' \
    "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$*"
}

log "TTS container starting"

# Defaults (overridable via environment)
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export MODEL_DIR="${MODEL_DIR:-/models}"
export OUTPUT_DIR="${OUTPUT_DIR:-/output}"
export INPUT_DIR="${INPUT_DIR:-/input}"
export CACHE_DIR="${CACHE_DIR:-/cache}"
export TTS_ENGINE="${TTS_ENGINE:-cosyvoice}"
export DEVICE="${DEVICE:-cpu}"
export MODEL_NAME="${MODEL_NAME:-FunAudioLLM/Fun-CosyVoice3-0.5B-2512}"
# COSYVOICE_MODEL overrides MODEL_NAME when set (handled in Settings)
export HF_HOME="${HF_HOME:-${CACHE_DIR}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_DIR}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_DIR}}"
export PYTHONUNBUFFERED=1

mkdir -p "${MODEL_DIR}" "${OUTPUT_DIR}" "${INPUT_DIR}" "${CACHE_DIR}" \
  "${INPUT_DIR}/speakers" "${HF_HOME}" "${TORCH_HOME}"

# Honour CosyVoice repo path for imports
export COSYVOICE_REPO="${COSYVOICE_REPO:-/opt/CosyVoice}"
export PYTHONPATH="${COSYVOICE_REPO}:${COSYVOICE_REPO}/third_party/Matcha-TTS:${PYTHONPATH:-}"

# Default prompt path (bundled with CosyVoice assets at image build time)
export DEFAULT_PROMPT_PATH="${DEFAULT_PROMPT_PATH:-${COSYVOICE_REPO}/asset/zero_shot_prompt.wav}"

if [[ ! -f "${DEFAULT_PROMPT_PATH}" ]]; then
  log "WARNING: default prompt audio missing at ${DEFAULT_PROMPT_PATH}"
fi

# Download model into mounted volume if not already present
if [[ "${SKIP_MODEL_DOWNLOAD:-false}" != "true" ]]; then
  log "Ensuring model is available (MODEL_NAME=${COSYVOICE_MODEL:-$MODEL_NAME})"
  python -m app.bootstrap ensure-model
else
  log "SKIP_MODEL_DOWNLOAD=true — not downloading models"
fi

log "Starting uvicorn on ${HOST}:${PORT} (engine=${TTS_ENGINE})"

exec uvicorn app.main:app \
  --host "${HOST}" \
  --port "${PORT}" \
  --log-level "$(echo "${LOG_LEVEL}" | tr '[:upper:]' '[:lower:]')" \
  --no-access-log \
  --proxy-headers \
  --forwarded-allow-ips='*'
