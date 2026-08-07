# syntax=docker/dockerfile:1.6
# CosyVoice 3 TTS API — Podman-compatible multi-arch image (linux/arm64 for Apple Silicon)
#
# Build (Apple Silicon / Podman):
#   podman build --platform=linux/arm64 -t cosyvoice-tts:latest .
#
# Models are NOT baked in — they download into /models on first run.

FROM python:3.10-slim-bookworm

LABEL org.opencontainers.image.title="cosyvoice-tts" \
      org.opencontainers.image.description="Production CosyVoice 3 TTS API (Podman / Apple Silicon)" \
      org.opencontainers.image.source="https://github.com/FunAudioLLM/CosyVoice" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    # Runtime defaults
    HOST=0.0.0.0 \
    PORT=8000 \
    LOG_LEVEL=INFO \
    TTS_ENGINE=cosyvoice \
    DEVICE=cpu \
    MODEL_DIR=/models \
    OUTPUT_DIR=/output \
    INPUT_DIR=/input \
    CACHE_DIR=/cache \
    MODEL_NAME=FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    MODEL_LOCAL_NAME=Fun-CosyVoice3-0.5B \
    COSYVOICE_REPO=/opt/CosyVoice \
    DEFAULT_PROMPT_PATH=/opt/CosyVoice/asset/zero_shot_prompt.wav \
    MAX_CHARS_PER_CHUNK=200 \
    DEFAULT_LANGUAGE=zh \
    OUTPUT_FORMAT=wav \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache \
    # Threading defaults suitable for container CPU inference
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false

# System packages: audio tooling + build deps for native wheels (pyworld, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        wget \
        git \
        git-lfs \
        ffmpeg \
        sox \
        libsox-dev \
        libsndfile1 \
        libsndfile1-dev \
        libgomp1 \
        build-essential \
        g++ \
        python3-dev \
        pkg-config \
        cmake \
        unzip \
        xz-utils \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Clone CosyVoice + Matcha-TTS submodule (code only — no model weights)
# COSYVOICE_GIT_REF can be a branch or tag (default: repo HEAD)
ARG COSYVOICE_GIT_REF=
RUN set -eux; \
    if [ -n "${COSYVOICE_GIT_REF}" ]; then \
      git clone --depth 1 --branch "${COSYVOICE_GIT_REF}" --recursive \
        https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice; \
    else \
      git clone --depth 1 --recursive \
        https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice; \
    fi; \
    cd /opt/CosyVoice; \
    git submodule update --init --recursive; \
    test -f /opt/CosyVoice/asset/zero_shot_prompt.wav; \
    # Drop VCS metadata to shrink the image (code remains)
    find /opt/CosyVoice -name .git -type d -prune -exec rm -rf {} +

WORKDIR /app

# CPU-only PyTorch for linux/arm64 and linux/amd64
# Using the official CPU wheel index (no CUDA).
ARG TORCH_VERSION=2.3.1
ARG TORCHAUDIO_VERSION=2.3.1
RUN pip install --upgrade pip setuptools wheel \
    && pip install \
        "torch==${TORCH_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
        --index-url https://download.pytorch.org/whl/cpu

# CosyVoice + application Python dependencies
COPY requirements-cosyvoice-cpu.txt /app/requirements-cosyvoice-cpu.txt
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements-cosyvoice-cpu.txt \
    && pip install -r /app/requirements.txt \
    && pip install huggingface_hub modelscope

# Application source
COPY app /app/app
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh \
    && mkdir -p /models /output /input /input/speakers /cache

# PYTHONPATH so CosyVoice + Matcha-TTS import cleanly
ENV PYTHONPATH=/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS:/app

# Persist model/audio volumes (declared for documentation; compose mounts them)
VOLUME ["/models", "/output", "/input", "/cache"]

EXPOSE 8000

# Healthcheck hits the FastAPI /health endpoint
# start-period is generous: first boot may download multi-GB weights
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

# Root is used for simpler Podman volume permissions on macOS.
# For hardened deploys, switch to a non-root user and fix volume UIDs.
ENTRYPOINT ["/app/entrypoint.sh"]
