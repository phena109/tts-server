# Convenience targets for Podman on Apple Silicon
#
#   make build / make run   — container (API :27755, UI :27756)
#   make ui                 — local static UI only (no model)
#   make smoke              — curl smoke tests against API

IMAGE ?= cosyvoice-tts:latest
PLATFORM ?= linux/arm64
NAME ?= cosyvoice-tts
UI_PORT ?= 27756

.PHONY: build run stop logs rm volumes up down smoke test ui

build:
	podman build --platform=$(PLATFORM) -t $(IMAGE) .

volumes:
	podman volume create cosyvoice-tts-models || true
	podman volume create cosyvoice-tts-output || true
	podman volume create cosyvoice-tts-input || true
	podman volume create cosyvoice-tts-cache || true

run: volumes
	podman run -d \
		--name $(NAME) \
		--platform=$(PLATFORM) \
		-p 27755:27755 \
		-p 27756:27756 \
		-e LOG_LEVEL=INFO \
		-e DEFAULT_LANGUAGE=yue \
		-v cosyvoice-tts-models:/models \
		-v cosyvoice-tts-output:/output \
		-v cosyvoice-tts-input:/input \
		-v cosyvoice-tts-cache:/cache \
		$(IMAGE)

stop:
	podman stop $(NAME) || true

logs:
	podman logs -f $(NAME)

rm: stop
	podman rm -f $(NAME) || true

up:
	podman compose up -d --build

down:
	podman compose down

smoke:
	bash scripts/smoke_curl.sh

ui:
	python3 -m app.ui_server --host 127.0.0.1 --port $(UI_PORT) --root web

test:
	python3 -m venv .venv 2>/dev/null || true
	.venv/bin/pip install -q numpy pytest
	.venv/bin/pytest tests/ -q
