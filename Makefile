# Convenience targets for Podman on Apple Silicon

IMAGE ?= cosyvoice-tts:latest
PLATFORM ?= linux/arm64
NAME ?= cosyvoice-tts

.PHONY: build run stop logs rm volumes up down smoke test

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
		-p 8000:8000 \
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

test:
	python3 -m venv .venv 2>/dev/null || true
	.venv/bin/pip install -q numpy pytest
	.venv/bin/pytest tests/ -q
