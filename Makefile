# Convenience targets for Podman on Apple Silicon
#
#   make build / make run   — container (API :27755, UI :27756)
#   make batch TEXT='…'     — one-shot CLI TTS (no server; runs until done)
#   make batch-file FILE=…  — same, from a host text file under ./input
#   make ui                 — local static UI only (no model)
#   make smoke              — curl smoke tests against API

IMAGE ?= cosyvoice-tts:latest
PLATFORM ?= linux/arm64
NAME ?= cosyvoice-tts
UI_PORT ?= 27756
# Batch CLI defaults (override on the command line)
TEXT ?=
FILE ?=
FORMAT ?= wav
LANGUAGE ?= yue
SPEAKER ?= default
SPEED ?= 1.0
OUT ?=
LONG ?=
# Podman VM memory for long CPU jobs (matches compose headroom)
MEM_LIMIT ?= 11g
SHM_SIZE ?= 1g

.PHONY: build run stop logs rm volumes up down smoke test ui batch batch-file shell

build:
	podman build --platform=$(PLATFORM) -t $(IMAGE) .

volumes:
	mkdir -p output input/speakers
	podman volume create cosyvoice-tts-models || true
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
		-v "$(CURDIR)/output:/output" \
		-v "$(CURDIR)/input:/input" \
		-v cosyvoice-tts-cache:/cache \
		$(IMAGE)

# One-shot batch TTS: no uvicorn/UI, no HTTP timeouts. Blocks until audio is written.
# Examples:
#   make batch TEXT='你好，我係測試。'
#   make batch TEXT='長文…' FORMAT=mp3 LONG=1 OUT=article.mp3
batch: volumes
	@if [ -z "$(TEXT)" ]; then echo 'Usage: make batch TEXT="你好"'; exit 2; fi
	podman run --rm \
		--name $(NAME)-batch \
		--platform=$(PLATFORM) \
		--memory=$(MEM_LIMIT) \
		--shm-size=$(SHM_SIZE) \
		-e LOG_LEVEL=INFO \
		-e DEFAULT_LANGUAGE=$(LANGUAGE) \
		-e DEFAULT_SPEAKER=$(SPEAKER) \
		-e OUTPUT_FORMAT=$(FORMAT) \
		-e OMP_NUM_THREADS=1 \
		-e MKL_NUM_THREADS=1 \
		-e MALLOC_ARENA_MAX=2 \
		-e COSYVOICE_STREAM=true \
		-e GLIBC_TUNABLES=glibc.cpu.hwcaps=-SVE,-SVE2,-I8MM,-BF16 \
		-e ATEN_CPU_CAPABILITY=default \
		-e TRANSFORMERS_ATTENTION_IMPLEMENTATION=eager \
		-e PYTHONFAULTHANDLER=1 \
		-v cosyvoice-tts-models:/models \
		-v "$(CURDIR)/output:/output" \
		-v "$(CURDIR)/input:/input" \
		-v cosyvoice-tts-cache:/cache \
		--entrypoint python \
		$(IMAGE) \
		-m app.cli tts \
		--text "$(TEXT)" \
		--language $(LANGUAGE) \
		--speaker $(SPEAKER) \
		--format $(FORMAT) \
		--speed $(SPEED) \
		$(if $(LONG),--long,) \
		$(if $(OUT),--output $(OUT),)

# Batch from a host text file. FILE is relative to the repo or under input/.
# Examples:
#   make batch-file FILE=input/article.txt FORMAT=mp3 LONG=1
#   make batch-file FILE=article.txt   # → /input/article.txt
batch-file: volumes
	@if [ -z "$(FILE)" ]; then echo 'Usage: make batch-file FILE=input/article.txt'; exit 2; fi
	@case "$(FILE)" in \
		/*) cfile="$(FILE)" ;; \
		input/*) cfile="/$(FILE)" ;; \
		./input/*) cfile="/input/$${FILE#./input/}" ;; \
		*) cfile="/input/$(FILE)" ;; \
	esac; \
	podman run --rm \
		--name $(NAME)-batch \
		--platform=$(PLATFORM) \
		--memory=$(MEM_LIMIT) \
		--shm-size=$(SHM_SIZE) \
		-e LOG_LEVEL=INFO \
		-e DEFAULT_LANGUAGE=$(LANGUAGE) \
		-e DEFAULT_SPEAKER=$(SPEAKER) \
		-e OUTPUT_FORMAT=$(FORMAT) \
		-e OMP_NUM_THREADS=1 \
		-e MKL_NUM_THREADS=1 \
		-e MALLOC_ARENA_MAX=2 \
		-e COSYVOICE_STREAM=true \
		-e GLIBC_TUNABLES=glibc.cpu.hwcaps=-SVE,-SVE2,-I8MM,-BF16 \
		-e ATEN_CPU_CAPABILITY=default \
		-e TRANSFORMERS_ATTENTION_IMPLEMENTATION=eager \
		-e PYTHONFAULTHANDLER=1 \
		-v cosyvoice-tts-models:/models \
		-v "$(CURDIR)/output:/output" \
		-v "$(CURDIR)/input:/input" \
		-v cosyvoice-tts-cache:/cache \
		--entrypoint python \
		$(IMAGE) \
		-m app.cli tts \
		--file "$$cfile" \
		--language $(LANGUAGE) \
		--speaker $(SPEAKER) \
		--format $(FORMAT) \
		--speed $(SPEED) \
		$(if $(LONG),--long,) \
		$(if $(OUT),--output $(OUT),)

# Interactive shell in the image (same volumes) for ad-hoc python -m app.cli
shell: volumes
	podman run --rm -it \
		--name $(NAME)-shell \
		--platform=$(PLATFORM) \
		--memory=$(MEM_LIMIT) \
		--shm-size=$(SHM_SIZE) \
		-e LOG_LEVEL=INFO \
		-e DEFAULT_LANGUAGE=$(LANGUAGE) \
		-v cosyvoice-tts-models:/models \
		-v "$(CURDIR)/output:/output" \
		-v "$(CURDIR)/input:/input" \
		-v cosyvoice-tts-cache:/cache \
		--entrypoint bash \
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
