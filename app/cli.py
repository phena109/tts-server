"""Batch TTS CLI — run synthesis in-process with no HTTP server.

Use this when the API/UI path is a poor fit: long CPU jobs, no timeouts,
and you just want audio under /output when the process finishes.

Examples (inside the container):

  python -m app.cli tts --text "你好，我係測試。"
  python -m app.cli tts --file /input/article.txt --format mp3 --long
  python -m app.cli ensure-model

From the host (one-shot, overrides the server entrypoint):

  podman run --rm \\
    --platform=linux/arm64 \\
    -v cosyvoice-tts-models:/models \\
    -v "$PWD/output:/output" \\
    -v "$PWD/input:/input" \\
    -v cosyvoice-tts-cache:/cache \\
    --entrypoint python \\
    cosyvoice-tts:latest \\
    -m app.cli tts --text "你好" --format wav
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from app.config.settings import get_settings
from app.engines.registry import create_engine
from app.models.schemas import TTSRequest
from app.services.audio_service import AudioService
from app.services.model_manager import ModelManager
from app.services.tts_service import TTSService, TTSServiceError
from app.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def _prepare_env(settings) -> None:
    """Mirror server boot: caches + CosyVoice import path."""
    os.environ.setdefault("HF_HOME", str(settings.cache_dir / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(settings.cache_dir / "torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str(settings.cache_dir))
    os.environ.setdefault("MODELSCOPE_CACHE", str(settings.cache_dir / "modelscope"))
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    # Prefer lower peak RSS on small Podman VMs (same defaults as entrypoint)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    os.environ.setdefault("COSYVOICE_STREAM", "true")
    os.environ.setdefault(
        "GLIBC_TUNABLES", "glibc.cpu.hwcaps=-SVE,-SVE2,-I8MM,-BF16"
    )
    os.environ.setdefault("ATEN_CPU_CAPABILITY", "default")
    os.environ.setdefault("TRANSFORMERS_ATTENTION_IMPLEMENTATION", "eager")

    repo = str(settings.cosyvoice_repo)
    matcha = str(settings.cosyvoice_repo / "third_party" / "Matcha-TTS")
    existing = os.environ.get("PYTHONPATH", "")
    parts = [p for p in existing.split(":") if p]
    for path in (repo, matcha):
        if path not in parts:
            parts.insert(0, path)
    os.environ["PYTHONPATH"] = ":".join(parts)
    # Also ensure import works in this process
    for path in (matcha, repo):
        if path not in sys.path:
            sys.path.insert(0, path)


def ensure_model_cmd() -> int:
    settings = get_settings()
    setup_logging(settings.log_level)
    settings.ensure_directories()
    _prepare_env(settings)
    manager = ModelManager(settings)
    path = manager.ensure_model()
    print(str(path))
    return 0


def _read_text(args: argparse.Namespace) -> str:
    if args.text is not None and args.file is not None:
        raise SystemExit("use either --text or --file, not both")
    if args.text is not None:
        text = args.text
    elif args.file is not None:
        path = Path(args.file)
        if not path.is_file():
            raise SystemExit(f"file not found: {path}")
        text = path.read_text(encoding="utf-8")
    else:
        if sys.stdin.isatty():
            raise SystemExit("provide --text, --file, or pipe text on stdin")
        text = sys.stdin.read()
    text = (text or "").strip()
    if not text:
        raise SystemExit("text is empty")
    return text


def _build_service() -> tuple[TTSService, object]:
    settings = get_settings()
    setup_logging(settings.log_level)
    settings.ensure_directories()
    _prepare_env(settings)

    manager = ModelManager(settings)
    if not manager.is_model_present():
        if settings.skip_model_download:
            raise SystemExit(
                f"model missing at {settings.model_path} and SKIP_MODEL_DOWNLOAD=true"
            )
        logger.info("Model missing; downloading (blocking, no timeout)…")
        manager.ensure_model()
    else:
        logger.info("Model present at %s", settings.model_path)

    logger.info(
        "Loading engine (blocking)…",
        extra={
            "engine": settings.tts_engine,
            "model": settings.resolved_model_name,
            "device": settings.device,
        },
    )
    engine = create_engine(settings)
    engine.load()
    service = TTSService(settings=settings, engine=engine, audio=AudioService())
    return service, settings


def tts_cmd(args: argparse.Namespace) -> int:
    text = _read_text(args)
    service, settings = _build_service()

    language = args.language or settings.default_language
    speaker = args.speaker or settings.default_speaker
    fmt = args.format or settings.output_format
    speed = args.speed

    logger.info(
        "Batch TTS starting",
        extra={
            "text_len": len(text),
            "language": language,
            "speaker": speaker,
            "format": fmt,
            "speed": speed,
            "long": bool(args.long),
        },
    )

    try:
        if args.long:
            result = service.synthesize_long(
                text,
                language=language,
                speaker=speaker,
                speed=speed,
                fmt=fmt,  # type: ignore[arg-type]
            )
        else:
            result = service.synthesize(
                TTSRequest(
                    text=text,
                    language=language,
                    speaker=speaker,
                    speed=speed,
                    format=fmt,  # type: ignore[arg-type]
                )
            )
    except TTSServiceError as exc:
        logger.error("TTS failed: %s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # TTSService already writes under OUTPUT_DIR; honour --output if set.
    out_path: Path
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = settings.output_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(result.content)
    else:
        out_path = settings.output_dir / result.filename

    print(str(out_path))
    logger.info(
        "Batch TTS done",
        extra={
            "path": str(out_path),
            "chunk_count": result.chunk_count,
            "generation_time_ms": result.generation_time_ms,
            "bytes": len(result.content),
        },
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description=(
            "Batch CosyVoice TTS — no HTTP server. "
            "Loads the model, synthesizes, writes audio, exits."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "ensure-model",
        help="Download model weights if missing (blocking)",
    )

    tts = sub.add_parser(
        "tts",
        help="Synthesize speech from text/file/stdin; print output path",
    )
    src = tts.add_mutually_exclusive_group()
    src.add_argument("--text", "-t", help="Text to speak")
    src.add_argument("--file", "-f", help="UTF-8 text file to speak")
    tts.add_argument(
        "--output",
        "-o",
        help="Output path (relative paths are under OUTPUT_DIR). "
        "Default: auto name under OUTPUT_DIR",
    )
    tts.add_argument(
        "--language",
        "-l",
        default=None,
        help="Language code (default: DEFAULT_LANGUAGE / yue)",
    )
    tts.add_argument(
        "--speaker",
        "-s",
        default=None,
        help="Speaker id (default: DEFAULT_SPEAKER / default)",
    )
    tts.add_argument(
        "--format",
        choices=("wav", "mp3"),
        default=None,
        help="Audio format (default: OUTPUT_FORMAT / wav)",
    )
    tts.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speaking rate (0.5–2.0, default 1.0)",
    )
    tts.add_argument(
        "--long",
        action="store_true",
        help="Long-form path (chunk + merge; good for articles)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "ensure-model":
        return ensure_model_cmd()
    if args.command == "tts":
        return tts_cmd(args)
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
