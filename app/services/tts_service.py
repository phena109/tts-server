"""High-level TTS orchestration: chunk → synthesize → merge → encode."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from app.config.settings import Settings
from app.engines.base import AudioResult, SynthesisRequest, TTSEngine
from app.models.schemas import TTSRequest
from app.services.audio_service import AudioService, extension_for, media_type_for
from app.utils.chunking import chunk_text
from app.utils.logging import get_logger

logger = get_logger(__name__)

AudioFormat = Literal["wav", "mp3"]


class TTSServiceError(Exception):
    """Domain error raised for client-facing failures."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(slots=True)
class SynthesizedAudio:
    content: bytes
    media_type: str
    filename: str
    chunk_count: int
    generation_time_ms: float
    model: str
    engine: str
    format: AudioFormat
    sample_rate: int
    language: str
    speaker: str


class TTSService:
    """Coordinates engines, chunking, and audio encoding."""

    def __init__(
        self,
        settings: Settings,
        engine: TTSEngine,
        audio: AudioService | None = None,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.audio = audio or AudioService()

    def ensure_ready(self) -> None:
        if not self.engine.is_ready():
            self.engine.load()

    def synthesize(self, request: TTSRequest) -> SynthesizedAudio:
        """Synthesize short/medium text (single or multi chunk as needed)."""
        return self._run(
            text=request.text,
            language=request.language or self.settings.default_language,
            speaker=request.speaker or self.settings.default_speaker,
            speed=request.speed,
            fmt=request.format or self.settings.output_format,  # type: ignore[arg-type]
            force_single_chunk=False,
        )

    def synthesize_long(
        self,
        text: str,
        *,
        language: str | None = None,
        speaker: str | None = None,
        speed: float = 1.0,
        fmt: AudioFormat = "mp3",
    ) -> SynthesizedAudio:
        """Article-oriented synthesis: always chunk + merge; default MP3."""
        return self._run(
            text=text,
            language=language or self.settings.default_language,
            speaker=speaker or self.settings.default_speaker,
            speed=speed,
            fmt=fmt,
            force_single_chunk=False,
            long_form=True,
        )

    def synthesize_file_text(
        self,
        text: str,
        *,
        language: str | None = None,
        speaker: str | None = None,
        speed: float = 1.0,
        fmt: AudioFormat | None = None,
        source_filename: str | None = None,
    ) -> SynthesizedAudio:
        result = self._run(
            text=text,
            language=language or self.settings.default_language,
            speaker=speaker or self.settings.default_speaker,
            speed=speed,
            fmt=fmt or self.settings.output_format,  # type: ignore[arg-type]
            force_single_chunk=False,
        )
        if source_filename:
            stem = Path(source_filename).stem or "speech"
            result.filename = f"{stem}.{extension_for(result.format)}"
        return result

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _run(
        self,
        *,
        text: str,
        language: str,
        speaker: str,
        speed: float,
        fmt: AudioFormat,
        force_single_chunk: bool,
        long_form: bool = False,
    ) -> SynthesizedAudio:
        text = (text or "").strip()
        if not text:
            raise TTSServiceError("text must not be empty")

        if fmt not in ("wav", "mp3"):
            raise TTSServiceError(f"unsupported format: {fmt}")

        self.ensure_ready()
        start = time.perf_counter()

        max_chars = self.settings.max_chars_per_chunk
        if force_single_chunk:
            chunks = [text]
        else:
            chunks = chunk_text(text, max_chars=max_chars)

        if not chunks:
            raise TTSServiceError("no synthesizable text after chunking")

        logger.info(
            "TTS job started",
            extra={
                "model": self.engine.model_id,
                "engine": self.engine.name,
                "chunk_count": len(chunks),
                "language": language,
                "speaker": speaker,
                "speed": speed,
                "format": fmt,
                "long_form": long_form,
                "text_len": len(text),
            },
        )

        pcm_parts: list[np.ndarray] = []
        sample_rate = self.settings.sample_rate

        try:
            for index, chunk in enumerate(chunks):
                req = SynthesisRequest(
                    text=chunk,
                    language=language,
                    speaker=speaker,
                    speed=speed,
                )
                result: AudioResult = self.engine.synthesize(req)
                pcm_parts.append(result.samples)
                sample_rate = result.sample_rate
                logger.info(
                    "Chunk synthesized",
                    extra={
                        "chunk_index": index,
                        "chunk_count": len(chunks),
                        "chunk_chars": len(chunk),
                        "model": self.engine.model_id,
                    },
                )
        except FileNotFoundError as exc:
            logger.error(
                "TTS synthesis failed",
                extra={
                    "model": self.engine.model_id,
                    "error": str(exc),
                    "chunk_count": len(chunks),
                },
            )
            raise TTSServiceError(str(exc), status_code=400) from exc
        except Exception as exc:
            logger.exception(
                "TTS synthesis failed",
                extra={
                    "model": self.engine.model_id,
                    "error": str(exc),
                    "chunk_count": len(chunks),
                },
            )
            raise TTSServiceError(
                f"synthesis failed: {exc}",
                status_code=500,
            ) from exc

        merged = self.audio.concatenate(pcm_parts, sample_rate=sample_rate)
        content = self.audio.encode(merged, sample_rate=sample_rate, fmt=fmt)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Optionally persist under /output for debugging / batch jobs
        out_name = f"tts_{uuid.uuid4().hex[:12]}.{extension_for(fmt)}"
        try:
            out_path = self.settings.output_dir / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(content)
        except OSError:
            logger.warning("Could not write output artifact", extra={"file": out_name})

        logger.info(
            "TTS job completed",
            extra={
                "model": self.engine.model_id,
                "engine": self.engine.name,
                "chunk_count": len(chunks),
                "generation_time_ms": round(elapsed_ms, 2),
                "format": fmt,
                "sample_rate": sample_rate,
                "language": language,
                "speaker": speaker,
                "bytes": len(content),
            },
        )

        return SynthesizedAudio(
            content=content,
            media_type=media_type_for(fmt),
            filename=out_name,
            chunk_count=len(chunks),
            generation_time_ms=round(elapsed_ms, 2),
            model=self.engine.model_id,
            engine=self.engine.name,
            format=fmt,
            sample_rate=sample_rate,
            language=language,
            speaker=speaker,
        )
