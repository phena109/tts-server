"""Pydantic request/response schemas for the public API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

AudioFormat = Literal["wav", "mp3"]


class TTSRequest(BaseModel):
    """JSON body for POST /tts."""

    text: str = Field(..., min_length=1, description="Text to synthesize")
    language: str = Field(
        default="yue",
        description="Language/dialect code, e.g. zh, yue, en, ja, ko",
    )
    speaker: str = Field(
        default="default",
        description="Speaker id; 'default' uses the bundled prompt voice",
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Relative speaking rate (1.0 = normal)",
    )
    format: AudioFormat = Field(
        default="wav",
        description="Output audio container format",
    )

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text must not be empty or whitespace-only")
        return cleaned

    @field_validator("language", "speaker")
    @classmethod
    def normalize_token(cls, value: str) -> str:
        return value.strip().lower()


class HealthResponse(BaseModel):
    status: str = "ok"
    engine: str | None = None
    model: str | None = None
    ready: bool = True


class TTSResultMeta(BaseModel):
    """Metadata returned in response headers / logs."""

    chunk_count: int
    generation_time_ms: float
    model: str
    engine: str
    format: AudioFormat
    sample_rate: int
    language: str
    speaker: str


class TTSFileResponseMeta(TTSResultMeta):
    source_filename: str | None = None
