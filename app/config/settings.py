"""Application configuration via environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration is driven by environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # --- Server ---
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=27755, alias="PORT")
    ui_port: int = Field(default=27756, alias="UI_PORT")
    ui_enabled: bool = Field(default=True, alias="UI_ENABLED")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    workers: int = Field(default=1, alias="WORKERS")

    # --- Engine selection (swap without changing the API) ---
    tts_engine: str = Field(default="cosyvoice", alias="TTS_ENGINE")

    # --- Model ---
    # COSYVOICE_MODEL is an alias for MODEL_NAME for convenience.
    model_name: str = Field(
        default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        alias="MODEL_NAME",
    )
    cosyvoice_model: str | None = Field(default=None, alias="COSYVOICE_MODEL")
    model_dir: Path = Field(default=Path("/models"), alias="MODEL_DIR")
    model_local_name: str = Field(
        default="Fun-CosyVoice3-0.5B",
        alias="MODEL_LOCAL_NAME",
    )
    cosyvoice_repo: Path = Field(
        default=Path("/opt/CosyVoice"),
        alias="COSYVOICE_REPO",
    )
    download_source: Literal["huggingface", "modelscope"] = Field(
        default="huggingface",
        alias="DOWNLOAD_SOURCE",
    )
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    skip_model_download: bool = Field(default=False, alias="SKIP_MODEL_DOWNLOAD")

    # --- Paths / volumes ---
    input_dir: Path = Field(default=Path("/input"), alias="INPUT_DIR")
    output_dir: Path = Field(default=Path("/output"), alias="OUTPUT_DIR")
    cache_dir: Path = Field(default=Path("/cache"), alias="CACHE_DIR")
    default_prompt_path: Path = Field(
        default=Path("/opt/CosyVoice/asset/zero_shot_prompt.wav"),
        alias="DEFAULT_PROMPT_PATH",
    )

    # --- Synthesis defaults ---
    default_language: str = Field(default="zh", alias="DEFAULT_LANGUAGE")
    default_speaker: str = Field(default="default", alias="DEFAULT_SPEAKER")
    output_format: Literal["wav", "mp3"] = Field(
        default="wav",
        alias="OUTPUT_FORMAT",
    )
    sample_rate: int = Field(default=24000, alias="SAMPLE_RATE")
    max_chars_per_chunk: int = Field(default=200, alias="MAX_CHARS_PER_CHUNK")
    device: str = Field(default="cpu", alias="DEVICE")
    load_jit: bool = Field(default=False, alias="LOAD_JIT")
    load_trt: bool = Field(default=False, alias="LOAD_TRT")
    fp16: bool = Field(default=False, alias="FP16")

    # --- Prompt / instruct templates ---
    system_prompt_prefix: str = Field(
        default="You are a helpful assistant.",
        alias="SYSTEM_PROMPT_PREFIX",
    )
    end_of_prompt_token: str = Field(
        default="<|endofprompt|>",
        alias="END_OF_PROMPT_TOKEN",
    )
    default_prompt_text: str = Field(
        default="希望你以后能够做的比我还好呦。",
        alias="DEFAULT_PROMPT_TEXT",
    )

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return value.upper()

    @property
    def resolved_model_name(self) -> str:
        """Prefer COSYVOICE_MODEL when set, otherwise MODEL_NAME."""
        override = (self.cosyvoice_model or "").strip()
        return override or self.model_name

    @property
    def model_path(self) -> Path:
        """Filesystem path where model weights live."""
        return self.model_dir / self.model_local_name

    def ensure_directories(self) -> None:
        """Create required volume mount points if missing."""
        for path in (
            self.model_dir,
            self.input_dir,
            self.output_dir,
            self.cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
