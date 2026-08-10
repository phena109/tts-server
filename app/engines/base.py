"""Engine-agnostic TTS interface.

Additional backends (MeloTTS, Fish Speech, Kokoro, …) implement
:class:`TTSEngine` and register in :mod:`app.engines.registry` without
changing the public REST API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(slots=True)
class SynthesisRequest:
    """Normalized synthesis request passed to every engine."""

    text: str
    language: str = "yue"
    speaker: str = "default"
    speed: float = 1.0
    # Optional path to a reference / prompt wav for zero-shot cloning
    prompt_audio_path: str | None = None
    prompt_text: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AudioResult:
    """Raw PCM audio produced by an engine (always float32 mono)."""

    samples: npt.NDArray[np.float32]
    sample_rate: int
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = np.asarray(self.samples, dtype=np.float32)
        if arr.ndim > 1:
            arr = np.mean(arr, axis=0 if arr.shape[0] < arr.shape[-1] else -1)
        self.samples = arr.reshape(-1).astype(np.float32, copy=False)


class TTSEngine(ABC):
    """Abstract TTS backend."""

    name: str = "base"

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory. Idempotent."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True when the engine can accept synthesis requests."""

    @abstractmethod
    def synthesize(self, request: SynthesisRequest) -> AudioResult:
        """Synthesize a single text segment (already chunked by the service)."""

    def shutdown(self) -> None:
        """Optional cleanup hook."""

    @property
    def model_id(self) -> str:
        return self.name
