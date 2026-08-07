"""Engine factory / registry for pluggable TTS backends."""

from __future__ import annotations

from typing import Callable

from app.config.settings import Settings
from app.engines.base import TTSEngine
from app.utils.logging import get_logger

logger = get_logger(__name__)

EngineFactory = Callable[[Settings], TTSEngine]

_REGISTRY: dict[str, EngineFactory] = {}


def register_engine(name: str, factory: EngineFactory) -> None:
    _REGISTRY[name.lower()] = factory


def list_engines() -> list[str]:
    _ensure_builtins()
    return sorted(_REGISTRY.keys())


def create_engine(settings: Settings) -> TTSEngine:
    """Instantiate the engine selected by ``settings.tts_engine``."""
    _ensure_builtins()
    key = settings.tts_engine.lower().strip()
    if key not in _REGISTRY:
        known = ", ".join(list_engines()) or "(none)"
        raise ValueError(f"Unknown TTS_ENGINE={settings.tts_engine!r}. Known: {known}")
    logger.info("Creating TTS engine", extra={"engine": key})
    return _REGISTRY[key](settings)


def _ensure_builtins() -> None:
    if _REGISTRY:
        return
    # Local imports keep optional heavy deps out of module import for tests
    from app.engines.cosyvoice.engine import CosyVoiceEngine

    register_engine("cosyvoice", CosyVoiceEngine.from_settings)

    # Placeholders for future backends — implement and uncomment to enable:
    # from app.engines.melo.engine import MeloTTSEngine
    # register_engine("melo", MeloTTSEngine.from_settings)
    # from app.engines.fish.engine import FishSpeechEngine
    # register_engine("fish", FishSpeechEngine.from_settings)
    # from app.engines.kokoro.engine import KokoroEngine
    # register_engine("kokoro", KokoroEngine.from_settings)
