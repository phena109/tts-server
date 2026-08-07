"""CosyVoice 3 engine adapter.

Wraps FunAudioLLM CosyVoice ``AutoModel`` behind the shared :class:`TTSEngine`
interface so the REST layer never depends on CosyVoice internals.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

from app.config.settings import Settings
from app.engines.base import AudioResult, SynthesisRequest, TTSEngine
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Language / dialect → natural-language instruct for CosyVoice3 inference_instruct2
LANGUAGE_INSTRUCT: dict[str, str] = {
    "zh": "请用普通话表达。",
    "zh-cn": "请用普通话表达。",
    "cn": "请用普通话表达。",
    "yue": "请用广东话表达。",
    "cantonese": "请用广东话表达。",
    "zh-yue": "请用广东话表达。",
    "zh-hk": "请用广东话表达。",
    "sc": "请用四川话表达。",
    "sichuan": "请用四川话表达。",
    "sh": "请用上海话表达。",
    "shanghai": "请用上海话表达。",
    "en": "Please speak in English.",
    "english": "Please speak in English.",
    "ja": "日本語で話してください。",
    "jp": "日本語で話してください。",
    "japanese": "日本語で話してください。",
    "ko": "한국어로 말해 주세요.",
    "korean": "한국어로 말해 주세요.",
    "de": "Bitte sprechen Sie auf Deutsch.",
    "german": "Bitte sprechen Sie auf Deutsch.",
    "es": "Por favor, hable en español.",
    "spanish": "Por favor, hable en español.",
    "fr": "Veuillez parler en français.",
    "french": "Veuillez parler en français.",
    "it": "Per favore, parli in italiano.",
    "italian": "Per favore, parli in italiano.",
    "ru": "Пожалуйста, говорите по-русски.",
    "russian": "Пожалуйста, говорите по-русски.",
}


def _speed_instruct(speed: float) -> str | None:
    if speed >= 1.35:
        return "请用尽可能快地语速说一句话。"
    if speed >= 1.15:
        return "请用较快的语速说。"
    if speed <= 0.65:
        return "请用尽可能慢地语速说一句话。"
    if speed <= 0.85:
        return "请用较慢的语速说。"
    return None


class CosyVoiceEngine(TTSEngine):
    """Production CosyVoice 3 backend (CPU-friendly container target)."""

    name = "cosyvoice"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None
        self._lock = threading.RLock()
        self._ready = False
        self._model_id = settings.resolved_model_name

    @classmethod
    def from_settings(cls, settings: Settings) -> CosyVoiceEngine:
        return cls(settings)

    @property
    def model_id(self) -> str:
        return self._model_id

    def is_ready(self) -> bool:
        return self._ready and self._model is not None

    def load(self) -> None:
        with self._lock:
            if self._ready and self._model is not None:
                return

            settings = self._settings
            model_path = settings.model_path
            if not model_path.exists():
                raise FileNotFoundError(
                    f"CosyVoice model not found at {model_path}. "
                    "Run model download first (entrypoint) or mount weights under MODEL_DIR."
                )

            self._ensure_cosyvoice_on_path(settings.cosyvoice_repo)
            logger.info(
                "Loading CosyVoice model",
                extra={
                    "model": self._model_id,
                    "path": str(model_path),
                    "device": settings.device,
                },
            )

            # Import only after path setup
            from cosyvoice.cli.cosyvoice import AutoModel  # type: ignore

            load_kwargs: dict[str, Any] = {"model_dir": str(model_path)}
            # AutoModel accepts optional load_jit / load_trt / fp16 on some versions
            for key, value in (
                ("load_jit", settings.load_jit),
                ("load_trt", settings.load_trt),
                ("fp16", settings.fp16),
            ):
                load_kwargs[key] = value

            try:
                self._model = AutoModel(**load_kwargs)
            except TypeError:
                # Older/newer signatures may not accept jit/trt/fp16
                self._model = AutoModel(model_dir=str(model_path))

            self._ready = True
            sample_rate = getattr(self._model, "sample_rate", settings.sample_rate)
            logger.info(
                "CosyVoice model ready",
                extra={
                    "model": self._model_id,
                    "sample_rate": sample_rate,
                },
            )

    def synthesize(self, request: SynthesisRequest) -> AudioResult:
        if not self.is_ready():
            self.load()

        assert self._model is not None
        settings = self._settings
        prompt_wav = self._resolve_prompt_audio(request)
        prompt_text = request.prompt_text or settings.default_prompt_text
        instruct = self._build_instruct(request.language, request.speed)

        logger.info(
            "CosyVoice synthesize",
            extra={
                "model": self._model_id,
                "language": request.language,
                "speaker": request.speaker,
                "speed": request.speed,
                "text_len": len(request.text),
                "mode": "instruct2" if instruct else "zero_shot",
            },
        )

        with self._lock:
            chunks: list[np.ndarray] = []
            if instruct:
                # CosyVoice3 dialect / speed control
                generator = self._model.inference_instruct2(
                    request.text,
                    instruct,
                    str(prompt_wav),
                    stream=False,
                )
            else:
                full_prompt = (
                    f"{settings.system_prompt_prefix}"
                    f"{settings.end_of_prompt_token}"
                    f"{prompt_text}"
                )
                generator = self._model.inference_zero_shot(
                    request.text,
                    full_prompt,
                    str(prompt_wav),
                    stream=False,
                )

            for item in generator:
                speech = item["tts_speech"]
                arr = self._to_numpy(speech)
                if arr.size:
                    chunks.append(arr)

        if not chunks:
            raise RuntimeError("CosyVoice returned no audio for the given text")

        samples = np.concatenate(chunks).astype(np.float32, copy=False)
        sample_rate = int(getattr(self._model, "sample_rate", settings.sample_rate))
        return AudioResult(
            samples=samples,
            sample_rate=sample_rate,
            meta={
                "model": self._model_id,
                "engine": self.name,
                "language": request.language,
                "speaker": request.speaker,
            },
        )

    def shutdown(self) -> None:
        with self._lock:
            self._model = None
            self._ready = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_prompt_audio(self, request: SynthesisRequest) -> Path:
        settings = self._settings
        if request.prompt_audio_path:
            path = Path(request.prompt_audio_path)
            if not path.is_file():
                raise FileNotFoundError(f"prompt audio not found: {path}")
            return path

        # speaker=default → bundled prompt
        if request.speaker in ("", "default", "none"):
            path = settings.default_prompt_path
            if not path.is_file():
                # Fallback search inside CosyVoice repo
                alt = settings.cosyvoice_repo / "asset" / "zero_shot_prompt.wav"
                if alt.is_file():
                    return alt
                raise FileNotFoundError(
                    f"Default prompt audio missing at {path}. "
                    "Set DEFAULT_PROMPT_PATH or mount a reference wav."
                )
            return path

        # Named speaker: look under /input/speakers/<name>.wav
        candidate = settings.input_dir / "speakers" / f"{request.speaker}.wav"
        if candidate.is_file():
            return candidate
        candidate_mp3 = settings.input_dir / "speakers" / f"{request.speaker}.mp3"
        if candidate_mp3.is_file():
            return candidate_mp3

        raise FileNotFoundError(
            f"Unknown speaker {request.speaker!r}. "
            f"Place a reference wav at {candidate} or use speaker='default'."
        )

    def _build_instruct(self, language: str, speed: float) -> str | None:
        settings = self._settings
        lang_key = (language or settings.default_language).lower().strip()
        lang_part = LANGUAGE_INSTRUCT.get(lang_key)
        speed_part = _speed_instruct(speed)

        if not lang_part and not speed_part:
            # Still use instruct for default zh to stabilize style? Prefer zero-shot.
            return None

        pieces = [settings.system_prompt_prefix]
        if lang_part:
            pieces.append(lang_part)
        if speed_part:
            pieces.append(speed_part)
        return " ".join(pieces) + settings.end_of_prompt_token

    @staticmethod
    def _ensure_cosyvoice_on_path(repo: Path) -> None:
        repo = repo.resolve()
        if not repo.is_dir():
            raise FileNotFoundError(
                f"CosyVoice repository not found at {repo}. "
                "Build the image with CosyVoice cloned to COSYVOICE_REPO."
            )
        paths = [str(repo), str(repo / "third_party" / "Matcha-TTS")]
        for p in paths:
            if p not in sys.path:
                sys.path.insert(0, p)

    @staticmethod
    def _to_numpy(speech: Any) -> np.ndarray:
        """Convert torch / numpy speech tensor to 1-D float32 numpy."""
        if hasattr(speech, "detach"):
            speech = speech.detach().cpu().float().numpy()
        arr = np.asarray(speech, dtype=np.float32)
        if arr.ndim > 1:
            # CosyVoice typically returns [1, T]
            arr = arr.reshape(-1)
        return arr
