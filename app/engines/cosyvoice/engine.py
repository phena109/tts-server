"""CosyVoice engine adapter (CosyVoice2 / CosyVoice3).

Wraps FunAudioLLM CosyVoice ``AutoModel`` behind the shared :class:`TTSEngine`
interface so the REST layer never depends on CosyVoice internals.

Default product model is ASLP-lab CosyVoice2-Yue-ZoengJyutGaai (Cantonese).
"""

from __future__ import annotations

import gc
import os
import sys
import threading
from pathlib import Path
from typing import Any, Literal

import numpy as np

from app.config.settings import Settings
from app.engines.base import AudioResult, SynthesisRequest, TTSEngine
from app.engines.cosyvoice.checkpoint_patch import install_checkpoint_compat
from app.engines.cosyvoice.mel_patch import install_safe_mel_spectrogram
from app.engines.cosyvoice.torch_patch import (
    ensure_linear_biases,
    install_safe_attention,
    install_safe_matmul,
)
from app.services.prompt_features import (
    ensure_prompt_features,
    load_prompt_features,
)
from app.services.wetext_assets import prepare_wetext_for_cosyvoice
from app.utils.logging import get_logger

logger = get_logger(__name__)

CosyVoiceGen = Literal["cosyvoice2", "cosyvoice3", "unknown"]

# CosyVoice2-Yue (ASLP-lab) official instruct phrasing for Cantonese.
_YUE_INSTRUCT_V2 = "用粤语说这句话"

# Language / dialect → natural-language instruct for CosyVoice3 inference_instruct2
LANGUAGE_INSTRUCT_V3: dict[str, str] = {
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

# CosyVoice2 instruct (plain natural language; no system/endofprompt tokens)
LANGUAGE_INSTRUCT_V2: dict[str, str] = {
    "zh": "用普通话说这句话",
    "zh-cn": "用普通话说这句话",
    "cn": "用普通话说这句话",
    "yue": _YUE_INSTRUCT_V2,
    "cantonese": _YUE_INSTRUCT_V2,
    "zh-yue": _YUE_INSTRUCT_V2,
    "zh-hk": _YUE_INSTRUCT_V2,
    "en": "Please say this in English.",
    "english": "Please say this in English.",
}

_YUE_LANG_KEYS = frozenset({"yue", "cantonese", "zh-yue", "zh-hk"})


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


def _configure_cpu_threads() -> None:
    """Keep peak RSS lower on small Podman VMs (16GB hosts)."""
    try:
        n = max(1, int(os.environ.get("OMP_NUM_THREADS", "2")))
    except ValueError:
        n = 2
    try:
        import torch

        torch.set_num_threads(n)
        torch.set_num_interop_threads(1)
    except Exception:
        logger.debug("torch thread config skipped", exc_info=True)


class CosyVoiceEngine(TTSEngine):
    """Production CosyVoice backend (CPU-friendly container target)."""

    name = "cosyvoice"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None
        self._lock = threading.RLock()
        self._ready = False
        self._model_id = settings.resolved_model_name
        self._prompt_extract_cache: dict[str, dict[str, Any]] = {}
        self._generation: CosyVoiceGen = "unknown"
        self._opencc: Any = None

    @classmethod
    def from_settings(cls, settings: Settings) -> CosyVoiceEngine:
        return cls(settings)

    @property
    def model_id(self) -> str:
        return self._model_id

    def is_ready(self) -> bool:
        return self._ready and self._model is not None

    @staticmethod
    def detect_generation(model_path: Path) -> CosyVoiceGen:
        if (model_path / "cosyvoice3.yaml").is_file():
            return "cosyvoice3"
        if (model_path / "cosyvoice2.yaml").is_file():
            return "cosyvoice2"
        return "unknown"

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

            self._generation = self.detect_generation(model_path)
            self._ensure_cosyvoice_on_path(settings.cosyvoice_repo)
            _configure_cpu_threads()
            # Must run before any Matcha/CosyVoice mel path (prompt features + TTS).
            install_safe_mel_spectrogram()
            # 3D+/4D torch.matmul SIGILLs on Apple Silicon Podman (flow attention).
            install_safe_matmul()
            # Must run before transformers Qwen2 is imported via AutoModel.
            install_safe_attention()

            # wetext FSTs before AutoModel → CosyVoiceFrontEnd builds TN models
            try:
                prepare_wetext_for_cosyvoice(settings.cache_dir)
            except Exception:
                logger.exception(
                    "wetext assets unavailable; CosyVoice will run without text frontend"
                )

            # Extract default prompt features BEFORE loading full weights so the
            # speech-tokenizer ONNX run does not stack on top of llm+flow RSS.
            default_prompt = self._default_prompt_path()
            if default_prompt is not None:
                ensure_prompt_features(
                    model_dir=model_path,
                    prompt_wav=default_prompt,
                    cache_dir=settings.cache_dir,
                    cosyvoice_repo=settings.cosyvoice_repo,
                    sample_rate=settings.sample_rate,
                )
                gc.collect()

            logger.info(
                "Loading CosyVoice model",
                extra={
                    "model": self._model_id,
                    "path": str(model_path),
                    "device": settings.device,
                    "generation": self._generation,
                },
            )

            # Import only after path setup
            from cosyvoice.cli.cosyvoice import AutoModel  # type: ignore

            # Patch CosyVoiceModel.load after cosyvoice is importable.
            install_checkpoint_compat()

            # CosyVoice3 accepts load_trt / fp16 / load_vllm — not load_jit.
            # CosyVoice1/2 historically accept load_jit / load_trt / fp16.
            load_kwargs: dict[str, Any] = {"model_dir": str(model_path)}
            if self._generation == "cosyvoice3":
                load_kwargs["load_trt"] = settings.load_trt
                load_kwargs["fp16"] = settings.fp16
            else:
                load_kwargs["load_jit"] = settings.load_jit
                load_kwargs["load_trt"] = settings.load_trt
                load_kwargs["fp16"] = settings.fp16

            try:
                self._model = AutoModel(**load_kwargs)
            except TypeError as exc:
                # Signature drift across CosyVoice versions — fall back to model_dir only.
                logger.warning(
                    "AutoModel rejected optional load flags; retrying with model_dir only",
                    extra={"error": str(exc)},
                )
                self._model = AutoModel(model_dir=str(model_path))

            # CosyVoice runs llm_job in a daemon Thread; on Apple Silicon Podman VMs
            # that path is flaky. Run LLM work inline instead.
            self._patch_sync_llm_tts()

            # torch F.linear without bias SIGILLs on this platform — add zeros.
            try:
                ensure_linear_biases(self._model)
                if hasattr(self._model, "model"):
                    ensure_linear_biases(self._model.model)
            except Exception:
                logger.exception("ensure_linear_biases failed")

            self._install_prompt_feature_cache()
            self._hydrate_prompt_cache_from_disk()
            self._init_opencc()

            self._ready = True
            sample_rate = getattr(self._model, "sample_rate", settings.sample_rate)
            frontend = getattr(self._model, "frontend", None)
            text_frontend = getattr(frontend, "text_frontend", None)
            logger.info(
                "CosyVoice model ready",
                extra={
                    "model": self._model_id,
                    "generation": self._generation,
                    "sample_rate": sample_rate,
                    "text_frontend": text_frontend or "none",
                    "prompt_cache_keys": len(self._prompt_extract_cache),
                    "opencc": self._opencc is not None,
                },
            )

    def synthesize(self, request: SynthesisRequest) -> AudioResult:
        if not self.is_ready():
            self.load()

        assert self._model is not None
        settings = self._settings
        prompt_wav = self._resolve_prompt_audio(request)
        prompt_text = request.prompt_text or settings.default_prompt_text
        lang_key = (request.language or settings.default_language).lower().strip()
        text = self._prepare_text(request.text, lang_key)
        instruct = self._build_instruct(lang_key, request.speed)

        logger.info(
            "CosyVoice synthesize",
            extra={
                "model": self._model_id,
                "generation": self._generation,
                "language": lang_key,
                "speaker": request.speaker,
                "speed": request.speed,
                "text_len": len(text),
                "mode": "instruct2" if instruct else "zero_shot",
            },
        )

        # Drop transient Python garbage before a multi-GB activation peak.
        gc.collect()
        try:
            import torch

            if hasattr(torch, "inference_mode"):
                inference_ctx = torch.inference_mode()
            else:
                inference_ctx = torch.no_grad()
        except Exception:
            from contextlib import nullcontext

            inference_ctx = nullcontext()

        with self._lock, inference_ctx:
            chunks: list[np.ndarray] = []
            # stream=True on CPU keeps flow/HiFT windows smaller (lower peak RSS).
            use_stream = str(os.environ.get("COSYVOICE_STREAM", "true")).lower() in (
                "1",
                "true",
                "yes",
            )
            if instruct:
                generator = self._model.inference_instruct2(
                    text,
                    instruct,
                    str(prompt_wav),
                    stream=use_stream,
                )
            else:
                if self._generation == "cosyvoice3":
                    full_prompt = (
                        f"{settings.system_prompt_prefix}"
                        f"{settings.end_of_prompt_token}"
                        f"{prompt_text}"
                    )
                else:
                    # CosyVoice2 zero-shot expects plain prompt transcript text.
                    full_prompt = prompt_text
                generator = self._model.inference_zero_shot(
                    text,
                    full_prompt,
                    str(prompt_wav),
                    stream=use_stream,
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
                "generation": self._generation,
                "language": lang_key,
                "speaker": request.speaker,
            },
        )

    def shutdown(self) -> None:
        with self._lock:
            self._model = None
            self._ready = False
            self._prompt_extract_cache.clear()
            gc.collect()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _patch_sync_llm_tts(self) -> None:
        """Replace CosyVoice model.tts so llm_job runs on the caller thread.

        Upstream CosyVoice runs ``llm_job`` in a daemon Thread and optionally
        streams ``token2wav`` in hop windows. On Apple Silicon Podman the
        worker thread is flaky, so we keep LLM work inline.

        Important: still honor ``stream=True``. The previous one-shot
        ``token2wav(finalize=True)`` path ignored streaming and ran flow/HiFT
        over the full token sequence at once — that spikes RSS and dies on
        12 GB VMs for longer text even when short lines succeed.
        """
        model = getattr(self._model, "model", None)
        if model is None or not hasattr(model, "tts") or not hasattr(model, "llm_job"):
            return
        if getattr(model.tts, "_tts_sync_patched", False):
            return

        import uuid

        import torch

        original_token2wav = model.token2wav
        lock = model.lock

        def tts_sync(
            text=torch.zeros(1, 0, dtype=torch.int32),
            flow_embedding=torch.zeros(0, 192),
            llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80),
            source_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            stream: bool = False,
            speed: float = 1.0,
            **kwargs: Any,
        ):
            this_uuid = str(uuid.uuid1())
            with lock:
                model.tts_speech_token_dict[this_uuid] = []
                model.llm_end_dict[this_uuid] = False
                model.hift_cache_dict[this_uuid] = None
            # Inline LLM / VC job (no threading.Thread).
            if source_speech_token.shape[1] == 0:
                model.llm_job(
                    text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid
                )
            else:
                model.vc_job(source_speech_token, this_uuid)

            tokens = model.tts_speech_token_dict[this_uuid]
            try:
                if stream and speed == 1.0 and hasattr(model, "token_hop_len"):
                    # Mirror CosyVoice2Model.tts stream branch, but tokens are
                    # already fully generated (llm_job finished above).
                    base_hop = int(getattr(model, "token_hop_len", 25) or 25)
                    max_hop = int(
                        getattr(model, "token_max_hop_len", 4 * base_hop) or (4 * base_hop)
                    )
                    scale = float(getattr(model, "stream_scale_factor", 2) or 2)
                    # Optional tighter hops for low-RAM hosts (env override).
                    try:
                        env_hop = int(os.environ.get("COSYVOICE_TOKEN_HOP_LEN", "0") or "0")
                    except ValueError:
                        env_hop = 0
                    hop = env_hop if env_hop > 0 else base_hop
                    if env_hop > 0:
                        max_hop = min(max_hop, max(hop, env_hop * 4))

                    pre_look = int(getattr(model.flow, "pre_lookahead_len", 0) or 0)
                    prompt_token_pad = int(
                        np.ceil(flow_prompt_speech_token.shape[1] / hop) * hop
                        - flow_prompt_speech_token.shape[1]
                    )
                    token_offset = 0
                    saved_hop = getattr(model, "token_hop_len", hop)
                    try:
                        model.token_hop_len = hop
                        while True:
                            this_hop = (
                                model.token_hop_len + prompt_token_pad
                                if token_offset == 0
                                else model.token_hop_len
                            )
                            if len(tokens) - token_offset >= this_hop + pre_look:
                                this_tts_speech_token = torch.tensor(
                                    tokens[
                                        : token_offset + this_hop + pre_look
                                    ]
                                ).unsqueeze(dim=0)
                                this_tts_speech = original_token2wav(
                                    token=this_tts_speech_token,
                                    prompt_token=flow_prompt_speech_token,
                                    prompt_feat=prompt_speech_feat,
                                    embedding=flow_embedding,
                                    token_offset=token_offset,
                                    uuid=this_uuid,
                                    stream=True,
                                    finalize=False,
                                )
                                token_offset += this_hop
                                model.token_hop_len = min(
                                    max_hop,
                                    int(model.token_hop_len * scale),
                                )
                                yield {"tts_speech": this_tts_speech.cpu()}
                                # Free hop activations between windows.
                                del this_tts_speech, this_tts_speech_token
                                gc.collect()
                            else:
                                break
                        # Remaining tokens (finalize).
                        this_tts_speech_token = torch.tensor(tokens).unsqueeze(dim=0)
                        this_tts_speech = original_token2wav(
                            token=this_tts_speech_token,
                            prompt_token=flow_prompt_speech_token,
                            prompt_feat=prompt_speech_feat,
                            embedding=flow_embedding,
                            token_offset=token_offset,
                            uuid=this_uuid,
                            finalize=True,
                        )
                        yield {"tts_speech": this_tts_speech.cpu()}
                    finally:
                        model.token_hop_len = saved_hop
                else:
                    # Non-stream / speed-change: one-shot finalize (upstream rule).
                    this_tts_speech_token = torch.tensor(tokens).unsqueeze(dim=0)
                    this_tts_speech = original_token2wav(
                        token=this_tts_speech_token,
                        prompt_token=flow_prompt_speech_token,
                        prompt_feat=prompt_speech_feat,
                        embedding=flow_embedding,
                        token_offset=0,
                        uuid=this_uuid,
                        finalize=True,
                        speed=speed,
                    )
                    yield {"tts_speech": this_tts_speech.cpu()}
            finally:
                with lock:
                    model.tts_speech_token_dict.pop(this_uuid, None)
                    model.llm_end_dict.pop(this_uuid, None)
                    model.hift_cache_dict.pop(this_uuid, None)

        tts_sync._tts_sync_patched = True  # type: ignore[attr-defined]
        model.tts = tts_sync  # type: ignore[method-assign]
        logger.info(
            "Patched CosyVoice model.tts to run LLM inline with stream-aware token2wav"
        )

    def _install_prompt_feature_cache(self) -> None:
        """Cache speech-tokenizer / campplus outputs per prompt wav path.

        Avoids re-running heavy ONNX / feature extract on every request. Disk-
        hydrated entries (default prompt) make the first Quick TTS safe.
        """
        frontend = getattr(self._model, "frontend", None)
        if frontend is None:
            return

        cache = self._prompt_extract_cache
        lock = self._lock

        def _lookup(prompt_wav: Any) -> dict[str, Any]:
            key = str(prompt_wav)
            if key in cache:
                return cache[key]
            # Also try basename / resolved forms
            try:
                p = Path(str(prompt_wav))
                for alt in (str(p.resolve()), p.name, f"/opt/CosyVoice/asset/{p.name}"):
                    if alt in cache:
                        return cache[alt]
            except Exception:
                pass
            return cache.setdefault(key, {})

        def _wrap(method_name: str, cache_key: str) -> None:
            original = getattr(frontend, method_name, None)
            if original is None or getattr(original, "_tts_cached", False):
                return

            def cached(prompt_wav: Any, *args: Any, **kwargs: Any) -> Any:
                with lock:
                    bucket = _lookup(prompt_wav)
                    if cache_key in bucket:
                        return bucket[cache_key]
                value = original(prompt_wav, *args, **kwargs)
                with lock:
                    _lookup(prompt_wav)[cache_key] = value
                return value

            cached._tts_cached = True  # type: ignore[attr-defined]
            setattr(frontend, method_name, cached)

        _wrap("_extract_speech_token", "speech_token")
        _wrap("_extract_speech_feat", "speech_feat")
        _wrap("_extract_spk_embedding", "spk_embedding")
        logger.info("Installed CosyVoice prompt feature cache")

    def _default_prompt_path(self) -> Path | None:
        settings = self._settings
        path = settings.default_prompt_path
        if path.is_file():
            return path
        alt = settings.cosyvoice_repo / "asset" / "zero_shot_prompt.wav"
        return alt if alt.is_file() else None

    def _hydrate_prompt_cache_from_disk(self) -> None:
        """Load disk-backed default prompt features into the in-memory cache."""
        settings = self._settings
        path = self._default_prompt_path()
        if path is None:
            return
        from app.services.prompt_features import prompt_feature_cache_path

        disk = prompt_feature_cache_path(
            settings.cache_dir, path, model_dir=settings.model_path
        )
        data = load_prompt_features(disk)
        if data is None:
            logger.warning(
                "No disk prompt features; first TTS will run ONNX extract",
                extra={"path": str(disk)},
            )
            return
        key = str(path)
        # Match keys used by CosyVoice (str paths as passed to frontend methods).
        self._prompt_extract_cache[key] = {
            "speech_token": data["speech_token"],
            "speech_feat": data["speech_feat"],
            "spk_embedding": data["spk_embedding"],
        }
        # Also index by resolved / alternate string forms callers might pass.
        self._prompt_extract_cache[str(path.resolve())] = self._prompt_extract_cache[key]
        logger.info(
            "Hydrated prompt feature cache from disk",
            extra={"path": str(disk), "prompt": key},
        )

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

    def _init_opencc(self) -> None:
        """Optional simplified→traditional converter for Cantonese (ASLP recommendation)."""
        self._opencc = None
        try:
            from opencc import OpenCC  # type: ignore

            self._opencc = OpenCC("s2t")
            logger.info("OpenCC s2t ready for Cantonese text normalization")
        except Exception:
            logger.info(
                "OpenCC unavailable; Cantonese will use input text as-is "
                "(install opencc-python-reimplemented for s2t)"
            )

    def _prepare_text(self, text: str, lang_key: str) -> str:
        """Normalize text for the active model (Yue: s2t when OpenCC is present)."""
        if lang_key not in _YUE_LANG_KEYS or self._opencc is None:
            return text
        try:
            converted = self._opencc.convert(text)
            if converted and converted != text:
                logger.debug(
                    "Applied OpenCC s2t for Yue",
                    extra={"before_len": len(text), "after_len": len(converted)},
                )
            return converted or text
        except Exception:
            logger.exception("OpenCC convert failed; using original text")
            return text

    def _build_instruct(self, language: str, speed: float) -> str | None:
        settings = self._settings
        lang_key = (language or settings.default_language).lower().strip()
        speed_part = _speed_instruct(speed)

        if self._generation == "cosyvoice3":
            lang_part = LANGUAGE_INSTRUCT_V3.get(lang_key)
            if not lang_part and not speed_part:
                return None
            pieces = [settings.system_prompt_prefix]
            if lang_part:
                pieces.append(lang_part)
            if speed_part:
                pieces.append(speed_part)
            return " ".join(pieces) + settings.end_of_prompt_token

        # CosyVoice2 (incl. Yue fine-tunes): plain instruct string.
        # Default product language is yue — always prefer instruct2 for Yue.
        lang_part = LANGUAGE_INSTRUCT_V2.get(lang_key)
        if not lang_part and not speed_part:
            # Default language for this product is yue; still instruct when unset.
            if not lang_key or lang_key in _YUE_LANG_KEYS:
                return _YUE_INSTRUCT_V2
            return None
        pieces: list[str] = []
        if lang_part:
            pieces.append(lang_part)
        if speed_part:
            pieces.append(speed_part)
        return " ".join(pieces) if pieces else None

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
