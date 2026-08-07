"""Model download and verification.

Models are NOT baked into the image. On first start the manager downloads
weights into the mounted ``/models`` volume and skips work when already present.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from app.config.settings import Settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Files that indicate a usable CosyVoice3 checkpoint tree.
# Upstream layout varies slightly; any strong signal is enough.
_MODEL_MARKERS: tuple[str, ...] = (
    "cosyvoice.yaml",
    "cosyvoice2.yaml",
    "cosyvoice3.yaml",
    "llm.pt",
    "llm.onnx",
    "flow.pt",
    "hift.pt",
    "config.json",
)


class ModelManager:
    """Ensure model weights exist under ``settings.model_path``."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def model_path(self) -> Path:
        return self.settings.model_path

    def is_model_present(self) -> bool:
        root = self.model_path
        if not root.is_dir():
            return False
        return any((root / name).exists() for name in _MODEL_MARKERS) or self._has_weights(
            root
        )

    def ensure_model(self) -> Path:
        """Download the model if missing; return local path."""
        self.settings.ensure_directories()
        path = self.model_path

        if self.is_model_present():
            logger.info(
                "Model already present, skipping download",
                extra={
                    "model": self.settings.resolved_model_name,
                    "path": str(path),
                },
            )
            return path

        if self.settings.skip_model_download:
            raise FileNotFoundError(
                f"Model missing at {path} and SKIP_MODEL_DOWNLOAD=true"
            )

        logger.info(
            "Downloading model",
            extra={
                "model": self.settings.resolved_model_name,
                "path": str(path),
                "source": self.settings.download_source,
            },
        )
        path.mkdir(parents=True, exist_ok=True)

        if self.settings.download_source == "modelscope":
            self._download_modelscope(path)
        else:
            self._download_huggingface(path)

        if not self.is_model_present():
            raise RuntimeError(
                f"Model download completed but no recognized weights found in {path}"
            )

        logger.info(
            "Model download complete",
            extra={
                "model": self.settings.resolved_model_name,
                "path": str(path),
            },
        )
        return path

    def _download_huggingface(self, local_dir: Path) -> None:
        # Prefer cache on the mounted volume so rebuilds stay warm
        cache = self.settings.cache_dir / "huggingface"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache / "hub"))

        from huggingface_hub import snapshot_download

        token = self.settings.hf_token or os.environ.get("HF_TOKEN")
        kwargs = {
            "repo_id": self.settings.resolved_model_name,
            "local_dir": str(local_dir),
            "token": token,
        }
        # resume_download removed in newer huggingface_hub; keep when available
        try:
            snapshot_download(**kwargs, resume_download=True)  # type: ignore[call-arg]
        except TypeError:
            snapshot_download(**kwargs)

    def _download_modelscope(self, local_dir: Path) -> None:
        # ModelScope id often mirrors FunAudioLLM/Fun-CosyVoice3-0.5B-2512
        from modelscope import snapshot_download as ms_snapshot_download

        ms_snapshot_download(
            self.settings.resolved_model_name,
            local_dir=str(local_dir),
        )

    @staticmethod
    def _has_weights(root: Path) -> bool:
        """Fallback: any reasonably large .pt/.onnx/.safetensors file."""
        patterns: Iterable[str] = ("*.pt", "*.pth", "*.onnx", "*.safetensors", "*.bin")
        for pattern in patterns:
            for file in root.rglob(pattern):
                try:
                    if file.stat().st_size > 1_000_000:
                        return True
                except OSError:
                    continue
        return False
