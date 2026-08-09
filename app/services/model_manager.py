"""Model download and verification.

Models are NOT baked into the image. On first start the manager downloads
weights into the mounted ``/models`` volume and skips work when already present.

Incomplete / interrupted downloads (common with multi-GB HF LFS assets) are
detected and resumed rather than treated as ready.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from app.config.settings import Settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Minimum expected sizes (bytes) for CosyVoice3 core weights.
# Values are intentionally well below published sizes so minor packing
# differences still pass, but partial/truncated files fail.
_COSYVOICE3_REQUIRED: dict[str, int] = {
    "cosyvoice3.yaml": 1_000,
    "llm.pt": 500_000_000,  # published ~2.0 GB
    "flow.pt": 200_000_000,  # published ~1.3 GB
    "hift.pt": 10_000_000,  # published ~83 MB
    "campplus.onnx": 1_000_000,  # published ~28 MB
    "speech_tokenizer_v3.onnx": 50_000_000,  # published ~969 MB
}

# CosyVoice / CosyVoice2 fall back markers (older layouts).
_LEGACY_WEIGHT_CANDIDATES: tuple[str, ...] = (
    "llm.pt",
    "llm.onnx",
    "flow.pt",
    "hift.pt",
)

_LEGACY_CONFIG_MARKERS: tuple[str, ...] = (
    "cosyvoice.yaml",
    "cosyvoice2.yaml",
)

# Qwen blank encoder weights required by CosyVoice3 yaml overrides.
_BLANK_EN_WEIGHTS: tuple[str, ...] = (
    "model.safetensors",
    "pytorch_model.bin",
)


class ModelManager:
    """Ensure model weights exist under ``settings.model_path``."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def model_path(self) -> Path:
        return self.settings.model_path

    def is_model_present(self) -> bool:
        """Return True only when the tree looks complete enough to load."""
        root = self.model_path
        if not root.is_dir():
            return False

        if self._has_interrupted_download(root):
            logger.warning(
                "Interrupted download detected under model path; will resume",
                extra={"path": str(root)},
            )
            return False

        # CosyVoice3 is the default product target.
        if (root / "cosyvoice3.yaml").is_file():
            return self._is_cosyvoice3_complete(root)

        # Older CosyVoice 1 / 2 layouts.
        if any((root / name).is_file() for name in _LEGACY_CONFIG_MARKERS):
            return self._is_legacy_complete(root)

        # Unknown layout: require at least one large weight file.
        return self._has_weights(root)

    def ensure_model(self) -> Path:
        """Download the model if missing or incomplete; return local path."""
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
                f"Model missing or incomplete at {path} and SKIP_MODEL_DOWNLOAD=true"
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
            missing = self._describe_missing(path)
            raise RuntimeError(
                f"Model download completed but checkpoint is still incomplete at {path}. "
                f"Missing or undersized: {missing}"
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
        # Modern huggingface_hub always resumes partial downloads when possible.
        snapshot_download(
            repo_id=self.settings.resolved_model_name,
            local_dir=str(local_dir),
            token=token,
        )

    def _download_modelscope(self, local_dir: Path) -> None:
        # ModelScope id often mirrors FunAudioLLM/Fun-CosyVoice3-0.5B-2512
        from modelscope import snapshot_download as ms_snapshot_download

        ms_snapshot_download(
            self.settings.resolved_model_name,
            local_dir=str(local_dir),
        )

    # ------------------------------------------------------------------
    # Completeness helpers
    # ------------------------------------------------------------------

    @classmethod
    def _is_cosyvoice3_complete(cls, root: Path) -> bool:
        for rel, min_size in _COSYVOICE3_REQUIRED.items():
            if not cls._file_ok(root / rel, min_size):
                return False
        blank = root / "CosyVoice-BlankEN"
        if not blank.is_dir():
            return False
        return any(cls._file_ok(blank / name, 50_000_000) for name in _BLANK_EN_WEIGHTS)

    @classmethod
    def _is_legacy_complete(cls, root: Path) -> bool:
        has_weight = any(
            cls._file_ok(root / name, 1_000_000) for name in _LEGACY_WEIGHT_CANDIDATES
        )
        return has_weight

    @staticmethod
    def _file_ok(path: Path, min_size: int) -> bool:
        try:
            return path.is_file() and path.stat().st_size >= min_size
        except OSError:
            return False

    @staticmethod
    def _has_interrupted_download(root: Path) -> bool:
        """Hugging Face leaves ``*.incomplete`` (and sometimes locks) mid-transfer."""
        try:
            for path in root.rglob("*"):
                name = path.name
                if name.endswith(".incomplete"):
                    return True
                # Stale lock files without a finished sibling weight indicate crash mid-download.
                if name.endswith(".lock") and path.is_file():
                    # Only treat locks under the HF download cache as interruption signals.
                    if ".cache" in path.parts or "download" in path.parts:
                        return True
        except OSError:
            return False
        return False

    @classmethod
    def _describe_missing(cls, root: Path) -> str:
        if not root.is_dir():
            return "directory missing"
        problems: list[str] = []
        if cls._has_interrupted_download(root):
            problems.append("interrupted *.incomplete / *.lock files")
        if (root / "cosyvoice3.yaml").is_file() or not any(
            (root / n).is_file() for n in _LEGACY_CONFIG_MARKERS
        ):
            for rel, min_size in _COSYVOICE3_REQUIRED.items():
                path = root / rel
                if not path.is_file():
                    problems.append(f"{rel} (missing)")
                else:
                    try:
                        size = path.stat().st_size
                    except OSError:
                        size = 0
                    if size < min_size:
                        problems.append(f"{rel} ({size} B < {min_size} B)")
            blank = root / "CosyVoice-BlankEN"
            if not blank.is_dir():
                problems.append("CosyVoice-BlankEN/ (missing)")
            elif not any(
                cls._file_ok(blank / name, 50_000_000) for name in _BLANK_EN_WEIGHTS
            ):
                problems.append("CosyVoice-BlankEN model weights (missing/undersized)")
        return ", ".join(problems) if problems else "unknown"

    @staticmethod
    def _has_weights(root: Path) -> bool:
        """Fallback: any reasonably large .pt/.onnx/.safetensors file."""
        patterns: Iterable[str] = ("*.pt", "*.pth", "*.onnx", "*.safetensors", "*.bin")
        for pattern in patterns:
            for file in root.rglob(pattern):
                # Ignore HF cache / incomplete scratch files
                if ".cache" in file.parts or file.name.endswith(".incomplete"):
                    continue
                try:
                    if file.stat().st_size > 1_000_000:
                        return True
                except OSError:
                    continue
        return False
