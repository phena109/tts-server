"""Model download and verification.

Models are NOT baked into the image. On first start the manager downloads
weights into the mounted ``/models`` volume and skips work when already present.

Incomplete / interrupted downloads (common with multi-GB HF LFS assets) are
detected and resumed rather than treated as ready.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable, Iterable

from app.config.settings import Settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Minimum expected sizes (bytes) for core weights.
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

# CosyVoice2 (incl. ASLP-lab Cantonese fine-tunes) core layout.
_COSYVOICE2_REQUIRED: dict[str, int] = {
    "cosyvoice2.yaml": 1_000,
    "llm.pt": 500_000_000,  # published ~2.0 GB
    "flow.pt": 200_000_000,  # published ~450 MB
    "hift.pt": 10_000_000,  # published ~83 MB
    "campplus.onnx": 1_000_000,  # published ~28 MB
    "speech_tokenizer_v2.onnx": 50_000_000,  # published ~496 MB
}

# CosyVoice 1 fall back markers.
_LEGACY_WEIGHT_CANDIDATES: tuple[str, ...] = (
    "llm.pt",
    "llm.onnx",
    "flow.pt",
    "hift.pt",
)

_LEGACY_CONFIG_MARKERS: tuple[str, ...] = (
    "cosyvoice.yaml",
)

# Qwen blank encoder weights required by CosyVoice2/3 yaml overrides.
_BLANK_EN_WEIGHTS: tuple[str, ...] = (
    "model.safetensors",
    "pytorch_model.bin",
)


class ModelManager:
    """Ensure model weights exist under ``settings.model_path``."""

    def __init__(
        self,
        settings: Settings,
        on_progress: Callable[..., None] | None = None,
    ) -> None:
        self.settings = settings
        self.on_progress = on_progress

    def _emit(self, **kwargs: object) -> None:
        if self.on_progress is None:
            return
        try:
            self.on_progress(**kwargs)
        except Exception:
            logger.exception("Progress callback failed")

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

        # CosyVoice3 layout.
        if (root / "cosyvoice3.yaml").is_file():
            return self._is_cosyvoice3_complete(root)

        # CosyVoice2 (default product target: ASLP-lab Yue fine-tunes).
        if (root / "cosyvoice2.yaml").is_file():
            return self._is_cosyvoice2_complete(root)

        # Older CosyVoice 1 layout.
        if any((root / name).is_file() for name in _LEGACY_CONFIG_MARKERS):
            return self._is_legacy_complete(root)

        # Unknown layout: require at least one large weight file.
        return self._has_weights(root)

    def describe_missing(self) -> str:
        """Human-readable summary of missing/undersized files under model_path."""
        return self._describe_missing(self.model_path)

    def ensure_model(self) -> Path:
        """Download the model if missing or incomplete; return local path."""
        self.settings.ensure_directories()
        path = self.model_path

        self._emit(phase="checking", message="Checking model files")

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

        self._emit(phase="downloading", message="Downloading model weights")

        if self.settings.download_source == "modelscope":
            self._download_modelscope(path)
        else:
            self._download_huggingface(path)

        self._emit(phase="verifying", message="Verifying model completeness")

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

        def _do_download() -> None:
            # Modern huggingface_hub always resumes partial downloads when possible.
            snapshot_download(
                repo_id=self.settings.resolved_model_name,
                local_dir=str(local_dir),
                token=token,
            )

        self._download_with_byte_scanner(local_dir, _do_download)

    def _download_modelscope(self, local_dir: Path) -> None:
        # ModelScope id often mirrors FunAudioLLM/Fun-CosyVoice3-0.5B-2512
        from modelscope import snapshot_download as ms_snapshot_download

        def _do_download() -> None:
            ms_snapshot_download(
                self.settings.resolved_model_name,
                local_dir=str(local_dir),
            )

        self._download_with_byte_scanner(local_dir, _do_download)

    def _download_with_byte_scanner(
        self, local_dir: Path, download_fn: Callable[[], None]
    ) -> None:
        stop = threading.Event()

        def loop() -> None:
            while not stop.wait(2.0):
                nbytes = self._scan_downloaded_bytes(local_dir)
                self._emit(
                    phase="downloading",
                    message=f"Downloading model weights ({nbytes} bytes so far)",
                    bytes_downloaded=nbytes,
                )

        t = threading.Thread(target=loop, name="model-dl-progress", daemon=True)
        t.start()
        try:
            download_fn()
        finally:
            stop.set()
            t.join(timeout=5)
            nbytes = self._scan_downloaded_bytes(local_dir)
            self._emit(
                phase="downloading",
                message=f"Download finished ({nbytes} bytes on disk)",
                bytes_downloaded=nbytes,
            )

    @staticmethod
    def _scan_downloaded_bytes(root: Path) -> int:
        """Sum sizes of files under root (include *.incomplete; skip .lock)."""
        total = 0
        if not root.is_dir():
            return 0
        try:
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                if path.name.endswith(".lock"):
                    continue
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
        except OSError:
            return total
        return total

    # ------------------------------------------------------------------
    # Completeness helpers
    # ------------------------------------------------------------------

    @classmethod
    def _is_cosyvoice3_complete(cls, root: Path) -> bool:
        for rel, min_size in _COSYVOICE3_REQUIRED.items():
            if not cls._file_ok(root / rel, min_size):
                return False
        return cls._blank_en_ok(root)

    @classmethod
    def _is_cosyvoice2_complete(cls, root: Path) -> bool:
        for rel, min_size in _COSYVOICE2_REQUIRED.items():
            if not cls._file_ok(root / rel, min_size):
                return False
        return cls._blank_en_ok(root)

    @classmethod
    def _blank_en_ok(cls, root: Path) -> bool:
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

        if (root / "cosyvoice2.yaml").is_file() and not (
            root / "cosyvoice3.yaml"
        ).is_file():
            required = _COSYVOICE2_REQUIRED
        else:
            required = _COSYVOICE3_REQUIRED

        for rel, min_size in required.items():
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
