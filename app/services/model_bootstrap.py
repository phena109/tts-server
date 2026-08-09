"""Coordinate model ensure + engine load with single-flight semantics."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from app.config.settings import Settings
from app.services.model_download_state import ModelDownloadState, ModelPhase
from app.services.model_manager import ModelManager
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ModelBootstrap:
    """Background single-flight model download + engine load coordinator."""

    def __init__(
        self,
        *,
        settings: Settings,
        manager: ModelManager,
        state: ModelDownloadState,
        load_engine: Callable[[], None],
    ) -> None:
        self.settings = settings
        self.manager = manager
        self.state = state
        self.load_engine = load_engine
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def _on_progress(self, **kwargs: Any) -> None:
        """Map manager ``_emit`` kwargs onto ``ModelDownloadState``."""
        phase = kwargs.get("phase")
        message = kwargs.get("message")
        progress = {
            k: kwargs[k]
            for k in ("bytes_downloaded", "bytes_total", "files_done", "files_total")
            if k in kwargs and kwargs[k] is not None
        }
        if phase is not None:
            try:
                self.state.set_phase(ModelPhase(str(phase)), message=message)
            except ValueError:
                if message is not None or progress:
                    self.state.set_progress(message=message, **progress)
                return
        if progress or message is not None:
            # When phase was set above, still apply byte/file progress without
            # overwriting the phase message unless only progress fields exist.
            self.state.set_progress(
                message=message if phase is None else None,
                **progress,
            )

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the background ensure finishes (or timeout)."""
        t = self._thread
        if t is None:
            return True
        t.join(timeout=timeout)
        return not t.is_alive()

    def ensure_async(self) -> dict[str, Any]:
        """Start ensure in background if needed. Returns flags for HTTP layer."""
        with self._lock:
            if self.state.is_ready() and self.manager.is_model_present():
                return {"already_ready": True, "started": False}

            if self.state.is_active() or (self._thread is not None and self._thread.is_alive()):
                return {"already_running": True, "started": False}

            if self.settings.skip_model_download and not self.manager.is_model_present():
                msg = (
                    f"Model missing or incomplete at {self.manager.model_path} "
                    "and SKIP_MODEL_DOWNLOAD=true"
                )
                self.state.set_error(msg)
                return {"conflict": True, "started": False, "error": msg}

            self.manager.on_progress = self._on_progress
            # Clear prior error (or idle) so Retry 202 responses show checking, not error.
            self.state.set_phase(ModelPhase.CHECKING, message="Checking model files")

            self._thread = threading.Thread(
                target=self._run,
                name="model-bootstrap",
                daemon=True,
            )
            self._thread.start()
            return {"started": True, "already_running": False, "already_ready": False}

    def _run(self) -> None:
        try:
            self.state.set_phase(ModelPhase.CHECKING, message="Checking model files")
            path = self.manager.ensure_model()
            self.state.set_phase(ModelPhase.VERIFYING, message="Verifying model completeness")
            if not self.manager.is_model_present():
                missing = self.manager.describe_missing()
                raise RuntimeError(f"Model incomplete after ensure: {missing}")

            self.state.set_phase(ModelPhase.LOADING_ENGINE, message="Loading TTS engine")
            self.load_engine()
            self.state.set_phase(ModelPhase.READY, message="Model ready")
            logger.info("Model bootstrap complete", extra={"path": str(path)})
        except Exception as exc:
            logger.exception("Model bootstrap failed")
            self.state.set_error(str(exc))
