"""Thread-safe live model download / readiness state."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class ModelPhase(StrEnum):
    IDLE = "idle"
    CHECKING = "checking"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    LOADING_ENGINE = "loading_engine"
    READY = "ready"
    ERROR = "error"


_ACTIVE_PHASES = frozenset(
    {
        ModelPhase.CHECKING,
        ModelPhase.DOWNLOADING,
        ModelPhase.VERIFYING,
        ModelPhase.LOADING_ENGINE,
    }
)


class ModelDownloadState:
    """Process-wide download/readiness status for API and UI polling."""

    def __init__(
        self,
        *,
        model: str,
        path: str,
        download_source: str,
    ) -> None:
        self._lock = threading.RLock()
        self._model = model
        self._path = path
        self._download_source = download_source
        self._phase: ModelPhase = ModelPhase.IDLE
        self._message: str = "Waiting"
        self._bytes_downloaded: int | None = None
        self._bytes_total: int | None = None
        self._files_done: int | None = None
        self._files_total: int | None = None
        self._error: str | None = None
        self._started_at: str | None = None
        self._updated_at: str = _utc_now_iso()

    @property
    def phase(self) -> ModelPhase:
        with self._lock:
            return self._phase

    def is_active(self) -> bool:
        with self._lock:
            return self._phase in _ACTIVE_PHASES

    def is_ready(self) -> bool:
        with self._lock:
            return self._phase == ModelPhase.READY

    def set_phase(self, phase: ModelPhase, *, message: str | None = None) -> None:
        with self._lock:
            now = _utc_now_iso()
            if phase in _ACTIVE_PHASES and self._phase not in _ACTIVE_PHASES:
                self._started_at = now
                self._error = None
            if phase == ModelPhase.READY:
                self._error = None
            self._phase = phase
            if message is not None:
                self._message = message
            self._updated_at = now

    def set_progress(
        self,
        *,
        bytes_downloaded: int | None = None,
        bytes_total: int | None = None,
        files_done: int | None = None,
        files_total: int | None = None,
        message: str | None = None,
    ) -> None:
        with self._lock:
            if bytes_downloaded is not None:
                self._bytes_downloaded = bytes_downloaded
            if bytes_total is not None:
                self._bytes_total = bytes_total
            if files_done is not None:
                self._files_done = files_done
            if files_total is not None:
                self._files_total = files_total
            if message is not None:
                self._message = message
            self._updated_at = _utc_now_iso()

    def set_error(self, error: str, *, message: str | None = None) -> None:
        with self._lock:
            self._phase = ModelPhase.ERROR
            self._error = error
            self._message = message if message is not None else error
            self._updated_at = _utc_now_iso()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            pct: float | None = None
            if (
                self._bytes_downloaded is not None
                and self._bytes_total is not None
                and self._bytes_total > 0
            ):
                pct = round(100.0 * self._bytes_downloaded / self._bytes_total, 1)
                pct = max(0.0, min(100.0, pct))
            return {
                "phase": str(self._phase),
                "ready": self._phase == ModelPhase.READY,
                "model": self._model,
                "path": self._path,
                "message": self._message,
                "bytes_downloaded": self._bytes_downloaded,
                "bytes_total": self._bytes_total,
                "progress_pct": pct,
                "files_done": self._files_done,
                "files_total": self._files_total,
                "error": self._error,
                "started_at": self._started_at,
                "updated_at": self._updated_at,
                "download_source": self._download_source,
            }
