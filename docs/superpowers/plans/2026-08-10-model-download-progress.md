# Model Download Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make multi‑GB model download visible and controllable via pollable API status, richer `/health`, background auto-download after early uvicorn bind, `POST /model/ensure`, and a web UI progress banner with retry.

**Architecture:** Process-wide thread-safe `ModelDownloadState` updated by `ModelManager` / a small `ModelBootstrap` coordinator. FastAPI lifespan no longer blocks on download; a background thread runs ensure → verify → load engine. Routes always serve status; TTS returns 503 until ready. Entrypoint skips blocking `ensure-model` by default.

**Tech Stack:** Python 3, FastAPI, uvicorn, pydantic, huggingface_hub `snapshot_download`, static web UI (`web/`), pytest.

**Spec:** `docs/superpowers/specs/2026-08-10-model-download-progress-design.md`

---

## File map

| File | Responsibility |
|------|----------------|
| **Create** `app/services/model_download_state.py` | Thread-safe phase/progress snapshot |
| **Create** `app/services/model_bootstrap.py` | Single-flight ensure worker: download → verify → load engine → update state |
| **Create** `app/api/routes_model.py` | `GET /model/status`, `POST /model/ensure` |
| **Create** `tests/test_model_download_state.py` | State unit tests |
| **Create** `tests/test_model_bootstrap.py` | Bootstrap/idempotency unit tests |
| **Create** `tests/test_model_routes.py` | API route tests with TestClient |
| **Modify** `app/models/schemas.py` | `ModelStatusResponse`, extend `HealthResponse` |
| **Modify** `app/services/model_manager.py` | Progress callbacks + optional disk byte scan during download |
| **Modify** `app/main.py` | Early-bind lifespan, wire bootstrap, register routes |
| **Modify** `app/api/routes_health.py` | Expose model phase fields |
| **Modify** `app/api/routes_tts.py` | 503 when TTS service missing / not ready |
| **Modify** `app/api/deps.py` | Optional helpers for state/bootstrap |
| **Modify** `entrypoint.sh` | Default: no blocking ensure-model; optional `ENSURE_MODEL_IN_ENTRYPOINT` |
| **Modify** `web/index.html`, `web/app.js`, `web/styles.css` | Banner, poll, retry, disable TTS |
| **Modify** `README.md` | Document new behavior and endpoints |
| **Modify** `tests/test_model_manager.py` | Keep existing; add progress-callback smoke if needed |

---

### Task 1: `ModelDownloadState` (TDD)

**Files:**
- Create: `app/services/model_download_state.py`
- Create: `tests/test_model_download_state.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Unit tests for ModelDownloadState."""

from __future__ import annotations

import threading

from app.services.model_download_state import ModelDownloadState, ModelPhase


def test_initial_snapshot_is_idle() -> None:
    state = ModelDownloadState(
        model="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        path="/models/Fun-CosyVoice3-0.5B",
        download_source="huggingface",
    )
    snap = state.snapshot()
    assert snap["phase"] == ModelPhase.IDLE
    assert snap["ready"] is False
    assert snap["model"] == "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
    assert snap["error"] is None
    assert snap["progress_pct"] is None
    assert snap["updated_at"]


def test_phase_transitions_and_progress() -> None:
    state = ModelDownloadState(model="m", path="/p", download_source="huggingface")
    state.set_phase(ModelPhase.DOWNLOADING, message="Downloading…")
    state.set_progress(bytes_downloaded=500, bytes_total=1000, files_done=1, files_total=2)
    snap = state.snapshot()
    assert snap["phase"] == ModelPhase.DOWNLOADING
    assert snap["message"] == "Downloading…"
    assert snap["bytes_downloaded"] == 500
    assert snap["bytes_total"] == 1000
    assert snap["progress_pct"] == 50.0
    assert snap["files_done"] == 1
    assert snap["files_total"] == 2
    assert snap["started_at"] is not None

    state.set_phase(ModelPhase.READY, message="Ready")
    snap = state.snapshot()
    assert snap["ready"] is True
    assert snap["phase"] == ModelPhase.READY


def test_set_error_sets_phase() -> None:
    state = ModelDownloadState(model="m", path="/p", download_source="huggingface")
    state.set_error("boom")
    snap = state.snapshot()
    assert snap["phase"] == ModelPhase.ERROR
    assert snap["error"] == "boom"
    assert snap["ready"] is False


def test_concurrent_snapshots_do_not_raise() -> None:
    state = ModelDownloadState(model="m", path="/p", download_source="huggingface")
    errors: list[BaseException] = []

    def writer() -> None:
        try:
            for i in range(200):
                state.set_progress(bytes_downloaded=i, bytes_total=200)
                state.set_phase(ModelPhase.DOWNLOADING, message=f"n={i}")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def reader() -> None:
        try:
            for _ in range(200):
                snap = state.snapshot()
                assert "phase" in snap
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model_download_state.py -v`  
Expected: FAIL with `ModuleNotFoundError` or import error for `model_download_state`.

- [ ] **Step 3: Implement `ModelDownloadState`**

Create `app/services/model_download_state.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model_download_state.py -v`  
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/model_download_state.py tests/test_model_download_state.py
git commit -m "feat: add thread-safe ModelDownloadState for progress polling"
```

---

### Task 2: Status schemas

**Files:**
- Modify: `app/models/schemas.py`

- [ ] **Step 1: Add response models**

Append to `app/models/schemas.py` (keep existing models):

```python
from typing import Literal  # already imported

class ModelStatusResponse(BaseModel):
    """Live model download / readiness snapshot."""

    phase: str
    ready: bool = False
    model: str | None = None
    path: str | None = None
    message: str = ""
    bytes_downloaded: int | None = None
    bytes_total: int | None = None
    progress_pct: float | None = None
    files_done: int | None = None
    files_total: int | None = None
    error: str | None = None
    started_at: str | None = None
    updated_at: str | None = None
    download_source: str | None = None
    # Present on POST /model/ensure responses
    already_running: bool | None = None
    already_ready: bool | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    engine: str | None = None
    model: str | None = None
    ready: bool = True
    model_phase: str | None = None
    model_ready: bool | None = None
```

Replace the existing `HealthResponse` class with the extended version above (do not leave two definitions).

- [ ] **Step 2: Quick import check**

Run: `python -c "from app.models.schemas import ModelStatusResponse, HealthResponse; print(ModelStatusResponse.model_json_schema()['properties'].keys())"`  
Expected: includes `phase`, `progress_pct`, and health includes `model_phase`.

- [ ] **Step 3: Commit**

```bash
git add app/models/schemas.py
git commit -m "feat: add ModelStatusResponse and extend HealthResponse"
```

---

### Task 3: Progress hooks on `ModelManager`

**Files:**
- Modify: `app/services/model_manager.py`
- Modify: `tests/test_model_manager.py` (small addition optional)

- [ ] **Step 1: Extend `ModelManager` constructor and `ensure_model`**

Changes to `app/services/model_manager.py`:

1. Import `Callable` and optional state types.
2. Accept optional `on_progress: Callable[..., None] | None = None` and store it.
3. Add helper methods used during download:

```python
from typing import Callable, Iterable  # update imports

# In __init__:
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
```

4. In `ensure_model`, before download:

```python
self._emit(phase="checking", message="Checking model files")
# ... existing present / skip checks ...
self._emit(phase="downloading", message="Downloading model weights")
# after download:
self._emit(phase="verifying", message="Verifying model completeness")
```

5. In `_download_huggingface`, pass a tqdm-compatible callback if the installed `huggingface_hub` supports it. Prefer:

```python
def _hf_progress_callback(self):
    """Return kwargs for snapshot_download progress if supported; else empty."""
    # Try tqdm_class that reports to on_progress; fall back to no hook.
    ...
```

Practical approach that works across hub versions:

- Wrap download in a daemon **progress scanner thread** that every 2s walks `local_dir` and sums sizes of non-`.incomplete` large files plus `*.incomplete`, calling:

```python
self._emit(
    phase="downloading",
    message="Downloading model weights",
    bytes_downloaded=total_bytes,
)
```

- Stop the scanner when `snapshot_download` returns.

Implement `_scan_downloaded_bytes(root: Path) -> int`:

```python
@staticmethod
def _scan_downloaded_bytes(root: Path) -> int:
    total = 0
    if not root.is_dir():
        return 0
    try:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if ".cache" in path.parts and not path.name.endswith(".incomplete"):
                # Count incomplete scratch; skip pure cache metadata if desired
                pass
            try:
                total += path.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total
```

Prefer counting: finished weight files under root **plus** `*.incomplete` under root (include `.cache` incompletes). Skip `.lock` files (size 0).

6. Same scanner around `_download_modelscope`.

Skeleton for scanner:

```python
def _download_with_byte_scanner(self, local_dir: Path, download_fn) -> None:
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
```

Add `import threading` at top of file.

- [ ] **Step 2: Run existing model manager tests**

Run: `pytest tests/test_model_manager.py -v`  
Expected: all PASS (constructor still works with one arg).

- [ ] **Step 3: Optional unit test for byte scanner**

Add to `tests/test_model_manager.py`:

```python
def test_scan_downloaded_bytes_sums_files(tmp_path: Path) -> None:
    root = tmp_path / "m"
    _write(root / "a.pt", 100)
    _write(root / "b.incomplete", 50)
    assert ModelManager._scan_downloaded_bytes(root) == 150
```

Run: `pytest tests/test_model_manager.py::test_scan_downloaded_bytes_sums_files -v`  
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add app/services/model_manager.py tests/test_model_manager.py
git commit -m "feat: emit download progress from ModelManager via callbacks"
```

---

### Task 4: `ModelBootstrap` coordinator (TDD)

**Files:**
- Create: `app/services/model_bootstrap.py`
- Create: `tests/test_model_bootstrap.py`

This owns: single-flight background ensure, wiring progress into state, and a pluggable “load engine” callback so tests never load CosyVoice.

- [ ] **Step 1: Write failing tests**

```python
"""Tests for ModelBootstrap single-flight ensure."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from app.config.settings import Settings
from app.services.model_bootstrap import ModelBootstrap
from app.services.model_download_state import ModelDownloadState, ModelPhase
from app.services.model_manager import ModelManager


def _settings(tmp_path: Path, **kwargs: object) -> Settings:
    base = dict(
        MODEL_DIR=str(tmp_path),
        MODEL_LOCAL_NAME="Fun-CosyVoice3-0.5B",
        SKIP_MODEL_DOWNLOAD=False,
        INPUT_DIR=str(tmp_path / "input"),
        OUTPUT_DIR=str(tmp_path / "output"),
        CACHE_DIR=str(tmp_path / "cache"),
    )
    base.update(kwargs)
    return Settings(**base)


def test_ensure_async_runs_download_and_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    state = ModelDownloadState(
        model=settings.resolved_model_name,
        path=str(settings.model_path),
        download_source=settings.download_source,
    )
    manager = ModelManager(settings)
    loaded = threading.Event()

    def fake_ensure() -> Path:
        state.set_phase(ModelPhase.DOWNLOADING, message="fake dl")
        time.sleep(0.05)
        return settings.model_path

    monkeypatch.setattr(manager, "ensure_model", fake_ensure)
    monkeypatch.setattr(manager, "is_model_present", lambda: True)

    def load_engine() -> None:
        loaded.set()

    boot = ModelBootstrap(settings=settings, manager=manager, state=state, load_engine=load_engine)
    result = boot.ensure_async()
    assert result["started"] is True
    assert boot.wait(timeout=5) is True
    assert loaded.is_set()
    assert state.is_ready()


def test_ensure_async_idempotent_while_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    state = ModelDownloadState(
        model="m", path=str(settings.model_path), download_source="huggingface"
    )
    manager = ModelManager(settings)
    gate = threading.Event()

    def fake_ensure() -> Path:
        gate.wait(timeout=5)
        return settings.model_path

    monkeypatch.setattr(manager, "ensure_model", fake_ensure)
    monkeypatch.setattr(manager, "is_model_present", lambda: True)

    boot = ModelBootstrap(
        settings=settings,
        manager=manager,
        state=state,
        load_engine=lambda: None,
    )
    r1 = boot.ensure_async()
    r2 = boot.ensure_async()
    assert r1["started"] is True
    assert r2["already_running"] is True
    gate.set()
    assert boot.wait(timeout=5) is True


def test_ensure_async_already_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    state = ModelDownloadState(
        model="m", path=str(settings.model_path), download_source="huggingface"
    )
    state.set_phase(ModelPhase.READY, message="ok")
    manager = ModelManager(settings)
    monkeypatch.setattr(manager, "is_model_present", lambda: True)
    boot = ModelBootstrap(
        settings=settings, manager=manager, state=state, load_engine=lambda: None
    )
    r = boot.ensure_async()
    assert r["already_ready"] is True


def test_skip_download_missing_returns_conflict(tmp_path: Path) -> None:
    settings = _settings(tmp_path, SKIP_MODEL_DOWNLOAD=True)
    state = ModelDownloadState(
        model="m", path=str(settings.model_path), download_source="huggingface"
    )
    manager = ModelManager(settings)
    boot = ModelBootstrap(
        settings=settings, manager=manager, state=state, load_engine=lambda: None
    )
    r = boot.ensure_async()
    assert r.get("conflict") is True
    assert state.phase == ModelPhase.ERROR or r["conflict"]
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `pytest tests/test_model_bootstrap.py -v`  
Expected: import error for `ModelBootstrap`.

- [ ] **Step 3: Implement `ModelBootstrap`**

Create `app/services/model_bootstrap.py`:

```python
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
        phase = kwargs.pop("phase", None)
        message = kwargs.pop("message", None)
        if phase is not None:
            try:
                self.state.set_phase(ModelPhase(phase), message=message)
            except ValueError:
                if message is not None:
                    self.state.set_progress(message=message, **{
                        k: kwargs[k] for k in list(kwargs) if k in {
                            "bytes_downloaded", "bytes_total", "files_done", "files_total"
                        }
                    })
        else:
            self.state.set_progress(message=message, **{
                k: v for k, v in kwargs.items()
                if k in {"bytes_downloaded", "bytes_total", "files_done", "files_total"}
            })
        # Always apply progress fields when present
        progress_keys = {
            k: kwargs[k]
            for k in ("bytes_downloaded", "bytes_total", "files_done", "files_total")
            if k in kwargs
        }
        if progress_keys or (message is not None and phase is None):
            self.state.set_progress(message=message if phase is None else None, **progress_keys)

    def wait(self, timeout: float | None = None) -> bool:
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

            # Wire progress callback for this run
            self.manager.on_progress = self._on_progress

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
                missing = self.manager._describe_missing(path)  # noqa: SLF001 — shared helper
                raise RuntimeError(f"Model incomplete after ensure: {missing}")

            self.state.set_phase(ModelPhase.LOADING_ENGINE, message="Loading TTS engine")
            self.load_engine()
            self.state.set_phase(ModelPhase.READY, message="Model ready")
            logger.info("Model bootstrap complete", extra={"path": str(path)})
        except Exception as exc:
            logger.exception("Model bootstrap failed")
            self.state.set_error(str(exc))
```

**Note:** Prefer adding a public `describe_missing(self) -> str` on `ModelManager` that wraps `_describe_missing(self.model_path)` and use that instead of `SLF001`. Add that one-liner in this task if not already present.

Simplify `_on_progress` during implementation so it is not double-applying fields — keep one clear path:

```python
def _on_progress(self, **kwargs: Any) -> None:
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
            if message is not None:
                self.state.set_progress(message=message, **progress)
                return
    if progress or message is not None:
        self.state.set_progress(message=message, **progress)
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_model_bootstrap.py tests/test_model_download_state.py -v`  
Expected: all PASS. Fix timing flakes with slightly longer sleeps if needed.

- [ ] **Step 5: Commit**

```bash
git add app/services/model_bootstrap.py app/services/model_manager.py tests/test_model_bootstrap.py
git commit -m "feat: add ModelBootstrap single-flight ensure coordinator"
```

---

### Task 5: Model API routes + health + TTS 503 (TDD)

**Files:**
- Create: `app/api/routes_model.py`
- Create: `tests/test_model_routes.py`
- Modify: `app/api/routes_health.py`
- Modify: `app/api/routes_tts.py`
- Modify: `app/main.py` (register router; lifespan skeleton enough for routes — full lifespan in Task 6)

- [ ] **Step 1: Write route tests with a minimal app**

```python
"""API tests for model status / ensure and health phase fields."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes_health import router as health_router
from app.api.routes_model import router as model_router
from app.api.routes_tts import router as tts_router
from app.config.settings import Settings
from app.services.model_bootstrap import ModelBootstrap
from app.services.model_download_state import ModelDownloadState, ModelPhase
from app.services.model_manager import ModelManager


def _settings(tmp_path: Path, **kwargs: object) -> Settings:
    base = dict(
        MODEL_DIR=str(tmp_path),
        MODEL_LOCAL_NAME="Fun-CosyVoice3-0.5B",
        SKIP_MODEL_DOWNLOAD=True,
        INPUT_DIR=str(tmp_path / "input"),
        OUTPUT_DIR=str(tmp_path / "output"),
        CACHE_DIR=str(tmp_path / "cache"),
    )
    base.update(kwargs)
    return Settings(**base)


def _app(tmp_path: Path, **kwargs: object) -> FastAPI:
    settings = _settings(tmp_path, **kwargs)
    state = ModelDownloadState(
        model=settings.resolved_model_name,
        path=str(settings.model_path),
        download_source=settings.download_source,
    )
    manager = ModelManager(settings)
    boot = ModelBootstrap(
        settings=settings,
        manager=manager,
        state=state,
        load_engine=lambda: None,
    )
    app = FastAPI()
    app.state.settings = settings
    app.state.model_state = state
    app.state.model_manager = manager
    app.state.model_bootstrap = boot
    app.state.engine = None
    app.state.tts_service = None
    app.include_router(health_router)
    app.include_router(model_router)
    app.include_router(tts_router)
    return app


def test_model_status_ok(tmp_path: Path) -> None:
    client = TestClient(_app(tmp_path))
    res = client.get("/model/status")
    assert res.status_code == 200
    body = res.json()
    assert body["phase"] == "idle"
    assert body["ready"] is False


def test_health_includes_model_phase(tmp_path: Path) -> None:
    app = _app(tmp_path)
    app.state.model_state.set_phase(ModelPhase.DOWNLOADING, message="dl")
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["model_phase"] == "downloading"
    assert body["ready"] is False
    assert body["status"] == "starting"


def test_ensure_conflict_when_skip_and_missing(tmp_path: Path) -> None:
    client = TestClient(_app(tmp_path, SKIP_MODEL_DOWNLOAD=True))
    res = client.post("/model/ensure")
    assert res.status_code == 409


def test_tts_503_when_not_ready(tmp_path: Path) -> None:
    client = TestClient(_app(tmp_path))
    res = client.post(
        "/tts",
        json={"text": "hello", "language": "en", "speaker": "default", "speed": 1.0, "format": "wav"},
    )
    assert res.status_code == 503
    assert "model" in res.json()["detail"].lower() or "status" in res.json()["detail"].lower()
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `pytest tests/test_model_routes.py -v`  
Expected: fail on missing `routes_model` and/or missing health fields / 503 behavior.

- [ ] **Step 3: Implement `routes_model.py`**

```python
"""Model download status and ensure endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.models.schemas import ModelStatusResponse

router = APIRouter(tags=["model"])


def _state(request: Request):
    return request.app.state.model_state


def _bootstrap(request: Request):
    return request.app.state.model_bootstrap


@router.get("/model/status", response_model=ModelStatusResponse)
async def model_status(request: Request) -> ModelStatusResponse:
    snap = _state(request).snapshot()
    return ModelStatusResponse(**snap)


@router.post("/model/ensure", response_model=ModelStatusResponse)
async def model_ensure(request: Request) -> JSONResponse:
    boot = _bootstrap(request)
    if boot is None:
        raise HTTPException(status_code=503, detail="model bootstrap not configured")
    result = boot.ensure_async()
    snap = _state(request).snapshot()
    if result.get("conflict"):
        body = {**snap, "already_running": False, "already_ready": False}
        return JSONResponse(status_code=409, content=body)
    status_code = 202 if result.get("started") else 200
    body = {
        **snap,
        "already_running": bool(result.get("already_running")),
        "already_ready": bool(result.get("already_ready")),
    }
    return JSONResponse(status_code=status_code, content=body)
```

- [ ] **Step 4: Update `routes_health.py`**

```python
@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    engine = getattr(request.app.state, "engine", None)
    settings = getattr(request.app.state, "settings", None)
    model_state = getattr(request.app.state, "model_state", None)

    ready = False
    engine_name = None
    model_name = None
    model_phase = None
    model_ready = None

    if model_state is not None:
        snap = model_state.snapshot()
        model_phase = snap.get("phase")
        model_ready = bool(snap.get("ready"))
        model_name = snap.get("model")
        ready = model_ready

    if engine is not None:
        engine_ready = bool(engine.is_ready())
        ready = ready and engine_ready if model_state is not None else engine_ready
        engine_name = engine.name
        model_name = getattr(engine, "model_id", None) or model_name

    if settings is not None and model_name is None:
        model_name = settings.resolved_model_name

    if model_state is None and engine is None:
        # Legacy / minimal apps: process up
        ready = True

    return HealthResponse(
        status="ok" if ready else "starting",
        engine=engine_name,
        model=model_name,
        ready=ready,
        model_phase=model_phase,
        model_ready=model_ready,
    )
```

- [ ] **Step 5: Update TTS routes for 503**

At the top of each TTS handler (or in `_service`):

```python
def _service(request: Request) -> TTSService:
    svc = getattr(request.app.state, "tts_service", None)
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail="TTS engine not ready; see GET /model/status",
        )
    return svc  # type: ignore[no-any-return]
```

- [ ] **Step 6: Register model router in `create_app` (even before full lifespan rewrite)**

In `app/main.py`:

```python
from app.api.routes_model import router as model_router
# ...
app.include_router(model_router)
```

- [ ] **Step 7: Run route tests**

Run: `pytest tests/test_model_routes.py -v`  
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add app/api/routes_model.py app/api/routes_health.py app/api/routes_tts.py app/main.py tests/test_model_routes.py app/models/schemas.py
git commit -m "feat: add /model/status and /model/ensure; 503 TTS until ready"
```

---

### Task 6: Lifespan rewrite — early API bind + auto bootstrap

**Files:**
- Modify: `app/main.py`
- Modify: `entrypoint.sh`

- [ ] **Step 1: Rewrite `lifespan` in `app/main.py`**

Replace blocking `manager.ensure_model()` with bootstrap wiring:

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    setup_logging(settings.log_level)
    settings.ensure_directories()

    import os

    os.environ.setdefault("HF_HOME", str(settings.cache_dir / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(settings.cache_dir / "torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str(settings.cache_dir))

    logger.info(
        "Starting TTS server",
        extra={
            "version": __version__,
            "engine": settings.tts_engine,
            "model": settings.resolved_model_name,
            "device": settings.device,
        },
    )

    state = ModelDownloadState(
        model=settings.resolved_model_name,
        path=str(settings.model_path),
        download_source=settings.download_source,
    )
    manager = ModelManager(settings)

    app.state.model_state = state
    app.state.model_manager = manager
    app.state.engine = None
    app.state.tts_service = None

    def load_engine() -> None:
        engine = create_engine(settings)
        engine.load()
        audio = AudioService()
        tts_service = TTSService(settings=settings, engine=engine, audio=audio)
        app.state.engine = engine
        app.state.tts_service = tts_service
        logger.info(
            "TTS engine loaded",
            extra={"engine": engine.name, "model": engine.model_id},
        )

    bootstrap = ModelBootstrap(
        settings=settings,
        manager=manager,
        state=state,
        load_engine=load_engine,
    )
    app.state.model_bootstrap = bootstrap

    if manager.is_model_present():
        try:
            state.set_phase(ModelPhase.LOADING_ENGINE, message="Loading TTS engine")
            load_engine()
            state.set_phase(ModelPhase.READY, message="Model ready")
        except Exception as exc:
            logger.exception("Engine load failed on startup")
            state.set_error(str(exc))
    elif not settings.skip_model_download:
        logger.info("Model missing or incomplete; starting background download")
        bootstrap.ensure_async()
    else:
        state.set_error(
            f"Model missing at {settings.model_path} and SKIP_MODEL_DOWNLOAD=true"
        )

    try:
        yield
    finally:
        logger.info("Shutting down TTS server")
        engine = getattr(app.state, "engine", None)
        if engine is not None:
            try:
                engine.shutdown()
            except Exception:
                logger.exception("Error during engine shutdown")
```

Add imports:

```python
from app.services.model_bootstrap import ModelBootstrap
from app.services.model_download_state import ModelDownloadState, ModelPhase
```

- [ ] **Step 2: Update `entrypoint.sh`**

Replace the blocking ensure block with:

```bash
# Optional legacy path: block in entrypoint before API (default off).
# Prefer in-process background download so /model/status is reachable immediately.
if [[ "${ENSURE_MODEL_IN_ENTRYPOINT:-false}" == "true" && "${SKIP_MODEL_DOWNLOAD:-false}" != "true" ]]; then
  log "ENSURE_MODEL_IN_ENTRYPOINT=true — downloading model before API start"
  python -m app.bootstrap ensure-model
elif [[ "${SKIP_MODEL_DOWNLOAD:-false}" == "true" ]]; then
  log "SKIP_MODEL_DOWNLOAD=true — API will not auto-download models"
else
  log "Model ensure deferred to API process (GET /model/status, POST /model/ensure)"
fi
```

Update the comment at the top of `entrypoint.sh` to describe early API bind + in-process download.

- [ ] **Step 3: Smoke import**

Run: `python -c "from app.main import create_app; create_app(); print('ok')"`  
Expected: `ok` (may construct settings from env; should not download).

- [ ] **Step 4: Run full unit suite**

Run: `pytest tests/ -v`  
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app/main.py entrypoint.sh
git commit -m "feat: start API before model download; bootstrap in background"
```

---

### Task 7: Web UI progress banner

**Files:**
- Modify: `web/index.html`
- Modify: `web/app.js`
- Modify: `web/styles.css`

- [ ] **Step 1: HTML — model status block inside Connection card**

After the health row in `web/index.html` (inside `#connection`), add:

```html
    <div id="modelBanner" class="model-banner" hidden>
      <div class="model-banner-head">
        <span id="modelPhase" class="status">—</span>
        <button type="button" id="btnEnsure" class="hidden">Retry download</button>
      </div>
      <p id="modelMessage" class="model-msg">—</p>
      <div class="progress-track" aria-hidden="true">
        <div id="modelProgressBar" class="progress-bar" style="width: 0%"></div>
      </div>
      <p id="modelProgressText" class="model-msg muted"></p>
    </div>
```

- [ ] **Step 2: CSS**

Append to `web/styles.css`:

```css
.model-banner {
  margin-top: 0.85rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--border);
}

.model-banner-head {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-wrap: wrap;
  margin-bottom: 0.35rem;
}

.model-msg {
  margin: 0.25rem 0;
  font-size: 0.9rem;
}

.model-msg.muted {
  color: var(--muted);
  font-size: 0.8rem;
}

.progress-track {
  height: 0.45rem;
  background: var(--border);
  border-radius: 999px;
  overflow: hidden;
  margin: 0.5rem 0 0.25rem;
}

.progress-bar {
  height: 100%;
  background: var(--accent);
  border-radius: 999px;
  transition: width 0.35s ease;
  width: 0%;
}

.progress-track.indeterminate .progress-bar {
  width: 35% !important;
  animation: progress-indeterminate 1.2s ease-in-out infinite;
}

@keyframes progress-indeterminate {
  0% { transform: translateX(-120%); }
  100% { transform: translateX(320%); }
}

button:disabled,
.primary:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}
```

- [ ] **Step 3: JS — poll status, banner, retry, disable TTS**

In `web/app.js`, add after element refs:

```javascript
  const modelBanner = $("modelBanner");
  const modelPhase = $("modelPhase");
  const modelMessage = $("modelMessage");
  const modelProgressBar = $("modelProgressBar");
  const modelProgressText = $("modelProgressText");
  const btnEnsure = $("btnEnsure");
  const synthButtons = [$("btnQuick"), $("btnFile"), $("btnLong")];

  let modelReady = false;
  let pollTimer = null;

  function setSynthEnabled(enabled) {
    for (const btn of synthButtons) {
      if (btn) btn.disabled = !enabled;
    }
  }

  function renderModelStatus(body) {
    if (!body) return;
    modelBanner.hidden = false;
    const phase = body.phase || "unknown";
    const ready = !!body.ready;
    modelReady = ready;
    modelPhase.textContent = ready
      ? `ready · ${body.model || "?"}`
      : `${phase}${body.progress_pct != null ? ` · ${body.progress_pct}%` : ""}`;
    modelPhase.className =
      "status " + (ready ? "ok" : phase === "error" ? "err" : "warn");
    modelMessage.textContent = body.message || body.error || "";
    const track = modelProgressBar.parentElement;
    if (body.progress_pct != null && !ready) {
      track.classList.remove("indeterminate");
      modelProgressBar.style.width = `${Math.max(0, Math.min(100, body.progress_pct))}%`;
      modelProgressText.textContent =
        body.bytes_downloaded != null
          ? `${body.bytes_downloaded} bytes` +
            (body.bytes_total != null ? ` / ${body.bytes_total}` : "")
          : "";
    } else if (!ready && phase !== "error") {
      track.classList.add("indeterminate");
      modelProgressBar.style.width = "35%";
      modelProgressText.textContent =
        body.bytes_downloaded != null ? `${body.bytes_downloaded} bytes so far` : "Working…";
    } else {
      track.classList.remove("indeterminate");
      modelProgressBar.style.width = ready ? "100%" : "0%";
      modelProgressText.textContent = ready ? "" : body.error || "";
    }
    btnEnsure.classList.toggle("hidden", !(phase === "error" || (!ready && phase === "idle")));
    setSynthEnabled(ready);
  }

  async function fetchModelStatus() {
    try {
      const res = await fetch(`${apiBase()}/model/status`);
      if (!res.ok) throw new Error(`status ${res.status}`);
      const body = await res.json();
      renderModelStatus(body);
      return body;
    } catch (err) {
      modelBanner.hidden = false;
      modelPhase.textContent = "api unreachable";
      modelPhase.className = "status err";
      modelMessage.textContent = err.message || String(err);
      setSynthEnabled(false);
      return null;
    }
  }

  function schedulePoll() {
    if (pollTimer) clearInterval(pollTimer);
    const tick = async () => {
      const body = await fetchModelStatus();
      if (body && body.ready) {
        clearInterval(pollTimer);
        pollTimer = setInterval(fetchModelStatus, 10000);
      }
    };
    tick();
    pollTimer = setInterval(tick, 1500);
  }

  btnEnsure.addEventListener("click", async () => {
    setBusy(btnEnsure, true);
    try {
      const res = await fetch(`${apiBase()}/model/ensure`, { method: "POST" });
      const body = await res.json().catch(() => ({}));
      if (res.status === 409) {
        showError(body.error || body.message || body.detail || "Ensure conflict");
      }
      renderModelStatus(body.phase ? body : await fetchModelStatus());
      schedulePoll();
    } catch (err) {
      showError(`Ensure failed: ${err.message}`);
    } finally {
      setBusy(btnEnsure, false);
    }
  });
```

Update `checkHealth` success branch to also call `fetchModelStatus()` (or rely on poll).

At bottom, replace bare `checkHealth()` with:

```javascript
  checkHealth();
  schedulePoll();
```

- [ ] **Step 4: Manual static check**

Open `web/index.html` structure: ids must match JS.  
Run: `python -m app.ui_server --host 127.0.0.1 --port 27756 --root web` (optional) and confirm page loads without JS errors when API is down (banner shows unreachable).

- [ ] **Step 5: Commit**

```bash
git add web/index.html web/app.js web/styles.css
git commit -m "feat: show model download progress banner in web UI"
```

---

### Task 8: README + bootstrap CLI note

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update docs**

1. First-boot section: API is available immediately; download runs in-process; watch `GET /model/status` or UI banner.
2. API section: document `GET /model/status` and `POST /model/ensure` with example curl and JSON fields.
3. Health example JSON: include `model_phase`, `model_ready`.
4. Web UI table: add Model status / Retry rows.
5. Configuration table: add `ENSURE_MODEL_IN_ENTRYPOINT` (default `false`).
6. Note `WORKERS=1` required for in-process download state.
7. How it works / model section: remove “entrypoint blocks until download” if present; say deferred to API lifespan.

Example curl block:

```bash
curl -s http://localhost:27755/model/status | jq
curl -s -X POST http://localhost:27755/model/ensure | jq
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document model status API and early-bind download"
```

---

### Task 9: Full verification

**Files:** none (run only)

- [ ] **Step 1: Run entire test suite**

Run: `pytest tests/ -v`  
Expected: all PASS.

- [ ] **Step 2: Lint sanity (if project uses nothing formal, skip)**

Run: `python -m compileall app tests`  
Expected: exit 0.

- [ ] **Step 3: Spec checklist self-verify**

Confirm against `docs/superpowers/specs/2026-08-10-model-download-progress-design.md`:

| Requirement | Task |
|-------------|------|
| Early API bind | Task 6 |
| Background auto-download | Task 6 |
| `GET /model/status` | Task 5 |
| `POST /model/ensure` | Task 5 |
| Health phase fields | Task 5 |
| Progress bytes/phase | Tasks 1, 3, 4 |
| TTS 503 | Task 5 |
| UI banner + retry | Task 7 |
| Entrypoint non-blocking | Task 6 |
| README | Task 8 |
| SKIP_MODEL_DOWNLOAD 409 | Tasks 4–5 |
| Single-flight ensure | Task 4 |

- [ ] **Step 4: Final commit only if uncommitted fixes remain**

```bash
git status
# if needed: git add -A && git commit -m "fix: address verification follow-ups for model progress"
```

---

## Self-review (plan vs spec)

1. **Spec coverage:** All design goals mapped to Tasks 1–8; verification in Task 9.
2. **Placeholders:** None intentional; progress “HF tqdm if available” is concretely replaced by byte scanner (always works) plus optional hooks later.
3. **Type consistency:** `ModelPhase` string values match API `phase`; snapshot keys match `ModelStatusResponse`; bootstrap return flags match route handling.

## Execution notes

- Prefer **subagent-driven-development** (one task per subagent + review).
- Do not download multi‑GB models in CI; all automated tests mock `ensure_model` / use `SKIP_MODEL_DOWNLOAD`.
- Default `WORKERS=1`; do not change to multi-worker without a shared status store.
