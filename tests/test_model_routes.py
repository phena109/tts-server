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
        json={
            "text": "hello",
            "language": "en",
            "speaker": "default",
            "speed": 1.0,
            "format": "wav",
        },
    )
    assert res.status_code == 503
    assert "model" in res.json()["detail"].lower() or "status" in res.json()["detail"].lower()


def test_ensure_starts_returns_202(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """POST ensure when download is allowed starts bootstrap -> 202."""
    import threading

    app = _app(tmp_path, SKIP_MODEL_DOWNLOAD=False)
    gate = threading.Event()
    manager = app.state.model_manager

    def fake_ensure() -> Path:
        gate.wait(timeout=5)
        return Path(app.state.settings.model_path)

    monkeypatch.setattr(manager, "ensure_model", fake_ensure)
    monkeypatch.setattr(manager, "is_model_present", lambda: True)

    client = TestClient(app)
    res = client.post("/model/ensure")
    assert res.status_code == 202
    body = res.json()
    assert body.get("already_ready") is False
    assert body.get("already_running") is False
    # Phase should already be checking (set under lock before thread work)
    assert body["phase"] == "checking"
    gate.set()
    assert app.state.model_bootstrap.wait(timeout=5) is True


def test_ensure_already_running_returns_200(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Second ensure while bootstrap is active -> 200 already_running."""
    import threading

    app = _app(tmp_path, SKIP_MODEL_DOWNLOAD=False)
    gate = threading.Event()
    manager = app.state.model_manager

    def fake_ensure() -> Path:
        gate.wait(timeout=5)
        return Path(app.state.settings.model_path)

    monkeypatch.setattr(manager, "ensure_model", fake_ensure)
    monkeypatch.setattr(manager, "is_model_present", lambda: True)

    client = TestClient(app)
    r1 = client.post("/model/ensure")
    assert r1.status_code == 202
    r2 = client.post("/model/ensure")
    assert r2.status_code == 200
    body = r2.json()
    assert body["already_running"] is True
    assert body.get("already_ready") is False
    gate.set()
    assert app.state.model_bootstrap.wait(timeout=5) is True


def test_ensure_already_ready_returns_200(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When model is ready, ensure -> 200 already_ready."""
    app = _app(tmp_path, SKIP_MODEL_DOWNLOAD=False)
    app.state.model_state.set_phase(ModelPhase.READY, message="ok")
    monkeypatch.setattr(app.state.model_manager, "is_model_present", lambda: True)

    client = TestClient(app)
    res = client.post("/model/ensure")
    assert res.status_code == 200
    body = res.json()
    assert body["already_ready"] is True
    assert body["ready"] is True
    assert body["phase"] == "ready"
