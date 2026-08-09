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
