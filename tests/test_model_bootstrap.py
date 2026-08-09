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
