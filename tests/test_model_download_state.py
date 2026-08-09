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
