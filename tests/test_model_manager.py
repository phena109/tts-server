"""Tests for model completeness detection (incomplete HF downloads)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config.settings import Settings
from app.services import model_manager as mm
from app.services.model_manager import ModelManager


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        MODEL_DIR=str(tmp_path),
        MODEL_LOCAL_NAME="Fun-CosyVoice3-0.5B",
        SKIP_MODEL_DOWNLOAD=True,
        INPUT_DIR=str(tmp_path / "input"),
        OUTPUT_DIR=str(tmp_path / "output"),
        CACHE_DIR=str(tmp_path / "cache"),
    )


def _write(path: Path, size: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


@pytest.fixture
def small_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink size floors so tests stay tiny on disk."""
    monkeypatch.setattr(
        mm,
        "_COSYVOICE3_REQUIRED",
        {
            "cosyvoice3.yaml": 4,
            "llm.pt": 10,
            "flow.pt": 10,
            "hift.pt": 10,
            "campplus.onnx": 10,
            "speech_tokenizer_v3.onnx": 10,
        },
    )
    # Blank-EN weight min size is hardcoded as 50_000_000 in _file_ok calls —
    # patch _is_cosyvoice3_complete via a thin override on min blank size by
    # writing real-sized tiny files and patching the method's threshold through
    # _file_ok wrapper is awkward; instead monkeypatch the whole completeness
    # helper's blank check by temporarily reducing via a custom path:
    original = ModelManager._is_cosyvoice3_complete

    def _complete_small(cls: type[ModelManager], root: Path) -> bool:  # noqa: ANN001
        for rel, min_size in mm._COSYVOICE3_REQUIRED.items():
            if not cls._file_ok(root / rel, min_size):
                return False
        blank = root / "CosyVoice-BlankEN"
        if not blank.is_dir():
            return False
        return any(cls._file_ok(blank / name, 10) for name in mm._BLANK_EN_WEIGHTS)

    monkeypatch.setattr(ModelManager, "_is_cosyvoice3_complete", classmethod(_complete_small))
    # keep original reachable for clarity
    assert original is not None


def test_incomplete_markers_not_present(tmp_path: Path, small_thresholds: None) -> None:
    root = tmp_path / "Fun-CosyVoice3-0.5B"
    # Mirrors the broken volume state: yaml + small onnx only, plus .incomplete
    _write(root / "cosyvoice3.yaml")
    _write(root / "config.json", 2)
    _write(root / "campplus.onnx", 20)
    _write(root / ".cache" / "huggingface" / "download" / "llm.pt.lock", 0)
    _write(root / ".cache" / "huggingface" / "download" / "abc.incomplete", 20)

    mgr = ModelManager(_settings(tmp_path))
    assert mgr.is_model_present() is False


def test_partial_cosyvoice3_without_weights_not_present(
    tmp_path: Path, small_thresholds: None
) -> None:
    root = tmp_path / "Fun-CosyVoice3-0.5B"
    _write(root / "cosyvoice3.yaml")
    _write(root / "campplus.onnx", 20)
    _write(root / "CosyVoice-BlankEN" / "config.json")
    # Missing llm.pt / flow.pt / hift.pt / tokenizer / blank weights

    mgr = ModelManager(_settings(tmp_path))
    assert mgr.is_model_present() is False


def test_complete_cosyvoice3_is_present(tmp_path: Path, small_thresholds: None) -> None:
    root = tmp_path / "Fun-CosyVoice3-0.5B"
    _write(root / "cosyvoice3.yaml")
    _write(root / "llm.pt", 20)
    _write(root / "flow.pt", 20)
    _write(root / "hift.pt", 20)
    _write(root / "campplus.onnx", 20)
    _write(root / "speech_tokenizer_v3.onnx", 20)
    _write(root / "CosyVoice-BlankEN" / "model.safetensors", 20)

    mgr = ModelManager(_settings(tmp_path))
    assert mgr.is_model_present() is True


def test_undersized_weight_not_present(tmp_path: Path, small_thresholds: None) -> None:
    root = tmp_path / "Fun-CosyVoice3-0.5B"
    _write(root / "cosyvoice3.yaml")
    _write(root / "llm.pt", 2)  # below threshold of 10
    _write(root / "flow.pt", 20)
    _write(root / "hift.pt", 20)
    _write(root / "campplus.onnx", 20)
    _write(root / "speech_tokenizer_v3.onnx", 20)
    _write(root / "CosyVoice-BlankEN" / "model.safetensors", 20)

    mgr = ModelManager(_settings(tmp_path))
    assert mgr.is_model_present() is False


def test_ensure_model_raises_when_skip_and_missing(tmp_path: Path) -> None:
    mgr = ModelManager(_settings(tmp_path))
    with pytest.raises(FileNotFoundError):
        mgr.ensure_model()
