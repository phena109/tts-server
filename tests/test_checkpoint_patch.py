"""Tests for CosyVoice training-checkpoint sanitization."""

from __future__ import annotations

from dataclasses import dataclass

from app.engines.cosyvoice.checkpoint_patch import sanitize_state_dict


@dataclass
class _FakeTensor:
    """Minimal stand-in so host tests do not need torch installed."""

    shape: tuple[int, ...] = (2, 2)
    dtype: str = "float32"


def test_strips_epoch_and_step() -> None:
    weight = _FakeTensor()
    raw = {
        "llm.weight": weight,
        "epoch": 12,
        "step": 3400,
    }
    cleaned = sanitize_state_dict(raw)
    assert set(cleaned.keys()) == {"llm.weight"}
    assert cleaned["llm.weight"] is weight


def test_unwraps_nested_state_dict() -> None:
    weight = _FakeTensor(shape=(1, 1))
    raw = {
        "epoch": 1,
        "state_dict": {"a.weight": weight},
        "optimizer": {"x": 1},
    }
    cleaned = sanitize_state_dict(raw)
    assert set(cleaned.keys()) == {"a.weight"}
    assert cleaned["a.weight"] is weight


def test_pure_weight_dict_unchanged() -> None:
    weight = _FakeTensor(shape=(3,))
    raw = {"w": weight}
    cleaned = sanitize_state_dict(raw)
    assert set(cleaned.keys()) == {"w"}
