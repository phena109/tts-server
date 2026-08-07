"""Audio encode/concat tests (ffmpeg optional for mp3)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.audio_service import AudioService


def test_encode_wav_roundtrip_header() -> None:
    svc = AudioService()
    sr = 24000
    t = np.linspace(0, 0.1, int(sr * 0.1), dtype=np.float32)
    samples = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    data = svc.encode(samples, sr, "wav")
    assert data[:4] == b"RIFF"
    assert b"WAVE" in data[:16]
    assert len(data) > 100


def test_concatenate_two_chunks() -> None:
    svc = AudioService()
    a = np.ones(1000, dtype=np.float32) * 0.1
    b = np.ones(1000, dtype=np.float32) * 0.2
    out = svc.concatenate([a, b], sample_rate=24000, crossfade_ms=5)
    assert out.dtype == np.float32
    assert out.size > 1000
    assert out.size < 2000  # crossfade shortens total vs pure concat


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_encode_mp3() -> None:
    svc = AudioService()
    sr = 16000
    samples = (0.1 * np.random.randn(sr).astype(np.float32))
    data = svc.encode(samples, sr, "mp3")
    assert len(data) > 100
    # MPEG frame sync often starts with 0xFFEx
    assert data[0] == 0xFF or data[:3] == b"ID3"
