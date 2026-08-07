"""Audio encoding, decoding, and concatenation via numpy + ffmpeg."""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt

from app.utils.logging import get_logger

logger = get_logger(__name__)

AudioFormat = Literal["wav", "mp3"]


class AudioServiceError(RuntimeError):
    """Raised when encoding/decoding fails."""


class AudioService:
    """Encode float32 mono PCM to wav/mp3 and merge multi-chunk audio."""

    def __init__(self, ffmpeg_bin: str = "ffmpeg") -> None:
        self.ffmpeg_bin = ffmpeg_bin
        if shutil.which(ffmpeg_bin) is None:
            logger.warning(
                "ffmpeg not found on PATH; mp3 encoding will fail",
                extra={"ffmpeg": ffmpeg_bin},
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        samples: npt.NDArray[np.float32],
        sample_rate: int,
        fmt: AudioFormat = "wav",
    ) -> bytes:
        pcm = self._normalize_pcm(samples)
        if fmt == "wav":
            return self._encode_wav(pcm, sample_rate)
        if fmt == "mp3":
            return self._encode_mp3(pcm, sample_rate)
        raise ValueError(f"Unsupported format: {fmt}")

    def concatenate(
        self,
        chunks: Sequence[npt.NDArray[np.float32]],
        sample_rate: int,
        crossfade_ms: float = 10.0,
    ) -> npt.NDArray[np.float32]:
        """Merge PCM chunks with a short equal-power crossfade."""
        cleaned = [self._normalize_pcm(c) for c in chunks if c is not None and np.size(c)]
        if not cleaned:
            return np.zeros(0, dtype=np.float32)
        if len(cleaned) == 1:
            return cleaned[0]

        fade = int(sample_rate * (crossfade_ms / 1000.0))
        out = cleaned[0]
        for nxt in cleaned[1:]:
            out = self._crossfade(out, nxt, fade)
        return out

    def write_file(
        self,
        path: Path,
        samples: npt.NDArray[np.float32],
        sample_rate: int,
        fmt: AudioFormat | None = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt is None:
            suffix = path.suffix.lower().lstrip(".")
            fmt = "mp3" if suffix == "mp3" else "wav"
        data = self.encode(samples, sample_rate, fmt)  # type: ignore[arg-type]
        path.write_bytes(data)
        return path

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_pcm(samples: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        # Peak-normalize only if clearly out of range
        peak = float(np.max(np.abs(arr))) if arr.size else 0.0
        if peak > 1.0:
            arr = arr / peak
        return arr

    @staticmethod
    def _encode_wav(samples: npt.NDArray[np.float32], sample_rate: int) -> bytes:
        # 16-bit PCM WAV
        clipped = np.clip(samples, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()

    def _encode_mp3(self, samples: npt.NDArray[np.float32], sample_rate: int) -> bytes:
        wav_bytes = self._encode_wav(samples, sample_rate)
        if shutil.which(self.ffmpeg_bin) is None:
            raise AudioServiceError(
                "ffmpeg is required for mp3 output but was not found on PATH"
            )

        with tempfile.TemporaryDirectory(prefix="tts-audio-") as tmp:
            in_path = Path(tmp) / "in.wav"
            out_path = Path(tmp) / "out.mp3"
            in_path.write_bytes(wav_bytes)
            cmd = [
                self.ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(in_path),
                "-codec:a",
                "libmp3lame",
                "-qscale:a",
                "2",
                str(out_path),
            ]
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                raise AudioServiceError(
                    f"ffmpeg mp3 encode failed: {exc.stderr or exc}"
                ) from exc
            return out_path.read_bytes()

    @staticmethod
    def _crossfade(
        a: npt.NDArray[np.float32],
        b: npt.NDArray[np.float32],
        fade_samples: int,
    ) -> npt.NDArray[np.float32]:
        if fade_samples <= 0 or a.size < fade_samples or b.size < fade_samples:
            return np.concatenate([a, b]).astype(np.float32, copy=False)

        fade = min(fade_samples, a.size, b.size)
        t = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        # Equal-power crossfade
        fade_out = np.cos(t * (np.pi / 2.0)).astype(np.float32)
        fade_in = np.sin(t * (np.pi / 2.0)).astype(np.float32)
        mixed = a[-fade:] * fade_out + b[:fade] * fade_in
        return np.concatenate([a[:-fade], mixed, b[fade:]]).astype(np.float32, copy=False)


def media_type_for(fmt: AudioFormat) -> str:
    return "audio/wav" if fmt == "wav" else "audio/mpeg"


def extension_for(fmt: AudioFormat) -> str:
    return "wav" if fmt == "wav" else "mp3"
