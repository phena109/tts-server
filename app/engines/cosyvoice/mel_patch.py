"""Patch audio feature paths that SIGILL under torch on some aarch64 VMs.

Apple Silicon Podman VMs (linux/arm64) crash with ``Illegal instruction`` in:

* ``matcha.utils.audio.mel_spectrogram`` (torch.stft → matmul)
* ``torchaudio.compliance.kaldi.fbank``
* ``whisper.log_mel_spectrogram``

Those kill CosyVoice during prompt feature extraction (first Quick TTS). We
replace them with pure NumPy / SciPy paths (no torch SIMD, no numba JIT).
"""

from __future__ import annotations

from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)

_patched = False


def _to_1d_float_numpy(y: Any) -> Any:
    import numpy as np
    import torch

    if torch.is_tensor(y):
        arr = y.detach().cpu().float().numpy()
    else:
        arr = np.asarray(y, dtype=np.float32)
    return np.ascontiguousarray(arr.reshape(-1), dtype=np.float32)


def _frame_signal(wav: Any, frame_length: int, hop_length: int) -> Any:
    import numpy as np

    if len(wav) < frame_length:
        wav = np.pad(wav, (0, frame_length - len(wav)))
    num_frames = 1 + (len(wav) - frame_length) // hop_length
    if num_frames < 1:
        num_frames = 1
    frames = np.lib.stride_tricks.as_strided(
        wav,
        shape=(num_frames, frame_length),
        strides=(wav.strides[0] * hop_length, wav.strides[0]),
        writeable=False,
    )
    return np.array(frames, copy=True)


def _stft_mag(
    wav: Any,
    n_fft: int,
    hop_length: int,
    win_length: int,
    center: bool = True,
) -> Any:
    """Onesided STFT magnitude via numpy.fft (no torch / numba)."""
    import numpy as np

    if center:
        pad = n_fft // 2
        wav = np.pad(wav, (pad, pad), mode="reflect")
    # Hann window
    window = np.hanning(win_length).astype(np.float32)
    if win_length < n_fft:
        window = np.pad(window, (0, n_fft - win_length))
    elif win_length > n_fft:
        window = window[:n_fft]
    frames = _frame_signal(wav, n_fft, hop_length)
    frames = frames * window[None, :]
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    mag = np.abs(spec).astype(np.float32).T  # [freq, frames]
    return mag


def _mel_filterbank(
    sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float | None
) -> Any:
    """HTK-style mel filterbank (freq bins x mels → we return mels x freqs)."""
    import numpy as np

    if fmax is None or fmax <= 0:
        fmax = sr / 2.0

    def hz_to_mel(f: float) -> float:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    n_freqs = n_fft // 2 + 1
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center == left:
            center += 1
        if right == center:
            right += 1
        right = min(right, n_freqs)
        for j in range(left, center):
            if center != left:
                fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right != center:
                fb[i, j] = (right - j) / (right - center)
    # Slaney-style normalize
    enorm = 2.0 / (mel_to_hz(mels[2 : n_mels + 2]) - mel_to_hz(mels[:n_mels]))
    fb *= enorm[:, None].astype(np.float32)
    return fb


def install_safe_mel_spectrogram() -> None:
    """Install all SIGILL-safe audio feature patches (idempotent)."""
    global _patched
    if _patched:
        return

    import numpy as np
    import torch

    def mel_spectrogram(
        y: Any,
        n_fft: int,
        num_mels: int,
        sampling_rate: int,
        hop_size: int,
        win_size: int,
        fmin: float,
        fmax: float,
        center: bool = False,
    ) -> torch.Tensor:
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, dtype=torch.float32)
        y = y.float()
        device = y.device
        wav = _to_1d_float_numpy(y)
        # Matcha pads before STFT when center=False
        pad = int((n_fft - hop_size) / 2)
        if pad > 0 and not center:
            wav = np.pad(wav, (pad, pad), mode="reflect")
        mag = _stft_mag(wav, n_fft=n_fft, hop_length=hop_size, win_length=win_size, center=False)
        mel_basis = _mel_filterbank(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel = mel_basis @ mag
        mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
        return torch.from_numpy(mel.astype(np.float32)).to(device=device)

    def kaldi_fbank(
        waveform: Any,
        blackman_coeff: float = 0.42,
        channel: int = -1,
        dither: float = 0.0,
        energy_floor: float = 1.0,
        frame_length: float = 25.0,
        frame_shift: float = 10.0,
        high_freq: float = 0.0,
        htk_compat: bool = False,
        low_freq: float = 20.0,
        min_duration: float = 0.0,
        num_mel_bins: int = 23,
        preemphasis_coefficient: float = 0.97,
        raw_energy: bool = True,
        remove_dc_offset: bool = True,
        round_to_power_of_two: bool = True,
        sample_frequency: float = 16000.0,
        snip_edges: bool = True,
        subtract_mean: bool = False,
        use_energy: bool = False,
        use_log_fbank: bool = True,
        use_power: bool = True,
        vtln_high: float = -500.0,
        vtln_low: float = 100.0,
        vtln_warp: float = 1.0,
        window_type: str = "povey",
    ) -> torch.Tensor:
        wav = _to_1d_float_numpy(waveform)
        sr = int(sample_frequency)
        win_length = int(sample_frequency * frame_length * 0.001)
        hop_length = int(sample_frequency * frame_shift * 0.001)
        n_fft = 1
        while n_fft < win_length:
            n_fft <<= 1
        if not round_to_power_of_two:
            n_fft = win_length
        fmax = sr / 2.0 if high_freq in (0.0, None) else float(abs(high_freq))
        # Pre-emphasis
        if preemphasis_coefficient and len(wav) > 1:
            wav = np.append(wav[0], wav[1:] - preemphasis_coefficient * wav[:-1])
        if remove_dc_offset:
            wav = wav - wav.mean()
        mag = _stft_mag(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=not snip_edges)
        if use_power:
            mag = mag ** 2
        mel_basis = _mel_filterbank(sr, n_fft, num_mel_bins, low_freq, fmax)
        mel = mel_basis @ mag
        if use_log_fbank:
            mel = np.log(np.maximum(mel, 1e-10))
        feat = mel.T.astype(np.float32)  # [frames, bins]
        if subtract_mean:
            feat = feat - feat.mean(axis=0, keepdims=True)
        return torch.from_numpy(feat)

    def whisper_log_mel_spectrogram(
        audio: Any,
        n_mels: int = 80,
        padding: int = 0,
        device: Any = None,
    ) -> torch.Tensor:
        # CosyVoice passes load_wav output shaped [1, T] and indexes feat.shape[2].
        batched = False
        if torch.is_tensor(audio):
            out_device = audio.device if device is None else device
            batched = audio.dim() == 2
            wav = _to_1d_float_numpy(audio)
        else:
            out_device = device or "cpu"
            arr = np.asarray(audio, dtype=np.float32)
            batched = arr.ndim == 2
            wav = np.ascontiguousarray(arr.reshape(-1))
        if padding > 0:
            wav = np.pad(wav, (0, int(padding)))
        # Whisper defaults: 16 kHz, n_fft=400, hop=160
        mag = _stft_mag(wav, n_fft=400, hop_length=160, win_length=400, center=True)
        mag = mag ** 2
        mel_basis = _mel_filterbank(16000, 400, n_mels, 0.0, 8000.0)
        mel = mel_basis @ mag
        log_spec = np.log10(np.maximum(mel, 1e-10))
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        t = torch.from_numpy(log_spec.astype(np.float32))
        if batched:
            t = t.unsqueeze(0)  # [1, n_mels, frames]
        if out_device is not None:
            t = t.to(out_device)
        return t

    try:
        import matcha.utils.audio as audio_mod

        audio_mod.mel_spectrogram = mel_spectrogram  # type: ignore[assignment]
    except ImportError:
        logger.warning("matcha.utils.audio not importable; mel patch skipped")

    try:
        import torchaudio.compliance.kaldi as kaldi_mod

        kaldi_mod.fbank = kaldi_fbank  # type: ignore[assignment]
    except ImportError:
        logger.warning("torchaudio.compliance.kaldi not importable; fbank patch skipped")

    try:
        import whisper.audio as whisper_audio
        import whisper

        whisper_audio.log_mel_spectrogram = whisper_log_mel_spectrogram  # type: ignore[assignment]
        whisper.log_mel_spectrogram = whisper_log_mel_spectrogram  # type: ignore[assignment]
    except ImportError:
        logger.warning("whisper not importable; log_mel patch skipped")

    _patched = True
    logger.info(
        "Installed SIGILL-safe audio patches (matcha mel, kaldi fbank, whisper mel)"
    )
