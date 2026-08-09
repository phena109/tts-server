"""Disk-backed CosyVoice prompt features (speech tokens / mel / embedding).

Extracting speech tokens runs a ~1GB ONNX session. Doing that **after** the full
LLM+flow+HiFT stack is loaded peaks past a 12GB Podman VM and kills the
process. We instead extract in an isolated subprocess that only constructs
``CosyVoiceFrontEnd``, write tensors under ``CACHE_DIR``, then inject them via
the engine's prompt-feature cache after AutoModel loads.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)


def prompt_feature_cache_path(cache_dir: Path, prompt_wav: Path) -> Path:
    """Stable path for a given prompt wav content."""
    prompt_wav = Path(prompt_wav)
    # Bundled CosyVoice default prompt → stable well-known name (ops can pre-seed).
    name = prompt_wav.name
    if name == "zero_shot_prompt.wav" or "zero_shot_prompt" in str(prompt_wav):
        return Path(cache_dir) / "prompt_features" / "default_zero_shot.pt"
    h = hashlib.sha256()
    h.update(str(prompt_wav.resolve()).encode())
    try:
        st = prompt_wav.stat()
        h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    except OSError:
        pass
    return Path(cache_dir) / "prompt_features" / f"{h.hexdigest()[:20]}.pt"


def load_prompt_features(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        import torch

        data = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(data, dict):
            return None
        for key in ("speech_token", "speech_feat", "spk_embedding"):
            if key not in data:
                return None
        return data
    except Exception:
        logger.exception("Failed to load prompt features", extra={"path": str(path)})
        return None


def ensure_prompt_features(
    *,
    model_dir: Path,
    prompt_wav: Path,
    cache_dir: Path,
    cosyvoice_repo: Path,
    sample_rate: int = 24000,
) -> Path | None:
    """Return path to on-disk features, building them in a child process if needed."""
    prompt_wav = Path(prompt_wav)
    if not prompt_wav.is_file():
        logger.warning("Prompt wav missing; skip feature cache", extra={"path": str(prompt_wav)})
        return None

    out = prompt_feature_cache_path(cache_dir, prompt_wav)
    if out.is_file():
        logger.info("Prompt features cache hit", extra={"path": str(out)})
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Building prompt features in isolated process",
        extra={"prompt": str(prompt_wav), "out": str(out)},
    )

    # Run as `python -m app.services.prompt_features` so imports resolve.
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{cosyvoice_repo}:{cosyvoice_repo}/third_party/Matcha-TTS:"
        f"/app:{env.get('PYTHONPATH', '')}"
    )
    cmd = [
        sys.executable,
        "-m",
        "app.services.prompt_features",
        "--model-dir",
        str(model_dir),
        "--prompt-wav",
        str(prompt_wav),
        "--out",
        str(out),
        "--cache-dir",
        str(cache_dir),
        "--sample-rate",
        str(sample_rate),
    ]
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.error("Prompt feature extraction timed out")
        return None

    if proc.returncode != 0 or not out.is_file():
        logger.error(
            "Prompt feature extraction failed",
            extra={
                "code": proc.returncode,
                "stdout": (proc.stdout or "")[-2000:],
                "stderr": (proc.stderr or "")[-2000:],
            },
        )
        return None

    logger.info("Prompt features written", extra={"path": str(out)})
    return out


def _extract_in_process(
    *,
    model_dir: Path,
    prompt_wav: Path,
    out: Path,
    cache_dir: Path,
    sample_rate: int,
) -> None:
    """Child-process entry: frontend only, no full TTS weights."""
    # Local imports — heavy stack only in the worker.
    import sys
    from pathlib import Path as _Path

    # Ensure CosyVoice + Matcha are importable even when invoked as -m.
    for p in (
        "/opt/CosyVoice",
        "/opt/CosyVoice/third_party/Matcha-TTS",
        "/app",
    ):
        if p not in sys.path:
            sys.path.insert(0, p)

    from hyperpyyaml import load_hyperpyyaml

    from app.engines.cosyvoice.mel_patch import install_safe_mel_spectrogram
    from app.services.wetext_assets import prepare_wetext_for_cosyvoice
    from cosyvoice.cli.frontend import CosyVoiceFrontEnd  # type: ignore

    install_safe_mel_spectrogram()
    prepare_wetext_for_cosyvoice(cache_dir)

    model_dir = Path(model_dir)
    hyper_yaml_path = model_dir / "cosyvoice3.yaml"
    if not hyper_yaml_path.is_file():
        # CosyVoice2 fallback
        hyper_yaml_path = model_dir / "cosyvoice2.yaml"
    if not hyper_yaml_path.is_file():
        raise FileNotFoundError(f"no cosyvoice yaml under {model_dir}")

    with open(hyper_yaml_path, "r", encoding="utf-8") as f:
        configs = load_hyperpyyaml(
            f,
            overrides={
                "qwen_pretrain_path": str(model_dir / "CosyVoice-BlankEN"),
            },
        )

    # tokenizer onnx name differs by version
    tok = model_dir / "speech_tokenizer_v3.onnx"
    if not tok.is_file():
        tok = model_dir / "speech_tokenizer_v2.onnx"

    frontend = CosyVoiceFrontEnd(
        configs["get_tokenizer"],
        configs["feat_extractor"],
        str(model_dir / "campplus.onnx"),
        str(tok),
        str(model_dir / "spk2info.pt"),
        configs["allowed_special"],
    )

    # Drop unused weight modules from hyperpyyaml so RSS stays low in the worker.
    for k in ("llm", "flow", "hift"):
        if k in configs:
            del configs[k]
    import gc

    gc.collect()

    wav = str(prompt_wav)
    # Return shapes match CosyVoiceFrontEnd methods (tuples for feat/token).
    speech_feat = frontend._extract_speech_feat(wav)
    speech_token = frontend._extract_speech_token(wav)
    spk_embedding = frontend._extract_spk_embedding(wav)

    import torch

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "speech_feat": speech_feat,
            "speech_token": speech_token,
            "spk_embedding": spk_embedding,
            "prompt_wav": str(prompt_wav),
            "sample_rate": sample_rate,
        },
        out,
    )
    print(f"wrote {out}", flush=True)


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Extract CosyVoice prompt features (frontend only)")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--prompt-wav", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--sample-rate", type=int, default=24000)
    args = p.parse_args(argv)
    _extract_in_process(
        model_dir=Path(args.model_dir),
        prompt_wav=Path(args.prompt_wav),
        out=Path(args.out),
        cache_dir=Path(args.cache_dir),
        sample_rate=args.sample_rate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
