"""Make CosyVoice tolerate training checkpoints with non-weight keys.

ASLP-lab CosyVoice2-Yue fine-tunes (and some other HF training dumps) ship
``llm.pt`` as a Lightning-style blob that includes scalar metadata such as
``epoch`` / ``step`` alongside real parameter tensors. Upstream CosyVoice
loads with::

    self.llm.load_state_dict(torch.load(...), strict=True)

which then fails with::

    Unexpected key(s) in state_dict: "epoch", "step".

We monkey-patch ``CosyVoiceModel.load`` to strip non-tensor / known training
keys before ``load_state_dict``. Idempotent.
"""

from __future__ import annotations

from typing import Any, Mapping

from app.utils.logging import get_logger

logger = get_logger(__name__)

_patched = False

# Keys that appear in training dumps but are not nn.Module parameters.
_TRAINING_META_KEYS = frozenset(
    {
        "epoch",
        "step",
        "global_step",
        "optimizer",
        "lr_scheduler",
        "scheduler",
        "scaler",
        "amp_scaler",
        "loss",
        "best_loss",
        "config",
        "hyper_parameters",
        "hparams",
        "pytorch-lightning_version",
        "loops",
        "callbacks",
        "lr_schedulers",
    }
)

_NESTED_STATE_KEYS = ("state_dict", "model", "module", "model_state_dict")


def _is_weight_tensor(value: Any) -> bool:
    """True for torch tensors / Parameter-like objects."""
    try:
        import torch

        if torch.is_tensor(value):
            return True
    except Exception:
        pass
    # Duck-type for unit tests without torch installed.
    return hasattr(value, "shape") and hasattr(value, "dtype")


def sanitize_state_dict(raw: Any) -> Mapping[str, Any]:
    """Return a pure parameter state_dict suitable for ``load_state_dict``.

    Accepts either a flat dict of tensors (plus optional meta keys) or a
    common nested training-checkpoint layout.
    """
    if not isinstance(raw, dict):
        raise TypeError(f"expected state_dict mapping, got {type(raw)!r}")

    state: Mapping[str, Any] = raw
    # Unwrap one level of common training wrappers when present.
    for key in _NESTED_STATE_KEYS:
        nested = state.get(key) if isinstance(state, dict) else None
        if isinstance(nested, dict) and nested:
            # Prefer nested only when it looks like weights (has tensors).
            if any(_is_weight_tensor(v) for v in nested.values()):
                state = nested
                break

    cleaned: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in state.items():
        if key in _TRAINING_META_KEYS:
            dropped.append(key)
            continue
        # Keep tensors / Parameter-like objects; drop scalars / nested junk.
        if _is_weight_tensor(value):
            cleaned[key] = value
            continue
        dropped.append(key)

    if dropped:
        logger.info(
            "Stripped non-weight keys from CosyVoice checkpoint",
            extra={"dropped": dropped[:20], "kept": len(cleaned)},
        )
    return cleaned


def install_checkpoint_compat() -> bool:
    """Patch CosyVoiceModel.load to sanitize training checkpoints. Idempotent."""
    global _patched
    if _patched:
        return False

    import torch
    from cosyvoice.cli.model import CosyVoiceModel  # type: ignore

    if getattr(CosyVoiceModel.load, "_tts_checkpoint_sanitized", False):
        _patched = True
        return False

    original_load = CosyVoiceModel.load

    def load(self: Any, llm_model: str, flow_model: str, hift_model: str) -> None:  # noqa: ANN401
        def _load_weights(path: str) -> Mapping[str, Any]:
            # CosyVoice uses weights_only=True; keep that when available.
            try:
                raw = torch.load(path, map_location=self.device, weights_only=True)
            except TypeError:
                raw = torch.load(path, map_location=self.device)
            return sanitize_state_dict(raw)

        self.llm.load_state_dict(_load_weights(llm_model), strict=True)
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(_load_weights(flow_model), strict=True)
        self.flow.to(self.device).eval()
        # Upstream strips a possible HiFi-GAN "generator." prefix.
        hift_state = {
            k.replace("generator.", ""): v for k, v in _load_weights(hift_model).items()
        }
        self.hift.load_state_dict(hift_state, strict=True)
        self.hift.to(self.device).eval()

    load._tts_checkpoint_sanitized = True  # type: ignore[attr-defined]
    CosyVoiceModel.load = load  # type: ignore[method-assign]
    _patched = True
    logger.info("Installed CosyVoice training-checkpoint load compatibility")
    return True
