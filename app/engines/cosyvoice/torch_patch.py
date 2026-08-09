"""Work around torch SIGILL bugs on Apple Silicon Podman (linux/arm64).

Two separate issues under Apple Virtualization:

1. ``torch.nn.functional.linear`` **without bias** SIGILLs (with a zero bias is OK).
2. ``torch.matmul`` / ``torch.bmm`` on 3D+ tensors SIGILL (2D and NumPy are fine).

We:
* add zero biases to every bias-free ``nn.Linear`` after model load
* replace Qwen2 ``eager_attention_forward`` QK/AV products with NumPy matmul
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from app.utils.logging import get_logger

logger = get_logger(__name__)

_attention_patched = False


def ensure_linear_biases(root: Any) -> int:
    """Add zero bias parameters to bias-free Linear modules. Returns count fixed.

    Walks ``nn.Module.modules()`` when available, otherwise scans attributes
    recursively (CosyVoice wrappers are plain objects holding ``.llm`` / ``.flow``).
    """
    import torch
    import torch.nn as nn

    fixed = 0
    seen: set[int] = set()

    def _visit(obj: Any, depth: int = 0) -> None:
        nonlocal fixed
        if obj is None or depth > 8:
            return
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)

        if isinstance(obj, nn.Linear):
            if obj.bias is None:
                obj.bias = nn.Parameter(
                    torch.zeros(
                        obj.out_features,
                        device=obj.weight.device,
                        dtype=obj.weight.dtype,
                    )
                )
                fixed += 1
            return

        if isinstance(obj, nn.Module):
            for child in obj.modules():
                if isinstance(child, nn.Linear) and child.bias is None:
                    cid = id(child)
                    if cid in seen:
                        continue
                    seen.add(cid)
                    child.bias = nn.Parameter(
                        torch.zeros(
                            child.out_features,
                            device=child.weight.device,
                            dtype=child.weight.dtype,
                        )
                    )
                    fixed += 1
            return

        # Plain CosyVoice containers
        for name in (
            "model",
            "llm",
            "flow",
            "hift",
            "frontend",
            "llm_model",
            "flow_model",
        ):
            if hasattr(obj, name):
                try:
                    _visit(getattr(obj, name), depth + 1)
                except Exception:
                    pass

    _visit(root)
    if fixed:
        logger.info(
            "Added zero biases to Linear modules (SIGILL workaround)",
            extra={"count": fixed},
        )
    return fixed


def install_safe_attention() -> None:
    global _attention_patched
    if _attention_patched:
        return

    import numpy as np
    import torch
    import torch.nn as nn

    def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def _matmul_bh(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Batch-head matmul via NumPy: (B,H,M,K) @ (B,H,K,N) -> (B,H,M,N)."""
        a_np = a.detach().cpu().float().numpy()
        b_np = b.detach().cpu().float().numpy()
        out = np.matmul(a_np, b_np)
        return torch.from_numpy(out).to(device=a.device, dtype=a.dtype)

    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_states = _repeat_kv(key, module.num_key_value_groups)
        value_states = _repeat_kv(value, module.num_key_value_groups)

        # Avoid torch 4D matmul (SIGILL on this platform)
        attn_weights = _matmul_bh(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=dropout, training=module.training
        )
        attn_output = _matmul_bh(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    try:
        import transformers.models.qwen2.modeling_qwen2 as qwen2_mod

        qwen2_mod.eager_attention_forward = eager_attention_forward  # type: ignore[assignment]
        reg = getattr(qwen2_mod, "ALL_ATTENTION_FUNCTIONS", None)
        if isinstance(reg, dict):
            reg["eager"] = eager_attention_forward
        _attention_patched = True
        logger.info("Installed SIGILL-safe Qwen2 eager attention (NumPy matmul)")
    except Exception:
        logger.exception("Failed to patch Qwen2 attention")