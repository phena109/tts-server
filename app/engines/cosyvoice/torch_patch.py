"""Work around torch SIGILL bugs on Apple Silicon Podman (linux/arm64).

Three separate issues under Apple Virtualization:

1. ``torch.nn.functional.linear`` **without bias** SIGILLs (with a zero bias is OK).
2. ``torch.matmul`` / ``torch.bmm`` on 3D+ tensors SIGILL (2D and NumPy are fine).
3. CosyVoice flow encoder attention (``RelPositionMultiHeadedAttention``) uses
   4D ``torch.matmul`` during ``token2wav`` — same bug, separate from Qwen2.

We:
* add zero biases to every bias-free ``nn.Linear`` after model load
* replace Qwen2 ``eager_attention_forward`` QK/AV products with NumPy matmul
* route global 3D+ ``torch.matmul`` / ``torch.bmm`` through NumPy
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from app.utils.logging import get_logger

logger = get_logger(__name__)

_attention_patched = False
_matmul_patched = False


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


def _numpy_matmul(a: Any, b: Any) -> Any:
    """Compute matmul via NumPy (broadcast-compatible with torch for batch dims)."""
    import numpy as np
    import torch

    a_dev = a.device
    a_dtype = a.dtype
    a_cpu = a.detach().cpu()
    b_cpu = b.detach().cpu()
    if a_cpu.is_floating_point():
        a_arr = a_cpu.float().numpy()
    else:
        a_arr = a_cpu.numpy()
    if b_cpu.is_floating_point():
        b_arr = b_cpu.float().numpy()
    else:
        b_arr = b_cpu.numpy()
    out = np.matmul(a_arr, b_arr)
    result = torch.from_numpy(np.ascontiguousarray(out))
    if a_dtype.is_floating_point:
        result = result.to(dtype=a_dtype)
    return result.to(device=a_dev)


def install_safe_matmul() -> None:
    """Route 3D+ torch.matmul/bmm through NumPy to avoid Apple Silicon SIGILL.

    CosyVoice flow attention (``transformer/attention.py``) crashes on
    ``torch.matmul`` of 4D tensors during ``token2wav``. 1D/2D stay on torch
    kernels (those are generally safe under Apple Virtualization).
    """
    global _matmul_patched
    if _matmul_patched:
        return

    import torch

    _orig_matmul = torch.matmul
    _orig_bmm = torch.bmm

    def safe_matmul(input: Any, other: Any, *, out: Any = None) -> Any:  # noqa: A002
        if (
            isinstance(input, torch.Tensor)
            and isinstance(other, torch.Tensor)
            and (input.dim() >= 3 or other.dim() >= 3)
        ):
            result = _numpy_matmul(input, other)
            if out is not None:
                out.copy_(result)
                return out
            return result
        if out is None:
            return _orig_matmul(input, other)
        return _orig_matmul(input, other, out=out)

    def safe_bmm(input: Any, mat2: Any, *, out: Any = None) -> Any:  # noqa: A002
        # bmm is always batch 3D — always use NumPy on this platform.
        if isinstance(input, torch.Tensor) and isinstance(mat2, torch.Tensor):
            result = _numpy_matmul(input, mat2)
            if out is not None:
                out.copy_(result)
                return out
            return result
        if out is None:
            return _orig_bmm(input, mat2)
        return _orig_bmm(input, mat2, out=out)

    def safe_tensor_matmul(self: Any, other: Any) -> Any:
        return safe_matmul(self, other)

    torch.matmul = safe_matmul  # type: ignore[assignment]
    torch.bmm = safe_bmm  # type: ignore[assignment]
    torch.Tensor.matmul = safe_tensor_matmul  # type: ignore[method-assign,assignment]
    # ``a @ b`` uses __matmul__ / __rmatmul__; patch for older torch paths too.
    torch.Tensor.__matmul__ = (  # type: ignore[method-assign,assignment]
        lambda self, other: safe_matmul(self, other)
    )
    torch.Tensor.__rmatmul__ = (  # type: ignore[method-assign,assignment]
        lambda self, other: safe_matmul(other, self)
    )

    _matmul_patched = True
    logger.info("Installed SIGILL-safe torch.matmul/bmm (NumPy for 3D+ tensors)")


def install_safe_attention() -> None:
    global _attention_patched
    if _attention_patched:
        return

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
        return _numpy_matmul(a, b)

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
