"""Tests for CosyVoice torch SIGILL workarounds (host-side, torch optional)."""

from __future__ import annotations

import pytest

from app.engines.cosyvoice import torch_patch as tp


def test_install_safe_matmul_idempotent() -> None:
    torch = pytest.importorskip("torch")

    # Reset module flag so we exercise install in this process.
    tp._matmul_patched = False
    tp.install_safe_matmul()
    tp.install_safe_matmul()  # second call is a no-op

    a = torch.randn(2, 3, 4, 5)
    b = torch.randn(2, 3, 5, 6)
    out = torch.matmul(a, b)
    assert out.shape == (2, 3, 4, 6)

    # 2D path still works
    x = torch.randn(4, 5)
    y = torch.randn(5, 3)
    z = torch.matmul(x, y)
    assert z.shape == (4, 3)

    # bmm
    ba = torch.randn(2, 4, 5)
    bb = torch.randn(2, 5, 3)
    bc = torch.bmm(ba, bb)
    assert bc.shape == (2, 4, 3)


def test_numpy_matmul_matches_expected_shape() -> None:
    torch = pytest.importorskip("torch")

    a = torch.randn(1, 2, 3, 4)
    b = torch.randn(1, 2, 4, 5)
    out = tp._numpy_matmul(a, b)
    assert out.shape == (1, 2, 3, 5)
    assert out.device == a.device
