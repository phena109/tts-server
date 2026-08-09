"""Tests for wetext FST asset bootstrap and snapshot_download patch."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services import wetext_assets as wa


def _fake_repo(root: Path) -> Path:
    repo = root / "wetext"
    for rel in wa._REQUIRED_FSTS:
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fst")
    return repo


def test_wetext_assets_ready(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    assert wa.wetext_assets_ready(repo) is True
    (repo / "zh/tn/tagger.fst").unlink()
    assert wa.wetext_assets_ready(repo) is False


def test_ensure_wetext_assets_uses_existing(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    # ensure_wetext_assets looks under cache_dir/wetext
    cache = tmp_path
    assert wa.ensure_wetext_assets(cache) == repo


def test_ensure_wetext_assets_clones_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(list(cmd))
        if cmd[:2] == ["git", "clone"]:
            dest = Path(cmd[-1])
            for rel in wa._REQUIRED_FSTS:
                p = dest / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"fst")
            return type("R", (), {"returncode": 0})()
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(wa.subprocess, "run", fake_run)
    repo = wa.ensure_wetext_assets(tmp_path)
    assert wa.wetext_assets_ready(repo)
    assert any(c[:2] == ["git", "clone"] for c in calls)


def test_install_wetext_snapshot_patch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _fake_repo(tmp_path)
    # Reset module flag so the test can reinstall.
    monkeypatch.setattr(wa, "_patch_installed", False)

    real_calls: list[str] = []

    def real_sd(model_id: str, *a, **k) -> str:  # type: ignore[no-untyped-def]
        real_calls.append(model_id)
        return "/elsewhere"

    import sys
    import types

    snap_mod = types.ModuleType("modelscope.hub.snapshot_download")
    snap_mod.snapshot_download = real_sd  # type: ignore[attr-defined]
    hub_mod = types.ModuleType("modelscope.hub")
    hub_mod.snapshot_download = snap_mod  # type: ignore[attr-defined]
    ms_mod = types.ModuleType("modelscope")
    ms_mod.snapshot_download = real_sd  # type: ignore[attr-defined]
    ms_mod.hub = hub_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "modelscope", ms_mod)
    monkeypatch.setitem(sys.modules, "modelscope.hub", hub_mod)
    monkeypatch.setitem(sys.modules, "modelscope.hub.snapshot_download", snap_mod)

    wa.install_wetext_snapshot_patch(repo)

    assert snap_mod.snapshot_download("pengzhendong/wetext") == str(repo.resolve())
    assert snap_mod.snapshot_download("other/model") == "/elsewhere"
    assert real_calls == ["other/model"]
