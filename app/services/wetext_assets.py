"""Ensure CosyVoice wetext (WeTextProcessing) FST assets are available.

``wetext.Normalizer`` calls ModelScope ``snapshot_download("pengzhendong/wetext")``.
In some environments that API returns ``Authentication token does not exist`` even
for the public model. Git clone from ModelScope still works, so we:

1. Clone (or reuse) the FST tree under ``CACHE_DIR/wetext``
2. Monkey-patch ``snapshot_download`` so ``wetext`` resolves to that tree

Must run **before** CosyVoice constructs ``CosyVoiceFrontEnd`` (first wetext import).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from app.utils.logging import get_logger

logger = get_logger(__name__)

WETEXT_MODEL_ID = "pengzhendong/wetext"
# ModelScope git endpoint (works when hub API auth fails for anonymous users).
WETEXT_GIT_URL = "https://www.modelscope.cn/pengzhendong/wetext.git"

# Files CosyVoice needs for zh/en TN (see wetext.Normalizer with lang=auto).
_REQUIRED_FSTS = (
    Path("zh/tn/tagger.fst"),
    Path("zh/tn/verbalizer.fst"),
    Path("en/tn/tagger.fst"),
    Path("en/tn/verbalizer.fst"),
)

_patch_installed = False
_real_snapshot_download = None  # type: ignore[var-annotated]


def wetext_repo_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "wetext"


def wetext_assets_ready(repo: Path) -> bool:
    return all((repo / rel).is_file() for rel in _REQUIRED_FSTS)


def ensure_wetext_assets(cache_dir: Path) -> Path:
    """Return local wetext FST repo, cloning from ModelScope if needed."""
    repo = wetext_repo_path(cache_dir)
    if wetext_assets_ready(repo):
        logger.info("wetext FST assets already present", extra={"path": str(repo)})
        return repo

    repo.parent.mkdir(parents=True, exist_ok=True)
    tmp = repo.parent / "wetext.git-tmp"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)

    logger.info(
        "Cloning wetext FST assets from ModelScope git",
        extra={"url": WETEXT_GIT_URL, "dest": str(repo)},
    )
    env = os.environ.copy()
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    # Prefer LFS smudge so .fst files materialize (image has git-lfs).
    try:
        subprocess.run(
            ["git", "lfs", "install", "--local"],
            cwd=str(repo.parent),
            env=env,
            check=False,
            capture_output=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            WETEXT_GIT_URL,
            str(tmp),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )

    # Drop .git to save space on the volume (FSTs are already checked out).
    git_dir = tmp / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir, ignore_errors=True)

    if repo.exists():
        shutil.rmtree(repo, ignore_errors=True)
    tmp.rename(repo)

    if not wetext_assets_ready(repo):
        missing = [str(rel) for rel in _REQUIRED_FSTS if not (repo / rel).is_file()]
        raise RuntimeError(
            f"wetext clone incomplete at {repo}; missing: {', '.join(missing)}"
        )

    logger.info("wetext FST assets ready", extra={"path": str(repo)})
    return repo


def install_wetext_snapshot_patch(repo: Path) -> None:
    """Make modelscope/wetext ``snapshot_download('pengzhendong/wetext')`` return *repo*."""
    global _patch_installed, _real_snapshot_download
    if _patch_installed:
        return
    if not wetext_assets_ready(repo):
        raise FileNotFoundError(f"wetext assets not ready at {repo}")

    repo_str = str(repo.resolve())

    def _patched(model_id: str, *args: object, **kwargs: object) -> str:
        mid = str(model_id).strip()
        if mid == WETEXT_MODEL_ID or mid.endswith("/wetext"):
            return repo_str
        real = _real_snapshot_download
        if real is None:
            raise RuntimeError("wetext snapshot patch not fully installed")
        return real(model_id, *args, **kwargs)  # type: ignore[operator]

    # Patch all common import paths before wetext binds the name.
    import modelscope
    import modelscope.hub.snapshot_download as snap_mod

    _real_snapshot_download = snap_mod.snapshot_download
    snap_mod.snapshot_download = _patched  # type: ignore[assignment]
    modelscope.snapshot_download = _patched  # type: ignore[attr-defined]

    try:
        import wetext.wetext as wetext_mod

        wetext_mod.snapshot_download = _patched  # type: ignore[assignment]
    except ImportError:
        pass

    _patch_installed = True
    logger.info(
        "Installed wetext snapshot_download patch",
        extra={"repo": repo_str},
    )


def prepare_wetext_for_cosyvoice(cache_dir: Path) -> Path:
    """Ensure assets + patch so CosyVoice frontend can construct wetext Normalizers."""
    repo = ensure_wetext_assets(cache_dir)
    install_wetext_snapshot_patch(repo)
    return repo
