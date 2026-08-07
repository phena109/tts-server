"""CLI helpers used by entrypoint.sh (model ensure, smoke checks)."""

from __future__ import annotations

import argparse
import sys

from app.config.settings import get_settings
from app.services.model_manager import ModelManager
from app.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def ensure_model() -> int:
    settings = get_settings()
    setup_logging(settings.log_level)
    settings.ensure_directories()
    manager = ModelManager(settings)
    path = manager.ensure_model()
    print(str(path))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TTS server bootstrap utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ensure-model", help="Download model weights if missing")

    args = parser.parse_args(argv)
    if args.command == "ensure-model":
        return ensure_model()
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
