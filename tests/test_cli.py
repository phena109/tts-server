"""CLI argument parsing (no CosyVoice / torch)."""

from __future__ import annotations

import pytest

from app.cli import build_parser


def test_tts_requires_source_when_no_args() -> None:
    parser = build_parser()
    args = parser.parse_args(["tts"])
    assert args.text is None
    assert args.file is None
    assert args.long is False


def test_tts_text_and_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "tts",
            "--text",
            "你好",
            "--language",
            "yue",
            "--speaker",
            "default",
            "--format",
            "mp3",
            "--speed",
            "1.1",
            "--long",
            "-o",
            "out.mp3",
        ]
    )
    assert args.command == "tts"
    assert args.text == "你好"
    assert args.language == "yue"
    assert args.speaker == "default"
    assert args.format == "mp3"
    assert args.speed == pytest.approx(1.1)
    assert args.long is True
    assert args.output == "out.mp3"


def test_tts_file_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["tts", "--file", "/input/a.txt"])
    assert args.file == "/input/a.txt"
    assert args.text is None


def test_ensure_model_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(["ensure-model"])
    assert args.command == "ensure-model"


def test_unknown_command_errors() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["nope"])
