"""Unit tests for intelligent text chunking (no model required)."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running tests without installing the package
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.chunking import chunk_text


def test_short_text_single_chunk() -> None:
    text = "你好，我係測試。"
    assert chunk_text(text, max_chars=200) == [text]


def test_paragraph_preference() -> None:
    # Each paragraph is longer than max_chars so they cannot pack together
    p1 = "第一段内容。还是第一段。这段足够长以便单独成块。"
    p2 = "第二段内容。还是第二段。这段也足够长以便单独成块。"
    text = f"{p1}\n\n{p2}"
    chunks = chunk_text(text, max_chars=24)
    assert len(chunks) >= 2
    assert any("第一段" in c for c in chunks)
    assert any("第二段" in c for c in chunks)


def test_sentence_split_cjk() -> None:
    text = "这是第一句。这是第二句！这是第三句？这是第四句。这是第五句。"
    chunks = chunk_text(text, max_chars=16)
    assert len(chunks) >= 2
    joined = "".join(chunks)
    assert "第一句" in joined and "第五句" in joined


def test_avoid_split_inside_parens_when_possible() -> None:
    text = "他说（这是括号内的内容，比较长一些）然后继续。"
    chunks = chunk_text(text, max_chars=80)
    # With generous max_chars the whole balanced sentence stays together
    assert any("（" in c and "）" in c for c in chunks) or len(chunks) == 1


def test_long_article_produces_multiple_chunks() -> None:
    para = "这是一句用于测试长文拆分的示例句子。"
    text = "\n\n".join([para * 3] * 5)
    chunks = chunk_text(text, max_chars=60)
    assert len(chunks) > 1
    assert all(len(c) <= 80 for c in chunks)  # hard-split may slightly exceed soft target


def test_empty_returns_empty() -> None:
    assert chunk_text("   ", max_chars=100) == []


def test_latin_sentences() -> None:
    text = "Hello world. This is a second sentence! And a third one follows?"
    chunks = chunk_text(text, max_chars=28)
    assert len(chunks) >= 2
