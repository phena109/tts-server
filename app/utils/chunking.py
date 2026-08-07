"""Intelligent text chunking for long-form TTS.

Preference order:
1. Paragraph boundaries (blank lines)
2. Sentence boundaries
3. Soft punctuation / whitespace fallback

Avoids splitting inside quotes or parentheses when possible.
"""

from __future__ import annotations

import re
from typing import List

# Sentence terminators for CJK + Latin scripts
_SENTENCE_END = re.compile(
    r"(?<=[。！？!?；;…])"  # CJK / fullwidth
    r"|(?<=[.!?])(?=\s|$)"  # Latin followed by space/end
)

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")


def _is_balanced(text: str) -> bool:
    """Return True when quotes/parentheses are balanced enough to split after."""
    pairs = {
        "(": ")",
        "[": "]",
        "{": "}",
        "（": "）",
        "【": "】",
        "「": "」",
        "『": "』",
    }
    # Track simple open/close counts; ignore mismatched nested complexity
    stack: list[str] = []
    quote_chars = {'"', "'", "“", "”", "‘", "’", "「", "」", "『", "』"}
    # Treat paired CJK quotes separately via stack; latin quotes toggle
    latin_double = 0
    latin_single = 0

    i = 0
    while i < len(text):
        ch = text[i]
        if ch in pairs:
            stack.append(pairs[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
        elif ch == '"':
            latin_double ^= 1
        elif ch == "'":
            # Apostrophes inside words are common; only toggle when likely a quote
            prev = text[i - 1] if i > 0 else " "
            nxt = text[i + 1] if i + 1 < len(text) else " "
            if prev.isspace() or nxt.isspace() or prev in "([{" or nxt in ".,;:!?)]":
                latin_single ^= 1
        i += 1

    return not stack and latin_double == 0 and latin_single == 0


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences while preserving terminators."""
    text = text.strip()
    if not text:
        return []

    parts: list[str] = []
    last = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.end()
        candidate = text[last:end].strip()
        if candidate:
            parts.append(candidate)
            last = end
    tail = text[last:].strip()
    if tail:
        parts.append(tail)
    return parts or [text]


def _hard_split(text: str, max_chars: int) -> List[str]:
    """Last-resort split at whitespace or fixed width, preferring balanced spans."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        window = remaining[:max_chars]
        # Prefer last whitespace inside window
        split_at = max(window.rfind(" "), window.rfind("\n"), window.rfind("\t"))
        # Prefer last CJK-friendly soft break (comma etc.) if no space
        if split_at < max_chars // 3:
            for sep in ("，", "、", ",", "；", ";", "：", ":"):
                pos = window.rfind(sep)
                if pos >= max_chars // 3:
                    split_at = pos + 1
                    break

        if split_at < max_chars // 4:
            # Try to extend slightly to close open quotes/parens
            extended = min(len(remaining), max_chars + max_chars // 4)
            for end in range(max_chars, extended + 1):
                if _is_balanced(remaining[:end]):
                    split_at = end
                    break
            else:
                split_at = max_chars

        piece = remaining[:split_at].strip()
        if piece:
            chunks.append(piece)
        remaining = remaining[split_at:].strip()

    return chunks


def _pack_units(units: List[str], max_chars: int) -> List[str]:
    """Greedily pack small units into chunks up to max_chars."""
    if not units:
        return []

    chunks: list[str] = []
    current = ""

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue

        if len(unit) > max_chars:
            # Flush current, then hard-split oversized unit
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split(unit, max_chars))
            continue

        if not current:
            current = unit
            continue

        joiner = "\n" if "\n" in unit or "\n" in current else ""
        # Prefer space join for latin-ish content without newlines
        if not joiner:
            joiner = "" if _ends_with_cjk(current) or _starts_with_cjk(unit) else " "

        candidate = f"{current}{joiner}{unit}"
        if len(candidate) <= max_chars and _is_balanced(candidate):
            current = candidate
        else:
            chunks.append(current)
            current = unit

    if current:
        chunks.append(current)
    return chunks


def _ends_with_cjk(text: str) -> bool:
    if not text:
        return False
    code = ord(text[-1])
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x3000 <= code <= 0x303F
        or 0xFF00 <= code <= 0xFFEF
    )


def _starts_with_cjk(text: str) -> bool:
    if not text:
        return False
    code = ord(text[0])
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x3000 <= code <= 0x303F
        or 0xFF00 <= code <= 0xFFEF
    )


def chunk_text(text: str, max_chars: int = 200) -> List[str]:
    """Split *text* into synthesis-friendly chunks.

    Parameters
    ----------
    text:
        Full input text (article, paragraph, etc.).
    max_chars:
        Soft maximum characters per chunk. Oversized sentences are hard-split.

    Returns
    -------
    list[str]
        Non-empty chunks ready for sequential TTS.
    """
    if max_chars < 16:
        raise ValueError("max_chars must be >= 16")

    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    # 1) Paragraphs first (always), even when the whole document is short —
    #    packing may still merge small paragraphs under max_chars.
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    # Fast path: single short balanced paragraph
    if (
        len(paragraphs) == 1
        and len(paragraphs[0]) <= max_chars
        and _is_balanced(paragraphs[0])
    ):
        return [paragraphs[0]]

    # Expand each paragraph into sentence-level units when needed
    sentence_units: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chars and _is_balanced(para):
            sentence_units.append(para)
            continue
        # 2) Sentences within paragraph
        sentences = _split_sentences(para)
        packed = _pack_units(sentences, max_chars)
        sentence_units.extend(packed)

    # Pack units, but never re-join content that came from different paragraphs
    # beyond the greedy packer (paragraphs were already separate units).
    final = _pack_units(sentence_units, max_chars)

    # Final safety pass for any residual oversize unit
    safe: list[str] = []
    for unit in final:
        if len(unit) <= max_chars:
            safe.append(unit)
        else:
            safe.extend(_hard_split(unit, max_chars))

    return [c for c in safe if c.strip()]
