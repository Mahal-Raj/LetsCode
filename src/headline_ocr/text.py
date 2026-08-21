"""OCR text normalization and quality heuristics."""

from __future__ import annotations

import re
import unicodedata

_SPACE = re.compile(r"\s+")
_DISALLOWED = re.compile(r"[^\w\s$%&'’+.,:;!?/\-]", flags=re.UNICODE)
_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:['’\-][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)*")


def normalize_text(text: str, *, uppercase: bool = True) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _DISALLOWED.sub(" ", normalized)
    normalized = _SPACE.sub(" ", normalized).strip(" -|_.,;:")
    return normalized.upper() if uppercase else normalized


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def alphabetic_word_count(text: str) -> int:
    return sum(any(character.isalpha() for character in word) for word in words(text))


def lexical_quality(text: str, confidence: float) -> float:
    """Dictionary-free score that does not reject names, places, or acronyms."""
    if not text:
        return 0.0
    visible = [char for char in text if not char.isspace()]
    alphanumeric_ratio = sum(char.isalnum() for char in visible) / max(len(visible), 1)
    length_saturation = min(len(words(text)) / 8.0, 1.0)
    return 0.72 * confidence + 0.18 * alphanumeric_ratio + 0.10 * length_saturation
