"""Evaluation metrics for clip-level headline extraction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from rapidfuzz.distance import Levenshtein

from .text import normalize_text, words


def character_error_rate(reference: str, hypothesis: str) -> float:
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)
    return Levenshtein.distance(reference, hypothesis) / max(len(reference), 1)


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = words(normalize_text(reference))
    hypothesis_words = words(normalize_text(hypothesis))
    return Levenshtein.distance(reference_words, hypothesis_words) / max(len(reference_words), 1)


def normalized_similarity(reference: str, hypothesis: str) -> float:
    return 1.0 - min(character_error_rate(reference, hypothesis), 1.0)


@dataclass(frozen=True)
class ClipScore:
    cer: float
    wer: float
    similarity: float
    exact: bool
    success: bool


def score_clip(reference: str, hypothesis: str, threshold: float = 0.80) -> ClipScore:
    cer = character_error_rate(reference, hypothesis)
    wer = word_error_rate(reference, hypothesis)
    similarity = 1.0 - min(cer, 1.0)
    return ClipScore(
        cer=cer,
        wer=wer,
        similarity=similarity,
        exact=normalize_text(reference) == normalize_text(hypothesis),
        success=similarity >= threshold,
    )


def bootstrap_paired_difference(
    baseline: list[bool], proposed: list[bool], *, seed: int = 4086, samples: int = 5000
) -> tuple[float, float, float]:
    if len(baseline) != len(proposed) or not baseline:
        raise ValueError("Paired bootstrap requires equal non-empty samples")
    baseline_array = np.asarray(baseline, dtype=float)
    proposed_array = np.asarray(proposed, dtype=float)
    observed = float(np.mean(proposed_array - baseline_array))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(baseline), size=(samples, len(baseline)))
    differences = np.mean(proposed_array[indices] - baseline_array[indices], axis=1)
    low, high = np.percentile(differences, [2.5, 97.5])
    return observed, float(low), float(high)


def mcnemar_exact(baseline: list[bool], proposed: list[bool]) -> float:
    """Two-sided exact McNemar test for paired binary outcomes."""
    baseline_only = sum(a and not b for a, b in zip(baseline, proposed, strict=True))
    proposed_only = sum(b and not a for a, b in zip(baseline, proposed, strict=True))
    discordant = baseline_only + proposed_only
    if discordant == 0:
        return 1.0
    tail = min(baseline_only, proposed_only)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)
