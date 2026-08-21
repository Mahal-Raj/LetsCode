"""Confidence-aware temporal consensus and ticker assembly."""

from __future__ import annotations

from collections import Counter

from rapidfuzz import fuzz

from .models import OCRCandidate
from .text import normalize_text, words


def _token_similarity(left: str, right: str) -> float:
    return fuzz.ratio(left, right)


def merge_overlapping_text(left: str, right: str, threshold: float = 78.0) -> str | None:
    """Merge chronological ticker windows using fuzzy token overlap."""
    left = normalize_text(left)
    right = normalize_text(right)
    if not left:
        return right
    if not right:
        return left
    if left in right:
        return right
    if right in left:
        return left

    left_words = words(left)
    right_words = words(right)
    max_overlap = min(len(left_words), len(right_words))
    for overlap in range(max_overlap, 0, -1):
        tail = left_words[-overlap:]
        head = right_words[:overlap]
        similarity = sum(_token_similarity(a, b) for a, b in zip(tail, head, strict=True))
        similarity /= overlap
        if similarity >= threshold:
            reconciled = [
                left_token if len(left_token) >= len(right_token) else right_token
                for left_token, right_token in zip(tail, head, strict=True)
            ]
            return " ".join([*left_words[:-overlap], *reconciled, *right_words[overlap:]])

    # Character overlap rescues words clipped at the ROI boundary.
    compact_left = " ".join(left_words)
    compact_right = " ".join(right_words)
    limit = min(len(compact_left), len(compact_right), 40)
    for overlap in range(limit, 5, -1):
        if fuzz.ratio(compact_left[-overlap:], compact_right[:overlap]) >= threshold:
            return compact_left + compact_right[overlap:]
    return None


def _deduplicate_time_order(candidates: list[OCRCandidate], threshold: float) -> list[OCRCandidate]:
    ordered = sorted(candidates, key=lambda item: (item.frame_index, -item.quality))
    by_frame: dict[int, OCRCandidate] = {}
    for candidate in ordered:
        previous = by_frame.get(candidate.frame_index)
        if previous is None or candidate.quality > previous.quality:
            by_frame[candidate.frame_index] = candidate
    sequence = [by_frame[index] for index in sorted(by_frame)]

    deduplicated: list[OCRCandidate] = []
    for candidate in sequence:
        if deduplicated and fuzz.ratio(candidate.text, deduplicated[-1].text) >= threshold:
            if candidate.quality > deduplicated[-1].quality:
                deduplicated[-1] = candidate
            continue
        deduplicated.append(candidate)
    return deduplicated


def _largest_support(candidates: list[OCRCandidate], threshold: float) -> int:
    support = 0
    for candidate in candidates:
        count = sum(fuzz.ratio(candidate.text, other.text) >= threshold for other in candidates)
        support = max(support, count)
    return support


def static_consensus(candidates: list[OCRCandidate]) -> str:
    """Choose the quality-weighted medoid rather than trusting one frame."""
    if not candidates:
        return ""
    best_text = ""
    best_score = -1.0
    for candidate in candidates:
        agreement = sum(
            other.quality * fuzz.ratio(candidate.text, other.text) / 100.0 for other in candidates
        )
        score = agreement + candidate.quality + min(len(words(candidate.text)), 12) / 20.0
        if score > best_score:
            best_text = candidate.text
            best_score = score
    return normalize_text(best_text)


def ticker_consensus(candidates: list[OCRCandidate], similarity_threshold: float) -> str:
    sequence = _deduplicate_time_order(candidates, similarity_threshold)
    if not sequence:
        return ""

    chains: list[tuple[str, float]] = []
    for candidate in sequence:
        best_index: int | None = None
        best_merge = candidate.text
        best_gain = 0
        for index, (chain, score) in enumerate(chains):
            merged = merge_overlapping_text(chain, candidate.text, similarity_threshold - 4)
            if merged is None:
                continue
            gain = len(words(merged)) - len(words(chain))
            if gain > best_gain or (
                gain == best_gain
                and score > (chains[best_index][1] if best_index is not None else -1)
            ):
                best_index = index
                best_merge = merged
                best_gain = gain
        if best_index is None:
            chains.append((candidate.text, candidate.quality))
        else:
            chain, score = chains[best_index]
            chains[best_index] = (best_merge, score + candidate.quality + max(best_gain, 0))

    return normalize_text(max(chains, key=lambda item: (len(words(item[0])), item[1]))[0])


def aggregate(
    candidates: list[OCRCandidate],
    *,
    mode: str,
    similarity_threshold: float,
    static_support_ratio: float,
) -> tuple[str, str]:
    if not candidates:
        return "", "none"
    frame_best = _deduplicate_time_order(candidates, 100.0)
    if mode == "auto":
        support = _largest_support(frame_best, similarity_threshold)
        word_counts = [len(words(candidate.text)) for candidate in frame_best]
        count_span = max(word_counts) - min(word_counts)
        support_ratio = support / max(len(frame_best), 1)
        inferred = (
            "ticker"
            if count_span >= 2 and support_ratio < 0.60
            else "static"
            if support_ratio >= static_support_ratio
            else "ticker"
        )
    else:
        inferred = mode
    if inferred == "static":
        return static_consensus(frame_best), inferred
    return ticker_consensus(frame_best, similarity_threshold), inferred


def variant_usage(candidates: list[OCRCandidate]) -> Counter[str]:
    return Counter(candidate.variant for candidate in candidates)
