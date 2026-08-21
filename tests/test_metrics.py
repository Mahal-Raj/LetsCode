import pytest

from headline_ocr.metrics import (
    bootstrap_paired_difference,
    character_error_rate,
    mcnemar_exact,
    score_clip,
    word_error_rate,
)


def test_exact_metrics_are_zero_error() -> None:
    assert character_error_rate("A NEWS HEADLINE", "A NEWS HEADLINE") == 0
    assert word_error_rate("A NEWS HEADLINE", "A NEWS HEADLINE") == 0


def test_clip_success_uses_normalized_similarity() -> None:
    score = score_clip("CITY COUNCIL APPROVES PLAN", "CITY COUNCIL APPROVES PLAN")
    assert score.success
    assert score.exact


def test_paired_statistics_are_deterministic() -> None:
    observed, low, high = bootstrap_paired_difference(
        [False, False, True, False], [True, True, True, False], samples=200
    )
    assert observed == pytest.approx(0.5)
    assert low <= observed <= high
    assert 0 <= mcnemar_exact([False, False, True], [True, True, True]) <= 1
