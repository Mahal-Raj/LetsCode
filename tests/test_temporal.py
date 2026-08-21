from headline_ocr.models import OCRCandidate
from headline_ocr.temporal import aggregate, merge_overlapping_text, static_consensus


def candidate(frame: int, text: str, quality: float = 0.9) -> OCRCandidate:
    return OCRCandidate(frame, frame / 2, text, 0.9, "raw", quality)


def test_merges_chronological_ticker_windows() -> None:
    merged = merge_overlapping_text(
        "CITY COUNCIL APPROVES NEW",
        "APPROVES NEW TRANSIT PLAN",
    )
    assert merged == "CITY COUNCIL APPROVES NEW TRANSIT PLAN"


def test_static_consensus_uses_agreement_not_outlier_length() -> None:
    candidates = [
        candidate(0, "CENTRAL BANK HOLDS RATE STEADY"),
        candidate(1, "CENTRAL BANK HOLDS RATE STEADY"),
        candidate(2, "ZZ CENTRAL BANK RANDOM RATE STEADY EXTRA WORDS", 0.6),
    ]
    assert static_consensus(candidates) == "CENTRAL BANK HOLDS RATE STEADY"


def test_auto_mode_identifies_static_support() -> None:
    candidates = [
        candidate(0, "REGIONAL HOSPITAL OPENS COMMUNITY CLINIC"),
        candidate(1, "REGIONAL HOSPITAL OPENS COMMUNITY CLINIC"),
        candidate(2, "REGIONAL HOSPITAL OPENS COMMUNITY CLINIC"),
    ]
    text, mode = aggregate(
        candidates,
        mode="auto",
        similarity_threshold=82,
        static_support_ratio=0.5,
    )
    assert mode == "static"
    assert text == "REGIONAL HOSPITAL OPENS COMMUNITY CLINIC"
