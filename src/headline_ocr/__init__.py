"""Temporal-consensus OCR for broadcast-news headlines."""

from .models import ExtractionResult, ExtractorConfig, NormalizedROI, OCRCandidate
from .pipeline import HeadlineExtractor

__all__ = [
    "ExtractionResult",
    "ExtractorConfig",
    "HeadlineExtractor",
    "NormalizedROI",
    "OCRCandidate",
]

__version__ = "1.0.0"
