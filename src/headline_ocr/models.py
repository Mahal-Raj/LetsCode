"""Data contracts shared by extraction, evaluation, and command-line layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class NormalizedROI:
    """Resolution-independent region of interest in the closed interval [0, 1]."""

    x: float = 0.02
    y: float = 0.64
    width: float = 0.96
    height: float = 0.32

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("ROI values must be between 0 and 1")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI width and height must be positive")
        if self.x + self.width > 1.0 or self.y + self.height > 1.0:
            raise ValueError("ROI must fit inside the frame")

    @classmethod
    def parse(cls, value: str) -> NormalizedROI:
        try:
            parts = tuple(float(item.strip()) for item in value.split(","))
        except ValueError as exc:
            raise ValueError("ROI must be x,y,width,height using normalized decimals") from exc
        if len(parts) != 4:
            raise ValueError("ROI must contain exactly four comma-separated values")
        return cls(*parts)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        frame_height, frame_width = frame.shape[:2]
        x0 = int(round(self.x * frame_width))
        y0 = int(round(self.y * frame_height))
        x1 = int(round((self.x + self.width) * frame_width))
        y1 = int(round((self.y + self.height) * frame_height))
        return frame[y0:y1, x0:x1]


@dataclass(frozen=True)
class ExtractorConfig:
    """Runtime configuration for one extraction pass."""

    roi: NormalizedROI = field(default_factory=NormalizedROI)
    sample_fps: float = 2.0
    mode: Literal["auto", "static", "ticker"] = "auto"
    preprocessing: tuple[str, ...] = ("raw", "clahe", "otsu")
    minimum_confidence: float = 0.35
    minimum_words: int = 3
    similarity_threshold: float = 82.0
    static_support_ratio: float = 0.34

    def __post_init__(self) -> None:
        if self.sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        if not 0 <= self.minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be between 0 and 1")
        if self.minimum_words < 1:
            raise ValueError("minimum_words must be at least one")


@dataclass(frozen=True)
class OCRCandidate:
    """One normalized OCR hypothesis from one sampled frame."""

    frame_index: int
    timestamp_seconds: float
    text: str
    confidence: float
    variant: str
    quality: float
    elapsed_seconds: float = 0.0


@dataclass
class ExtractionResult:
    """Serializable result of extracting one video."""

    video: str
    headline: str
    detected: bool
    mode_used: str
    sampled_frames: int
    duration_seconds: float
    processing_seconds: float
    candidates: list[OCRCandidate]
    configuration: ExtractorConfig

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["video"] = str(Path(self.video))
        return payload
