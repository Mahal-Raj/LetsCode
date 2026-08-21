"""OCR engine abstraction and RapidOCR implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class OCRReading:
    text: str
    confidence: float


class OCREngine(Protocol):
    def recognize(self, image: np.ndarray) -> OCRReading: ...


class RapidOCREngine:
    """CPU-friendly ONNX recognizer for a cropped, single-line headline band."""

    def __init__(self) -> None:
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:  # pragma: no cover - installation error path
            raise RuntimeError("Install the project dependencies before running OCR") from exc
        self._engine = RapidOCR()

    def recognize(self, image: np.ndarray) -> OCRReading:
        # The user supplies a headline ROI, so running a full-frame text detector
        # is both slower and less reproducible than line recognition on the crop.
        result = self._engine(image, use_det=False, use_cls=False, use_rec=True)
        if not result or not result.txts:
            return OCRReading("", 0.0)
        joined = " ".join(str(text) for text in result.txts)
        weighted_sum = sum(
            float(score) * max(len(str(text)), 1)
            for text, score in zip(result.txts, result.scores, strict=False)
        )
        total_length = sum(max(len(str(text)), 1) for text in result.txts)
        return OCRReading(joined, weighted_sum / total_length)
