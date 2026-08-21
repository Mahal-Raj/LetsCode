"""Image preprocessing hypotheses used before OCR."""

from __future__ import annotations

import cv2
import numpy as np

SUPPORTED_VARIANTS = ("raw", "clahe", "otsu", "adaptive")


def _upscale(image: np.ndarray, factor: float = 2.0) -> np.ndarray:
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def preprocess(image: np.ndarray, variant: str) -> np.ndarray:
    """Return one deterministic preprocessing hypothesis."""
    if image.size == 0:
        raise ValueError("Cannot preprocess an empty ROI")
    if variant not in SUPPORTED_VARIANTS:
        supported = ", ".join(SUPPORTED_VARIANTS)
        raise ValueError(f"Unknown preprocessing variant {variant!r}; choose from {supported}")

    if variant == "raw":
        return _upscale(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = _upscale(gray)
    if variant == "clahe":
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    if variant == "otsu":
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )
    return cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
