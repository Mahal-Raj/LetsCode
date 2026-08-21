import numpy as np
import pytest

from headline_ocr.models import NormalizedROI


def test_normalized_roi_crops_resolution_independently() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = NormalizedROI(0.25, 0.20, 0.50, 0.40).crop(frame)
    assert crop.shape == (40, 100, 3)


def test_invalid_roi_is_rejected() -> None:
    with pytest.raises(ValueError):
        NormalizedROI(0.8, 0.2, 0.3, 0.4)


def test_roi_parser_requires_four_values() -> None:
    with pytest.raises(ValueError):
        NormalizedROI.parse("0.1,0.2,0.3")
