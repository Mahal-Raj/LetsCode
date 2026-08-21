"""Generate paper figures from the deterministic benchmark artifacts."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from headline_ocr.benchmark import BENCHMARK_ROI

ROOT = Path(__file__).resolve().parents[1]


def read_frame(path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {path}")
    return frame


def label_panel(image: np.ndarray, label: str) -> np.ndarray:
    panel = cv2.copyMakeBorder(image, 48, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(
        panel,
        label,
        (18, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return panel


def ticker_sequence() -> Path:
    video = ROOT / "benchmark_artifacts/videos/h3_ticker_blur_compression.avi"
    frames = []
    for frame_index, label in ((6, "t = 1.0 s"), (12, "t = 2.0 s"), (18, "t = 3.0 s")):
        crop = BENCHMARK_ROI.crop(read_frame(video, frame_index))
        crop = cv2.resize(crop, (720, 180), interpolation=cv2.INTER_CUBIC)
        frames.append(label_panel(crop, label))
    separator = np.full((232, 14, 3), 255, dtype=np.uint8)
    canvas = frames[0]
    for frame in frames[1:]:
        canvas = np.hstack((canvas, separator, frame))
    destination = ROOT / "paper/figures/ticker_sequence.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), canvas):
        raise RuntimeError(f"Could not write {destination}")
    return destination


if __name__ == "__main__":
    print(ticker_sequence())
