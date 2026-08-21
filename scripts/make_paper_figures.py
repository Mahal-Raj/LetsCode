"""Generate paper figures from the deterministic benchmark artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from headline_ocr.benchmark import BENCHMARK_ROI

ROOT = Path(__file__).resolve().parents[1]


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 1.0,
    color: tuple[int, int, int] = (29, 36, 48),
    thickness: int = 2,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


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


def benchmark_comparison() -> Path:
    """Plot the headline-level metrics used in the manuscript from summary.csv."""
    with (ROOT / "results/summary.csv").open(newline="", encoding="utf-8") as source:
        rows = {row["approach"]: row for row in csv.DictReader(source)}

    baseline = rows["frame_baseline"]
    temporal = rows["temporal_consensus"]
    metrics = (
        ("CER", "lower is better", "mean_cer"),
        ("WER", "lower is better", "mean_wer"),
        ("Exact", "higher is better", "exact_match_rate"),
        ("Success @ 0.80", "higher is better", "success_at_0_80"),
    )

    canvas = np.full((600, 1800, 3), 255, dtype=np.uint8)
    plot_left, plot_right = 390, 1690
    plot_top, plot_bottom = 95, 485
    baseline_color = (178, 114, 0)  # Okabe-Ito blue (#0072B2), stored as BGR.
    temporal_color = (0, 94, 213)  # Okabe-Ito vermillion (#D55E00), stored as BGR.
    grid_color = (222, 222, 222)

    put_text(canvas, "Center frame", (510, 58), scale=1.0, thickness=2)
    cv2.rectangle(canvas, (455, 31), (495, 59), baseline_color, -1)
    put_text(canvas, "Temporal consensus", (980, 58), scale=1.0, thickness=2)
    cv2.rectangle(canvas, (925, 31), (965, 59), temporal_color, -1)

    for tick in (0, 25, 50, 75, 100):
        x = int(plot_left + (plot_right - plot_left) * tick / 100)
        cv2.line(canvas, (x, plot_top), (x, plot_bottom), grid_color, 2)
        put_text(canvas, f"{tick}%", (x - 27, 82), scale=0.68, color=(90, 90, 90), thickness=1)

    for index, (label, direction, key) in enumerate(metrics):
        center_y = 135 + index * 90
        put_text(canvas, label, (35, center_y - 3), scale=0.92, thickness=2)
        put_text(
            canvas,
            direction,
            (35, center_y + 34),
            scale=0.60,
            color=(100, 100, 100),
            thickness=1,
        )
        values = (
            (float(baseline[key]) * 100, baseline_color, center_y - 31),
            (float(temporal[key]) * 100, temporal_color, center_y + 9),
        )
        for value, color, y in values:
            x_end = int(plot_left + (plot_right - plot_left) * value / 100)
            cv2.rectangle(canvas, (plot_left, y), (max(x_end, plot_left + 3), y + 28), color, -1)
            put_text(
                canvas,
                f"{value:.2f}%",
                (min(x_end + 14, plot_right - 115), y + 23),
                scale=0.68,
                color=(30, 30, 30),
                thickness=2,
            )

    cv2.line(canvas, (plot_left, plot_bottom), (plot_right, plot_bottom), (80, 80, 80), 2)
    put_text(
        canvas,
        "50 controlled clips | 40 positive + 10 controls | fixed similarity threshold: 0.80",
        (365, 565),
        scale=0.72,
        color=(75, 75, 75),
        thickness=1,
    )

    destination = ROOT / "paper/figures/benchmark_comparison.png"
    if not cv2.imwrite(str(destination), canvas):
        raise RuntimeError(f"Could not write {destination}")
    return destination


if __name__ == "__main__":
    print(ticker_sequence())
    print(benchmark_comparison())
