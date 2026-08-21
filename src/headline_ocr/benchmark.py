"""Deterministic synthetic-video benchmark and paired ablation study."""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

from .metrics import (
    bootstrap_paired_difference,
    mcnemar_exact,
    score_clip,
)
from .models import ExtractorConfig, NormalizedROI, OCRCandidate
from .pipeline import HeadlineExtractor
from .temporal import aggregate

HEADLINES = (
    "CITY COUNCIL APPROVES NEW TRANSIT PLAN",
    "REGIONAL HOSPITAL OPENS COMMUNITY CLINIC",
    "RESEARCHERS RELEASE OPEN CLIMATE DATASET",
    "CENTRAL BANK HOLDS INTEREST RATE STEADY",
    "LOCAL TEAM ADVANCES TO CHAMPIONSHIP FINAL",
)
CONDITIONS = ("clean", "low_contrast", "gaussian_noise", "blur_compression")
MODES = ("static", "ticker")
NEGATIVE_LABELS = ("", "LIVE", "HD", "12:45", "WEATHER", "SPORTS", "LIVE HD", "8:30 PM")
BENCHMARK_ROI = NormalizedROI(0.03, 0.68, 0.94, 0.21)


@dataclass(frozen=True)
class ClipSpec:
    clip_id: str
    path: str
    ground_truth: str
    mode: str
    condition: str
    seed: int


def _base_frame(width: int, height: int, frame_index: int, rng: np.random.Generator) -> np.ndarray:
    x_gradient = np.linspace(0, 75, width, dtype=np.uint8)
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = 24 + x_gradient
    frame[:, :, 1] = 38 + np.roll(x_gradient, frame_index * 3)
    frame[:, :, 2] = 65 + x_gradient // 2
    for _ in range(7):
        x0 = int(rng.integers(0, width - 40))
        y0 = int(rng.integers(0, int(height * 0.58)))
        rectangle_width = int(rng.integers(30, 150))
        rectangle_height = int(rng.integers(20, 90))
        color = tuple(int(value) for value in rng.integers(25, 180, size=3))
        cv2.rectangle(frame, (x0, y0), (x0 + rectangle_width, y0 + rectangle_height), color, -1)
    return frame


def _fit_scale(text: str, maximum_width: int, initial: float = 0.86) -> float:
    scale = initial
    while scale > 0.45:
        width = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)[0][0]
        if width <= maximum_width:
            return scale
        scale -= 0.03
    return scale


def _degrade(frame: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    if condition == "gaussian_noise":
        noise = rng.normal(0, 28, frame.shape).astype(np.int16)
        return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if condition == "blur_compression":
        kernel = np.zeros((7, 7), dtype=np.float32)
        kernel[3, :] = 1.0 / 7.0
        blurred = cv2.filter2D(frame, -1, kernel)
        ok, encoded = cv2.imencode(".jpg", blurred, [cv2.IMWRITE_JPEG_QUALITY, 25])
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else blurred
    return frame


def _render_clip(spec: ClipSpec, width: int = 640, height: int = 360) -> None:
    fps = 6.0
    frame_count = 24
    destination = Path(spec.path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not initialize the MJPG benchmark writer")

    band_x, band_y = 20, 246
    band_width, band_height = 600, 72
    rng = np.random.default_rng(spec.seed)
    try:
        for frame_index in range(frame_count):
            frame = _base_frame(width, height, frame_index, rng)
            if spec.condition == "low_contrast":
                band_color = (74, 78, 82)
                text_color = (104, 108, 112)
            else:
                band_color = (12, 20, 35)
                text_color = (245, 247, 250)
            cv2.rectangle(
                frame,
                (band_x, band_y),
                (band_x + band_width, band_y + band_height),
                band_color,
                -1,
            )
            cv2.line(
                frame,
                (band_x, band_y),
                (band_x + band_width, band_y),
                (30, 90, 235),
                4,
            )

            text = spec.ground_truth
            if text:
                if spec.mode == "static":
                    scale = _fit_scale(text, band_width - 24)
                    text_width, text_height = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_DUPLEX, scale, 2
                    )[0]
                    x_position = band_x + (band_width - text_width) // 2
                else:
                    scale = 1.14
                    text_width, text_height = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_DUPLEX, scale, 2
                    )[0]
                    progress = frame_index / max(frame_count - 1, 1)
                    x_position = int(band_x + band_width - progress * (band_width + text_width))
                y_position = band_y + (band_height + text_height) // 2
                cv2.putText(
                    frame,
                    text,
                    (x_position, y_position),
                    cv2.FONT_HERSHEY_DUPLEX,
                    scale,
                    text_color,
                    2,
                    cv2.LINE_AA,
                )
            else:
                label = NEGATIVE_LABELS[spec.seed % len(NEGATIVE_LABELS)]
                if label:
                    cv2.putText(
                        frame,
                        label,
                        (band_x + 24, band_y + 47),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.75,
                        text_color,
                        2,
                        cv2.LINE_AA,
                    )
            writer.write(_degrade(frame, spec.condition, rng))
    finally:
        writer.release()


def generate_benchmark(root: str | Path, *, regenerate: bool = False) -> list[ClipSpec]:
    root = Path(root)
    video_directory = root / "videos"
    specifications: list[ClipSpec] = []
    seed = 4086
    for headline_index, headline in enumerate(HEADLINES):
        for mode in MODES:
            for condition in CONDITIONS:
                clip_id = f"h{headline_index + 1}_{mode}_{condition}"
                path = video_directory / f"{clip_id}.avi"
                specifications.append(ClipSpec(clip_id, str(path), headline, mode, condition, seed))
                seed += 1
    for negative_index in range(10):
        condition = CONDITIONS[negative_index % len(CONDITIONS)]
        clip_id = f"negative_{negative_index + 1}_{condition}"
        path = video_directory / f"{clip_id}.avi"
        specifications.append(ClipSpec(clip_id, str(path), "", "negative", condition, seed))
        seed += 1

    for spec in specifications:
        if regenerate or not Path(spec.path).is_file():
            _render_clip(spec)
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps([asdict(item) for item in specifications], indent=2), encoding="utf-8"
    )
    return specifications


def _best_at_frame(
    candidates: list[OCRCandidate], variants: set[str], midpoint_frame: int = 12
) -> str:
    eligible = [candidate for candidate in candidates if candidate.variant in variants]
    if not eligible:
        return ""
    nearest_distance = min(abs(candidate.frame_index - midpoint_frame) for candidate in eligible)
    eligible = [
        candidate
        for candidate in eligible
        if abs(candidate.frame_index - midpoint_frame) == nearest_distance
    ]
    return max(eligible, key=lambda item: item.quality).text if eligible else ""


def _approaches(candidates: list[OCRCandidate], config: ExtractorConfig) -> dict[str, str]:
    raw = _best_at_frame(candidates, {"raw"})
    ensemble = _best_at_frame(candidates, set(config.preprocessing))
    temporal, _ = aggregate(
        candidates,
        mode=config.mode,
        similarity_threshold=config.similarity_threshold,
        static_support_ratio=config.static_support_ratio,
    )
    return {"frame_baseline": raw, "preprocess_ensemble": ensemble, "temporal_consensus": temporal}


def _summary(rows: list[dict]) -> list[dict]:
    summary: list[dict] = []
    for approach in ("frame_baseline", "preprocess_ensemble", "temporal_consensus"):
        positives = [row for row in rows if row["is_positive"] and row["approach"] == approach]
        negatives = [row for row in rows if not row["is_positive"] and row["approach"] == approach]
        true_positive = sum(row["success"] for row in positives)
        false_negative = len(positives) - true_positive
        false_positive = sum(bool(row["hypothesis"]) for row in negatives)
        true_negative = len(negatives) - false_positive
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        summary.append(
            {
                "approach": approach,
                "positive_clips": len(positives),
                "negative_clips": len(negatives),
                "mean_cer": float(np.mean([row["cer"] for row in positives])),
                "mean_wer": float(np.mean([row["wer"] for row in positives])),
                "exact_match_rate": float(np.mean([row["exact"] for row in positives])),
                "success_at_0_80": float(np.mean([row["success"] for row in positives])),
                "precision_at_0_80": precision,
                "recall_at_0_80": recall,
                "f1_at_0_80": f1,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
            }
        )
    return summary


def _stratified_summary(rows: list[dict], field: str) -> list[dict]:
    output: list[dict] = []
    values = sorted({row[field] for row in rows if row["is_positive"]})
    for approach in ("frame_baseline", "preprocess_ensemble", "temporal_consensus"):
        for value in values:
            group = [
                row
                for row in rows
                if row["is_positive"] and row["approach"] == approach and row[field] == value
            ]
            output.append(
                {
                    "approach": approach,
                    field: value,
                    "clips": len(group),
                    "mean_cer": float(np.mean([row["cer"] for row in group])),
                    "mean_wer": float(np.mean([row["wer"] for row in group])),
                    "exact_match_rate": float(np.mean([row["exact"] for row in group])),
                    "success_at_0_80": float(np.mean([row["success"] for row in group])),
                }
            )
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_latex_macros(path: Path, summary: list[dict], metadata: dict) -> None:
    indexed = {row["approach"]: row for row in summary}
    baseline = indexed["frame_baseline"]
    ensemble = indexed["preprocess_ensemble"]
    temporal = indexed["temporal_consensus"]
    paired = metadata["paired_statistics"]

    def percent(value: float, digits: int = 1) -> str:
        return f"{100 * value:.{digits}f}\\%"

    p_value = paired["mcnemar_exact_p"]
    commands = {
        "BenchmarkClips": str(metadata["clip_count"]),
        "PositiveClips": str(metadata["positive_clips"]),
        "NegativeClips": str(metadata["negative_clips"]),
        "BaselineCER": percent(baseline["mean_cer"], 2),
        "TemporalCER": percent(temporal["mean_cer"], 2),
        "BaselineWER": percent(baseline["mean_wer"], 2),
        "TemporalWER": percent(temporal["mean_wer"], 2),
        "EnsembleCER": percent(ensemble["mean_cer"], 2),
        "EnsembleWER": percent(ensemble["mean_wer"], 2),
        "BaselineExact": percent(baseline["exact_match_rate"]),
        "TemporalExact": percent(temporal["exact_match_rate"]),
        "EnsembleExact": percent(ensemble["exact_match_rate"]),
        "BaselineSuccess": percent(baseline["success_at_0_80"]),
        "TemporalSuccess": percent(temporal["success_at_0_80"]),
        "EnsembleSuccess": percent(ensemble["success_at_0_80"]),
        "BaselineFOne": percent(baseline["f1_at_0_80"], 2),
        "EnsembleFOne": percent(ensemble["f1_at_0_80"], 2),
        "TemporalFOne": percent(temporal["f1_at_0_80"], 2),
        "SuccessDifference": f"{100 * paired['success_rate_difference']:.1f}",
        "BootstrapLow": f"{100 * paired['bootstrap_95_percent_ci'][0]:.1f}",
        "BootstrapHigh": f"{100 * paired['bootstrap_95_percent_ci'][1]:.1f}",
        "McNemarP": f"{p_value:.2e}",
        "MedianRuntime": f"{metadata['median_full_pipeline_seconds_per_clip']:.2f}",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in commands.items())
        + "\n",
        encoding="utf-8",
    )


def _paired_statistics(rows: list[dict]) -> dict:
    baseline_rows = {
        row["clip_id"]: bool(row["success"])
        for row in rows
        if row["is_positive"] and row["approach"] == "frame_baseline"
    }
    proposed_rows = {
        row["clip_id"]: bool(row["success"])
        for row in rows
        if row["is_positive"] and row["approach"] == "temporal_consensus"
    }
    clip_ids = sorted(set(baseline_rows) & set(proposed_rows))
    baseline = [baseline_rows[clip_id] for clip_id in clip_ids]
    proposed = [proposed_rows[clip_id] for clip_id in clip_ids]
    observed, low, high = bootstrap_paired_difference(baseline, proposed)
    return {
        "paired_positive_clips": len(clip_ids),
        "success_rate_difference": observed,
        "bootstrap_95_percent_ci": [low, high],
        "mcnemar_exact_p": mcnemar_exact(baseline, proposed),
        "bootstrap_seed": 4086,
        "bootstrap_samples": 5000,
    }


def run_benchmark(
    artifact_root: str | Path = "benchmark_artifacts",
    output_root: str | Path = "results",
    *,
    limit: int | None = None,
    regenerate: bool = False,
) -> dict:
    artifact_root = Path(artifact_root)
    output_root = Path(output_root)
    specs = generate_benchmark(artifact_root, regenerate=regenerate)
    if limit is not None:
        specs = specs[:limit]

    config = ExtractorConfig(
        roi=BENCHMARK_ROI,
        sample_fps=2.0,
        mode="auto",
        preprocessing=("raw", "clahe", "otsu"),
        minimum_confidence=0.30,
        minimum_words=3,
        similarity_threshold=80.0,
        static_support_ratio=0.34,
    )
    extractor = HeadlineExtractor(config=config)
    rows: list[dict] = []
    timings: list[float] = []
    for index, spec in enumerate(specs, start=1):
        started = time.perf_counter()
        candidates, sampled_frames, duration = extractor.collect_candidates(spec.path)
        processing_seconds = time.perf_counter() - started
        timings.append(processing_seconds)
        for approach, hypothesis in _approaches(candidates, config).items():
            if spec.ground_truth:
                score = score_clip(spec.ground_truth, hypothesis)
                cer, wer = score.cer, score.wer
                similarity, exact, success = score.similarity, score.exact, score.success
            else:
                cer = wer = 0.0 if not hypothesis else 1.0
                similarity, exact, success = (
                    (1.0, True, True) if not hypothesis else (0.0, False, False)
                )
            rows.append(
                {
                    "clip_id": spec.clip_id,
                    "condition": spec.condition,
                    "mode": spec.mode,
                    "is_positive": bool(spec.ground_truth),
                    "approach": approach,
                    "reference": spec.ground_truth,
                    "hypothesis": hypothesis,
                    "cer": round(cer, 6),
                    "wer": round(wer, 6),
                    "similarity": round(similarity, 6),
                    "exact": exact,
                    "success": success,
                    "sampled_frames": sampled_frames,
                    "processing_seconds_full_pipeline": round(processing_seconds, 6),
                    "video_duration_seconds": round(duration, 3),
                }
            )
        print(
            f"[{index:02d}/{len(specs):02d}] {spec.clip_id}: {processing_seconds:.2f}s", flush=True
        )

    summary = _summary(rows)
    statistics = _paired_statistics(rows) if all(spec.ground_truth for spec in specs[:1]) else {}
    metadata = {
        "benchmark_version": "1.0",
        "clip_count": len(specs),
        "positive_clips": sum(bool(spec.ground_truth) for spec in specs),
        "negative_clips": sum(not spec.ground_truth for spec in specs),
        "conditions": list(CONDITIONS),
        "modes": list(MODES),
        "frame_size": [640, 360],
        "fps": 6,
        "frames_per_clip": 24,
        "sample_fps": config.sample_fps,
        "median_full_pipeline_seconds_per_clip": float(np.median(timings)),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": version("numpy"),
            "opencv": version("opencv-python"),
            "onnxruntime": version("onnxruntime"),
            "rapidocr": version("rapidocr"),
        },
        "configuration": asdict(config),
        "paired_statistics": statistics,
    }
    _write_csv(output_root / "per_clip.csv", rows)
    _write_csv(output_root / "summary.csv", summary)
    _write_csv(output_root / "by_condition.csv", _stratified_summary(rows, "condition"))
    _write_csv(output_root / "by_mode.csv", _stratified_summary(rows, "mode"))
    (output_root / "benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    _write_latex_macros(Path("paper/generated/results_macros.tex"), summary, metadata)
    return {"summary": summary, "metadata": metadata}
