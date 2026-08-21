"""Video sampling and headline extraction pipeline."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from .models import ExtractionResult, ExtractorConfig, OCRCandidate
from .ocr import OCREngine, RapidOCREngine
from .preprocessing import preprocess
from .temporal import aggregate
from .text import alphabetic_word_count, lexical_quality, normalize_text


class HeadlineExtractor:
    """Extract a static lower-third or assemble a scrolling news ticker."""

    def __init__(self, config: ExtractorConfig | None = None, engine: OCREngine | None = None):
        self.config = config or ExtractorConfig()
        self.engine = engine or RapidOCREngine()

    def _candidate(
        self,
        image,
        *,
        frame_index: int,
        timestamp: float,
        variant: str,
    ) -> OCRCandidate | None:
        started = time.perf_counter()
        reading = self.engine.recognize(preprocess(image, variant))
        elapsed = time.perf_counter() - started
        text = normalize_text(reading.text)
        quality = lexical_quality(text, reading.confidence)
        if reading.confidence < self.config.minimum_confidence:
            return None
        if alphabetic_word_count(text) < self.config.minimum_words:
            return None
        return OCRCandidate(
            frame_index=frame_index,
            timestamp_seconds=timestamp,
            text=text,
            confidence=reading.confidence,
            variant=variant,
            quality=quality,
            elapsed_seconds=elapsed,
        )

    def collect_candidates(self, video_path: str | Path) -> tuple[list[OCRCandidate], int, float]:
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path}")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open video: {path}")

        source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
        frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_total / source_fps if frame_total > 0 else 0.0
        interval = max(1, round(source_fps / self.config.sample_fps))
        candidates: list[OCRCandidate] = []
        sampled_frames = 0
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % interval != 0:
                    frame_index += 1
                    continue
                sampled_frames += 1
                roi = self.config.roi.crop(frame)
                for variant in self.config.preprocessing:
                    candidate = self._candidate(
                        roi,
                        frame_index=frame_index,
                        timestamp=frame_index / source_fps,
                        variant=variant,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
                frame_index += 1
        finally:
            capture.release()
        return candidates, sampled_frames, duration

    def extract(self, video_path: str | Path) -> ExtractionResult:
        started = time.perf_counter()
        candidates, sampled_frames, duration = self.collect_candidates(video_path)
        headline, mode_used = aggregate(
            candidates,
            mode=self.config.mode,
            similarity_threshold=self.config.similarity_threshold,
            static_support_ratio=self.config.static_support_ratio,
        )
        return ExtractionResult(
            video=str(Path(video_path).resolve()),
            headline=headline,
            detected=bool(headline),
            mode_used=mode_used,
            sampled_frames=sampled_frames,
            duration_seconds=duration,
            processing_seconds=time.perf_counter() - started,
            candidates=candidates,
            configuration=self.config,
        )
