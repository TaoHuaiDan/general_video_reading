"""Replacement-finalized content must reach OCR despite inherited typewriter score.

End-to-end through ``engine.run``: a fixed geometry whose burned-in subtitle
line is replaced every sampled frame (the classic hard-sub pattern). Each
replacement finalizes the previous content track; those tracks inherit a
rising typewriter_score and must not be dropped by the defer.
"""

from __future__ import annotations

import numpy as np

from temporal_ocr.config import EngineConfig
from temporal_ocr.engine import TemporalOCREngine
from temporal_ocr.types import (
    DetectionObservation,
    DetectionRequest,
    FramePacket,
    OCRResult,
)

LINES = ["A", "B", "C", "D", "E", "F", "G", "H"]
POLYGON = ((200.0, 300.0), (420.0, 300.0), (420.0, 344.0), (200.0, 344.0))


def _frame(index: int) -> np.ndarray:
    """Black frame whose subtitle band carries a unique noisy pattern."""
    rng = np.random.default_rng(index)
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    base = 60 + 25 * index
    # Pattern mean identifies the line; the random texture makes every line's
    # perceptual signature far (>threshold) from every other line's.
    band = np.clip(base - 40 + rng.integers(0, 80, size=(44, 220)), 0, 255).astype(np.uint8)
    image[300:344, 200:420] = band[:, :, None]
    return image


class _SingleBoxDetector:
    name = "single-box-detector"

    def detect(self, frame: FramePacket, request: DetectionRequest):
        return [
            DetectionObservation(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                polygon=POLYGON,
                confidence=0.99,
                tier=request.tier,
            )
        ]


class _MeanTextRecognizer:
    name = "mean-text-recognizer"
    cache_namespace = "mean-text"

    def __init__(self, calls: list[int]) -> None:
        self.calls = calls

    def recognize_batch(self, tasks):
        self.calls.extend(round((float(np.mean(t.candidates[0].image)) - 60) / 25) for t in tasks)
        return [
            OCRResult(
                content_id=task.content_id,
                revision=task.revision,
                text=LINES[round((float(np.mean(task.candidates[0].image)) - 60) / 25)],
                confidence=0.95,
                backend="fake",
            )
            for task in tasks
        ]


def test_replacement_finalized_tracks_are_not_dropped_by_typewriter_defer() -> None:
    calls: list[int] = []
    recognizer = _MeanTextRecognizer(calls)

    config = EngineConfig()
    config.detection.track_guided_local = False
    engine = TemporalOCREngine(
        _SingleBoxDetector(),
        recognizer,
        config=config,
    )

    frames = [FramePacket(index, float(index), _frame(index)) for index in range(len(LINES))]
    result = engine.run(frames)

    recognized = {
        event.text_raw
        for event in result.events
        if event.text_raw
    }
    missing = [text for text in LINES if text not in recognized]
    assert missing == [], (
        f"subtitle lines lost to typewriter defer: {missing}; "
        f"deferred={result.profile.counters.get('ocr_deferred_typewriter')}"
    )
