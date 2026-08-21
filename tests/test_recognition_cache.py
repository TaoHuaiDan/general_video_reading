"""Regression tests for exact OCR caching and bounded cache retention."""

from __future__ import annotations

import numpy as np

from temporal_ocr.backends import CallableRecognizer, NullDetector
from temporal_ocr.config import EngineConfig
from temporal_ocr.engine import TemporalOCREngine
from temporal_ocr.geometry import exact_signature, image_signature
from temporal_ocr.profiling import Profiler
from temporal_ocr.recognition import RecognitionCache
from temporal_ocr.types import CanonicalObservation, OCRResult


def test_exact_signature_separates_content_and_is_stable() -> None:
    low = np.full((48, 120), 100, dtype=np.uint8)
    high = np.full((48, 120), 200, dtype=np.uint8)

    assert exact_signature(low) == exact_signature(low.copy())
    assert exact_signature(low) != exact_signature(high)
    assert exact_signature(low) != exact_signature(np.full((48, 121), 100, dtype=np.uint8))
    assert exact_signature(low) != exact_signature(np.full((48, 120), 100, dtype=np.float32))


def _uniform_observation(
    geometry_id: int,
    timestamp: float,
    value: int,
) -> CanonicalObservation:
    image = np.full((48, 120), value, dtype=np.uint8)
    return CanonicalObservation(
        geometry_id=geometry_id,
        frame_id=int(timestamp * 10),
        timestamp=timestamp,
        image=image,
        # Uniform crops collapse to the same perceptual dHash bits.
        signature=image_signature(image),
        sharpness=0.9,
        contrast=0.9,
        completeness=0.9,
        occlusion=0.0,
    )


def test_cache_does_not_reuse_result_across_perceptually_identical_crops() -> None:
    batch_sizes: list[int] = []

    def recognize(tasks):
        batch_sizes.append(len(tasks))
        return [
            OCRResult(
                content_id=task.content_id,
                text="low" if float(np.mean(task.candidates[0].image)) < 150 else "high",
                confidence=0.99,
                backend="fake",
            )
            for task in tasks
        ]

    engine = TemporalOCREngine(
        NullDetector(),
        CallableRecognizer(recognize, name="fake"),
        config=EngineConfig(),
    )
    engine.cache = RecognitionCache()
    profiler = Profiler()

    track_low = engine.content._new_track(_uniform_observation(1, 0.0, 100))
    engine._enqueue_content(track_low, profiler)
    engine._flush_ocr(16, profiler, flush_all=True)
    assert engine._results[track_low.content_id].text == "low"

    track_high = engine.content._new_track(_uniform_observation(2, 1.0, 200))
    assert track_high.latest_signature == track_low.latest_signature
    engine._enqueue_content(track_high, profiler)
    engine._flush_ocr(16, profiler, flush_all=True)

    assert engine._results[track_high.content_id].text == "high"
    assert len(batch_sizes) == 2

    track_again = engine.content._new_track(_uniform_observation(3, 2.0, 100))
    engine._enqueue_content(track_again, profiler)
    engine._flush_ocr(16, profiler, flush_all=True)

    assert engine._results[track_again.content_id].text == "low"
    assert track_again.content_id in engine._cached_content_ids
    assert len(batch_sizes) == 2


def test_recognition_cache_is_bounded_lru() -> None:
    cache = RecognitionCache(max_entries=2)

    def result(value: str) -> OCRResult:
        return OCRResult(content_id=1, text=value, confidence=1.0, backend="fake")

    cache.put("a", b"1", result("one"))
    cache.put("a", b"2", result("two"))
    assert len(cache) == 2

    cache.get("a", b"1")
    cache.put("a", b"3", result("three"))

    assert len(cache) == 2
    assert cache.get("a", b"1") is not None
    assert cache.get("a", b"2") is None
    assert cache.get("a", b"3") is not None

    unbounded = RecognitionCache(max_entries=None)
    for index in range(10):
        unbounded.put("a", bytes([index]), result(str(index)))
    assert len(unbounded) == 10
