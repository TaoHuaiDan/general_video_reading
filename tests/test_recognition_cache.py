"""Regression tests for exact OCR caching and bounded cache retention."""

from __future__ import annotations

import numpy as np

from temporal_ocr.backends import CallableRecognizer, NullDetector, NullRecognizer
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


def test_engine_keeps_explicitly_provided_empty_cache() -> None:
    custom = RecognitionCache(max_entries=None)

    engine = TemporalOCREngine(NullDetector(), NullRecognizer(), cache=custom)

    assert engine.cache is custom


_SCRIPTED_OCR = {
    100: ("8", 0.60),
    150: ("5", 0.70),
    200: ("3", 0.96),
}


def _scripted_recognizer(call_log: list[list[int]]) -> CallableRecognizer:
    """Mimic the RapidOCR fallback: final text = highest OCR confidence."""

    def recognize(tasks):
        call_log.append([len(task.candidates) for task in tasks])
        results: list[OCRResult] = []
        for task in tasks:
            best_text, best_score = "", -1.0
            for candidate in task.candidates:
                text, score = _SCRIPTED_OCR[int(float(np.mean(candidate.image)))]
                if score > best_score:
                    best_text, best_score = text, score
            results.append(
                OCRResult(
                    content_id=task.content_id,
                    text=best_text,
                    confidence=best_score,
                    backend="fake",
                )
            )
        return results

    return CallableRecognizer(recognize, name="fake")


def _candidate(
    geometry_id: int,
    timestamp: float,
    value: int,
    quality: float,
) -> CanonicalObservation:
    image = np.full((48, 120), value, dtype=np.uint8)
    return CanonicalObservation(
        geometry_id=geometry_id,
        frame_id=int(timestamp * 10),
        timestamp=timestamp,
        image=image,
        # Uniform crops share one perceptual signature, so appending a
        # fallback candidate keeps the same content track.
        signature=image_signature(image),
        sharpness=quality,
        contrast=quality,
        completeness=quality,
        occlusion=0.0,
    )


def _run_task(engine: TemporalOCREngine, candidates: list[CanonicalObservation]) -> None:
    profiler = Profiler()
    track = engine.content._new_track(candidates[0])
    for extra in candidates[1:]:
        engine.content.update(extra)
    engine._enqueue_content(track, profiler)
    engine._flush_ocr(16, profiler, flush_all=True)


def track_was_cached(engine: TemporalOCREngine, content_id: int) -> bool:
    return content_id in engine._cached_content_ids


def test_fallback_winning_candidate_does_not_pollute_primary_exact_key() -> None:
    call_log: list[list[int]] = []
    engine = TemporalOCREngine(
        NullDetector(),
        _scripted_recognizer(call_log),
        config=EngineConfig(),
    )

    # Primary A has the best quality but weak OCR; fallback B wins.
    _run_task(
        engine,
        [
            _candidate(1, 0.0, 100, quality=0.9),
            _candidate(1, 1.0, 200, quality=0.5),
        ],
    )
    assert engine.content.tracks[1].recognized_text == "3"

    # A second task whose primary is the exact same crop A, but without the
    # fallback, must not inherit B's result through the cache.
    _run_task(engine, [_candidate(2, 2.0, 100, quality=0.9)])

    assert engine.content.tracks[2].recognized_text == "8"


def test_identical_candidate_set_is_reused_exactly() -> None:
    call_log: list[list[int]] = []
    engine = TemporalOCREngine(
        NullDetector(),
        _scripted_recognizer(call_log),
        config=EngineConfig(),
    )

    _run_task(
        engine,
        [
            _candidate(1, 0.0, 100, quality=0.9),
            _candidate(1, 1.0, 200, quality=0.5),
        ],
    )
    _run_task(
        engine,
        [
            _candidate(2, 2.0, 100, quality=0.9),
            _candidate(2, 3.0, 200, quality=0.5),
        ],
    )

    assert engine.content.tracks[2].recognized_text == "3"
    assert track_was_cached(engine, 2)


def test_different_fallback_set_gets_fresh_comparison() -> None:
    call_log: list[list[int]] = []
    engine = TemporalOCREngine(
        NullDetector(),
        _scripted_recognizer(call_log),
        config=EngineConfig(),
    )

    _run_task(
        engine,
        [
            _candidate(1, 0.0, 100, quality=0.9),
            _candidate(1, 1.0, 200, quality=0.5),
        ],
    )
    assert engine.content.tracks[1].recognized_text == "3"

    # Same primary A, different fallback C: the comparison must run again.
    _run_task(
        engine,
        [
            _candidate(2, 2.0, 100, quality=0.9),
            _candidate(2, 3.0, 150, quality=0.5),
        ],
    )

    assert engine.content.tracks[2].recognized_text == "5"
