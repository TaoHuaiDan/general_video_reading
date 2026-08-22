"""Adversarial regressions for OCR task/result revision identity.

Task identity is ``(content_id, revision)``.  These tests drive the real
OCRBatchQueue / _flush_ocr lifecycle, not just ContentTracker helpers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from temporal_ocr.backends import CallableRecognizer, NullDetector
from temporal_ocr.config import EngineConfig
from temporal_ocr.engine import TemporalOCREngine
from temporal_ocr.geometry import image_signature
from temporal_ocr.profiling import Profiler
from temporal_ocr.recognition import candidate_set_signature, recognize_tasks
from temporal_ocr.types import (
    DetectionObservation,
    DetectionTier,
    OCRResult,
    OCRTask,
)

_REVISION_TEXT = {0: "A", 1: "B", 2: "C"}


def _obs(timestamp: float, value: int, signature: bytes) -> Any:
    from temporal_ocr.types import CanonicalObservation

    return CanonicalObservation(
        geometry_id=1,
        frame_id=int(timestamp * 10),
        timestamp=timestamp,
        image=np.full((48, 120), value, dtype=np.uint8),
        signature=signature,
        sharpness=0.9,
        contrast=0.9,
        completeness=0.9,
        occlusion=0.0,
    )


def _drifted(signature: bytes, steps: int) -> bytes:
    # Each step flips 24/256 bits (0.094 < 0.16); three steps cross twice.
    return bytes(b | 0xFF if index < 3 * steps else b for index, b in enumerate(signature))


def _make_engine(script):
    recognizer = CallableRecognizer(script, name="scripted")
    return TemporalOCREngine(
        NullDetector(),
        recognizer,
        config=EngineConfig(),
    )


def _scripted_results_from_tasks(tasks) -> list[OCRResult]:
    return _scripted_results([(task.content_id, task.revision) for task in tasks])


def _scripted_results(identities: list[tuple[int, int]]) -> list[OCRResult]:
    return [
        OCRResult(
            content_id=cid,
            revision=rev,
            text=_REVISION_TEXT[rev],
            confidence=0.9,
            backend="scripted",
        )
        for cid, rev in identities
    ]


def _seed_geometry(engine: TemporalOCREngine) -> None:
    engine.geometry.update(
        [
            DetectionObservation(
                frame_id=0,
                timestamp=0.0,
                polygon=((10.0, 20.0), (70.0, 20.0), (70.0, 46.0), (10.0, 46.0)),
                confidence=0.99,
                tier=DetectionTier.AUDIT,
            )
        ],
        frame_size=(320, 180),
    )


def _drive_two_revisions(engine: TemporalOCREngine, profiler: Profiler) -> Any:
    """Queue rev0 and rev1 for the same content track without flushing."""
    anchor = image_signature(np.full((48, 120), 100, dtype=np.uint8))
    track = engine.content._new_track(_obs(0.0, 100, anchor))
    engine.content.update(_obs(1.0, 100, anchor))
    engine._enqueue_content(track, profiler)
    assert len(engine.ocr_queue) == 1

    engine.content.update(_obs(2.0, 200, _drifted(anchor, 1)))
    engine.content.update(_obs(3.0, 200, _drifted(anchor, 2)))
    engine.content.update(_obs(4.0, 200, _drifted(anchor, 2)))
    engine._enqueue_content(track, profiler)
    assert len(engine.ocr_queue) == 2
    return track


def test_same_content_two_revisions_in_one_batch_applies_only_current() -> None:
    engine = _make_engine(_scripted_results_from_tasks)
    profiler = Profiler()
    _seed_geometry(engine)
    track = _drive_two_revisions(engine, profiler)

    engine._flush_ocr(16, profiler, flush_all=True)

    assert track.recognized_text == "B"
    assert profiler.profile.counters.get("ocr_stale_results_discarded") == 1.0
    # Cache entries stay bound to the exact crop sets that produced them.
    low = np.full((48, 120), 100, dtype=np.uint8)
    high = np.full((48, 120), 200, dtype=np.uint8)
    rev0_key = candidate_set_signature([low, low])
    rev1_key = candidate_set_signature([high, high])
    cached_rev0 = engine.cache.get(engine._cache_namespace, rev0_key)
    cached_rev1 = engine.cache.get(engine._cache_namespace, rev1_key)
    assert cached_rev0 is not None and cached_rev0.text == "A"
    assert cached_rev1 is not None and cached_rev1.text == "B"


def test_reverse_result_order_still_applies_only_current_revision() -> None:
    def reversed_script(tasks):
        return list(reversed(_scripted_results_from_tasks(tasks)))

    engine = _make_engine(reversed_script)
    profiler = Profiler()
    _seed_geometry(engine)
    track = _drive_two_revisions(engine, profiler)

    engine._flush_ocr(16, profiler, flush_all=True)

    assert track.recognized_text == "B"
    assert track.confidence > 0.0
    assert profiler.profile.counters.get("ocr_stale_results_discarded") == 1.0


def test_missing_result_identity_is_rejected() -> None:
    tasks = [
        OCRTask(content_id=7, geometry_id=1, candidates=(), revision=0),
        OCRTask(content_id=7, geometry_id=1, candidates=(), revision=1),
    ]
    results = [_scripted_results([(7, 1)])]

    with pytest.raises(RuntimeError):
        recognize_tasks(CallableRecognizer(lambda t: results[0], name="bad"), tasks)


def test_duplicate_result_identity_is_rejected() -> None:
    tasks = [
        OCRTask(content_id=7, geometry_id=1, candidates=(), revision=0),
        OCRTask(content_id=7, geometry_id=1, candidates=(), revision=1),
    ]

    def duplicate(_tasks):
        return [r for r in _scripted_results([(7, 1), (7, 1)])]

    with pytest.raises(RuntimeError):
        recognize_tasks(CallableRecognizer(duplicate, name="bad"), tasks)


def test_wrong_revision_result_is_rejected() -> None:
    tasks = [
        OCRTask(content_id=7, geometry_id=1, candidates=(), revision=0),
        OCRTask(content_id=7, geometry_id=1, candidates=(), revision=1),
    ]
    wrong = _scripted_results([(7, 0), (7, 2)])

    with pytest.raises(RuntimeError):
        recognize_tasks(CallableRecognizer(lambda _t: wrong, name="bad"), tasks)


def test_three_revisions_can_share_the_queue_without_cross_contamination() -> None:
    def three_rev_script(tasks):
        return list(reversed(_scripted_results_from_tasks(tasks)))

    engine = _make_engine(three_rev_script)
    profiler = Profiler()
    anchor = image_signature(np.full((48, 120), 100, dtype=np.uint8))

    track = engine.content._new_track(_obs(0.0, 100, anchor))
    engine.content.update(_obs(1.0, 100, anchor))
    engine._enqueue_content(track, profiler)

    # re-arm -> rev1 queued while rev0 still queued.
    engine.content.update(_obs(2.0, 200, _drifted(anchor, 1)))
    engine.content.update(_obs(3.0, 200, _drifted(anchor, 2)))
    engine.content.update(_obs(4.0, 200, _drifted(anchor, 2)))
    engine._enqueue_content(track, profiler)

    # re-arm again -> rev2 also queued.
    engine.content.update(_obs(5.0, 200, _drifted(anchor, 3)))
    engine.content.update(_obs(6.0, 200, _drifted(anchor, 4)))
    engine.content.update(_obs(7.0, 200, _drifted(anchor, 4)))
    engine._enqueue_content(track, profiler)

    assert len(engine.ocr_queue) == 3
    assert engine._queued_revisions == {track.content_id: 2}

    engine._flush_ocr(16, profiler, flush_all=True)

    assert track.revision == 2
    assert track.recognized_text == "C"
    assert track.confidence > 0.0
    assert profiler.profile.counters.get("ocr_stale_results_discarded") == 2.0
    # Stale tasks must not clear the current revision's bookkeeping; after a
    # full flush the current entry is the only one removed.
    assert engine._queued_revisions == {}
    assert engine._results[track.content_id].revision == 2


def test_eof_flush_processes_current_revision_despite_stale_task() -> None:
    engine = _make_engine(_scripted_results_from_tasks)
    profiler = Profiler()
    _seed_geometry(engine)
    track = _drive_two_revisions(engine, profiler)

    # EOF-equivalent final flush plus event build.
    engine._flush_ocr(16, profiler, flush_all=True)
    events = engine._build_events()

    assert [event.text_raw for event in events] == ["B"]
    assert events[0].content_id == track.content_id
