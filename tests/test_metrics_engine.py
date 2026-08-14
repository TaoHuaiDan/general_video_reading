from __future__ import annotations

import numpy as np

from temporal_ocr.backends import CallableDetector, CallableRecognizer
from temporal_ocr.config import EngineConfig
from temporal_ocr.engine import TemporalOCREngine
from temporal_ocr.metrics import evaluate_events
from temporal_ocr.types import (
    DetectionObservation,
    DetectionTier,
    FramePacket,
    OCRResult,
    TextEvent,
)

POLYGON = ((10.0, 10.0), (110.0, 10.0), (110.0, 50.0), (10.0, 50.0))


def event(event_id: int, text: str, start: float = 0.0, end: float = 2.0) -> TextEvent:
    return TextEvent(
        event_id=event_id,
        geometry_id=1,
        content_id=event_id,
        start=start,
        end=end,
        text_raw=text,
        text_normalized=text,
        confidence=0.9,
        polygon_history=((start, POLYGON), (end, POLYGON)),
        source_frame_ids=(1,),
    )


def test_event_metrics_cover_recall_accuracy_duplicates_and_throughput() -> None:
    reference = [event(1, "今天去学校。")]
    predicted = [event(2, "今天去学校。"), event(3, "今天去学校。")]

    report = evaluate_events(reference, predicted, video_sec=60.0, wall_sec=10.0)

    assert report.event_recall == 1.0
    assert report.text_accuracy == 1.0
    assert report.duplicate_rate == 0.5
    assert report.mean_temporal_iou == 1.0
    assert report.mean_spatial_iou > 0.99
    assert report.video_realtime == 6.0


def test_engine_runs_with_pluggable_backends_and_tracks_moving_text() -> None:
    frames: list[FramePacket] = []
    for frame_id, timestamp in enumerate((0.0, 0.5, 1.0, 1.5)):
        image = np.zeros((120, 220, 3), dtype=np.uint8)
        x = 10 + frame_id * 4
        image[20:60, x : x + 100] = 255
        frames.append(
            FramePacket(
                frame_id=frame_id,
                timestamp=timestamp,
                image=image,
                metadata={"x": x},
            )
        )

    def detect(frame, request):
        x = float(frame.metadata["x"])
        return [
            DetectionObservation(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                polygon=((x, 20.0), (x + 100.0, 20.0), (x + 100.0, 60.0), (x, 60.0)),
                confidence=0.99,
                tier=request.tier,
            )
        ]

    def recognize(tasks):
        return [
            OCRResult(
                content_id=task.content_id,
                text="moving text",
                confidence=0.98,
                backend="fake",
            )
            for task in tasks
        ]

    config = EngineConfig()
    config.detection.audit_interval_sec = 5.0
    config.content.stable_observations = 2
    engine = TemporalOCREngine(
        CallableDetector(detect, name="fake-detector"),
        CallableRecognizer(recognize, name="fake"),
        config=config,
    )

    result = engine.run(frames)

    assert len(result.events) == 1
    assert result.events[0].text_raw == "moving text"
    assert len(result.events[0].polygon_history) >= 2
    assert result.profile.ocr_tasks == 1
    assert result.profile.output_events == 1


def test_engine_defers_transient_typewriter_states_but_flushes_final_state() -> None:
    frames: list[FramePacket] = []
    for frame_id, timestamp in enumerate((0.0, 1.0, 2.0)):
        image = np.zeros((120, 220, 3), dtype=np.uint8)
        image[20:60, 10 + frame_id * 30 : 110 + frame_id * 30] = 255
        frames.append(FramePacket(frame_id, timestamp, image))

    def detect(frame, request):
        return [
            DetectionObservation(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                polygon=POLYGON,
                confidence=0.99,
                tier=request.tier,
            )
        ]

    def recognize(tasks):
        return [
            OCRResult(content_id=task.content_id, text="final", confidence=0.99, backend="fake")
            for task in tasks
        ]

    config = EngineConfig()
    config.content.stable_observations = 2
    config.content.change_threshold = 0.01
    config.content.typewriter_skip_score = 0.1
    engine = TemporalOCREngine(
        CallableDetector(detect, name="fake-detector"),
        CallableRecognizer(recognize, name="fake"),
        config=config,
    )
    result = engine.run(frames)

    assert result.profile.output_events <= 2
    assert result.profile.counters.get("ocr_deferred_typewriter", 0.0) >= 1.0


def test_idle_heartbeat_does_not_extend_stale_content_tracks() -> None:
    frames: list[FramePacket] = []
    for frame_id, timestamp in enumerate((0.0, 1.0, 2.0)):
        image = np.zeros((120, 220, 3), dtype=np.uint8)
        image[20:60, 10:110] = 255 if frame_id < 2 else 0
        frames.append(FramePacket(frame_id, timestamp, image))

    def detect(frame, request):
        return [
            DetectionObservation(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                polygon=POLYGON,
                confidence=0.99,
                tier=request.tier,
            )
        ]

    def recognize(tasks):
        return [OCRResult(task.content_id, "text", 0.99, backend="fake") for task in tasks]

    result = TemporalOCREngine(
        CallableDetector(detect, name="fake-detector"),
        CallableRecognizer(recognize, name="fake"),
    ).run(frames)

    assert result.events
    assert max(event.end for event in result.events) <= 2.0


def test_merge_observations_suppresses_local_fragments_inside_fast_line() -> None:
    line = DetectionObservation(
        frame_id=0,
        timestamp=0.0,
        polygon=((10.0, 10.0), (210.0, 10.0), (210.0, 50.0), (10.0, 50.0)),
        confidence=0.90,
        tier=DetectionTier.FAST,
    )
    fragments = [
        DetectionObservation(
            frame_id=0,
            timestamp=0.0,
            polygon=((x, 12.0), (x + 24.0, 12.0), (x + 24.0, 48.0), (x, 48.0)),
            confidence=0.99,
            tier=DetectionTier.LOCAL,
        )
        for x in (20.0, 55.0, 90.0)
    ]

    merged = TemporalOCREngine._merge_observations([*fragments, line])

    assert merged == [line]


def test_fast_detection_prevents_redundant_local_detection() -> None:
    line = DetectionObservation(
        frame_id=0,
        timestamp=0.0,
        polygon=((10.0, 10.0), (210.0, 10.0), (210.0, 50.0), (10.0, 50.0)),
        confidence=0.90,
        tier=DetectionTier.FAST,
    )
    covered = ((20.0, 12.0), (60.0, 12.0), (60.0, 48.0), (20.0, 48.0))
    missed = ((250.0, 12.0), (290.0, 12.0), (290.0, 48.0), (250.0, 48.0))

    assert TemporalOCREngine._uncovered_scopes((covered, missed), [line]) == (missed,)

    encompassing = ((0.0, 0.0), (220.0, 0.0), (220.0, 60.0), (0.0, 60.0))
    assert TemporalOCREngine._uncovered_scopes((encompassing,), [line]) == ()
