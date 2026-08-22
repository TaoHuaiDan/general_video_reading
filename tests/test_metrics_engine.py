from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from temporal_ocr.backends import CallableDetector, CallableRecognizer
from temporal_ocr.change import TileChangeDetector
from temporal_ocr.config import EngineConfig
from temporal_ocr.engine import TemporalOCREngine
from temporal_ocr.metrics import evaluate_events
from temporal_ocr.types import (
    CanonicalObservation,
    DetectionObservation,
    DetectionRequest,
    DetectionTier,
    FramePacket,
    MotionEstimate,
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


def test_event_matching_maximizes_cardinality_before_quality() -> None:
    # R1 can match P1 (best score) or P2; R2 can only match P1.  Greedy
    # score-first matching takes R1-P1 and loses R2 entirely; a
    # completeness-first matcher must find two pairs.
    def poly(left: float, right: float) -> tuple:
        return ((left, 0.0), (right, 0.0), (right, 50.0), (left, 50.0))

    def make(event_id: int, polygon: tuple) -> TextEvent:
        return TextEvent(
            event_id=event_id,
            geometry_id=event_id,
            content_id=event_id,
            start=0.0,
            end=10.0,
            text_raw="hello",
            text_normalized="hello",
            confidence=0.9,
            polygon_history=((0.0, polygon), (10.0, polygon)),
            source_frame_ids=(1,),
        )

    reference = [make(1, poly(0.0, 200.0)), make(2, poly(160.0, 200.0))]
    predicted = [make(3, poly(0.0, 180.0)), make(4, poly(0.0, 20.0))]

    report = evaluate_events(reference, predicted)

    assert report.matched_events == 2
    assert report.event_recall == 1.0


def test_matching_maximizes_total_score_among_maximum_cardinality_solutions() -> None:
    # Both maximum-cardinality solutions exist; the score-greedy-but-feasible
    # choice (R1-P1 + R2-P2) totals less than (R1-P2 + R2-P1).
    from temporal_ocr.metrics import _minimum_cost_maximum_matching

    chosen = _minimum_cost_maximum_matching(
        [
            (0, 0, 0.90),
            (0, 1, 0.80),
            (1, 0, 0.80),
            (1, 1, 0.10),
        ]
    )

    assert sorted(chosen) == [(0, 1), (1, 0)]


def test_event_matching_prefers_higher_total_score_pairing() -> None:
    def poly(left: float, right: float) -> tuple:
        return ((left, 0.0), (right, 0.0), (right, 50.0), (left, 50.0))

    def make(event_id: int, polygon: tuple) -> TextEvent:
        return TextEvent(
            event_id=event_id,
            geometry_id=event_id,
            content_id=event_id,
            start=0.0,
            end=10.0,
            text_raw="hello",
            text_normalized="hello",
            confidence=0.9,
            polygon_history=((0.0, polygon), (10.0, polygon)),
            source_frame_ids=(1,),
        )

    reference = [make(1, poly(0.0, 100.0)), make(2, poly(0.0, 84.0))]
    predicted = [make(3, poly(0.0, 95.0)), make(4, poly(15.0, 100.0))]

    report = evaluate_events(reference, predicted)

    assert report.matched_events == 2
    assert report.event_recall == 1.0
    # The optimal pairing (R1-P2, R2-P1) has clearly higher spatial quality
    # than the score-greedy pairing (R1-P1, R2-P2).
    assert report.mean_spatial_iou == pytest.approx(
        (85.0 / 100.0 + 84.0 / 95.0) / 2.0,
        abs=1e-3,
    )


def test_misrecognized_text_still_counts_as_a_detected_event() -> None:
    # Perfect spatio-temporal alignment must count as detection even when OCR
    # produced wrong text; the error belongs to Text Accuracy, not Recall.
    reference = [event(1, "今天去学校。")]
    predicted = [event(2, "完全不同的文字")]

    report = evaluate_events(reference, predicted)

    assert report.matched_events == 1
    assert report.event_recall == 1.0
    assert report.text_accuracy < 0.5


def test_no_matches_reports_zero_accuracy_instead_of_perfect() -> None:
    left = ((10.0, 10.0), (110.0, 10.0), (110.0, 50.0), (10.0, 50.0))
    far_right = ((500.0, 300.0), (600.0, 300.0), (600.0, 340.0), (500.0, 340.0))

    def timed(event_id: int, text: str, polygon: tuple, start: float) -> TextEvent:
        return TextEvent(
            event_id=event_id,
            geometry_id=event_id,
            content_id=event_id,
            start=start,
            end=start + 1.0,
            text_raw=text,
            text_normalized=text,
            confidence=0.9,
            polygon_history=((start, polygon),),
            source_frame_ids=(1,),
        )

    reference = [timed(1, "甲事件文本", left, 0.0)]
    predicted = [timed(2, "乙别处内容", far_right, 50.0)]

    report = evaluate_events(reference, predicted)

    assert report.matched_events == 0
    assert report.event_recall == 0.0
    assert report.text_accuracy == 0.0


def test_report_includes_matched_text_accuracy_and_event_precision() -> None:
    reference = [event(1, "今天去学校。")]
    predicted = [event(2, "今天去学校。"), event(3, "今天去学校。")]

    report = evaluate_events(reference, predicted)

    assert report.matched_text_accuracy == 1.0
    assert report.event_precision == pytest.approx(0.5)
    payload = report.to_dict()
    assert "matched_text_accuracy" in payload
    assert "event_precision" in payload


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


def test_engine_excludes_declared_watermark_region() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    watermark = ((150.0, 70.0), (195.0, 70.0), (195.0, 95.0), (150.0, 95.0))
    real_text = ((10.0, 10.0), (110.0, 10.0), (110.0, 45.0), (10.0, 45.0))

    def detect(frame, request):
        return [
            DetectionObservation(frame.frame_id, frame.timestamp, watermark, 0.99, request.tier),
            DetectionObservation(frame.frame_id, frame.timestamp, real_text, 0.99, request.tier),
        ]

    def recognize(tasks):
        return [
            OCRResult(task.content_id, "dialogue", 0.99, backend="fake")
            for task in tasks
        ]

    result = TemporalOCREngine(
        CallableDetector(detect, name="fake-detector"),
        CallableRecognizer(recognize, name="fake"),
        exclude_regions=((0.70, 0.60, 1.0, 1.0),),
    ).run([FramePacket(0, 0.0, image)])

    assert [item.text_raw for item in result.events] == ["dialogue"]
    assert result.profile.counters.get("excluded_observations", 0.0) >= 1.0


def test_change_detector_ignores_excluded_watermark_pixels() -> None:
    previous = np.zeros((100, 200, 3), dtype=np.uint8)
    current = previous.copy()
    current[70:95, 150:195] = 255
    excluded = ((150.0, 60.0), (200.0, 60.0), (200.0, 100.0), (150.0, 100.0))

    result = TileChangeDetector().compare(
        previous,
        current,
        MotionEstimate(np.eye(3), valid=True, confidence=1.0),
        excluded_polygons=(excluded,),
    )

    assert result.changed_ratio == 0.0
    assert result.scopes == ()


def test_excluded_tiles_do_not_dilute_change_ratio() -> None:
    # The left half is a declared ignore region; every remaining valid tile
    # changes completely.  The ratio must describe the valid tiles only.
    previous = np.zeros((100, 200, 3), dtype=np.uint8)
    current = previous.copy()
    current[0:100, 100:200] = 255
    left_half = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))

    result = TileChangeDetector().compare(
        previous,
        current,
        MotionEstimate(np.eye(3), valid=True, confidence=1.0),
        excluded_polygons=(left_half,),
    )

    assert result.changed_ratio == 1.0
    assert result.score > 0.0


def test_fully_excluded_frame_has_well_defined_zero_statistics() -> None:
    previous = np.zeros((100, 200, 3), dtype=np.uint8)
    current = previous.copy()
    current[0:100, 0:200] = 255
    everything = ((0.0, 0.0), (200.0, 0.0), (200.0, 100.0), (0.0, 100.0))

    result = TileChangeDetector().compare(
        previous,
        current,
        MotionEstimate(np.eye(3), valid=True, confidence=1.0),
        excluded_polygons=(everything,),
    )

    assert result.score == 0.0
    assert result.changed_ratio == 0.0
    assert result.changed_tiles == ()
    assert result.scopes == ()


def test_replaced_intermediate_states_surface_but_final_state_still_flushes() -> None:
    # With change_threshold near zero every frame is a content replacement.
    # Replaced finalized states are complete texts whose geometry moved on,
    # so they must reach OCR instead of being dropped by the typewriter
    # defer; the final state still flushes at end-of-run.
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
    config.detection.track_guided_local = False
    engine = TemporalOCREngine(
        CallableDetector(detect, name="fake-detector"),
        CallableRecognizer(recognize, name="fake"),
        config=config,
    )
    result = engine.run(frames)

    assert result.profile.output_events == len(frames)
    assert result.events[-1].end == 2.0
    assert all(event.text_raw == "final" for event in result.events)


def test_engine_run_does_not_mutate_caller_config_and_logs_max_wait() -> None:
    frames: list[FramePacket] = []
    for frame_id, timestamp in enumerate((0.0, 0.5, 1.0)):
        image = np.zeros((120, 220, 3), dtype=np.uint8)
        image[20:60, 10:110] = 255
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

    config = EngineConfig()
    config.content.stable_wait_sec = 0.90
    config.content.maximum_wait_sec = 3.0
    engine = TemporalOCREngine(
        CallableDetector(detect, name="fake-detector"),
        CallableRecognizer(recognize, name="fake"),
        config=config,
    )

    result = engine.run(frames)

    assert config.content.stable_wait_sec == 0.90
    assert config.content.maximum_wait_sec == 3.0
    assert engine.content.config is not config.content
    assert result.policy_changes
    assert all("maximum_wait_sec" in entry for entry in result.policy_changes)


def test_runtime_signals_ignore_finalized_tracks_and_reset_without_active_content() -> None:
    from temporal_ocr.backends import NullDetector

    engine = TemporalOCREngine(
        NullDetector(),
        CallableRecognizer(lambda _tasks: [], name="fake"),
    )

    def seed_track(geometry_id: int, timestamp: float) -> Any:
        image = np.zeros((48, 120), dtype=np.uint8)
        return engine.content._new_track(
            CanonicalObservation(
                geometry_id=geometry_id,
                frame_id=geometry_id,
                timestamp=timestamp,
                image=image,
                signature=b"sig",
                sharpness=0.9,
                contrast=0.9,
                completeness=0.9,
                occlusion=0.0,
            )
        )

    stale = seed_track(1, 0.0)
    stale.typewriter_score = 1.0
    stale.last_seen = 10.0
    engine.content.finalize_geometry(1)

    assert engine._content_runtime_signals() == (0.0, 0.0)

    active = seed_track(2, 1.0)
    active.typewriter_score = 0.8
    active.last_seen = 3.0

    assert engine._content_runtime_signals() == (2.0, 0.8)


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


def test_small_observation_inside_large_scope_does_not_skip_the_scope() -> None:
    # A large LOCAL change scope may contain text the small observation never
    # covered; containment of the observation by the scope is not evidence
    # that the whole scope was already detected.
    line = DetectionObservation(
        frame_id=0,
        timestamp=0.0,
        polygon=((10.0, 10.0), (110.0, 10.0), (110.0, 50.0), (10.0, 50.0)),
        confidence=0.90,
        tier=DetectionTier.FAST,
    )
    wrapped_dialogue_scope = ((0.0, 0.0), (400.0, 0.0), (400.0, 200.0), (0.0, 200.0))

    assert TemporalOCREngine._uncovered_scopes((wrapped_dialogue_scope,), [line]) == (
        wrapped_dialogue_scope,
    )


def test_line_covering_a_local_scope_still_skips_redundant_local_work() -> None:
    line = DetectionObservation(
        frame_id=0,
        timestamp=0.0,
        polygon=((10.0, 10.0), (210.0, 10.0), (210.0, 50.0), (10.0, 50.0)),
        confidence=0.90,
        tier=DetectionTier.FAST,
    )
    small_scope = ((20.0, 12.0), (60.0, 12.0), (60.0, 48.0), (20.0, 48.0))

    assert TemporalOCREngine._uncovered_scopes((small_scope,), [line]) == ()


def test_tracked_geometry_can_cover_a_local_change_without_model_detection() -> None:
    engine = TemporalOCREngine(
        CallableDetector(lambda _frame, _request: [], name="fake-detector"),
        CallableRecognizer(lambda _tasks: [], name="fake"),
    )
    seed = DetectionObservation(
        frame_id=0,
        timestamp=0.0,
        polygon=POLYGON,
        confidence=0.9,
        tier=DetectionTier.AUDIT,
    )
    engine.geometry.update([seed], frame_size=(320, 180))
    request = DetectionRequest(
        tier=DetectionTier.LOCAL,
        reason="changed",
        target_width=1600,
        scopes=(POLYGON,),
    )

    projected, overflow, overflow_ids = engine._tracked_local_observations(
        FramePacket(1, 1.0, np.zeros((180, 320, 3), dtype=np.uint8)),
        request,
        motion=MotionEstimate(np.eye(3), valid=True, confidence=0.9),
    )

    assert len(projected) == 1
    assert overflow == ()
    assert overflow_ids == ()
    assert projected[0].polygon == POLYGON
    assert projected[0].tier == DetectionTier.LOCAL


_REFRESH_BOX = ((10.0, 20.0), (70.0, 20.0), (70.0, 46.0), (10.0, 46.0))


def _refresh_scene(*, far_block: bool = False, bands: bool = False) -> np.ndarray:
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    image[20:46, 10:70] = 255
    if far_block:
        # Outside the guided overflow padding rings of the tracked line.
        image[30:40, 95:115] = 255
    if bands:
        # Envelope-scale change above and below the line: the scope becomes
        # structurally much larger than the tracked box.
        image[0:18, 0:240] = 255
        image[52:80, 0:240] = 255
    return image


def _detect_same_polygon(frame, request):
    return [
        DetectionObservation(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            polygon=_REFRESH_BOX,
            confidence=0.99,
            tier=request.tier,
        )
    ]


def _detect_audit_only(frame, request):
    if request.tier == DetectionTier.AUDIT:
        return _detect_same_polygon(frame, request)
    return []


def test_geometry_refresh_keeps_identity_when_text_is_redetected() -> None:
    def recognize(tasks):
        return [OCRResult(task.content_id, "text", 0.99, backend="fake") for task in tasks]

    frames = [
        FramePacket(0, 0.0, _refresh_scene()),
        FramePacket(1, 0.5, _refresh_scene(far_block=True)),
        FramePacket(2, 1.0, _refresh_scene(far_block=True, bands=True)),
    ]
    result = TemporalOCREngine(
        CallableDetector(_detect_same_polygon, name="fake-detector"),
        CallableRecognizer(recognize, name="fake"),
    ).run(frames)

    # The refresh re-detected the same line: it must not become a duplicate
    # second event.
    assert [event.text_raw for event in result.events] == ["text"]


def test_geometry_refresh_still_ocrs_unstable_content_when_text_disappears() -> None:
    def recognize(tasks):
        return [OCRResult(task.content_id, "text", 0.99, backend="fake") for task in tasks]

    frames = [
        FramePacket(0, 0.0, _refresh_scene()),
        FramePacket(1, 1.0, _refresh_scene(bands=True)),
    ]
    result = TemporalOCREngine(
        CallableDetector(_detect_audit_only, name="fake-detector"),
        CallableRecognizer(recognize, name="fake"),
    ).run(frames)

    # The refresh did not re-detect the line and the content was never
    # stable; its best candidate must still be recognized, not dropped.
    assert [event.text_raw for event in result.events] == ["text"]


def test_tracked_geometry_requests_refresh_when_pixels_overflow_the_box() -> None:
    engine = TemporalOCREngine(
        CallableDetector(lambda _frame, _request: [], name="fake-detector"),
        CallableRecognizer(lambda _tasks: [], name="fake"),
    )
    seed = DetectionObservation(
        frame_id=0,
        timestamp=0.0,
        polygon=POLYGON,
        confidence=0.9,
        tier=DetectionTier.AUDIT,
    )
    engine.geometry.update([seed], frame_size=(320, 180))
    request = DetectionRequest(
        tier=DetectionTier.LOCAL,
        reason="changed",
        target_width=1600,
        scopes=(POLYGON,),
    )
    pixel_delta = np.zeros((180, 320), dtype=np.float32)
    pixel_delta[50:85, 10:110] = 1.0

    projected, overflow, overflow_ids = engine._tracked_local_observations(
        FramePacket(1, 1.0, np.zeros((180, 320, 3), dtype=np.uint8)),
        request,
        motion=MotionEstimate(np.eye(3), valid=True, confidence=0.9),
        pixel_delta=pixel_delta,
    )

    assert projected == []
    assert len(overflow) == 1
    assert overflow_ids == (1,)
    assert overflow[0][0][0] < POLYGON[0][0]
    assert overflow[0][1][0] > POLYGON[1][0]
    assert overflow[0][0][1] < POLYGON[0][1]
    assert overflow[0][2][1] > POLYGON[2][1]


def test_structurally_larger_scope_requests_geometry_refresh_without_pixel_overflow() -> None:
    # A high-confidence single-line track inside a change scope that could
    # hold several wrapped dialogue lines must refresh its geometry even when
    # no changed pixels spill just outside the current box.
    engine = TemporalOCREngine(
        CallableDetector(lambda _frame, _request: [], name="fake-detector"),
        CallableRecognizer(lambda _tasks: [], name="fake"),
    )
    seed = DetectionObservation(
        frame_id=0,
        timestamp=0.0,
        polygon=POLYGON,
        confidence=0.9,
        tier=DetectionTier.AUDIT,
    )
    engine.geometry.update([seed], frame_size=(640, 360))
    wrapped_scope = ((0.0, 0.0), (400.0, 0.0), (400.0, 320.0), (0.0, 320.0))
    request = DetectionRequest(
        tier=DetectionTier.LOCAL,
        reason="changed",
        target_width=1600,
        scopes=(wrapped_scope,),
    )

    projected, overflow, overflow_ids = engine._tracked_local_observations(
        FramePacket(1, 1.0, np.zeros((360, 640, 3), dtype=np.uint8)),
        request,
        motion=MotionEstimate(np.eye(3), valid=True, confidence=0.9),
        pixel_delta=np.zeros((360, 640), dtype=np.float32),
    )

    assert projected == []
    assert len(overflow) == 1
    assert overflow_ids == (1,)
    assert overflow[0][0][1] < POLYGON[0][1]
    assert overflow[0][2][1] > POLYGON[2][1]


def test_ordinary_typewriter_line_is_not_refreshed_by_a_similar_scope() -> None:
    # The normal guided path must stay detector-free: a scope comparable to
    # the tracked line neither overflows nor is structurally much larger.
    engine = TemporalOCREngine(
        CallableDetector(lambda _frame, _request: [], name="fake-detector"),
        CallableRecognizer(lambda _tasks: [], name="fake"),
    )
    seed = DetectionObservation(
        frame_id=0,
        timestamp=0.0,
        polygon=POLYGON,
        confidence=0.9,
        tier=DetectionTier.AUDIT,
    )
    engine.geometry.update([seed], frame_size=(640, 360))
    similar_scope = ((8.0, 8.0), (120.0, 8.0), (120.0, 60.0), (8.0, 60.0))
    request = DetectionRequest(
        tier=DetectionTier.LOCAL,
        reason="changed",
        target_width=1600,
        scopes=(similar_scope,),
    )

    projected, overflow, overflow_ids = engine._tracked_local_observations(
        FramePacket(1, 1.0, np.zeros((360, 640, 3), dtype=np.uint8)),
        request,
        motion=MotionEstimate(np.eye(3), valid=True, confidence=0.9),
        pixel_delta=np.zeros((360, 640), dtype=np.float32),
    )

    assert len(projected) == 1
    assert overflow == ()
    assert overflow_ids == ()
