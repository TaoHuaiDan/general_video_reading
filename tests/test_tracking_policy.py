from __future__ import annotations

import numpy as np

from temporal_ocr.config import ContentConfig, DetectionConfig, PolicyConfig, TrackingConfig
from temporal_ocr.detection import HierarchicalDetectionPlanner
from temporal_ocr.motion import identity_motion
from temporal_ocr.policy import RuleBasedPolicyScheduler
from temporal_ocr.selection import ComplementaryCandidateSelector
from temporal_ocr.tracking import ContentTracker, GeometryTracker
from temporal_ocr.types import (
    CanonicalObservation,
    DetectionObservation,
    DetectionTier,
    PolicyDecision,
    RuntimeSignals,
)


def observation(frame_id: int, timestamp: float, x: float) -> DetectionObservation:
    return DetectionObservation(
        frame_id=frame_id,
        timestamp=timestamp,
        polygon=((x, 20.0), (x + 60.0, 20.0), (x + 60.0, 50.0), (x, 50.0)),
        confidence=0.95,
        tier=DetectionTier.FAST,
    )


def canonical(geometry_id: int, timestamp: float, signature: bytes, quality: float = 0.8):
    return CanonicalObservation(
        geometry_id=geometry_id,
        frame_id=int(timestamp * 10),
        timestamp=timestamp,
        image=np.zeros((48, 120), dtype=np.uint8),
        signature=signature,
        sharpness=quality,
        contrast=quality,
        completeness=quality,
        occlusion=0.0,
    )


def test_geometry_track_survives_motion_when_content_is_separate() -> None:
    tracker = GeometryTracker(TrackingConfig(min_iou=0.5, max_center_distance=0.05))
    first = tracker.update([observation(0, 0.0, 10.0)], frame_size=(400, 200))
    motion = identity_motion()
    motion.matrix = np.asarray([[1.0, 0.0, 100.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    second = tracker.update(
        [observation(1, 0.5, 110.0)],
        motion=motion,
        frame_size=(400, 200),
    )

    assert first.created == (1,)
    assert not second.created
    assert list(second.assignments) == [1]


def test_content_track_stays_stable_independent_of_geometry_motion() -> None:
    tracker = ContentTracker(ContentConfig(stable_observations=2, change_threshold=0.2))
    first = tracker.update(canonical(7, 0.0, b"\x00\x00"))
    second = tracker.update(canonical(7, 0.5, b"\x00\x00"))
    changed = tracker.update(canonical(7, 1.0, b"\xff\xff"))

    assert first.ready_task is None
    assert second.ready_task is not None
    assert changed.finalized is not None
    assert changed.active.content_id != second.active.content_id


def test_candidate_selector_keeps_complementary_frame() -> None:
    selector = ComplementaryCandidateSelector(limit=2, diversity_weight=0.5)
    best = canonical(1, 0.0, b"\x00", 0.95)
    redundant = canonical(1, 0.1, b"\x00", 0.90)
    complementary = canonical(1, 0.2, b"\xff", 0.75)

    selected = selector.select([best, redundant], complementary)

    assert best in selected
    assert complementary in selected
    assert redundant not in selected


def test_policy_scheduler_reacts_to_pressure_and_missed_audits() -> None:
    scheduler = RuleBasedPolicyScheduler(
        PolicyConfig(queue_pressure_threshold=10, default_batch_size=8),
        DetectionConfig(audit_interval_sec=10.0),
    )
    decision = scheduler.decide(
        RuntimeSignals(
            audit_new_text_yield=0.2,
            ocr_queue_length=20,
            motion_confidence=0.3,
            global_motion_magnitude=12.0,
        )
    )

    assert decision.batch_size > 8
    assert decision.audit_interval_sec < 10.0
    assert decision.fast_detection_width >= 1600
    assert "queue_pressure" in decision.reason


def test_hierarchical_detection_planner_emits_all_tiers() -> None:
    planner = HierarchicalDetectionPlanner(DetectionConfig(audit_interval_sec=10.0))
    decision = PolicyDecision(0.2, 10.0, 960, 0.4, 1.5, 8, 30)
    audit = planner.plan(
        timestamp=0.0,
        scene_cut=True,
        changed_scopes=(),
        changed_ratio=1.0,
        has_active_tracks=False,
        decision=decision,
        motion_reliable=True,
    )
    scope = (((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),)
    regular = planner.plan(
        timestamp=0.5,
        scene_cut=False,
        changed_scopes=scope,
        changed_ratio=0.01,
        has_active_tracks=False,
        decision=decision,
        motion_reliable=True,
    )

    assert [item.tier for item in audit] == [DetectionTier.AUDIT]
    assert {item.tier for item in regular} == {DetectionTier.FAST, DetectionTier.LOCAL}


def test_active_tracks_use_local_detection_for_small_changes() -> None:
    planner = HierarchicalDetectionPlanner(DetectionConfig(audit_interval_sec=10.0))
    decision = PolicyDecision(0.2, 10.0, 960, 0.4, 1.5, 8, 30)
    scope = (((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),)

    requests = planner.plan(
        timestamp=0.0,
        scene_cut=True,
        changed_scopes=(),
        changed_ratio=1.0,
        has_active_tracks=False,
        decision=decision,
        motion_reliable=True,
    )
    assert [item.tier for item in requests] == [DetectionTier.AUDIT]

    requests = planner.plan(
        timestamp=0.5,
        scene_cut=False,
        changed_scopes=scope,
        changed_ratio=0.01,
        has_active_tracks=True,
        decision=decision,
        motion_reliable=True,
    )
    assert [item.tier for item in requests] == [DetectionTier.LOCAL]

    requests = planner.plan(
        timestamp=1.0,
        scene_cut=False,
        changed_scopes=scope,
        changed_ratio=0.50,
        has_active_tracks=True,
        decision=decision,
        motion_reliable=True,
    )
    assert {item.tier for item in requests} == {DetectionTier.FAST, DetectionTier.LOCAL}


def test_active_text_change_interrupts_probe_cooldown() -> None:
    planner = HierarchicalDetectionPlanner(DetectionConfig(audit_interval_sec=10.0))
    decision = PolicyDecision(1.25, 10.0, 960, 0.4, 1.5, 8, 30)
    scope = (((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),)

    planner.plan(
        timestamp=0.0,
        scene_cut=True,
        changed_scopes=(),
        changed_ratio=1.0,
        has_active_tracks=False,
        decision=decision,
        motion_reliable=True,
    )
    requests = planner.plan(
        timestamp=1.0,
        scene_cut=False,
        changed_scopes=scope,
        changed_ratio=0.01,
        has_active_tracks=True,
        decision=decision,
        motion_reliable=True,
    )

    assert [item.tier for item in requests] == [DetectionTier.LOCAL]
    assert requests[0].reason == "urgent_local_change"


def test_motion_unreliable_audit_is_rate_limited() -> None:
    planner = HierarchicalDetectionPlanner(
        DetectionConfig(audit_interval_sec=10.0, min_audit_interval_sec=3.0)
    )
    decision = PolicyDecision(0.2, 10.0, 960, 0.4, 1.5, 8, 30)

    first = planner.plan(
        timestamp=0.0,
        scene_cut=False,
        changed_scopes=(),
        changed_ratio=0.0,
        has_active_tracks=False,
        decision=decision,
        motion_reliable=False,
    )
    too_soon = planner.plan(
        timestamp=0.5,
        scene_cut=False,
        changed_scopes=(),
        changed_ratio=0.0,
        has_active_tracks=False,
        decision=decision,
        motion_reliable=False,
    )
    due = planner.plan(
        timestamp=3.1,
        scene_cut=False,
        changed_scopes=(),
        changed_ratio=0.0,
        has_active_tracks=False,
        decision=decision,
        motion_reliable=False,
    )

    assert [item.tier for item in first] == [DetectionTier.AUDIT]
    assert [item.tier for item in too_soon] == [DetectionTier.FAST]
    assert [item.tier for item in due] == [DetectionTier.AUDIT]
