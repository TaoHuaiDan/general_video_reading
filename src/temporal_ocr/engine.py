"""End-to-end temporal orchestration over pluggable detector/OCR backends."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from temporal_ocr.backends import TextDetector, TextRecognizer
from temporal_ocr.change import ChangeMapResult, TileChangeDetector
from temporal_ocr.config import EngineConfig
from temporal_ocr.detection import HierarchicalDetectionPlanner
from temporal_ocr.geometry import (
    candidate_quality,
    canonicalize_crop,
    image_signature,
    polygon_area,
    polygon_bbox,
    polygon_coverage,
    polygon_intersection_over_smaller,
    polygon_iou,
    to_gray,
    transform_polygon,
)
from temporal_ocr.motion import GlobalMotionEstimator, identity_motion
from temporal_ocr.policy import RuleBasedPolicyScheduler
from temporal_ocr.profiling import Profiler, RunProfile
from temporal_ocr.recognition import OCRBatchQueue, RecognitionCache, recognize_tasks
from temporal_ocr.tracking import ContentTracker, GeometryTracker
from temporal_ocr.types import (
    CanonicalObservation,
    ContentTrack,
    DetectionObservation,
    DetectionRequest,
    DetectionTier,
    FramePacket,
    OCRResult,
    PolicyDecision,
    Polygon,
    RuntimeSignals,
    TextEvent,
)

_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WS_RE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


@dataclass(slots=True)
class EngineResult:
    events: list[TextEvent]
    profile: RunProfile
    policy_changes: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [event.to_dict() for event in self.events],
            "profile": self.profile.to_dict(),
            "policy_changes": self.policy_changes,
        }


class TemporalOCREngine:
    """Generic temporal OCR engine with no platform or network dependencies."""

    def __init__(
        self,
        detector: TextDetector,
        recognizer: TextRecognizer,
        *,
        config: EngineConfig | None = None,
        cache: RecognitionCache | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.detector = detector
        self.recognizer = recognizer
        self.cache = cache or RecognitionCache()
        self._reset_run_state()

    def _reset_run_state(self) -> None:
        """Reset temporal state while preserving the optional cross-run cache."""
        self.motion = GlobalMotionEstimator(self.config.motion)
        self.change = TileChangeDetector(
            self.config.detection.tile_rows,
            self.config.detection.tile_cols,
            self.config.detection.tile_change_threshold,
        )
        self.planner = HierarchicalDetectionPlanner(self.config.detection)
        self.geometry = GeometryTracker(self.config.tracking)
        self.content = ContentTracker(self.config.content)
        self.policy = RuleBasedPolicyScheduler(self.config.policy, self.config.detection)
        self.ocr_queue = OCRBatchQueue()
        self._cached_content_ids: set[int] = set()
        self._queued_content_ids: set[int] = set()
        self._results: dict[int, OCRResult] = {}

    @staticmethod
    def _merge_observations(
        observations: list[DetectionObservation],
    ) -> list[DetectionObservation]:
        priority = {
            DetectionTier.AUDIT: 3,
            DetectionTier.LOCAL: 2,
            DetectionTier.FAST: 1,
        }
        selected: list[DetectionObservation] = []
        # Larger line-level proposals are considered before contained local
        # fragments. Confidence and detection tier break ties between boxes of
        # roughly the same scale.
        for observation in sorted(
            observations,
            key=lambda item: (
                polygon_area(item.polygon),
                priority[item.tier],
                item.confidence,
            ),
            reverse=True,
        ):
            duplicate = False
            for item in selected:
                if polygon_iou(observation.polygon, item.polygon) >= 0.75:
                    duplicate = True
                    break
                containment = polygon_intersection_over_smaller(
                    observation.polygon,
                    item.polygon,
                )
                _, top_a, _, bottom_a = polygon_bbox(observation.polygon)
                _, top_b, _, bottom_b = polygon_bbox(item.polygon)
                height_ratio = max(bottom_a - top_a, bottom_b - top_b) / max(
                    min(bottom_a - top_a, bottom_b - top_b),
                    1e-6,
                )
                if containment >= 0.80 and height_ratio <= 2.20:
                    duplicate = True
                    break
            if duplicate:
                continue
            selected.append(observation)
        return selected

    @staticmethod
    def _uncovered_scopes(
        scopes: tuple[Polygon, ...],
        observations: list[DetectionObservation],
    ) -> tuple[Polygon, ...]:
        return tuple(
            scope
            for scope in scopes
            if not any(
                polygon_coverage(scope, item.polygon) >= 0.55
                or polygon_coverage(item.polygon, scope) >= 0.55
                for item in observations
            )
        )

    def _tracked_local_observations(
        self,
        frame: FramePacket,
        request: DetectionRequest,
        *,
        motion: Any,
    ) -> list[DetectionObservation]:
        """Project active geometry tracks into a changed local scope.

        Local change detection is commonly caused by new pixels inside an
        already-known subtitle box. Re-running a neural detector for that box
        is redundant: the geometry can be propagated with the global motion
        estimate and the current crop still goes through content tracking and
        OCR. Periodic full-frame passes remain the discovery safety net for
        genuinely new text regions.
        """
        config = self.config.detection
        if not config.track_guided_local or not request.scopes:
            return []
        observations: list[DetectionObservation] = []
        height, width = np.asarray(frame.image).shape[:2]
        diagonal = max(float(np.hypot(width, height)), 1.0)
        for track in self.geometry.tracks.values():
            if not track.samples or track.state.value == "ended":
                continue
            if track.tracking_confidence < config.track_guided_min_confidence:
                continue
            if float(np.hypot(*track.velocity)) / diagonal > config.track_guided_max_velocity_ratio:
                continue
            polygon = track.latest.polygon
            if getattr(motion, "valid", False):
                polygon = transform_polygon(polygon, np.asarray(motion.matrix))
            if not any(
                polygon_coverage(polygon, scope) >= 0.55
                or polygon_coverage(scope, polygon) >= 0.55
                for scope in request.scopes
            ):
                continue
            observations.append(
                DetectionObservation(
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    polygon=polygon,
                    confidence=track.tracking_confidence,
                    tier=DetectionTier.LOCAL,
                )
            )
        return observations

    def _enqueue_content(
        self,
        track: ContentTrack,
        profiler: Profiler,
        *,
        force: bool = False,
    ) -> None:
        if not track.candidates or track.recognized_text is not None:
            return
        if (
            not force
            and track.stable_observations < self.config.content.stable_observations
            and track.typewriter_score >= self.config.content.typewriter_skip_score
        ):
            # Intermediate typewriter states are superseded by the next state;
            # recognizing each one wastes a batch slot and creates noisy events.
            profiler.count("ocr_deferred_typewriter")
            return
        if track.content_id in self._queued_content_ids:
            return
        best = max(track.candidates, key=lambda item: item.quality)
        cached = self.cache.get(self.recognizer.name, best.signature)
        if cached is not None:
            result = OCRResult(
                content_id=track.content_id,
                text=cached.text,
                confidence=cached.confidence,
                alternatives=cached.alternatives,
                backend=cached.backend,
                inference_sec=0.0,
            )
            self.content.apply_result(result)
            self._results[result.content_id] = result
            self._cached_content_ids.add(result.content_id)
            profiler.profile.cache_hits += 1
            return
        from temporal_ocr.types import OCRTask

        task = OCRTask(
            content_id=track.content_id,
            geometry_id=track.geometry_id,
            candidates=tuple(track.candidates),
        )
        self.content.mark_queued(track.content_id)
        self.ocr_queue.push(task)
        self._queued_content_ids.add(track.content_id)
        profiler.profile.ocr_tasks += 1

    def _flush_ocr(
        self,
        batch_size: int,
        profiler: Profiler,
        *,
        flush_all: bool = False,
    ) -> None:
        while len(self.ocr_queue) and (flush_all or len(self.ocr_queue) >= batch_size):
            tasks = self.ocr_queue.pop_batch(batch_size)
            with profiler.stage("ocr_inference"):
                results = recognize_tasks(self.recognizer, tasks)
            profiler.profile.ocr_batches += 1
            task_by_id = {task.content_id: task for task in tasks}
            for result in results:
                self._queued_content_ids.discard(result.content_id)
                track = self.content.apply_result(result)
                self._results[result.content_id] = result
                task = task_by_id[result.content_id]
                best = max(task.candidates, key=lambda item: item.quality)
                self.cache.put(self.recognizer.name, best.signature, result)
                track.last_seen = max(track.last_seen, best.timestamp)

    def _build_events(self) -> list[TextEvent]:
        events: list[TextEvent] = []
        for track in sorted(self.content.tracks.values(), key=lambda item: item.first_seen):
            if not track.recognized_text:
                continue
            geometry = self.geometry.tracks.get(track.geometry_id)
            if geometry is None:
                continue
            history = tuple(
                (sample.timestamp, sample.polygon)
                for sample in geometry.samples
                if track.first_seen - 1e-6 <= sample.timestamp <= track.last_seen + 1e-6
            )
            if not history and geometry.samples:
                sample = min(
                    geometry.samples,
                    key=lambda item: abs(item.timestamp - track.first_seen),
                )
                history = ((sample.timestamp, sample.polygon),)
            result = self._results.get(track.content_id)
            events.append(
                TextEvent(
                    event_id=len(events) + 1,
                    geometry_id=track.geometry_id,
                    content_id=track.content_id,
                    start=track.first_seen,
                    end=max(track.first_seen, track.last_seen),
                    text_raw=track.recognized_text,
                    text_normalized=normalize_text(track.recognized_text),
                    confidence=track.confidence,
                    polygon_history=history,
                    source_frame_ids=tuple(item.frame_id for item in track.candidates),
                    cached=track.content_id in self._cached_content_ids,
                    recognition_level=2 if result and result.alternatives else 1,
                    alternatives=result.alternatives if result else (),
                )
            )
        return events

    def run(self, frames: Iterable[FramePacket]) -> EngineResult:
        self._reset_run_state()
        profiler = Profiler()
        previous_luma: np.ndarray | None = None
        first_timestamp: float | None = None
        last_timestamp = 0.0
        signals = RuntimeSignals()
        decision = self.policy.decide(signals)
        policy_changes: list[dict[str, Any]] = []
        last_decision: PolicyDecision | None = None

        for frame in frames:
            profiler.profile.frames_decoded += 1
            first_timestamp = frame.timestamp if first_timestamp is None else first_timestamp
            last_timestamp = frame.timestamp
            image = np.asarray(frame.image)
            luma = np.asarray(frame.luma) if frame.luma is not None else to_gray(image)

            if previous_luma is None:
                motion = identity_motion()
                change = ChangeMapResult(1.0, 1.0, (), ())
                scene_cut = True
            else:
                excluded = tuple(
                    track.latest.polygon
                    for track in self.geometry.tracks.values()
                    if track.samples and track.state.value != "ended"
                )
                with profiler.stage("global_motion"):
                    motion = self.motion.estimate(
                        previous_luma,
                        luma,
                        excluded_polygons=excluded,
                    )
                profiler.profile.motion_estimates += 1
                profiler.profile.valid_motion_estimates += int(motion.valid)
                with profiler.stage("change_map"):
                    change = self.change.compare(previous_luma, luma, motion)
                scene_cut = (
                    change.score >= self.config.detection.scene_change_score_threshold
                    and change.changed_ratio >= self.config.detection.scene_change_ratio_threshold
                )

            active_tracks = [
                track
                for track in self.geometry.tracks.values()
                if track.samples and track.state.value != "ended"
            ]
            moving = sum(
                1
                for track in active_tracks
                if float(np.hypot(track.velocity[0], track.velocity[1])) >= 2.0
            )
            signals.timestamp = frame.timestamp
            signals.scene_change_rate = 0.85 * signals.scene_change_rate + 0.15 * float(scene_cut)
            signals.global_motion_magnitude = self.motion.magnitude(motion)
            signals.motion_confidence = motion.confidence
            signals.layout_stability = 1.0 - change.changed_ratio
            signals.moving_text_ratio = moving / max(len(active_tracks), 1)
            signals.detection_queue_length = 0
            signals.ocr_queue_length = len(self.ocr_queue)
            if self.content.tracks:
                signals.average_text_lifetime = float(
                    np.mean(
                        [item.last_seen - item.first_seen for item in self.content.tracks.values()]
                    )
                )
                signals.typewriter_score = float(
                    np.mean([item.typewriter_score for item in self.content.tracks.values()])
                )
            decision = self.policy.decide(signals)
            self.content.config.stable_wait_sec = decision.stable_wait_sec
            self.content.config.maximum_wait_sec = decision.maximum_wait_sec
            if last_decision != decision:
                policy_changes.append(
                    {
                        "timestamp": frame.timestamp,
                        "probe_interval_sec": decision.probe_interval_sec,
                        "audit_interval_sec": decision.audit_interval_sec,
                        "fast_detection_width": decision.fast_detection_width,
                        "stable_wait_sec": decision.stable_wait_sec,
                        "batch_size": decision.batch_size,
                        "batch_wait_ms": decision.batch_wait_ms,
                        "reason": list(decision.reason),
                    }
                )
                last_decision = decision

            # Invalid feature-based motion is common on flat slides and synthetic
            # backgrounds. It is only risky when a large part of the compensated
            # frame changed; otherwise forcing an audit merely wastes inference.
            motion_reliable = motion.valid or change.changed_ratio < 0.25
            requests = self.planner.plan(
                timestamp=frame.timestamp,
                scene_cut=scene_cut,
                changed_scopes=change.scopes,
                changed_ratio=change.changed_ratio,
                has_active_tracks=bool(active_tracks),
                decision=decision,
                motion_reliable=motion_reliable,
            )
            observations: list[DetectionObservation] = []
            with profiler.stage("text_detection"):
                for request in requests:
                    if request.tier == DetectionTier.FAST:
                        profiler.profile.detection_requests_fast += 1
                    elif request.tier == DetectionTier.LOCAL:
                        guided = self._tracked_local_observations(
                            frame,
                            request,
                            motion=motion,
                        )
                        if guided:
                            observations.extend(guided)
                            profiler.count("track_guided_local_observations", len(guided))
                        uncovered = self._uncovered_scopes(request.scopes, observations)
                        if not uncovered:
                            continue
                        profiler.profile.detection_requests_local += 1
                        request = DetectionRequest(
                            tier=request.tier,
                            reason=request.reason,
                            target_width=request.target_width,
                            scopes=uncovered,
                        )
                    else:
                        profiler.profile.detection_requests_audit += 1
                    observations.extend(self.detector.detect(frame, request))
            observations = self._merge_observations(observations)
            profiler.profile.detection_observations += len(observations)
            profiler.profile.frames_probed += int(bool(requests))

            if requests:
                geometry_update = self.geometry.update(
                    observations,
                    motion=motion,
                    frame_size=(image.shape[1], image.shape[0]),
                )
                profiler.profile.geometry_tracks_created += len(geometry_update.created)
                signals.track_birth_rate = 0.8 * signals.track_birth_rate + 0.2 * len(
                    geometry_update.created
                )
                signals.track_loss_rate = 0.8 * signals.track_loss_rate + 0.2 * len(
                    geometry_update.ended
                )
                # The bootstrap audit discovers the initial scene; it cannot
                # represent text missed by the fast/local tiers.
                if active_tracks and any(
                    request.tier == DetectionTier.AUDIT for request in requests
                ):
                    current_yield = len(geometry_update.created) / max(
                        len(observations), 1
                    )
                    signals.audit_new_text_yield = (
                        0.8 * signals.audit_new_text_yield + 0.2 * current_yield
                    )

                for geometry_id in geometry_update.ended:
                    finalized = self.content.finalize_geometry(geometry_id)
                    if finalized is not None:
                        self._enqueue_content(finalized, profiler, force=True)

                for geometry_id, observation in geometry_update.assignments.items():
                    with profiler.stage("perspective_normalization"):
                        crop, transform = canonicalize_crop(image, observation.polygon)
                        signature = image_signature(crop, self.config.content.signature_size)
                        sharpness, contrast, completeness, occlusion = candidate_quality(crop)
                    canonical = CanonicalObservation(
                        geometry_id=geometry_id,
                        frame_id=frame.frame_id,
                        timestamp=frame.timestamp,
                        image=crop,
                        signature=signature,
                        sharpness=sharpness,
                        contrast=contrast,
                        completeness=completeness,
                        occlusion=occlusion,
                        transform=transform,
                    )
                    before_count = len(self.content.tracks)
                    update = self.content.update(canonical)
                    profiler.profile.content_tracks_created += len(self.content.tracks) - before_count
                    if update.finalized is not None:
                        self._enqueue_content(update.finalized, profiler)
                    if update.ready_task is not None:
                        self._enqueue_content(update.active, profiler)
            else:
                # Detection is intentionally skipped on idle frames. Keep the
                # temporal extent of active events correct without creating a
                # new image crop or OCR task; disappearance/change will produce
                # a non-empty change map and re-enter detection.
                for track in self.geometry.tracks.values():
                    if track.state.value != "ended":
                        track.last_seen = frame.timestamp
                for content_id in set(self.content.by_geometry.values()):
                    track = self.content.tracks.get(content_id)
                    if track is not None and track.state.value != "finalized":
                        track.last_seen = frame.timestamp

            if self.ocr_queue.should_flush(decision.batch_size, decision.batch_wait_ms):
                # Age-based flushing must also release a partial batch; otherwise
                # low-traffic videos would wait until end-of-file.
                self._flush_ocr(decision.batch_size, profiler, flush_all=True)
            previous_luma = luma

        for geometry_id in list(self.content.by_geometry):
            finalized = self.content.finalize_geometry(geometry_id)
            if finalized is not None:
                self._enqueue_content(finalized, profiler, force=True)
        self._flush_ocr(decision.batch_size, profiler, flush_all=True)
        events = self._build_events()
        profiler.profile.output_events = len(events)
        processed_video_sec = 0.0
        if first_timestamp is not None:
            processed_video_sec = max(0.0, last_timestamp - first_timestamp)
        profile = profiler.finish(processed_video_sec)
        return EngineResult(events, profile, policy_changes)
