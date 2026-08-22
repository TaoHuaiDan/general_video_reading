"""End-to-end temporal orchestration over pluggable detector/OCR backends."""

from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, replace
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
    normalized_region_polygon,
    polygon_area,
    polygon_bbox,
    polygon_coverage,
    polygon_intersection_over_smaller,
    polygon_iou,
    to_gray,
    transform_polygon,
    validate_normalized_regions,
)
from temporal_ocr.motion import GlobalMotionEstimator, identity_motion
from temporal_ocr.policy import RuleBasedPolicyScheduler
from temporal_ocr.profiling import Profiler, RunProfile
from temporal_ocr.recognition import (
    OCRBatchQueue,
    RecognitionCache,
    candidate_set_signature,
    recognize_tasks,
)
from temporal_ocr.tracking import ContentTracker, GeometryTracker
from temporal_ocr.types import (
    CanonicalObservation,
    ContentState,
    ContentTrack,
    DetectionObservation,
    DetectionRequest,
    DetectionTier,
    FramePacket,
    OCRResult,
    OCRTask,
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
        exclude_regions: Iterable[Iterable[float]] | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.detector = detector
        self.recognizer = recognizer
        self.cache = cache if cache is not None else RecognitionCache()
        # Semantic namespace of the recognizer configuration; two recognizer
        # configurations that could produce different results for the same
        # crops must never share exact cache entries.
        self._cache_namespace = (
            getattr(self.recognizer, "cache_namespace", None) or self.recognizer.name
        )
        self.exclude_regions = validate_normalized_regions(exclude_regions)
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
        # The content tracker owns a runtime copy of ContentConfig; the
        # caller's canonical EngineConfig is never mutated by a run.
        self.content = ContentTracker(replace(self.config.content))
        self.policy = RuleBasedPolicyScheduler(
            self.config.policy,
            self.config.detection,
            self.config.content,
        )
        self.ocr_queue = OCRBatchQueue()
        self._cached_content_ids: set[int] = set()
        # content_id -> revision of the task currently queued for it, so a
        # re-armed (newer) revision can enqueue while an old task is in flight.
        self._queued_revisions: dict[int, int] = {}
        self._results: dict[int, OCRResult] = {}
        self._geometry_refresh_cooldown_until: dict[int, float] = {}
        self._geometry_refresh_last_refresh_at: dict[int, float] = {}

    def _content_runtime_signals(self) -> tuple[float, float]:
        """Return (average_text_lifetime, typewriter_score) for active content.

        Finalized tracks belong to scenes that already ended; including them
        let a single typewriter scene pollute policy decisions for the rest
        of a long video.  With no active content both signals are explicitly
        zero.
        """
        active = [
            track
            for track in self.content.tracks.values()
            if track.state != ContentState.FINALIZED
        ]
        if not active:
            return 0.0, 0.0
        average_lifetime = float(
            np.mean([item.last_seen - item.first_seen for item in active])
        )
        typewriter_score = float(np.mean([item.typewriter_score for item in active]))
        return average_lifetime, typewriter_score

    def _exclude_polygons(self, width: int, height: int) -> tuple[Polygon, ...]:
        """Return the caller's normalized ignore rectangles in pixel space."""
        return tuple(
            normalized_region_polygon(region, width, height)
            for region in self.exclude_regions
        )

    @staticmethod
    def _is_excluded(
        polygon: Polygon,
        excluded_polygons: tuple[Polygon, ...],
        *,
        minimum_overlap: float = 0.35,
    ) -> bool:
        return any(
            polygon_coverage(polygon, excluded) >= minimum_overlap
            for excluded in excluded_polygons
        )

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
        """Keep only scopes that existing observations themselves cover.

        "Covered" means the observation overlaps enough of the scope.  An
        observation merely contained inside a large scope says nothing about
        the rest of that scope, so counting containment as coverage silently
        dropped undetected text in large change regions.
        """
        return tuple(
            scope
            for scope in scopes
            if not any(
                polygon_coverage(scope, item.polygon) >= 0.55
                for item in observations
            )
        )

    @staticmethod
    def _coalesce_scopes(scopes: Iterable[Polygon]) -> tuple[Polygon, ...]:
        """Remove duplicate or nearly-contained local detector crops.

        Multiple overflow tracks can expand to the same dialogue envelope in
        one frame.  Running RapidOCR once per copy is pure duplicate work; the
        largest envelope already contains every text proposal the smaller crop
        could return.  Keep genuinely separate regions independent.
        """
        selected: list[Polygon] = []
        for scope in sorted(scopes, key=polygon_area, reverse=True):
            if any(
                polygon_iou(scope, item) >= 0.75
                or polygon_coverage(scope, item) >= 0.80
                for item in selected
            ):
                continue
            selected.append(scope)
        return tuple(selected)

    def _tracked_local_observations(
        self,
        frame: FramePacket,
        request: DetectionRequest,
        *,
        motion: Any,
        pixel_delta: np.ndarray | None = None,
        profiler: Profiler | None = None,
    ) -> tuple[list[DetectionObservation], tuple[Polygon, ...], tuple[int, ...]]:
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
            return [], (), ()
        observations: list[DetectionObservation] = []
        overflow_scopes: list[Polygon] = []
        overflow_geometry_ids: list[int] = []
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
            if self._is_excluded(polygon, request.exclude_regions):
                continue
            matching_scopes = tuple(
                scope
                for scope in request.scopes
                if polygon_coverage(polygon, scope) >= 0.55
                or polygon_coverage(scope, polygon) >= 0.55
            )
            if not matching_scopes:
                continue
            refresh_confident = track.tracking_confidence >= 0.60
            # An urgent change by itself is not evidence that the geometry is
            # stale: it is usually just new glyph pixels inside an existing
            # line.  Refresh when the change spills outside the projected box
            # or the scope is structurally much larger than the tracked line
            # (a wrapped-dialogue envelope); otherwise the guided observation
            # path remains detector-free.
            force_local_refresh = False
            content_id = self.content.by_geometry.get(track.geometry_id)
            content_track = self.content.tracks.get(content_id) if content_id is not None else None
            last_refresh_at = self._geometry_refresh_last_refresh_at.get(
                track.geometry_id,
                float("-inf"),
            )
            pixel_overflow = (
                pixel_delta is not None
                and refresh_confident
                and self._has_track_overflow(polygon, pixel_delta)
            )
            cooldown_until = self._geometry_refresh_cooldown_until.get(
                track.geometry_id,
                float("-inf"),
            )
            cooldown_active = frame.timestamp < cooldown_until
            # A new content state that appeared after the last refresh is a
            # stronger signal than the cooldown alone.  Permit one early
            # geometry refresh when changed pixels also spill outside the box;
            # this catches a newly wrapped line without reopening a detector
            # call on every ordinary typewriter frame.
            early_refresh = (
                cooldown_active
                and pixel_overflow
                and content_track is not None
                and content_track.last_changed > last_refresh_at + 1e-6
                and frame.timestamp - last_refresh_at
                >= self.config.detection.track_guided_existing_refresh_cooldown_sec
            )
            refresh_allowed = not cooldown_active or early_refresh
            if profiler is not None:
                profiler.count("local_early_refresh_candidates", early_refresh)
            geometry_scope_refresh = refresh_allowed and self._scope_geometry_refresh(
                polygon,
                matching_scopes,
                tracking_confidence=track.tracking_confidence,
            )
            if profiler is not None:
                profiler.count("local_geometry_scope_refresh_candidates", geometry_scope_refresh)
                profiler.count("local_pixel_overflow_candidates", pixel_overflow)
            if refresh_allowed and (
                force_local_refresh or pixel_overflow or geometry_scope_refresh
            ):
                if profiler is not None:
                    profiler.count("local_geometry_scope_refresh_tracks", geometry_scope_refresh)
                    profiler.count("local_pixel_overflow_refresh_tracks", pixel_overflow)
                    profiler.count("local_early_refresh_tracks", early_refresh)
                overflow_geometry_ids.append(track.geometry_id)
                overflow_scopes.extend(
                    self._expanded_overflow_scopes(polygon, matching_scopes)
                )
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
        return observations, tuple(overflow_scopes), tuple(overflow_geometry_ids)

    def _has_track_overflow(
        self,
        polygon: Polygon,
        pixel_delta: np.ndarray,
    ) -> bool:
        """Detect changed pixels just outside a projected text quadrilateral."""
        config = self.config.detection
        x1, y1, x2, y2 = polygon_bbox(polygon)
        box_width = max(2.0, x2 - x1)
        box_height = max(2.0, y2 - y1)
        padding_x = max(6.0, box_width * 0.35)
        padding_y = max(8.0, box_height * config.track_guided_overflow_padding_ratio)
        image_height, image_width = pixel_delta.shape[:2]

        def bounds(left: float, top: float, right: float, bottom: float) -> tuple[int, int, int, int]:
            return (
                max(0, min(image_width - 1, int(np.floor(left)))),
                max(0, min(image_height - 1, int(np.floor(top)))),
                max(1, min(image_width, int(np.ceil(right)))),
                max(1, min(image_height, int(np.ceil(bottom)))),
            )

        inner_left, inner_top, inner_right, inner_bottom = bounds(x1, y1, x2, y2)
        outer_left, outer_top, outer_right, outer_bottom = bounds(
            x1 - padding_x,
            y1 - padding_y,
            x2 + padding_x,
            y2 + padding_y,
        )
        threshold = config.track_guided_pixel_change_threshold
        vertical_parts = (
            pixel_delta[outer_top:inner_top, outer_left:outer_right],
            pixel_delta[inner_bottom:outer_bottom, outer_left:outer_right],
        )
        horizontal_parts = (
            pixel_delta[inner_top:inner_bottom, outer_left:inner_left],
            pixel_delta[inner_top:inner_bottom, inner_right:outer_right],
        )

        def changed_ratio(parts: tuple[np.ndarray, ...]) -> float:
            area = sum(part.size for part in parts)
            if area <= 0:
                return 0.0
            changed = sum(np.count_nonzero(part >= threshold) for part in parts)
            return float(changed) / area

        return max(changed_ratio(vertical_parts), changed_ratio(horizontal_parts)) >= (
            config.track_guided_overflow_ratio
        )

    @staticmethod
    def _scope_geometry_refresh(
        polygon: Polygon,
        scopes: tuple[Polygon, ...],
        *,
        tracking_confidence: float,
    ) -> bool:
        """Refresh a high-confidence line when its change scope can hold wraps."""
        if tracking_confidence < 0.60:
            return False
        x1, y1, x2, y2 = polygon_bbox(polygon)
        box_width = max(2.0, x2 - x1)
        box_height = max(2.0, y2 - y1)
        if box_height < 16.0:
            return False
        for scope in scopes:
            sx1, sy1, sx2, sy2 = polygon_bbox(scope)
            if (
                sx2 - sx1 >= 2.0 * box_width
                and sy2 - sy1 >= 4.0 * box_height
            ):
                return True
        return False

    @staticmethod
    def _expanded_overflow_scopes(
        polygon: Polygon,
        scopes: tuple[Polygon, ...],
    ) -> tuple[Polygon, ...]:
        """Make a local refresh wide/tall enough to recover wrapped dialogue."""
        x1, y1, x2, y2 = polygon_bbox(polygon)
        box_width = max(2.0, x2 - x1)
        box_height = max(2.0, y2 - y1)
        # The tracked box can be only one line of a wrapped dialogue.  Keep
        # the changed local component as the horizontal/vertical envelope so
        # the refresh can discover sibling lines, but still issue one model
        # call instead of detecting the original and expanded scopes twice.
        scope_boxes = [polygon_bbox(scope) for scope in scopes]
        if scope_boxes:
            scope_x1 = min(item[0] for item in scope_boxes)
            scope_y1 = min(item[1] for item in scope_boxes)
            scope_x2 = max(item[2] for item in scope_boxes)
            scope_y2 = max(item[3] for item in scope_boxes)
        else:
            scope_x1, scope_y1, scope_x2, scope_y2 = x1, y1, x2, y2
        expanded = (
            min(x1 - 0.6 * box_width, scope_x1),
            min(y1 - 2.0 * box_height, scope_y1),
            max(x2 + 0.6 * box_width, scope_x2),
            max(y2 + 2.0 * box_height, scope_y2),
        )
        left, top, right, bottom = expanded
        return (
            (
                (left, top),
                (right, top),
                (right, bottom),
                (left, bottom),
            ),
        )

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
        if self._queued_revisions.get(track.content_id) == track.revision:
            # A task for exactly this content revision is already queued.
            return
        # Exact reuse must key on the complete computation input: the final
        # result can come from any fallback candidate, so the key covers the
        # whole candidate set, not just the primary crop.
        cache_key = candidate_set_signature([item.image for item in track.candidates])
        cached = self.cache.get(self._cache_namespace, cache_key)
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
        task = OCRTask(
            content_id=track.content_id,
            geometry_id=track.geometry_id,
            candidates=tuple(track.candidates),
            revision=track.revision,
        )
        self.content.mark_queued(track.content_id)
        self.ocr_queue.push(task)
        self._queued_revisions[track.content_id] = track.revision
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
            # Task identity is (content_id, revision): one batch may contain
            # several revisions of the same content track.
            task_by_identity = {(task.content_id, task.revision): task for task in tasks}
            for result in results:
                task = task_by_identity[(result.content_id, result.revision)]
                # The cache is keyed by the exact crop set, so the entry stays
                # valid for that input even if the task's revision went stale.
                self.cache.put(
                    self._cache_namespace,
                    candidate_set_signature([item.image for item in task.candidates]),
                    result,
                )
                queued_revision = self._queued_revisions.get(result.content_id)
                if queued_revision == task.revision:
                    del self._queued_revisions[result.content_id]
                track = self.content.tracks.get(result.content_id)
                if track is None or track.revision != task.revision:
                    # The content was re-armed to a newer revision while this
                    # task was in flight; the stale text must not overwrite it.
                    profiler.count("ocr_stale_results_discarded")
                    continue
                self.content.apply_result(result)
                self._results[result.content_id] = result
                best = max(task.candidates, key=lambda item: item.quality)
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
            excluded_polygons = self._exclude_polygons(image.shape[1], image.shape[0])

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
                        excluded_polygons=(*excluded, *excluded_polygons),
                    )
                profiler.profile.motion_estimates += 1
                profiler.profile.valid_motion_estimates += int(motion.valid)
                with profiler.stage("change_map"):
                    change = self.change.compare(
                        previous_luma,
                        luma,
                        motion,
                        excluded_polygons=excluded_polygons,
                    )
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
            # The current detector runs synchronously inside the frame loop,
            # so no real detection queue exists. This stays a reserved,
            # always-zero signal for a future async detector; it must not be
            # fabricated into a fake queue depth.
            signals.detection_queue_length = 0
            signals.ocr_queue_length = len(self.ocr_queue)
            (
                signals.average_text_lifetime,
                signals.typewriter_score,
            ) = self._content_runtime_signals()
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
                        "maximum_wait_sec": decision.maximum_wait_sec,
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
            requests = [
                replace(request, exclude_regions=excluded_polygons)
                for request in requests
            ]
            observations: list[DetectionObservation] = []
            overflow_scopes: list[Polygon] = []
            overflow_geometry_ids: list[int] = []
            with profiler.stage("text_detection"):
                for request in requests:
                    if request.tier == DetectionTier.FAST:
                        profiler.profile.detection_requests_fast += 1
                    elif request.tier == DetectionTier.LOCAL:
                        if request.reason == "urgent_local_change":
                            profiler.count("urgent_local_requests")
                        guided, request_overflow, request_overflow_ids = (
                            self._tracked_local_observations(
                            frame,
                            request,
                            motion=motion,
                            pixel_delta=change.pixel_delta,
                            profiler=profiler,
                            )
                        )
                        if guided:
                            observations.extend(guided)
                            profiler.count("track_guided_local_observations", len(guided))
                        overflow_scopes.extend(request_overflow)
                        overflow_geometry_ids.extend(request_overflow_ids)
                        profiler.count("local_overflow_refresh_tracks", len(request_overflow_ids))
                        uncovered = self._uncovered_scopes(request.scopes, observations)
                        if request_overflow:
                            # An overflow refresh is a replacement for the original
                            # local scope.  Keeping both would invoke RapidOCR twice
                            # on overlapping crops and erase the speed benefit.
                            uncovered = tuple(
                                scope
                                for scope in uncovered
                                if not any(
                                    polygon_coverage(scope, refresh) >= 0.55
                                    or polygon_coverage(refresh, scope) >= 0.55
                                    for refresh in request_overflow
                                )
                            )
                            uncovered += tuple(request_overflow)
                        # A change scope fully covered by an ignore region is
                        # pure watermark noise; do not launch a local detector
                        # call for it.  Large scopes that merely contain a
                        # corner watermark are retained for nearby text.
                        uncovered = tuple(
                            scope
                            for scope in uncovered
                            if not any(
                                polygon_coverage(scope, excluded) >= 0.80
                                for excluded in request.exclude_regions
                            )
                        )
                        uncovered = self._coalesce_scopes(uncovered)
                        if not uncovered:
                            profiler.count("local_requests_fully_guided")
                            continue
                        profiler.count("local_uncovered_scopes", len(uncovered))
                        profiler.profile.detection_requests_local += 1
                        request = DetectionRequest(
                            tier=request.tier,
                            reason=request.reason,
                            target_width=request.target_width,
                            scopes=uncovered,
                            exclude_regions=request.exclude_regions,
                        )
                    else:
                        profiler.profile.detection_requests_audit += 1
                    # Keep detector-level telemetry separate from the aggregate
                    # text_detection stage.  This makes it possible to tell
                    # whether a slow run is caused by too many calls, overly
                    # broad local scopes, or the detector itself.
                    tier_name = request.tier.value
                    scope_area = sum(polygon_area(scope) for scope in request.scopes)
                    if not request.scopes:
                        scope_area = float(image.shape[0] * image.shape[1])
                    detector_started = time.perf_counter()
                    detected = self.detector.detect(frame, request)
                    before_exclusion_count = len(detected)
                    detected = [
                        observation
                        for observation in detected
                        if not self._is_excluded(
                            observation.polygon,
                            request.exclude_regions,
                        )
                    ]
                    profiler.count(
                        "excluded_observations",
                        before_exclusion_count - len(detected),
                    )
                    profiler.count(f"detector_calls_{tier_name}")
                    profiler.count(
                        f"detector_sec_{tier_name}",
                        time.perf_counter() - detector_started,
                    )
                    profiler.count(f"detector_scope_count_{tier_name}", len(request.scopes) or 1)
                    profiler.count(f"detector_scope_area_{tier_name}", scope_area)
                    profiler.count(f"detector_empty_calls_{tier_name}", not detected)
                    observations.extend(detected)
            observations = self._merge_observations(observations)
            profiler.profile.detection_observations += len(observations)
            profiler.profile.frames_probed += int(bool(requests))

            if requests:
                overflow_ids = set(overflow_geometry_ids)
                geometry_update = self.geometry.update(
                    observations,
                    motion=motion,
                    frame_size=(image.shape[1], image.shape[0]),
                )
                # A refresh re-detects the geometry envelope; it does not prove
                # the old text vanished.  Tracks the refresh re-detected keep
                # their geometry/content identity, so the same line neither
                # duplicates into a second event nor loses its lifecycle.  Only
                # tracks with no matching detection are finalized now, and
                # their best candidate must still reach OCR instead of being
                # silently discarded.
                reassociated = overflow_ids.intersection(geometry_update.assignments)
                lost_refresh_ids = tuple(overflow_ids - reassociated)
                for geometry_id in self.geometry.end_ids(lost_refresh_ids):
                    finalized = self.content.finalize_geometry(geometry_id)
                    if finalized is not None:
                        profiler.count("local_refresh_lost_tracks")
                        self._enqueue_content(finalized, profiler, force=True)
                if overflow_ids and geometry_update.assignments:
                    for geometry_id, assigned in geometry_update.assignments.items():
                        assigned_in_refresh = any(
                            polygon_coverage(assigned.polygon, scope) >= 0.55
                            or polygon_coverage(scope, assigned.polygon) >= 0.55
                            for scope in overflow_scopes
                        )
                        if not assigned_in_refresh:
                            continue
                        refresh_observation_count = sum(
                            any(
                                polygon_coverage(item.polygon, scope) >= 0.55
                                or polygon_coverage(scope, item.polygon) >= 0.55
                                for scope in overflow_scopes
                            )
                            for item in observations
                        )
                        cooldown_sec = (
                            self.config.detection.track_guided_refresh_cooldown_sec
                            if refresh_observation_count >= 2
                            else self.config.detection.track_guided_existing_refresh_cooldown_sec
                        )
                        cooldown_until = frame.timestamp + max(
                            cooldown_sec,
                            self.config.content.maximum_wait_sec + 0.45,
                        )
                        self._geometry_refresh_cooldown_until[geometry_id] = cooldown_until
                        self._geometry_refresh_last_refresh_at[geometry_id] = frame.timestamp
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
