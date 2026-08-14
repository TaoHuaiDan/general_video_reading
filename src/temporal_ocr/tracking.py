"""Geometry and content tracking kept as separate state machines."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from temporal_ocr.config import ContentConfig, TrackingConfig
from temporal_ocr.geometry import (
    polygon_center,
    polygon_iou,
    signature_distance,
    transform_polygon,
)
from temporal_ocr.selection import ComplementaryCandidateSelector
from temporal_ocr.types import (
    CanonicalObservation,
    ContentState,
    ContentTrack,
    DetectionObservation,
    GeometrySample,
    GeometryState,
    GeometryTrack,
    MotionEstimate,
    OCRResult,
    OCRTask,
)


@dataclass(slots=True)
class GeometryUpdate:
    assignments: dict[int, DetectionObservation]
    created: tuple[int, ...]
    ended: tuple[int, ...]


class GeometryTracker:
    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()
        self.tracks: dict[int, GeometryTrack] = {}
        self._next_id = 1

    def update(
        self,
        observations: list[DetectionObservation],
        *,
        motion: MotionEstimate | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> GeometryUpdate:
        active = [track for track in self.tracks.values() if track.state != GeometryState.ENDED]
        candidates: list[tuple[float, int, int]] = []
        diagonal = 1.0
        if frame_size is not None:
            diagonal = max(1.0, math.hypot(frame_size[0], frame_size[1]))

        for track in active:
            predicted = track.latest.polygon
            if motion is not None and motion.valid:
                predicted = transform_polygon(predicted, np.asarray(motion.matrix))
            predicted_center = polygon_center(predicted)
            for index, observation in enumerate(observations):
                iou = polygon_iou(predicted, observation.polygon)
                center = polygon_center(observation.polygon)
                distance = math.hypot(
                    center[0] - predicted_center[0],
                    center[1] - predicted_center[1],
                ) / diagonal
                if iou >= self.config.min_iou or distance <= self.config.max_center_distance:
                    cost = 0.7 * (1.0 - iou) + 0.3 * distance
                    candidates.append((cost, track.geometry_id, index))

        assignments: dict[int, DetectionObservation] = {}
        used_tracks: set[int] = set()
        used_observations: set[int] = set()
        for _cost, geometry_id, observation_index in sorted(candidates):
            if geometry_id in used_tracks or observation_index in used_observations:
                continue
            used_tracks.add(geometry_id)
            used_observations.add(observation_index)
            assignments[geometry_id] = observations[observation_index]

        created: list[int] = []
        for index, observation in enumerate(observations):
            if index in used_observations:
                continue
            geometry_id = self._next_id
            self._next_id += 1
            track = GeometryTrack(
                geometry_id=geometry_id,
                state=GeometryState.DETECTED,
                first_seen=observation.timestamp,
                last_seen=observation.timestamp,
            )
            self.tracks[geometry_id] = track
            assignments[geometry_id] = observation
            created.append(geometry_id)

        for geometry_id, observation in assignments.items():
            track = self.tracks[geometry_id]
            previous_center = polygon_center(track.latest.polygon) if track.samples else None
            track.samples.append(
                GeometrySample(
                    frame_id=observation.frame_id,
                    timestamp=observation.timestamp,
                    polygon=observation.polygon,
                    confidence=observation.confidence,
                )
            )
            if previous_center is not None:
                center = polygon_center(observation.polygon)
                track.velocity = (center[0] - previous_center[0], center[1] - previous_center[1])
            track.last_seen = observation.timestamp
            track.missed_frames = 0
            track.state = GeometryState.TRACKING
            track.tracking_confidence = observation.confidence

        ended: list[int] = []
        for track in active:
            if track.geometry_id in assignments:
                continue
            track.missed_frames += 1
            track.state = GeometryState.LOST
            if track.missed_frames > self.config.max_missed_frames:
                track.state = GeometryState.ENDED
                ended.append(track.geometry_id)
        return GeometryUpdate(assignments, tuple(created), tuple(ended))


@dataclass(slots=True)
class ContentUpdate:
    active: ContentTrack
    ready_task: OCRTask | None = None
    finalized: ContentTrack | None = None


class ContentTracker:
    def __init__(self, config: ContentConfig | None = None) -> None:
        self.config = config or ContentConfig()
        self.selector = ComplementaryCandidateSelector(self.config.candidate_limit)
        self.tracks: dict[int, ContentTrack] = {}
        self.by_geometry: dict[int, int] = {}
        self._next_id = 1

    def _new_track(self, observation: CanonicalObservation) -> ContentTrack:
        track = ContentTrack(
            content_id=self._next_id,
            geometry_id=observation.geometry_id,
            state=ContentState.UNKNOWN,
            first_seen=observation.timestamp,
            last_seen=observation.timestamp,
            last_changed=observation.timestamp,
            latest_signature=observation.signature,
            stable_observations=1,
            candidates=[observation],
        )
        self._next_id += 1
        self.tracks[track.content_id] = track
        self.by_geometry[observation.geometry_id] = track.content_id
        return track

    def update(self, observation: CanonicalObservation) -> ContentUpdate:
        content_id = self.by_geometry.get(observation.geometry_id)
        if content_id is None:
            return ContentUpdate(self._new_track(observation))

        track = self.tracks[content_id]
        difference = signature_distance(track.latest_signature, observation.signature)
        if difference >= self.config.change_threshold:
            previous = track
            previous.state = ContentState.FINALIZED
            new_track = self._new_track(observation)
            new_track.state = ContentState.CHANGING
            new_track.typewriter_score = min(1.0, previous.typewriter_score * 0.7 + 0.3)
            return ContentUpdate(new_track, finalized=previous)

        track.last_seen = observation.timestamp
        track.latest_signature = observation.signature
        track.stable_observations += 1
        track.candidates = self.selector.select(track.candidates, observation)
        elapsed_stable = observation.timestamp - track.last_changed
        ready = (
            track.state not in {ContentState.QUEUED, ContentState.RECOGNIZED}
            and (
                track.stable_observations >= self.config.stable_observations
                or elapsed_stable >= self.config.stable_wait_sec
                or observation.timestamp - track.first_seen >= self.config.maximum_wait_sec
            )
        )
        if ready:
            track.state = ContentState.STABLE
            return ContentUpdate(
                track,
                ready_task=OCRTask(
                    content_id=track.content_id,
                    geometry_id=track.geometry_id,
                    candidates=tuple(track.candidates),
                ),
            )
        return ContentUpdate(track)

    def mark_queued(self, content_id: int) -> None:
        self.tracks[content_id].state = ContentState.QUEUED

    def apply_result(self, result: OCRResult) -> ContentTrack:
        track = self.tracks[result.content_id]
        track.recognized_text = result.text
        track.confidence = result.confidence
        track.state = ContentState.RECOGNIZED
        return track

    def finalize_geometry(self, geometry_id: int) -> ContentTrack | None:
        content_id = self.by_geometry.pop(geometry_id, None)
        if content_id is None:
            return None
        track = self.tracks[content_id]
        track.state = ContentState.FINALIZED
        return track
