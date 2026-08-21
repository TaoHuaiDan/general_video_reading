"""Shared domain types for the temporal OCR pipeline.

The most important boundary in this module is the separation between geometry
and content. A GeometryTrack may move, rotate or scale without creating a new
ContentTrack. Conversely, a fixed subtitle box may host many content tracks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, TypeAlias

Point: TypeAlias = tuple[float, float]
Polygon: TypeAlias = tuple[Point, Point, Point, Point]


class DetectionTier(str, Enum):
    FAST = "fast"
    LOCAL = "local"
    AUDIT = "audit"


class GeometryState(str, Enum):
    DETECTED = "detected"
    TRACKING = "tracking"
    OCCLUDED = "occluded"
    LOST = "lost"
    ENDED = "ended"


class ContentState(str, Enum):
    UNKNOWN = "unknown"
    CHANGING = "changing"
    STABLE = "stable"
    QUEUED = "queued"
    RECOGNIZED = "recognized"
    FINALIZED = "finalized"


@dataclass(slots=True)
class FramePacket:
    frame_id: int
    timestamp: float
    image: Any
    luma: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MotionEstimate:
    matrix: Any
    model: str = "identity"
    confidence: float = 0.0
    inlier_ratio: float = 0.0
    residual_error: float = 0.0
    valid: bool = False


@dataclass(slots=True)
class DetectionRequest:
    tier: DetectionTier
    reason: str
    target_width: int
    scopes: tuple[Polygon, ...] = ()
    # Pixel-space polygons that the detector must ignore for this frame.
    # The engine populates these from normalized MCP/API regions.
    exclude_regions: tuple[Polygon, ...] = ()


@dataclass(slots=True)
class DetectionObservation:
    frame_id: int
    timestamp: float
    polygon: Polygon
    confidence: float
    tier: DetectionTier
    orientation_deg: float = 0.0
    preliminary_text: str | None = None


@dataclass(slots=True)
class GeometrySample:
    frame_id: int
    timestamp: float
    polygon: Polygon
    confidence: float


@dataclass(slots=True)
class GeometryTrack:
    geometry_id: int
    state: GeometryState
    first_seen: float
    last_seen: float
    samples: list[GeometrySample] = field(default_factory=list)
    missed_frames: int = 0
    velocity: Point = (0.0, 0.0)
    scale_rate: float = 0.0
    rotation_rate: float = 0.0
    tracking_confidence: float = 0.0

    @property
    def latest(self) -> GeometrySample:
        return self.samples[-1]


@dataclass(slots=True)
class CanonicalObservation:
    geometry_id: int
    frame_id: int
    timestamp: float
    image: Any
    signature: bytes
    sharpness: float
    contrast: float
    completeness: float
    occlusion: float
    transform: Any | None = None

    @property
    def quality(self) -> float:
        return max(
            0.0,
            0.40 * self.sharpness
            + 0.25 * self.contrast
            + 0.25 * self.completeness
            - 0.10 * self.occlusion,
        )


@dataclass(slots=True)
class ContentTrack:
    content_id: int
    geometry_id: int
    state: ContentState
    first_seen: float
    last_seen: float
    last_changed: float
    latest_signature: bytes
    # Signature baseline for accumulated sub-threshold drift.  It is set at
    # track creation and refreshed whenever recognition is re-armed, so slow
    # gradual changes (a slow typewriter) cannot hide behind the per-step
    # change threshold forever.
    anchor_signature: bytes = b""
    stable_observations: int = 0
    typewriter_score: float = 0.0
    candidates: list[CanonicalObservation] = field(default_factory=list)
    recognized_text: str | None = None
    confidence: float = 0.0


@dataclass(slots=True)
class OCRTask:
    content_id: int
    geometry_id: int
    candidates: tuple[CanonicalObservation, ...]
    priority: int = 0


@dataclass(slots=True)
class OCRResult:
    content_id: int
    text: str
    confidence: float
    alternatives: tuple[str, ...] = ()
    backend: str = "unknown"
    inference_sec: float = 0.0


@dataclass(slots=True)
class TextEvent:
    event_id: int
    geometry_id: int
    content_id: int
    start: float
    end: float
    text_raw: str
    text_normalized: str
    confidence: float
    polygon_history: tuple[tuple[float, Polygon], ...]
    source_frame_ids: tuple[int, ...]
    cached: bool = False
    recognition_level: int = 1
    alternatives: tuple[str, ...] = ()
    type_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RuntimeSignals:
    timestamp: float = 0.0
    scene_change_rate: float = 0.0
    global_motion_magnitude: float = 0.0
    motion_confidence: float = 1.0
    layout_stability: float = 1.0
    average_text_lifetime: float = 0.0
    moving_text_ratio: float = 0.0
    track_birth_rate: float = 0.0
    track_loss_rate: float = 0.0
    typewriter_score: float = 0.0
    audit_new_text_yield: float = 0.0
    cache_hit_rate: float = 0.0
    detection_queue_length: int = 0
    ocr_queue_length: int = 0
    # ``None`` means no sampler is connected.  A missing GPU sampler must not
    # be mistaken for an idle GPU and influence batching decisions.
    cpu_utilization: float | None = None
    gpu_utilization: float | None = None


@dataclass(slots=True)
class PolicyDecision:
    probe_interval_sec: float
    audit_interval_sec: float
    fast_detection_width: int
    stable_wait_sec: float
    maximum_wait_sec: float
    batch_size: int
    batch_wait_ms: int
    enable_local_detection: bool = True
    reason: tuple[str, ...] = ()
