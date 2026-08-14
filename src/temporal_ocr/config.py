"""Configuration with conservative, benchmarkable defaults."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class MotionConfig:
    max_width: int = 640
    min_features: int = 24
    min_inlier_ratio: float = 0.45
    max_residual_error: float = 3.5
    ransac_threshold: float = 3.0


@dataclass(slots=True)
class DetectionConfig:
    fast_width: int = 960
    local_width: int = 1600
    audit_width: int = 2560
    audit_interval_sec: float = 20.0
    min_audit_interval_sec: float = 3.0
    max_audit_interval_sec: float = 40.0
    tile_rows: int = 10
    tile_cols: int = 16
    tile_change_threshold: float = 0.035
    # A small local change can be handled by LOCAL detection when geometry
    # tracks already exist. FAST is reserved for bootstrap and broad changes.
    fast_trigger_change_ratio: float = 0.12
    scene_change_score_threshold: float = 0.085
    scene_change_ratio_threshold: float = 0.60


@dataclass(slots=True)
class TrackingConfig:
    min_iou: float = 0.25
    max_center_distance: float = 0.20
    max_missed_frames: int = 6


@dataclass(slots=True)
class ContentConfig:
    change_threshold: float = 0.16
    stable_observations: int = 2
    stable_wait_sec: float = 0.45
    maximum_wait_sec: float = 1.8
    candidate_limit: int = 3
    signature_size: int = 16
    typewriter_skip_score: float = 0.60


@dataclass(slots=True)
class PolicyConfig:
    min_probe_interval_sec: float = 0.10
    max_probe_interval_sec: float = 0.75
    default_probe_interval_sec: float = 0.25
    min_batch_size: int = 1
    max_batch_size: int = 64
    default_batch_size: int = 16
    min_batch_wait_ms: int = 10
    max_batch_wait_ms: int = 80
    default_batch_wait_ms: int = 35
    queue_pressure_threshold: int = 48


@dataclass(slots=True)
class EngineConfig:
    motion: MotionConfig = field(default_factory=MotionConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    content: ContentConfig = field(default_factory=ContentConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EngineConfig:
        return cls(
            motion=MotionConfig(**data.get("motion", {})),
            detection=DetectionConfig(**data.get("detection", {})),
            tracking=TrackingConfig(**data.get("tracking", {})),
            content=ContentConfig(**data.get("content", {})),
            policy=PolicyConfig(**data.get("policy", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> EngineConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
