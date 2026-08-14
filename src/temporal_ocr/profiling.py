"""Low-overhead stage timings and counters written with every engine run."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RunProfile:
    total_sec: float = 0.0
    processed_video_sec: float = 0.0
    video_realtime: float = 0.0
    frames_decoded: int = 0
    frames_probed: int = 0
    motion_estimates: int = 0
    valid_motion_estimates: int = 0
    detection_requests_fast: int = 0
    detection_requests_local: int = 0
    detection_requests_audit: int = 0
    detection_observations: int = 0
    geometry_tracks_created: int = 0
    content_tracks_created: int = 0
    ocr_tasks: int = 0
    ocr_batches: int = 0
    cache_hits: int = 0
    output_events: int = 0
    stage_sec: dict[str, float] = field(default_factory=dict)
    counters: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Profiler:
    def __init__(self) -> None:
        self.profile = RunProfile()
        self.started = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.profile.stage_sec[name] = self.profile.stage_sec.get(name, 0.0) + (
                time.perf_counter() - started
            )

    def count(self, name: str, value: float = 1.0) -> None:
        self.profile.counters[name] = self.profile.counters.get(name, 0.0) + value

    def finish(self, processed_video_sec: float) -> RunProfile:
        self.profile.total_sec = time.perf_counter() - self.started
        self.profile.processed_video_sec = processed_video_sec
        self.profile.video_realtime = processed_video_sec / max(self.profile.total_sec, 1e-9)
        return self.profile
