"""Reproducible benchmark helpers for engine and policy A/B comparisons."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from temporal_ocr.engine import TemporalOCREngine
from temporal_ocr.metrics import EvaluationReport, evaluate_events
from temporal_ocr.types import FramePacket, TextEvent


@dataclass(slots=True)
class BenchmarkRun:
    name: str
    engine_profile: dict[str, Any]
    quality: EvaluationReport
    policy_changes: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "engine_profile": self.engine_profile,
            "quality": asdict(self.quality),
            "policy_changes": self.policy_changes,
        }


def run_benchmark(
    name: str,
    engine: TemporalOCREngine,
    frames: Iterable[FramePacket],
    reference_events: list[TextEvent],
) -> BenchmarkRun:
    """Run one engine configuration and evaluate quality plus throughput."""
    result = engine.run(frames)
    quality = evaluate_events(
        reference_events,
        result.events,
        video_sec=result.profile.processed_video_sec,
        wall_sec=result.profile.total_sec,
    )
    return BenchmarkRun(
        name=name,
        engine_profile=result.profile.to_dict(),
        quality=quality,
        policy_changes=result.policy_changes,
    )


def rank_runs(runs: list[BenchmarkRun]) -> list[BenchmarkRun]:
    """Prefer event completeness, then text accuracy, then throughput."""
    return sorted(
        runs,
        key=lambda run: (
            run.quality.event_recall,
            run.quality.text_accuracy,
            -run.quality.duplicate_rate,
            run.quality.video_realtime,
        ),
        reverse=True,
    )
