"""Shared video OCR execution helpers used by the CLI and the MCP server."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from temporal_ocr.config import EngineConfig
from temporal_ocr.engine import TemporalOCREngine
from temporal_ocr.output import write_events, write_run_metadata
from temporal_ocr.rapidocr_backend import (
    RapidOCRBatchRecognizer,
    RapidOCRDetector,
    RapidOCRRuntime,
)
from temporal_ocr.sources import PyAVFrameSource


@dataclass(slots=True)
class OCRExecution:
    """Small, serializable handle for one completed OCR run."""

    events_path: Path
    metadata_path: Path
    event_count: int
    profile: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": str(self.events_path),
            "metadata": str(self.metadata_path),
            "event_count": self.event_count,
            "video_realtime": self.profile.get("video_realtime", 0.0),
            "profile": self.profile,
        }


def _take_time_range(
    source: Iterable[Any],
    *,
    start_sec: float | None,
    end_sec: float | None,
) -> Iterator[Any]:
    for frame in source:
        if start_sec is not None and frame.timestamp < start_sec:
            continue
        if end_sec is not None and frame.timestamp > end_sec:
            break
        yield frame


def run_video_ocr(
    video: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
    sample_fps: float | None = None,
    max_width: int | None = None,
    thread_type: str = "AUTO",
    frame_id_offset: int = 0,
    runtime: RapidOCRRuntime | None = None,
    intra_op_num_threads: int | None = None,
    inter_op_num_threads: int | None = None,
) -> OCRExecution:
    """Run the existing engine and write its stable JSONL/JSON artifacts."""
    video_path = Path(video).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if start_sec is not None and start_sec < 0:
        raise ValueError("start_sec must be non-negative")
    if end_sec is not None and end_sec < 0:
        raise ValueError("end_sec must be non-negative")
    if start_sec is not None and end_sec is not None and start_sec > end_sec:
        raise ValueError("start_sec must not be greater than end_sec")
    if frame_id_offset < 0:
        raise ValueError("frame_id_offset must be non-negative")
    if intra_op_num_threads is not None and intra_op_num_threads <= 0:
        raise ValueError("intra_op_num_threads must be positive")
    if inter_op_num_threads is not None and inter_op_num_threads <= 0:
        raise ValueError("inter_op_num_threads must be positive")

    config = EngineConfig.load(config_path) if config_path else EngineConfig()
    source = PyAVFrameSource(
        video_path,
        thread_type=thread_type,
        sample_fps=sample_fps,
        max_width=max_width,
        start_sec=start_sec,
        end_sec=end_sec,
        frame_id_offset=frame_id_offset,
    )
    frames = _take_time_range(source, start_sec=start_sec, end_sec=end_sec)
    if runtime is None:
        runtime = RapidOCRRuntime(
            params={
                "EngineConfig.onnxruntime.intra_op_num_threads": (
                    intra_op_num_threads
                    if intra_op_num_threads is not None
                    else config.ocr.intra_op_num_threads
                ),
                "EngineConfig.onnxruntime.inter_op_num_threads": (
                    inter_op_num_threads
                    if inter_op_num_threads is not None
                    else config.ocr.inter_op_num_threads
                ),
            }
        )
    engine = TemporalOCREngine(
        RapidOCRDetector(runtime=runtime),
        RapidOCRBatchRecognizer(runtime=runtime),
        config=config,
    )
    result = engine.run(frames)

    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    events_path = write_events(target_dir / "events.jsonl", result.events)
    metadata = result.to_dict()
    metadata.pop("events", None)
    metadata["input"] = {
        "video": str(video_path),
        "config": str(Path(config_path).expanduser().resolve()) if config_path else None,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "sample_fps": sample_fps,
        "max_width": max_width,
        "thread_type": thread_type,
        "frame_id_offset": frame_id_offset,
        "intra_op_num_threads": intra_op_num_threads,
        "inter_op_num_threads": inter_op_num_threads,
    }
    metadata["artifacts"] = {
        "events": str(events_path),
        "metadata": str(target_dir / "run.json"),
    }
    metadata_path = write_run_metadata(target_dir / "run.json", metadata)
    return OCRExecution(
        events_path=events_path,
        metadata_path=metadata_path,
        event_count=len(result.events),
        profile=result.profile.to_dict(),
    )
