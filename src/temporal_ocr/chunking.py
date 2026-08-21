"""Parallel time-chunk orchestration for long-video OCR runs.

Chunks are logical seek windows rather than re-encoded video files.  Each
window has a small overlap with its neighbours so the temporal engine can
finish events that cross a boundary.  The final merge is deliberately
conservative: only events from different chunks and from the overlap vicinity
can be merged.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from temporal_ocr.config import EngineConfig
from temporal_ocr.geometry import as_polygon, polygon_iou
from temporal_ocr.metrics import text_similarity
from temporal_ocr.output import write_run_metadata
from temporal_ocr.rapidocr_backend import RapidOCRRuntime
from temporal_ocr.runner import OCRExecution, run_video_ocr

_FRAME_ID_CHUNK_STRIDE = 10_000_000
DEFAULT_AUTO_CHUNK_SEC = 180.0
DEFAULT_SHORT_VIDEO_THRESHOLD_SEC = 300.0


@dataclass(frozen=True, slots=True)
class OCRChunk:
    index: int
    core_start_sec: float
    core_end_sec: float
    read_start_sec: float
    read_end_sec: float


def _probe_duration(video: Path) -> float:
    """Return a local video's duration using PyAV, with an OpenCV fallback."""
    duration = 0.0
    try:
        import av

        with av.open(str(video)) as container:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is not None and stream.duration is not None and stream.time_base:
                duration = float(stream.duration * stream.time_base)
                if duration > 0:
                    return duration
            if container.duration:
                duration = float(container.duration) / 1_000_000.0
                if duration > 0:
                    return duration
    except Exception:  # noqa: BLE001 - probe fallback is intentionally best effort
        duration = 0.0

    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not probe video duration: {video}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    finally:
        capture.release()
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"video duration is unavailable: {video}")
    return frame_count / fps


def make_chunks(
    duration_sec: float,
    *,
    chunk_sec: float,
    overlap_sec: float,
    start_sec: float = 0.0,
) -> list[OCRChunk]:
    """Build non-overlapping core ranges with overlapping read windows."""
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    if chunk_sec <= 0:
        raise ValueError("chunk_sec must be positive")
    if overlap_sec < 0 or overlap_sec >= chunk_sec / 2.0:
        raise ValueError("overlap_sec must be >= 0 and less than half of chunk_sec")
    if start_sec < 0 or start_sec >= duration_sec:
        raise ValueError("start_sec must be inside the video duration")

    chunks: list[OCRChunk] = []
    core_start = start_sec
    index = 0
    while core_start < duration_sec - 1e-9:
        core_end = min(duration_sec, core_start + chunk_sec)
        chunks.append(
            OCRChunk(
                index=index,
                core_start_sec=core_start,
                core_end_sec=core_end,
                read_start_sec=max(start_sec, core_start - overlap_sec),
                read_end_sec=min(duration_sec, core_end + overlap_sec),
            )
        )
        index += 1
        core_start = core_end
    return chunks


def choose_chunk_sec(
    duration_sec: float,
    *,
    ideal_chunk_sec: float = DEFAULT_AUTO_CHUNK_SEC,
    short_video_threshold_sec: float = DEFAULT_SHORT_VIDEO_THRESHOLD_SEC,
) -> float | None:
    """Choose an automatic chunk width, or bypass chunking for short videos.

    The target width comes from the full-video benchmark on the reference
    machine: around 180 seconds gives one useful wave of four workers for a
    roughly twelve-minute input.  Dividing by the rounded chunk count keeps
    the final tail from becoming an under-utilized second wave on other
    durations.
    """
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    if ideal_chunk_sec <= 0:
        raise ValueError("ideal_chunk_sec must be positive")
    if short_video_threshold_sec < 0:
        raise ValueError("short_video_threshold_sec must be non-negative")
    if duration_sec <= short_video_threshold_sec:
        return None
    chunk_count = max(2, math.ceil(duration_sec / ideal_chunk_sec))
    return duration_sec / chunk_count


def _read_events(path: Path, *, chunk_index: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            # Local tracker IDs repeat in every independently initialized
            # chunk. Remap them before the merge while retaining source chunk
            # provenance for diagnostics.
            event["geometry_id"] = int(event.get("geometry_id", 0)) + (
                chunk_index * _FRAME_ID_CHUNK_STRIDE
            )
            event["content_id"] = int(event.get("content_id", 0)) + (
                chunk_index * _FRAME_ID_CHUNK_STRIDE
            )
            event["_chunk_index"] = chunk_index
            events.append(event)
    return events


def _write_events(path: Path, events: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            payload = {key: value for key, value in event.items() if not key.startswith("_")}
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
    return path


def _event_text(event: dict[str, Any]) -> str:
    return str(event.get("text_normalized") or event.get("text_raw") or "").strip()


def _event_polygon(event: dict[str, Any]) -> Any | None:
    history = event.get("polygon_history") or []
    if not history:
        return None
    try:
        return as_polygon(history[-1][1])
    except (IndexError, TypeError, ValueError):
        return None


def _spatial_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_polygon = _event_polygon(left)
    right_polygon = _event_polygon(right)
    if left_polygon is None or right_polygon is None:
        return 0.0
    try:
        return polygon_iou(left_polygon, right_polygon)
    except (TypeError, ValueError):
        return 0.0


def _is_prefix(left: str, right: str) -> bool:
    left = "".join(left.split())
    right = "".join(right.split())
    return bool(left and right) and (left.startswith(right) or right.startswith(left))


def _should_merge(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    overlap_sec: float,
) -> bool:
    if left.get("_chunk_index") == right.get("_chunk_index"):
        return False
    left_start = float(left.get("start", 0.0))
    left_end = float(left.get("end", left_start))
    right_start = float(right.get("start", 0.0))
    right_end = float(right.get("end", right_start))
    # Chunked reads overlap around every boundary, so two observations of the
    # same underlying event always coexist in time.  A temporal gap means the
    # event ended before the other chunk even began reading it: that is a
    # genuine reappearance, not a boundary duplicate.  Event recall therefore
    # wins over duplicate suppression here.
    temporal_overlap = min(left_end, right_end) - max(left_start, right_start)
    if temporal_overlap < -1e-6:
        return False

    left_text = _event_text(left)
    right_text = _event_text(right)
    if not left_text or not right_text:
        return False
    text_score = text_similarity(left_text, right_text)
    spatial_score = _spatial_similarity(left, right)
    if _event_polygon(left) is not None and _event_polygon(right) is not None:
        # Identical text at a clearly different location is a distinct event,
        # no matter how much it overlaps in time.
        if spatial_score < 0.08:
            return False
    elif temporal_overlap <= 0.0:
        # Without geometry evidence, require actual coexistence.
        return False
    if text_score >= 0.92:
        return True
    return spatial_score >= 0.25 and (text_score >= 0.72 or _is_prefix(left_text, right_text))


def _merge_history(
    left: list[Any] | tuple[Any, ...],
    right: list[Any] | tuple[Any, ...],
) -> list[Any]:
    values = [*left, *right]
    values.sort(key=lambda item: float(item[0]) if item else 0.0)
    merged: list[Any] = []
    for item in values:
        if not item:
            continue
        timestamp = float(item[0])
        if merged and abs(timestamp - float(merged[-1][0])) < 1e-6:
            continue
        merged.append(item)
    return merged


def _choose_text(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_text = _event_text(left)
    right_text = _event_text(right)
    if _is_prefix(left_text, right_text):
        return right if len(right_text) >= len(left_text) else left
    left_confidence = float(left.get("confidence", 0.0))
    right_confidence = float(right.get("confidence", 0.0))
    return right if right_confidence > left_confidence else left


def _merge_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    chosen = _choose_text(left, right)
    merged = dict(chosen)
    merged["start"] = min(float(left.get("start", 0.0)), float(right.get("start", 0.0)))
    merged["end"] = max(float(left.get("end", 0.0)), float(right.get("end", 0.0)))
    merged["confidence"] = max(
        float(left.get("confidence", 0.0)), float(right.get("confidence", 0.0))
    )
    merged["polygon_history"] = _merge_history(
        left.get("polygon_history", []), right.get("polygon_history", [])
    )
    merged["source_frame_ids"] = sorted(
        {
            int(value)
            for value in [
                *left.get("source_frame_ids", []),
                *right.get("source_frame_ids", []),
            ]
        }
    )
    merged["alternatives"] = list(
        dict.fromkeys(
            [*left.get("alternatives", []), *right.get("alternatives", [])]
        )
    )
    merged["cached"] = bool(left.get("cached")) or bool(right.get("cached"))
    merged["recognition_level"] = max(
        int(left.get("recognition_level", 1)), int(right.get("recognition_level", 1))
    )
    merged["_chunk_index"] = min(
        int(left.get("_chunk_index", 0)), int(right.get("_chunk_index", 0))
    )
    return merged


def merge_chunk_events(
    events: Iterable[dict[str, Any]],
    *,
    overlap_sec: float,
    start_sec: float,
    end_sec: float,
) -> tuple[list[dict[str, Any]], int]:
    """Sort and conservatively deduplicate events from overlapping chunks."""
    candidates = [
        event
        for event in events
        if float(event.get("end", 0.0)) >= start_sec - 1e-6
        and float(event.get("start", 0.0)) <= end_sec + 1e-6
    ]
    candidates.sort(key=lambda item: (float(item.get("start", 0.0)), int(item.get("_chunk_index", 0))))
    merged: list[dict[str, Any]] = []
    duplicate_count = 0
    for event in candidates:
        match_index: int | None = None
        for index, existing in enumerate(merged):
            if _should_merge(existing, event, overlap_sec=overlap_sec):
                match_index = index
                break
        if match_index is None:
            merged.append(dict(event))
        else:
            merged[match_index] = _merge_pair(merged[match_index], event)
            duplicate_count += 1

    merged.sort(key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))))
    for event_id, event in enumerate(merged, start=1):
        event["event_id"] = event_id
        event.pop("_chunk_index", None)
    return merged, duplicate_count


def _aggregate_profiles(
    profiles: Iterable[dict[str, Any]],
    *,
    core_duration_sec: float,
    wall_sec: float,
    chunk_count: int,
) -> dict[str, Any]:
    values = list(profiles)
    totals: dict[str, Any] = {
        "total_sec": wall_sec,
        "processed_video_sec": core_duration_sec,
        "video_realtime": core_duration_sec / max(wall_sec, 1e-9),
        "chunk_count": chunk_count,
        "work_sec_sum": sum(float(item.get("total_sec", 0.0)) for item in values),
        "frames_decoded": sum(int(item.get("frames_decoded", 0)) for item in values),
        "frames_probed": sum(int(item.get("frames_probed", 0)) for item in values),
        "motion_estimates": sum(int(item.get("motion_estimates", 0)) for item in values),
        "valid_motion_estimates": sum(
            int(item.get("valid_motion_estimates", 0)) for item in values
        ),
        "detection_requests_fast": sum(
            int(item.get("detection_requests_fast", 0)) for item in values
        ),
        "detection_requests_local": sum(
            int(item.get("detection_requests_local", 0)) for item in values
        ),
        "detection_requests_audit": sum(
            int(item.get("detection_requests_audit", 0)) for item in values
        ),
        "detection_observations": sum(
            int(item.get("detection_observations", 0)) for item in values
        ),
        "geometry_tracks_created": sum(
            int(item.get("geometry_tracks_created", 0)) for item in values
        ),
        "content_tracks_created": sum(
            int(item.get("content_tracks_created", 0)) for item in values
        ),
        "ocr_tasks": sum(int(item.get("ocr_tasks", 0)) for item in values),
        "ocr_batches": sum(int(item.get("ocr_batches", 0)) for item in values),
        "cache_hits": sum(int(item.get("cache_hits", 0)) for item in values),
        "output_events": 0,
    }
    stage_names = {
        name
        for item in values
        for name in (item.get("stage_sec") or {})
    }
    totals["stage_sec"] = {
        name: sum(float(item.get("stage_sec", {}).get(name, 0.0)) for item in values)
        for name in sorted(stage_names)
    }
    counter_names = {
        name
        for item in values
        for name in (item.get("counters") or {})
    }
    totals["counters"] = {
        name: sum(float(item.get("counters", {}).get(name, 0.0)) for item in values)
        for name in sorted(counter_names)
    }
    return totals


def run_video_ocr_chunked(
    video: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
    chunk_sec: float | None = None,
    overlap_sec: float = 4.0,
    workers: int | None = None,
    sample_fps: float | None = 1.0,
    max_width: int | None = 1280,
    thread_type: str = "AUTO",
    ocr_threads_per_worker: int | None = None,
    short_video_threshold_sec: float = DEFAULT_SHORT_VIDEO_THRESHOLD_SEC,
    exclude_regions: list[list[float]] | tuple[tuple[float, ...], ...] | None = None,
) -> OCRExecution:
    """Run automatic or explicit seekable chunks and merge their events."""
    video_path = Path(video).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    duration = _probe_duration(video_path)
    requested_start = 0.0 if start_sec is None else float(start_sec)
    requested_end = duration if end_sec is None else min(float(end_sec), duration)
    if requested_start < 0 or requested_end <= requested_start:
        raise ValueError("require 0 <= start_sec < end_sec within video duration")
    if overlap_sec < 0:
        raise ValueError("overlap_sec must be non-negative")
    requested_duration = requested_end - requested_start
    automatic_chunking = chunk_sec is None
    effective_chunk_sec = (
        choose_chunk_sec(
            requested_duration,
            short_video_threshold_sec=short_video_threshold_sec,
        )
        if automatic_chunking
        else chunk_sec
    )
    if effective_chunk_sec is None:
        target_dir = Path(output_dir).expanduser().resolve()
        execution = run_video_ocr(
            video_path,
            target_dir,
            config_path=config_path,
            start_sec=requested_start,
            end_sec=requested_end,
            sample_fps=sample_fps,
            max_width=max_width,
            thread_type=thread_type,
            exclude_regions=exclude_regions,
        )
        metadata = json.loads(execution.metadata_path.read_text(encoding="utf-8"))
        metadata["mode"] = "auto_continuous"
        metadata["chunking"] = {
            "strategy": "short_video_bypass",
            "requested_chunk_sec": None,
            "effective_chunk_sec": None,
            "short_video_threshold_sec": short_video_threshold_sec,
            "chunk_count": 1,
            "exclude_regions": exclude_regions or [],
        }
        write_run_metadata(execution.metadata_path, metadata)
        return execution
    if overlap_sec >= effective_chunk_sec / 2.0:
        raise ValueError("overlap_sec must be less than half of the effective chunk width")
    chunk_list = make_chunks(
        requested_end,
        chunk_sec=effective_chunk_sec,
        overlap_sec=overlap_sec,
        start_sec=requested_start,
    )
    worker_count = workers or min(4, len(chunk_list))
    if worker_count <= 0 or worker_count > 8:
        raise ValueError("workers must be between 1 and 8")
    worker_count = min(worker_count, len(chunk_list))

    config = EngineConfig.load(config_path) if config_path else EngineConfig()
    effective_ocr_threads = (
        ocr_threads_per_worker
        if ocr_threads_per_worker is not None
        else (min(config.ocr.intra_op_num_threads, 4) if worker_count > 1 else config.ocr.intra_op_num_threads)
    )
    if effective_ocr_threads <= 0:
        raise ValueError("ocr_threads_per_worker must be positive")
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = target_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    thread_state = threading.local()

    def get_runtime() -> RapidOCRRuntime:
        runtime = getattr(thread_state, "runtime", None)
        if runtime is None:
            runtime = RapidOCRRuntime(
                params={
                    "EngineConfig.onnxruntime.intra_op_num_threads": effective_ocr_threads,
                    "EngineConfig.onnxruntime.inter_op_num_threads": config.ocr.inter_op_num_threads,
                }
            )
            thread_state.runtime = runtime
        return runtime

    def execute_chunk(chunk: OCRChunk) -> OCRExecution:
        return run_video_ocr(
            video_path,
            parts_dir / f"part-{chunk.index:04d}",
            config_path=config_path,
            start_sec=chunk.read_start_sec,
            end_sec=chunk.read_end_sec,
            sample_fps=sample_fps,
            max_width=max_width,
            thread_type=thread_type,
            frame_id_offset=chunk.index * _FRAME_ID_CHUNK_STRIDE,
            runtime=get_runtime(),
            intra_op_num_threads=effective_ocr_threads,
            inter_op_num_threads=config.ocr.inter_op_num_threads,
            exclude_regions=exclude_regions,
        )

    started = time.perf_counter()
    executions: dict[int, OCRExecution] = {}
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="ocr-chunk") as pool:
        futures = {pool.submit(execute_chunk, chunk): chunk for chunk in chunk_list}
        try:
            for future in as_completed(futures):
                chunk = futures[future]
                executions[chunk.index] = future.result()
        except Exception:
            for future in futures:
                future.cancel()
            raise

    all_events: list[dict[str, Any]] = []
    chunk_details: list[dict[str, Any]] = []
    for chunk in chunk_list:
        execution = executions[chunk.index]
        all_events.extend(_read_events(execution.events_path, chunk_index=chunk.index))
        chunk_details.append(
            {
                "index": chunk.index,
                "core_start_sec": chunk.core_start_sec,
                "core_end_sec": chunk.core_end_sec,
                "read_start_sec": chunk.read_start_sec,
                "read_end_sec": chunk.read_end_sec,
                "events": str(execution.events_path),
                "metadata": str(execution.metadata_path),
                "event_count": execution.event_count,
                "profile": execution.profile,
            }
        )

    merged_events, duplicate_count = merge_chunk_events(
        all_events,
        overlap_sec=overlap_sec,
        start_sec=requested_start,
        end_sec=requested_end,
    )
    events_path = _write_events(target_dir / "events.jsonl", merged_events)
    wall_sec = time.perf_counter() - started
    profile = _aggregate_profiles(
        [execution.profile for execution in executions.values()],
        core_duration_sec=requested_end - requested_start,
        wall_sec=wall_sec,
        chunk_count=len(chunk_list),
    )
    profile["output_events"] = len(merged_events)
    metadata = {
        "mode": "chunked",
        "input": {
            "video": str(video_path),
            "config": str(Path(config_path).expanduser().resolve()) if config_path else None,
            "duration_sec": duration,
            "start_sec": requested_start,
            "end_sec": requested_end,
            "sample_fps": sample_fps,
            "max_width": max_width,
            "thread_type": thread_type,
            "exclude_regions": exclude_regions or [],
        },
        "chunking": {
            "strategy": "automatic" if automatic_chunking else "explicit",
            "requested_chunk_sec": chunk_sec,
            "effective_chunk_sec": effective_chunk_sec,
            "overlap_sec": overlap_sec,
            "chunk_count": len(chunk_list),
            "workers": worker_count,
            "ocr_threads_per_worker": effective_ocr_threads,
            "short_video_threshold_sec": short_video_threshold_sec,
            "parts_dir": str(parts_dir),
            "parts": chunk_details,
        },
        "merge": {
            "input_event_count": len(all_events),
            "output_event_count": len(merged_events),
            "duplicate_count": duplicate_count,
        },
        "profile": profile,
        "artifacts": {
            "events": str(events_path),
            "metadata": str(target_dir / "run.json"),
        },
    }
    metadata_path = write_run_metadata(target_dir / "run.json", metadata)
    return OCRExecution(
        events_path=events_path,
        metadata_path=metadata_path,
        event_count=len(merged_events),
        profile=profile,
    )
