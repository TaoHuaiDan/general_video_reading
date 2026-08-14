"""Local MCP adapter for the independent temporal OCR engine.

The server is intentionally local-file-only.  Bilibili acquisition and subtitle
APIs stay in their own MCP; this adapter starts OCR jobs over paths that already
exist on disk and returns compact artifact references instead of dumping every
OCR event into the conversation.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from mcp.server.fastmcp import FastMCP

from temporal_ocr.chunking import run_video_ocr_chunked
from temporal_ocr.runner import OCRExecution, run_video_ocr


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_video(video: str | Path) -> Path:
    path = Path(video).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"video file does not exist: {path}")
    return path


def _compact_execution(execution: OCRExecution) -> dict[str, Any]:
    profile = execution.profile
    return {
        "events": str(execution.events_path),
        "metadata": str(execution.metadata_path),
        "event_count": execution.event_count,
        "processed_video_sec": profile.get("processed_video_sec", 0.0),
        "wall_sec": profile.get("total_sec", 0.0),
        "video_realtime": profile.get("video_realtime", 0.0),
    }


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


@dataclass(slots=True)
class _Job:
    job_id: str
    kind: str
    output_dir: Path
    created_at: str
    status: str = "queued"
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    future: Future[None] | None = field(default=None, repr=False)


class _JobStore:
    """Small in-process job registry with one bounded OCR worker by default."""

    def __init__(self) -> None:
        configured_root = os.environ.get("TEMPORAL_OCR_MCP_OUTPUT_ROOT")
        self.output_root = (
            Path(configured_root).expanduser().resolve()
            if configured_root
            else (Path.cwd() / "artifacts" / "ocr-mcp").resolve()
        )
        self.output_root.mkdir(parents=True, exist_ok=True)
        configured_workers = int(os.environ.get("TEMPORAL_OCR_MCP_WORKERS", "1"))
        self._executor = ThreadPoolExecutor(max_workers=max(1, min(configured_workers, 4)))
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.RLock()

    def allocate_output_dir(self, requested: str | None, job_id: str) -> Path:
        target = Path(requested).expanduser().resolve() if requested else self.output_root / job_id
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(f"output directory is not empty: {target}")
        target.mkdir(parents=True, exist_ok=True)
        return target

    def submit(
        self,
        kind: str,
        output_dir: Path,
        task: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        job_id = f"{kind}-{uuid.uuid4().hex[:12]}"
        return self.submit_with_id(job_id, kind, output_dir, task)

    def submit_with_id(
        self,
        job_id: str,
        kind: str,
        output_dir: Path,
        task: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        job = _Job(
            job_id=job_id,
            kind=kind,
            output_dir=output_dir,
            created_at=_utc_now(),
        )
        with self._lock:
            self._jobs[job_id] = job
        future = self._executor.submit(self._run, job_id, task)
        with self._lock:
            job.future = future
        return self.public(job_id)

    def _run(self, job_id: str, task: Callable[[], dict[str, Any]]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.started_at = _utc_now()
        try:
            result = task()
        except Exception as exc:  # noqa: BLE001 - preserve the task failure for MCP clients
            with self._lock:
                job.status = "failed"
                job.finished_at = _utc_now()
                job.error = f"{type(exc).__name__}: {exc}"
            return
        with self._lock:
            job.status = "completed"
            job.finished_at = _utc_now()
            job.result = result

    def get(self, job_id: str) -> _Job:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError as exc:
                raise KeyError(f"unknown OCR job: {job_id}") from exc

    def public(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        payload: dict[str, Any] = {
            "job_id": job.job_id,
            "kind": job.kind,
            "status": job.status,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "output_dir": str(job.output_dir),
        }
        if job.started_at and not job.finished_at:
            payload["elapsed_sec"] = max(
                0.0,
                time.time() - datetime.fromisoformat(job.started_at).timestamp(),
            )
        if job.error:
            payload["error"] = job.error
        if job.result:
            payload["result"] = self._public_result(job.result)
        return payload

    @staticmethod
    def _public_result(result: dict[str, Any]) -> dict[str, Any]:
        if "runs" in result:
            return {
                "runs": result["runs"],
                "output_dir": result.get("output_dir"),
            }
        return {
            key: result[key]
            for key in (
                "events",
                "metadata",
                "event_count",
                "processed_video_sec",
                "wall_sec",
                "video_realtime",
            )
            if key in result
        }

    def list_completed(self, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda item: item.created_at, reverse=True)
        return [self.public(job.job_id) for job in jobs[: max(1, min(limit, 100))]]

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


mcp = FastMCP("temporal-ocr")
_JOBS = _JobStore()
atexit.register(_JOBS.shutdown)


@mcp.tool()
def inspect_video(video: str) -> dict[str, Any]:
    """Inspect a local video without running OCR."""
    path = _validate_video(video)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
    finally:
        capture.release()
    codec = "".join(chr((fourcc_value >> shift) & 0xFF) for shift in (0, 8, 16, 24))
    return {
        "video": str(path),
        "size_bytes": path.stat().st_size,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": frame_count / fps if fps > 0 else None,
        "codec": codec.strip("\x00 "),
    }


def _submit_ocr(
    *,
    kind: str,
    video: str,
    output_dir: str | None,
    config_path: str | None,
    start_sec: float | None,
    end_sec: float | None,
    sample_fps: float | None,
    max_width: int | None,
    thread_type: str,
) -> dict[str, Any]:
    video_path = _validate_video(video)
    job_id = f"{kind}-{uuid.uuid4().hex[:12]}"
    target = _JOBS.allocate_output_dir(output_dir, job_id)

    def task() -> dict[str, Any]:
        execution = run_video_ocr(
            video_path,
            target,
            config_path=config_path,
            start_sec=start_sec,
            end_sec=end_sec,
            sample_fps=sample_fps,
            max_width=max_width,
            thread_type=thread_type,
        )
        return _compact_execution(execution)

    return _JOBS.submit_with_id(job_id, kind, target, task)


@mcp.tool()
def ocr_video(
    video: str,
    output_dir: str | None = None,
    config_path: str | None = None,
    end_sec: float | None = None,
    sample_fps: float | None = 1.0,
    max_width: int | None = 1280,
    thread_type: str = "AUTO",
) -> dict[str, Any]:
    """Start asynchronous full-video OCR over a local video file."""
    return _submit_ocr(
        kind="ocr",
        video=video,
        output_dir=output_dir,
        config_path=config_path,
        start_sec=None,
        end_sec=end_sec,
        sample_fps=sample_fps,
        max_width=max_width,
        thread_type=thread_type,
    )


@mcp.tool()
def ocr_video_chunked(
    video: str,
    output_dir: str | None = None,
    config_path: str | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
    chunk_sec: float = 120.0,
    overlap_sec: float = 4.0,
    workers: int | None = None,
    sample_fps: float | None = 1.0,
    max_width: int | None = 1280,
    thread_type: str = "AUTO",
    ocr_threads_per_worker: int | None = None,
) -> dict[str, Any]:
    """Start parallel overlapping-chunk OCR over a local video file."""
    video_path = _validate_video(video)
    if start_sec is not None and start_sec < 0:
        raise ValueError("start_sec must be non-negative")
    if end_sec is not None and end_sec <= 0:
        raise ValueError("end_sec must be positive")
    if chunk_sec <= 0:
        raise ValueError("chunk_sec must be positive")
    if overlap_sec < 0 or overlap_sec >= chunk_sec / 2.0:
        raise ValueError("overlap_sec must be >= 0 and less than half of chunk_sec")
    if workers is not None and not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    if ocr_threads_per_worker is not None and ocr_threads_per_worker <= 0:
        raise ValueError("ocr_threads_per_worker must be positive")
    job_id = f"ocr-chunked-{uuid.uuid4().hex[:12]}"
    target = _JOBS.allocate_output_dir(output_dir, job_id)

    def task() -> dict[str, Any]:
        execution = run_video_ocr_chunked(
            video_path,
            target,
            config_path=config_path,
            start_sec=start_sec,
            end_sec=end_sec,
            chunk_sec=chunk_sec,
            overlap_sec=overlap_sec,
            workers=workers,
            sample_fps=sample_fps,
            max_width=max_width,
            thread_type=thread_type,
            ocr_threads_per_worker=ocr_threads_per_worker,
        )
        return _compact_execution(execution)

    return _JOBS.submit_with_id(job_id, "ocr_chunked", target, task)


@mcp.tool()
def ocr_segment(
    video: str,
    start_sec: float,
    end_sec: float,
    output_dir: str | None = None,
    config_path: str | None = None,
    sample_fps: float | None = 1.0,
    max_width: int | None = 1280,
    thread_type: str = "AUTO",
) -> dict[str, Any]:
    """Start asynchronous OCR for a bounded video segment."""
    if start_sec < 0 or end_sec < 0 or start_sec > end_sec:
        raise ValueError("require 0 <= start_sec <= end_sec")
    return _submit_ocr(
        kind="segment",
        video=video,
        output_dir=output_dir,
        config_path=config_path,
        start_sec=start_sec,
        end_sec=end_sec,
        sample_fps=sample_fps,
        max_width=max_width,
        thread_type=thread_type,
    )


@mcp.tool()
def benchmark_ocr(
    video: str,
    config_paths: list[str] | None = None,
    output_dir: str | None = None,
    end_sec: float | None = None,
    sample_fps: float | None = 1.0,
    max_width: int | None = 1280,
    thread_type: str = "AUTO",
) -> dict[str, Any]:
    """Start an asynchronous A/B benchmark over several config files."""
    video_path = _validate_video(video)
    paths = config_paths or [None]
    if len(paths) > 8:
        raise ValueError("benchmark_ocr accepts at most 8 config paths")
    job_id = f"benchmark-{uuid.uuid4().hex[:12]}"
    target = _JOBS.allocate_output_dir(output_dir, job_id)

    def task() -> dict[str, Any]:
        runs: list[dict[str, Any]] = []
        for index, config_path in enumerate(paths):
            label = Path(config_path).stem if config_path else "default"
            run_dir = target / f"{index:02d}-{label}"
            execution = run_video_ocr(
                video_path,
                run_dir,
                config_path=config_path,
                end_sec=end_sec,
                sample_fps=sample_fps,
                max_width=max_width,
                thread_type=thread_type,
            )
            runs.append({"name": label, **_compact_execution(execution)})
        return {"output_dir": str(target), "runs": runs}

    return _JOBS.submit_with_id(job_id, "benchmark", target, task)


@mcp.tool()
def get_run_status(job_id: str) -> dict[str, Any]:
    """Return compact status for an OCR or benchmark job."""
    return _JOBS.public(job_id)


@mcp.tool()
def get_run_result(
    job_id: str,
    include_events: bool = False,
    max_events: int = 100,
    event_offset: int = 0,
) -> dict[str, Any]:
    """Read a completed run's profile and optionally a bounded event preview."""
    job = _JOBS.get(job_id)
    if job.status != "completed" or not job.result:
        return _JOBS.public(job_id)
    result = dict(job.result)
    if "runs" in result:
        return {
            "job_id": job_id,
            "status": job.status,
            "output_dir": result.get("output_dir"),
            "runs": result["runs"],
        }
    metadata_path = Path(str(result["metadata"]))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    profile = metadata.get("profile", {})
    payload: dict[str, Any] = {
        "job_id": job_id,
        "status": job.status,
        "input": metadata.get("input", {}),
        "profile": profile,
        "artifacts": metadata.get(
            "artifacts",
            {"events": result.get("events"), "metadata": result.get("metadata")},
        ),
        "event_count": result.get("event_count", profile.get("output_events", 0)),
    }
    if include_events:
        if max_events < 1 or max_events > 1000:
            raise ValueError("max_events must be between 1 and 1000")
        if event_offset < 0:
            raise ValueError("event_offset must be non-negative")
        event_path = Path(str(result["events"]))
        rows = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        payload["events"] = rows[event_offset : event_offset + max_events]
        payload["events_offset"] = event_offset
        payload["events_truncated"] = event_offset + max_events < len(rows)
    return payload


@mcp.tool()
def list_runs(limit: int = 20) -> dict[str, Any]:
    """List jobs known by this server process."""
    return {"output_root": str(_JOBS.output_root), "runs": _JOBS.list_completed(limit)}


@mcp.tool()
def cleanup_run(job_id: str, confirm: bool = False) -> dict[str, Any]:
    """Delete one completed run directory after explicit confirmation."""
    job = _JOBS.get(job_id)
    if not confirm:
        return {
            "job_id": job_id,
            "requires_confirmation": True,
            "output_dir": str(job.output_dir),
        }
    if job.status in {"queued", "running"}:
        raise RuntimeError("cannot clean up a running OCR job")
    target = job.output_dir.resolve()
    if not _within(target, _JOBS.output_root):
        raise PermissionError("cleanup is limited to the MCP output root")
    if target.exists():
        shutil.rmtree(target)
    return {"job_id": job_id, "removed": str(target)}


def main() -> None:
    """Run the MCP server over stdio, as expected by desktop MCP clients."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
