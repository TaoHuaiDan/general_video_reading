"""Video frame sources. Network acquisition intentionally does not live here."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, cast

import cv2

from temporal_ocr.types import FramePacket


class IterableFrameSource:
    def __init__(self, frames: Iterable[FramePacket]) -> None:
        self.frames = frames

    def __iter__(self) -> Iterator[FramePacket]:
        return iter(self.frames)


class PyAVFrameSource:
    def __init__(
        self,
        path: str | Path,
        *,
        thread_type: str = "AUTO",
        sample_fps: float | None = None,
        max_width: int | None = None,
        start_sec: float | None = None,
        end_sec: float | None = None,
        frame_id_offset: int = 0,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.thread_type = thread_type.upper()
        if sample_fps is not None and sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        if max_width is not None and max_width <= 0:
            raise ValueError("max_width must be positive")
        if start_sec is not None and start_sec < 0:
            raise ValueError("start_sec must be non-negative")
        if end_sec is not None and end_sec < 0:
            raise ValueError("end_sec must be non-negative")
        if start_sec is not None and end_sec is not None and start_sec > end_sec:
            raise ValueError("start_sec must not be greater than end_sec")
        if frame_id_offset < 0:
            raise ValueError("frame_id_offset must be non-negative")
        self.sample_fps = sample_fps
        self.max_width = max_width
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.frame_id_offset = frame_id_offset

    def __iter__(self) -> Iterator[FramePacket]:
        try:
            import av
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyAV is required: pip install -e .[video]") from exc
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with av.open(str(self.path)) as container:
            stream = cast(
                Any,
                next((item for item in container.streams if item.type == "video"), None),
            )
            if stream is None:
                raise RuntimeError("input has no video stream")
            try:
                stream.thread_type = self.thread_type
            except (AttributeError, ValueError):
                pass
            if self.start_sec is not None and self.start_sec > 0:
                time_base = float(stream.time_base or 0.0)
                if time_base > 0:
                    # Seek to the nearest preceding keyframe. The exact start
                    # boundary is still enforced below while decoding the
                    # small keyframe pre-roll.
                    container.seek(
                        max(0, int(self.start_sec / time_base)),
                        stream=stream,
                        backward=True,
                        any_frame=False,
                    )
            if self.sample_fps is not None:
                source_fps = float(stream.average_rate or 0.0)
                if source_fps > self.sample_fps * 2.0:
                    # High-FPS screen recordings commonly contain a large
                    # non-reference frame population. At low OCR sample rates,
                    # keeping reference frames retains timestamp coverage while
                    # avoiding most of the decode work before OCR.
                    try:
                        stream.codec_context.skip_frame = (
                            "NONREF" if self.sample_fps <= 1.5 else "BIDIR"
                        )
                    except (AttributeError, ValueError):
                        pass
            next_sample_timestamp: float | None = None
            emitted_frame_id = 0
            for raw_frame in container.decode(stream):
                frame = cast(Any, raw_frame)
                if frame.time is not None:
                    timestamp = float(frame.time)
                elif frame.pts is not None and stream.time_base is not None:
                    timestamp = float(frame.pts * stream.time_base)
                else:
                    rate = float(stream.average_rate or 30.0)
                    timestamp = (self.frame_id_offset + emitted_frame_id) / max(rate, 1e-9)
                if self.start_sec is not None and timestamp < self.start_sec:
                    continue
                if self.end_sec is not None and timestamp > self.end_sec:
                    break
                if self.sample_fps is not None:
                    interval = 1.0 / self.sample_fps
                    if (
                        next_sample_timestamp is not None
                        and timestamp + 1e-9 < next_sample_timestamp
                    ):
                        continue
                    next_sample_timestamp = timestamp + interval
                image = frame.to_ndarray(format="bgr24")
                if self.max_width is not None and image.shape[1] > self.max_width:
                    scale = self.max_width / image.shape[1]
                    image = cv2.resize(
                        image,
                        (self.max_width, max(2, round(image.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                yield FramePacket(
                    frame_id=self.frame_id_offset + emitted_frame_id,
                    timestamp=timestamp,
                    image=image,
                    luma=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                )
                emitted_frame_id += 1
