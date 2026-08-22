"""Video frame sources. Network acquisition intentionally does not live here."""

from __future__ import annotations

import itertools
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


def frame_timestamp(
    frame: Any,
    *,
    time_base: float | None,
    average_rate: float,
    decoded_frame_index: int,
    fallback_origin: float = 0.0,
) -> float:
    """Return the presentation timestamp of one decoded frame.

    When the container provides no timing information, the fallback derives
    the timestamp from the segment's temporal origin plus the local decode
    position and source frame rate.  ``frame_id_offset`` is a chunk output
    namespace and must never influence the time axis; the decode index
    advances for every decoded frame (including sampled-out ones), so
    sampling cannot stall the clock.  Without any timing information and an
    unknown frame rate there is no provable timeline, so this fails instead
    of fabricating absurd timestamps.
    """
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is not None and time_base:
        return float(frame.pts * time_base)
    if average_rate <= 0:
        raise ValueError(
            "cannot derive frame timestamps: frame carries no time/pts and the"
            " source frame rate is unknown"
        )
    return fallback_origin + decoded_frame_index / average_rate


def iter_cadence_provable_decode(
    decode: Any,
    *,
    decoder_skip_available: bool,
) -> Iterator[Any]:
    """Yield frames from a decode pass whose cadence the fallback can trust.

    ``decode(with_decoder_skip)`` must return a fresh frame iterator.  The
    timestamp fallback counts decoded frames, so it is only valid while the
    iterator still follows the full source frame cadence.  When a
    decoder-level skip (NONREF/BIDIR) is active but the first frame carries
    no authoritative timestamp, that pass has already dropped media frames:
    restart at full cadence instead of fabricating timestamps from a broken
    ordinal.
    """
    raw_frames = decode(decoder_skip_available)
    first = next(raw_frames, None)
    if (
        first is not None
        and decoder_skip_available
        and first.time is None
        and first.pts is None
    ):
        raw_frames = decode(False)
        first = next(raw_frames, None)
    if first is not None:
        yield first
    yield from raw_frames


def iter_sampled_packets(
    raw_frames: Iterable[Any],
    *,
    time_base: float | None,
    average_rate: float,
    sample_fps: float | None,
    max_width: int | None,
    start_sec: float | None,
    end_sec: float | None,
    frame_id_offset: int,
    fallback_origin: float = 0.0,
) -> Iterator[FramePacket]:
    """Turn decoded frames into sampled :class:`FramePacket` instances."""
    next_sample_timestamp: float | None = None
    emitted_frame_id = 0
    decoded_frame_index = 0
    for raw_frame in raw_frames:
        timestamp = frame_timestamp(
            raw_frame,
            time_base=time_base,
            average_rate=average_rate,
            decoded_frame_index=decoded_frame_index,
            fallback_origin=fallback_origin,
        )
        decoded_frame_index += 1
        if start_sec is not None and timestamp < start_sec:
            continue
        if end_sec is not None and timestamp > end_sec:
            break
        if sample_fps is not None:
            interval = 1.0 / sample_fps
            if (
                next_sample_timestamp is not None
                and timestamp + 1e-9 < next_sample_timestamp
            ):
                continue
            next_sample_timestamp = timestamp + interval
        image = raw_frame.to_ndarray(format="bgr24")
        if max_width is not None and image.shape[1] > max_width:
            scale = max_width / image.shape[1]
            image = cv2.resize(
                image,
                (max_width, max(2, round(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        yield FramePacket(
            frame_id=frame_id_offset + emitted_frame_id,
            timestamp=timestamp,
            image=image,
            luma=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        )
        emitted_frame_id += 1


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
            skip_mode: str | None = None
            if self.sample_fps is not None:
                source_fps = float(stream.average_rate or 0.0)
                if source_fps > self.sample_fps * 2.0:
                    # High-FPS screen recordings commonly contain a large
                    # non-reference frame population. At low OCR sample rates,
                    # keeping reference frames retains timestamp coverage while
                    # avoiding most of the decode work before OCR.
                    skip_mode = "NONREF" if self.sample_fps <= 1.5 else "BIDIR"
            average_rate = float(stream.average_rate or 0.0)
            time_base = float(stream.time_base) if stream.time_base else None

            def decode(with_decoder_skip: bool) -> Iterator[Any]:
                if with_decoder_skip and skip_mode is not None:
                    try:
                        stream.codec_context.skip_frame = skip_mode
                    except (AttributeError, ValueError):
                        pass
                if self.start_sec is not None and self.start_sec > 0:
                    seek_time_base = float(stream.time_base or 0.0)
                    if seek_time_base > 0:
                        # Seek to the nearest preceding keyframe. The exact
                        # start boundary is still enforced below while decoding
                        # the small keyframe pre-roll.
                        container.seek(
                            max(0, int(self.start_sec / seek_time_base)),
                            stream=stream,
                            backward=True,
                            any_frame=False,
                        )
                return container.decode(stream)

            raw_frames = iter_cadence_provable_decode(
                decode,
                decoder_skip_available=skip_mode is not None,
            )
            fallback_origin = 0.0
            first = next(raw_frames, None)
            if first is not None:
                if first.time is None and first.pts is None:
                    if self.start_sec is not None and self.start_sec > 0:
                        # A keyframe seek lands at or before start_sec, and the
                        # fallback clock cannot be anchored to the media
                        # timeline without any timing. Restart from the
                        # beginning so the fallback timeline stays provably
                        # correct (start_sec filtering still applies).
                        container.seek(0, stream=stream, backward=True, any_frame=False)
                        raw_frames = itertools.chain(container.decode(stream), ())
                    else:
                        raw_frames = itertools.chain([first], raw_frames)
                else:
                    fallback_origin = frame_timestamp(
                        first,
                        time_base=time_base,
                        average_rate=average_rate,
                        decoded_frame_index=0,
                    )
                    raw_frames = itertools.chain([first], raw_frames)
            yield from iter_sampled_packets(
                raw_frames,
                time_base=time_base,
                average_rate=average_rate,
                sample_fps=self.sample_fps,
                max_width=self.max_width,
                start_sec=self.start_sec,
                end_sec=self.end_sec,
                frame_id_offset=self.frame_id_offset,
                fallback_origin=fallback_origin,
            )
