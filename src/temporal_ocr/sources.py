"""Video frame sources. Network acquisition intentionally does not live here."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2

from temporal_ocr.types import FramePacket


@dataclass(frozen=True, slots=True)
class DecodePassResult:
    """A decode pass plus whether the ordinal timestamp fallback is provable."""

    frames: Any
    ordinal_fallback_allowed: bool


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


def resolve_decode_pass(
    decode: Any,
    *,
    decoder_skip_available: bool,
    seek_to_start: bool,
) -> Any:
    """Pick a decode pass whose timeline the engine can prove.

    ``decode(with_decoder_skip, from_zero)`` must return a fresh frame
    iterator starting at the requested origin.  The timestamp fallback counts
    decoded frames, so it is only valid while the iterator follows the full
    source frame cadence:

    - A decoder-level skip (NONREF/BIDIR) drops media frames.  It may stay
      active only while every output frame carries its own authoritative
      time/pts (probed on the first frame); any later untimed frame then
      fails loudly instead of using an untrustworthy ordinal.
    - Without timing, neither a skipped stream nor a keyframe-seek origin is
      provable: restart fresh at full cadence from the true media start.
    """
    frames = decode(with_decoder_skip=decoder_skip_available, from_zero=False)
    first = next(frames, None)
    if first is None:
        return DecodePassResult(iter(()), ordinal_fallback_allowed=True)
    timed = first.time is not None or first.pts is not None
    if not timed and (decoder_skip_available or seek_to_start):
        # Untimed frames under a decoder-level skip, or after an unanchored
        # keyframe seek, cannot use the ordinal clock. Restart at full
        # cadence from the provable media start.
        return DecodePassResult(
            decode(with_decoder_skip=False, from_zero=True),
            ordinal_fallback_allowed=True,
        )
    if decoder_skip_available:
        return DecodePassResult(
            itertools.chain([first], frames),
            ordinal_fallback_allowed=False,
        )
    return DecodePassResult(itertools.chain([first], frames), ordinal_fallback_allowed=True)


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
    ordinal_fallback_allowed: bool = True,
) -> Iterator[FramePacket]:
    """Turn decoded frames into sampled :class:`FramePacket` instances."""
    next_sample_timestamp: float | None = None
    emitted_frame_id = 0
    decoded_frame_index = 0
    for raw_frame in raw_frames:
        if (
            not ordinal_fallback_allowed
            and raw_frame.time is None
            and raw_frame.pts is None
        ):
            raise ValueError(
                "decoder-level frame skipping is active and a decoded frame"
                " carries no time/pts: the timestamp fallback requires a"
                " full-cadence decode"
            )
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

            def decode(with_decoder_skip: bool, from_zero: bool) -> Iterator[Any]:
                # Always assign the target mode: a restart pass must really
                # clear a previously set NONREF/BIDIR, not keep filtering.
                target = skip_mode if with_decoder_skip and skip_mode else "DEFAULT"
                try:
                    stream.codec_context.skip_frame = target
                except (AttributeError, ValueError):
                    pass
                if from_zero:
                    container.seek(0, stream=stream, backward=True, any_frame=False)
                elif self.start_sec is not None and self.start_sec > 0:
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

            pass_result = resolve_decode_pass(
                decode,
                decoder_skip_available=skip_mode is not None,
                seek_to_start=self.start_sec is not None and self.start_sec > 0,
            )
            raw_frames = pass_result.frames
            fallback_origin = 0.0
            first = next(raw_frames, None)
            if first is not None and (first.time is not None or first.pts is not None):
                fallback_origin = frame_timestamp(
                    first,
                    time_base=time_base,
                    average_rate=average_rate,
                    decoded_frame_index=0,
                )
            if first is not None:
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
                ordinal_fallback_allowed=pass_result.ordinal_fallback_allowed,
            )
