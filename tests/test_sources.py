"""Regression tests for the PyAV timestamp fallback and frame-id namespace."""

from __future__ import annotations

import numpy as np
import pytest

from temporal_ocr.sources import iter_sampled_packets


class _FakeFrame:
    def __init__(self, *, time: float | None = None, pts: int | None = None) -> None:
        self.time = time
        self.pts = pts

    def to_ndarray(self, format: str) -> np.ndarray:
        return np.zeros((4, 6, 3), dtype=np.uint8)


def test_fallback_timestamp_ignores_frame_id_offset() -> None:
    frames = [_FakeFrame() for _ in range(3)]

    packets = list(
        iter_sampled_packets(
            frames,
            time_base=None,
            average_rate=30.0,
            sample_fps=None,
            max_width=None,
            start_sec=None,
            end_sec=None,
            frame_id_offset=10_000_000,
        )
    )

    assert [packet.timestamp for packet in packets] == [0.0, 1.0 / 30.0, 2.0 / 30.0]
    # The chunk namespace only affects frame ids, never timestamps.
    assert [packet.frame_id for packet in packets] == [
        10_000_000,
        10_000_001,
        10_000_002,
    ]


def test_fallback_timeline_advances_across_sampling_skips() -> None:
    # 90 decoded frames at 30 fps sampled at 1 fps: skipped frames must keep
    # the fallback timeline moving, otherwise sampling stalls forever.
    frames = [_FakeFrame() for _ in range(90)]

    packets = list(
        iter_sampled_packets(
            frames,
            time_base=None,
            average_rate=30.0,
            sample_fps=1.0,
            max_width=None,
            start_sec=None,
            end_sec=None,
            frame_id_offset=0,
        )
    )

    assert [packet.timestamp for packet in packets[:3]] == [0.0, 1.0, 2.0]
    assert len(packets) == 3
    # Emitted frame ids stay sequential within the chunk namespace.
    assert [packet.frame_id for packet in packets] == [0, 1, 2]


def test_explicit_frame_time_and_pts_take_precedence() -> None:
    frames = [_FakeFrame(time=12.5), _FakeFrame(pts=250)]

    packets = list(
        iter_sampled_packets(
            frames,
            time_base=0.04,
            average_rate=30.0,
            sample_fps=None,
            max_width=None,
            start_sec=None,
            end_sec=None,
            frame_id_offset=7,
        )
    )

    assert [packet.timestamp for packet in packets] == [12.5, 10.0]


def test_fallback_timeline_anchors_at_segment_origin() -> None:
    # After a keyframe seek (chunk read_start_sec=120) the decode index starts
    # at zero locally; the fallback must anchor at the segment origin instead
    # of restarting the media timeline at t=0.
    frames = [_FakeFrame() for _ in range(3)]

    packets = list(
        iter_sampled_packets(
            frames,
            time_base=None,
            average_rate=30.0,
            sample_fps=None,
            max_width=None,
            start_sec=120.0,
            end_sec=None,
            frame_id_offset=10_000_000,
            fallback_origin=120.0,
        )
    )

    assert packets[0].timestamp == 120.0
    assert packets[1].timestamp == 120.0 + 1.0 / 30.0
    assert packets[2].timestamp == 120.0 + 2.0 / 30.0
    # The chunk namespace still only affects frame ids.
    assert packets[0].frame_id == 10_000_000


def test_frame_timestamp_fallback_uses_segment_origin() -> None:
    from temporal_ocr.sources import frame_timestamp

    frame = _FakeFrame()

    assert (
        frame_timestamp(
            frame,
            time_base=None,
            average_rate=30.0,
            decoded_frame_index=2,
            fallback_origin=120.0,
        )
        == 120.0 + 2.0 / 30.0
    )


def test_missing_timing_and_unknown_rate_fails_instead_of_absurd_timeline() -> None:
    import pytest

    from temporal_ocr.sources import frame_timestamp

    with pytest.raises(ValueError):
        frame_timestamp(
            _FakeFrame(),
            time_base=None,
            average_rate=0.0,
            decoded_frame_index=0,
            fallback_origin=0.0,
        )


def test_decoder_skip_is_disabled_when_fallback_cadence_is_required() -> None:
    # Media cadence is 30 fps; the decoder-level NONREF filter only hands
    # frames 0, 3, 6, 9 downstream, and those frames carry no time/pts.
    # The restart must really disable decoder-level skipping AND rewind to
    # the media start: continuing from consumed decoder state would drop
    # frame 0 from the "full cadence" pass.
    from temporal_ocr.sources import resolve_decode_pass

    skipped_frames = [_FakeFrame() for _ in range(4)]
    full_frames = [_FakeFrame() for _ in range(12)]
    decode_calls: list[tuple[bool, bool]] = []

    def decode(with_decoder_skip: bool, from_zero: bool):
        decode_calls.append((with_decoder_skip, from_zero))
        return iter(list(skipped_frames if with_decoder_skip else full_frames))

    result = resolve_decode_pass(
        decode, decoder_skip_available=True, seek_to_start=False
    )

    assert decode_calls == [(True, False), (False, True)]
    assert result.ordinal_fallback_allowed is True
    assert len(list(result.frames)) == 12


def test_decoder_skip_kept_only_while_every_frame_is_timed() -> None:
    # With authoritative timing on the skip path the optimization stays on,
    # but ordinal fallback is forbidden for any later untimed frame.
    from temporal_ocr.sources import resolve_decode_pass

    timed_frames = [_FakeFrame(time=float(index)) for index in range(4)]
    decode_calls: list[tuple[bool, bool]] = []

    def decode(with_decoder_skip: bool, from_zero: bool):
        decode_calls.append((with_decoder_skip, from_zero))
        return iter(list(timed_frames))

    result = resolve_decode_pass(
        decode, decoder_skip_available=True, seek_to_start=False
    )

    assert decode_calls == [(True, False)]
    assert result.ordinal_fallback_allowed is False
    assert len(list(result.frames)) == 4


def test_full_cadence_decode_is_used_when_no_skip_is_available() -> None:
    from temporal_ocr.sources import resolve_decode_pass

    frames = [_FakeFrame() for _ in range(5)]
    decode_calls: list[tuple[bool, bool]] = []

    def decode(with_decoder_skip: bool, from_zero: bool):
        decode_calls.append((with_decoder_skip, from_zero))
        return iter(list(frames))

    result = resolve_decode_pass(
        decode, decoder_skip_available=False, seek_to_start=False
    )

    assert decode_calls == [(False, False)]
    assert result.ordinal_fallback_allowed is True
    assert len(list(result.frames)) == 5


def test_untimed_seek_restart_rewinds_to_media_start_without_skip() -> None:
    from temporal_ocr.sources import resolve_decode_pass

    frames = [_FakeFrame() for _ in range(6)]
    decode_calls: list[tuple[bool, bool]] = []

    def decode(with_decoder_skip: bool, from_zero: bool):
        decode_calls.append((with_decoder_skip, from_zero))
        return iter(list(frames))

    result = resolve_decode_pass(
        decode, decoder_skip_available=False, seek_to_start=True
    )

    assert decode_calls == [(False, False), (False, True)]
    assert result.ordinal_fallback_allowed is True
    assert len(list(result.frames)) == 6


class _FakeCodecContext:
    def __init__(self) -> None:
        self.skip_frame: str | None = None


class _FakeStream:
    def __init__(self, frames: list[_FakeFrame], rate: float = 30.0) -> None:
        self.frames = frames
        self.average_rate = rate
        self.time_base = None
        self.type = "video"
        self.codec_context = _FakeCodecContext()
        self.position: int = 0


class _FakeContainer:
    """Decoder stand-in: NONREF keeps every third source frame."""

    def __init__(self, stream: _FakeStream) -> None:
        self.stream = stream
        self.streams = [stream]
        self.seeks: list[float] = []

    def seek(self, timestamp, stream: _FakeStream | None = None, backward=False, any_frame=False) -> None:
        self.seeks.append(float(timestamp))
        if stream is not None:
            stream.position = 0

    def decode(self, stream: _FakeStream):
        step = 3 if stream.codec_context.skip_frame == "NONREF" else 1
        collected = [
            frame
            for index in range(stream.position, len(stream.frames))
            if index % step == 0
            for frame in [stream.frames[index]]
        ]
        stream.position = len(stream.frames)
        return iter(collected)


class _FakeContext:
    def __init__(self, container: _FakeContainer) -> None:
        self.container = container

    def __enter__(self):
        return self.container

    def __exit__(self, *args):
        return False


class _FakeAvModule:
    def __init__(self, container: _FakeContainer) -> None:
        self._container = container

    def open(self, path):
        return _FakeContext(self._container)


def test_pyav_source_restarts_full_cadence_with_skip_cleared(
    tmp_path, monkeypatch
) -> None:
    import sys

    from temporal_ocr.sources import PyAVFrameSource

    dummy = tmp_path / "video.mp4"
    dummy.write_bytes(b"fake")
    # 90 frames @ 30 fps = 3 s of media time.
    stream = _FakeStream([_FakeFrame() for _ in range(90)])
    container = _FakeContainer(stream)
    monkeypatch.setitem(sys.modules, "av", _FakeAvModule(container))

    source = PyAVFrameSource(dummy, sample_fps=1.0)
    packets = list(source)

    # Second pass really cleared the decoder-level skip and rewound.
    assert stream.codec_context.skip_frame == "DEFAULT"
    assert container.seeks == [0.0]
    # The ordinal clock maps onto true media time: samples land exactly at
    # 0 s / 1 s / 2 s. A filtered-cadence ordinal timeline would cap below
    # 1 s and emit a single packet.
    assert [packet.timestamp for packet in packets] == [0.0, 1.0, 2.0]
    # Emitted frame ids stay sequential within the chunk namespace.
    assert [packet.frame_id for packet in packets] == [0, 1, 2]


def test_pyav_source_fails_when_timed_skip_pass_hides_untimed_frames(
    tmp_path, monkeypatch
) -> None:
    import sys

    from temporal_ocr.sources import PyAVFrameSource

    dummy = tmp_path / "video.mp4"
    dummy.write_bytes(b"fake")
    frames = [_FakeFrame(time=0.0)] + [_FakeFrame() for _ in range(8)]
    stream = _FakeStream(frames)
    container = _FakeContainer(stream)
    monkeypatch.setitem(sys.modules, "av", _FakeAvModule(container))

    source = PyAVFrameSource(dummy, sample_fps=1.0)

    with pytest.raises(ValueError):
        list(source)
