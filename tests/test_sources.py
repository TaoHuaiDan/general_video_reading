"""Regression tests for the PyAV timestamp fallback and frame-id namespace."""

from __future__ import annotations

import numpy as np

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
