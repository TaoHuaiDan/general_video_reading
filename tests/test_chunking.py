from __future__ import annotations

from temporal_ocr.chunking import choose_chunk_sec, make_chunks, merge_chunk_events


def test_choose_chunk_sec_bypasses_short_video_and_balances_long_video() -> None:
    assert choose_chunk_sec(240.0) is None
    selected = choose_chunk_sec(717.899)
    assert selected is not None
    assert 179.0 < selected < 180.0
    assert choose_chunk_sec(360.0) == 180.0


def _event(
    *,
    chunk_index: int,
    start: float,
    end: float,
    text: str,
    confidence: float = 0.9,
) -> dict:
    return {
        "event_id": 1,
        "geometry_id": chunk_index * 10_000_000 + 1,
        "content_id": chunk_index * 10_000_000 + 1,
        "start": start,
        "end": end,
        "text_raw": text,
        "text_normalized": text,
        "confidence": confidence,
        "polygon_history": [[start, [[0, 0], [100, 0], [100, 40], [0, 40]]]],
        "source_frame_ids": [chunk_index * 10_000_000 + 1],
        "alternatives": [],
        "cached": False,
        "recognition_level": 1,
        "_chunk_index": chunk_index,
    }


def test_make_chunks_has_core_ranges_and_overlapping_read_windows() -> None:
    chunks = make_chunks(300.0, chunk_sec=120.0, overlap_sec=4.0)
    assert [(item.core_start_sec, item.core_end_sec) for item in chunks] == [
        (0.0, 120.0),
        (120.0, 240.0),
        (240.0, 300.0),
    ]
    assert chunks[0].read_start_sec == 0.0
    assert chunks[0].read_end_sec == 124.0
    assert chunks[1].read_start_sec == 116.0
    assert chunks[1].read_end_sec == 244.0
    assert chunks[-1].read_end_sec == 300.0


def test_merge_chunk_events_removes_boundary_duplicate() -> None:
    merged, duplicate_count = merge_chunk_events(
        [
            _event(chunk_index=0, start=118.0, end=123.0, text="继续前进"),
            _event(chunk_index=1, start=119.0, end=124.0, text="继续前进", confidence=0.95),
        ],
        overlap_sec=4.0,
        start_sec=0.0,
        end_sec=240.0,
    )
    assert duplicate_count == 1
    assert len(merged) == 1
    assert merged[0]["event_id"] == 1
    assert merged[0]["start"] == 118.0
    assert merged[0]["end"] == 124.0
    assert merged[0]["confidence"] == 0.95
