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
    x: float = 0.0,
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
        "polygon_history": [
            [start, [[x, 0], [x + 100, 0], [x + 100, 40], [x, 40]]]
        ],
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


def test_merge_keeps_repeated_same_text_at_the_same_place_as_two_events() -> None:
    # "保存成功" genuinely appearing twice within seconds must survive the
    # merge even though both instances come from neighbouring chunks.
    merged, duplicate_count = merge_chunk_events(
        [
            _event(chunk_index=0, start=116.0, end=118.0, text="保存成功"),
            _event(chunk_index=1, start=120.0, end=124.0, text="保存成功"),
        ],
        overlap_sec=4.0,
        start_sec=0.0,
        end_sec=240.0,
    )

    assert duplicate_count == 0
    assert len(merged) == 2


def test_merge_keeps_simultaneous_same_text_at_different_places() -> None:
    # Left "暂停" and right "暂停" at the same time are two real events;
    # text similarity alone must not merge them.
    merged, duplicate_count = merge_chunk_events(
        [
            _event(chunk_index=0, start=118.0, end=123.0, text="暂停", x=0.0),
            _event(chunk_index=1, start=119.0, end=124.0, text="暂停", x=300.0),
        ],
        overlap_sec=4.0,
        start_sec=0.0,
        end_sec=240.0,
    )

    assert duplicate_count == 0
    assert len(merged) == 2


def test_merge_still_removes_true_overlap_boundary_duplicate() -> None:
    merged, duplicate_count = merge_chunk_events(
        [
            _event(chunk_index=0, start=118.0, end=123.0, text="继续前进"),
            _event(chunk_index=1, start=119.5, end=124.0, text="继续前进", confidence=0.95),
        ],
        overlap_sec=4.0,
        start_sec=0.0,
        end_sec=240.0,
    )

    assert duplicate_count == 1
    assert len(merged) == 1
    assert merged[0]["start"] == 118.0
    assert merged[0]["end"] == 124.0


def test_merged_composite_keeps_full_chunk_provenance() -> None:
    merged, _duplicate_count = merge_chunk_events(
        [
            _event(chunk_index=0, start=118.0, end=123.0, text="继续前进"),
            _event(chunk_index=1, start=119.5, end=124.0, text="继续前进", confidence=0.95),
        ],
        overlap_sec=4.0,
        start_sec=0.0,
        end_sec=240.0,
    )

    assert merged[0]["_source_chunks"] == {0, 1}


def test_composite_event_does_not_swallow_later_event_from_same_chunk() -> None:
    first_pass, _count = merge_chunk_events(
        [
            _event(chunk_index=0, start=118.0, end=123.0, text="继续前进"),
            _event(chunk_index=1, start=119.5, end=124.0, text="继续前进", confidence=0.95),
        ],
        overlap_sec=4.0,
        start_sec=0.0,
        end_sec=240.0,
    )
    assert len(first_pass) == 1

    # A later event originating from chunk 1 shares provenance with the
    # composite; it must not be treated as a cross-chunk duplicate of it.
    late = _event(chunk_index=1, start=120.0, end=124.5, text="继续前进", confidence=0.97)
    second_pass, duplicate_count = merge_chunk_events(
        [*first_pass, late],
        overlap_sec=4.0,
        start_sec=0.0,
        end_sec=240.0,
    )

    assert duplicate_count == 0
    assert len(second_pass) == 2


def _moving_event(
    *,
    chunk_index: int,
    start: float,
    end: float,
    text: str,
    track: list[tuple[float, float]],
    confidence: float = 0.9,
) -> dict:
    """Event whose text line moves horizontally: track maps time -> left x."""
    return {
        "event_id": 1,
        "geometry_id": chunk_index * 10_000_000 + 1,
        "content_id": chunk_index * 10_000_000 + 1,
        "start": start,
        "end": end,
        "text_raw": text,
        "text_normalized": text,
        "confidence": confidence,
        "polygon_history": [
            [t, [[x, 0], [x + 40, 0], [x + 40, 40], [x, 40]]] for t, x in track
        ],
        "source_frame_ids": [chunk_index * 10_000_000 + 1],
        "alternatives": [],
        "cached": False,
        "recognition_level": 1,
        "_chunk_index": chunk_index,
    }


def test_merge_aligns_moving_text_polygons_inside_temporal_overlap() -> None:
    # Same moving line observed by two neighbouring chunks.  Each chunk's
    # final polygon differs (the line kept moving), but inside the shared
    # overlap the two trajectories coincide and must merge.
    merged, duplicate_count = merge_chunk_events(
        [
            _moving_event(
                chunk_index=0,
                start=0.0,
                end=10.0,
                text="滚动字幕",
                track=[(0.0, 0.0), (5.0, 50.0), (10.0, 100.0)],
            ),
            _moving_event(
                chunk_index=1,
                start=5.0,
                end=15.0,
                text="滚动字幕",
                confidence=0.95,
                track=[(5.0, 50.0), (10.0, 100.0), (15.0, 150.0)],
            ),
        ],
        overlap_sec=4.0,
        start_sec=0.0,
        end_sec=240.0,
    )

    assert duplicate_count == 1
    assert len(merged) == 1
    assert merged[0]["start"] == 0.0
    assert merged[0]["end"] == 15.0


def test_merge_still_rejects_same_text_at_different_aligned_positions() -> None:
    # Identical text at clearly different positions at every aligned time is
    # two distinct events even though both are "moving".
    merged, duplicate_count = merge_chunk_events(
        [
            _moving_event(
                chunk_index=0,
                start=5.0,
                end=10.0,
                text="暂停",
                track=[(5.0, 0.0), (10.0, 50.0)],
            ),
            _moving_event(
                chunk_index=1,
                start=5.0,
                end=10.0,
                text="暂停",
                track=[(5.0, 300.0), (10.0, 350.0)],
            ),
        ],
        overlap_sec=4.0,
        start_sec=0.0,
        end_sec=240.0,
    )

    assert duplicate_count == 0
    assert len(merged) == 2


def test_internal_provenance_is_not_written_to_artifacts(tmp_path) -> None:
    from temporal_ocr.chunking import _write_events

    events = [
        {
            "event_id": 1,
            "start": 0.0,
            "end": 1.0,
            "_chunk_index": 3,
            "_source_chunks": {2, 3},
            "text_normalized": "text",
        }
    ]

    path = _write_events(tmp_path / "events.jsonl", events)
    content = path.read_text(encoding="utf-8")

    assert "_chunk_index" not in content
    assert "_source_chunks" not in content
