"""Quality, completeness, duplication, throughput and latency metrics."""

from __future__ import annotations

import statistics
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

from temporal_ocr.geometry import as_polygon, polygon_iou
from temporal_ocr.types import Polygon, TextEvent


def normalize_for_metric(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        for col, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[col] + 1,
                    previous[col - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def text_similarity(left: str, right: str) -> float:
    left_norm = normalize_for_metric(left)
    right_norm = normalize_for_metric(right)
    denominator = max(len(left_norm), len(right_norm), 1)
    return max(0.0, 1.0 - edit_distance(left_norm, right_norm) / denominator)


def temporal_iou(left: TextEvent, right: TextEvent) -> float:
    intersection = max(0.0, min(left.end, right.end) - max(left.start, right.start))
    union = max(left.end, right.end) - min(left.start, right.start)
    return intersection / max(union, 1e-9)


def _nearest_polygon(history: tuple[tuple[float, Polygon], ...], timestamp: float) -> Polygon:
    return min(history, key=lambda item: abs(item[0] - timestamp))[1]


def mean_spatial_iou(left: TextEvent, right: TextEvent) -> float:
    if not left.polygon_history or not right.polygon_history:
        return 0.0
    overlap_start = max(left.start, right.start)
    overlap_end = min(left.end, right.end)
    if overlap_end < overlap_start:
        return 0.0
    timestamps = {
        timestamp
        for timestamp, _polygon in (*left.polygon_history, *right.polygon_history)
        if overlap_start <= timestamp <= overlap_end
    }
    if not timestamps:
        timestamps = {(overlap_start + overlap_end) / 2.0}
    return sum(
        polygon_iou(
            _nearest_polygon(left.polygon_history, timestamp),
            _nearest_polygon(right.polygon_history, timestamp),
        )
        for timestamp in timestamps
    ) / len(timestamps)


@dataclass(slots=True)
class EvaluationReport:
    reference_events: int
    predicted_events: int
    matched_events: int
    event_recall: float
    text_accuracy: float
    duplicate_rate: float
    mean_temporal_iou: float
    mean_spatial_iou: float
    video_realtime: float
    throughput_video_sec_per_wall_sec: float
    latency_p50_sec: float | None = None
    latency_p95_sec: float | None = None
    latency_p99_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percentile)))
    return float(ordered[index])


def _max_matching_size(pairs: list[tuple[int, int]]) -> int:
    """Return the maximum bipartite matching size via Kuhn's algorithm.

    The benchmark contract is completeness-first: greedy score-first pairing
    cannot guarantee maximum cardinality and would under-report Event Recall
    whenever two references compete for one high-scoring prediction.  Inputs
    are tiny (events per run), so a small augmenting-path matcher is enough.
    """
    adjacency: dict[int, list[int]] = {}
    for ref_index, pred_index in pairs:
        adjacency.setdefault(ref_index, []).append(pred_index)

    match_of_pred: dict[int, int] = {}

    def try_assign(ref_index: int, visited: set[int]) -> bool:
        for pred_index in adjacency.get(ref_index, ()):
            if pred_index in visited:
                continue
            visited.add(pred_index)
            if pred_index not in match_of_pred or try_assign(match_of_pred[pred_index], visited):
                match_of_pred[pred_index] = ref_index
                return True
        return False

    matched = 0
    for ref_index in sorted(adjacency):
        if try_assign(ref_index, set()):
            matched += 1
    return matched


def evaluate_events(
    reference: list[TextEvent],
    predicted: list[TextEvent],
    *,
    video_sec: float = 0.0,
    wall_sec: float = 0.0,
    latencies_sec: list[float] | None = None,
    min_text_similarity: float = 0.55,
    min_temporal_iou: float = 0.10,
    min_spatial_iou: float = 0.05,
) -> EvaluationReport:
    pairs: list[tuple[float, int, int, float, float, float]] = []
    for ref_index, ref_event in enumerate(reference):
        for pred_index, pred_event in enumerate(predicted):
            text_score = text_similarity(ref_event.text_normalized, pred_event.text_normalized)
            time_score = temporal_iou(ref_event, pred_event)
            space_score = mean_spatial_iou(ref_event, pred_event)
            if (
                text_score >= min_text_similarity
                and time_score >= min_temporal_iou
                and space_score >= min_spatial_iou
            ):
                score = 0.55 * text_score + 0.25 * time_score + 0.20 * space_score
                pairs.append((score, ref_index, pred_index, text_score, time_score, space_score))

    # First maximize the number of matched pairs, then pick higher-quality
    # pairs among maximum-cardinality solutions.  A candidate is accepted
    # only if the remaining graph can still reach the target cardinality.
    target_matches = _max_matching_size(
        [(ref_index, pred_index) for _s, ref_index, pred_index, *_ in pairs]
    )
    matched_ref: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int, float, float]] = []
    ordered_pairs = sorted(pairs, key=lambda item: (-item[0], item[1], item[2]))
    for _score, ref_index, pred_index, _text, time_score, space_score in ordered_pairs:
        if len(matches) >= target_matches:
            break
        if ref_index in matched_ref or pred_index in matched_pred:
            continue
        matched_ref.add(ref_index)
        matched_pred.add(pred_index)
        remaining = [
            (candidate_ref, candidate_pred)
            for _s, candidate_ref, candidate_pred, *_ in pairs
            if candidate_ref not in matched_ref and candidate_pred not in matched_pred
        ]
        if len(matches) + 1 + _max_matching_size(remaining) < target_matches:
            matched_ref.discard(ref_index)
            matched_pred.discard(pred_index)
            continue
        matches.append((ref_index, pred_index, time_score, space_score))

    total_reference_chars = 0
    total_character_errors = 0
    for ref_index, pred_index, _time_score, _space_score in matches:
        ref_text = normalize_for_metric(reference[ref_index].text_normalized)
        pred_text = normalize_for_metric(predicted[pred_index].text_normalized)
        total_reference_chars += max(len(ref_text), 1)
        total_character_errors += edit_distance(ref_text, pred_text)

    duplicate_predictions: set[int] = set()
    for _score, ref_index, pred_index, _text, _time, _space in pairs:
        if pred_index in matched_pred:
            continue
        if ref_index in matched_ref:
            duplicate_predictions.add(pred_index)

    temporal_scores = [item[2] for item in matches]
    spatial_scores = [item[3] for item in matches]
    latencies = latencies_sec or []
    throughput = video_sec / max(wall_sec, 1e-9) if video_sec > 0 and wall_sec > 0 else 0.0
    return EvaluationReport(
        reference_events=len(reference),
        predicted_events=len(predicted),
        matched_events=len(matches),
        event_recall=len(matches) / max(len(reference), 1),
        text_accuracy=max(
            0.0,
            1.0 - total_character_errors / max(total_reference_chars, 1),
        ),
        duplicate_rate=len(duplicate_predictions) / max(len(predicted), 1),
        mean_temporal_iou=statistics.fmean(temporal_scores) if temporal_scores else 0.0,
        mean_spatial_iou=statistics.fmean(spatial_scores) if spatial_scores else 0.0,
        video_realtime=throughput,
        throughput_video_sec_per_wall_sec=throughput,
        latency_p50_sec=_percentile(latencies, 0.50),
        latency_p95_sec=_percentile(latencies, 0.95),
        latency_p99_sec=_percentile(latencies, 0.99),
    )


def event_from_dict(data: dict[str, Any]) -> TextEvent:
    history = tuple(
        (float(timestamp), as_polygon(polygon))
        for timestamp, polygon in data.get("polygon_history", [])
    )
    return TextEvent(
        event_id=int(data.get("event_id", 0)),
        geometry_id=int(data.get("geometry_id", 0)),
        content_id=int(data.get("content_id", 0)),
        start=float(data["start"]),
        end=float(data["end"]),
        text_raw=str(data.get("text_raw", data.get("text_normalized", ""))),
        text_normalized=str(data.get("text_normalized", data.get("text_raw", ""))),
        confidence=float(data.get("confidence", 0.0)),
        polygon_history=history,
        source_frame_ids=tuple(int(item) for item in data.get("source_frame_ids", [])),
        cached=bool(data.get("cached", False)),
        recognition_level=int(data.get("recognition_level", 1)),
        alternatives=tuple(str(item) for item in data.get("alternatives", [])),
        type_hint=data.get("type_hint"),
    )
