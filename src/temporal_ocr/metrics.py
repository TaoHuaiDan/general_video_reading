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
    # OCR quality restricted to matched events; decoupled from detection.
    matched_text_accuracy: float = 0.0
    # Fraction of predictions that correspond to a reference event.
    event_precision: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percentile)))
    return float(ordered[index])


def _minimum_cost_maximum_matching(
    edges: list[tuple[int, int, float]],
) -> list[tuple[int, int]]:
    """Return a maximum-cardinality matching with maximum total edge score.

    The benchmark contract is completeness-first, so the primary objective is
    the number of matched reference/prediction pairs; total match quality is
    only the second objective.  Greedy score-first pairing cannot guarantee
    either.  This implements successive shortest augmenting paths
    (Bellman-Ford) on a unit-capacity bipartite flow network with edge cost
    ``-score``: after each augmentation the flow of that value has minimal
    total cost, so the resulting maximum flow is score-optimal among all
    maximum-cardinality matchings.  Benchmark graphs are tiny and the edge
    insertion order makes tie-breaking deterministic.
    """
    if not edges:
        return []
    refs = sorted({ref for ref, _pred, _score in edges})
    preds = sorted({pred for _ref, pred, _score in edges})
    source = 0
    sink = len(refs) + len(preds) + 1
    ref_node = {ref: index + 1 for index, ref in enumerate(refs)}
    pred_node = {pred: len(refs) + 1 + index for index, pred in enumerate(preds)}

    # Residual graph entries: [to, capacity, cost, reverse_index, payload].
    # ``payload >= 0`` marks a real ref/pred edge so saturated edges can be
    # recovered after the flow.
    graph: list[list[list[Any]]] = [[] for _ in range(sink + 1)]
    payloads: list[tuple[int, int]] = []

    def add_edge(u: int, v: int, cost: float, payload: int = -1) -> None:
        if payload >= 0:
            payloads.append((ref_of_node[u], pred_of_node[v]))
        graph[u].append([v, 1, cost, len(graph[v]), payload])
        graph[v].append([u, 0, -cost, len(graph[u]) - 1, -1])

    ref_of_node = {node: ref for ref, node in ref_node.items()}
    pred_of_node = {node: pred for pred, node in pred_node.items()}

    for ref in refs:
        add_edge(source, ref_node[ref], 0.0)
    for pred in preds:
        add_edge(pred_node[pred], sink, 0.0)
    for ref, pred, score in sorted(edges, key=lambda item: (-item[2], item[0], item[1])):
        add_edge(ref_node[ref], pred_node[pred], -float(score), payload=len(payloads))

    infinity = float("inf")
    while True:
        distance = [infinity] * (sink + 1)
        distance[source] = 0.0
        path_to: list[tuple[int, int]] = [(-1, -1)] * (sink + 1)
        for _ in range(sink + 1):
            improved = False
            for u in range(sink + 1):
                if distance[u] == infinity:
                    continue
                for index, edge in enumerate(graph[u]):
                    to, capacity, cost = edge[0], edge[1], edge[2]
                    if capacity > 0 and distance[u] + cost < distance[to] - 1e-12:
                        distance[to] = distance[u] + cost
                        path_to[to] = (u, index)
                        improved = True
            if not improved:
                break
        if distance[sink] == infinity:
            break
        node = sink
        while node != source:
            u, index = path_to[node]
            edge = graph[u][index]
            edge[1] -= 1
            graph[edge[0]][edge[3]][1] += 1
            node = u

    matched: list[tuple[int, int]] = []
    for u in range(sink + 1):
        for edge in graph[u]:
            if edge[4] >= 0 and edge[1] == 0:
                matched.append(payloads[edge[4]])
    return sorted(set(matched))


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
    """Evaluate predictions against references, completeness-first.

    Matching eligibility is spatio-temporal only: a prediction that localizes
    a reference event in time and space counts as detection even when the OCR
    text is wrong — that failure belongs to Text Accuracy.  Text similarity
    still weights the combined match score used to choose among candidate
    pairs (``min_text_similarity`` is kept for API compatibility but no
    longer gates matching).  Unmatched reference characters count as
    deletion errors so "no detections" can never masquerade as perfect text.
    """
    pairs: list[tuple[float, int, int, float, float, float]] = []
    for ref_index, ref_event in enumerate(reference):
        for pred_index, pred_event in enumerate(predicted):
            text_score = text_similarity(ref_event.text_normalized, pred_event.text_normalized)
            time_score = temporal_iou(ref_event, pred_event)
            space_score = mean_spatial_iou(ref_event, pred_event)
            if (
                time_score >= min_temporal_iou
                and space_score >= min_spatial_iou
            ):
                score = 0.55 * text_score + 0.25 * time_score + 0.20 * space_score
                pairs.append((score, ref_index, pred_index, text_score, time_score, space_score))

    # Completeness-first matching: maximum cardinality first, then maximum
    # total match quality among those solutions (deterministic).
    pair_scores = {(ref_index, pred_index): (time_score, space_score)
                   for _s, ref_index, pred_index, _t, time_score, space_score in pairs}
    matched_pairs = _minimum_cost_maximum_matching(
        [(ref_index, pred_index, score) for score, ref_index, pred_index, *_ in pairs]
    )
    matched_ref: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int, float, float]] = []
    for ref_index, pred_index in matched_pairs:
        time_score, space_score = pair_scores[(ref_index, pred_index)]
        matched_ref.add(ref_index)
        matched_pred.add(pred_index)
        matches.append((ref_index, pred_index, time_score, space_score))

    total_reference_chars = sum(
        max(len(normalize_for_metric(ref_event.text_normalized)), 1)
        for ref_event in reference
    )
    matched_reference_chars = 0
    matched_character_errors = 0
    matched_ref_ids = {item[0] for item in matches}
    for ref_index, pred_index, _time_score, _space_score in matches:
        ref_text = normalize_for_metric(reference[ref_index].text_normalized)
        pred_text = normalize_for_metric(predicted[pred_index].text_normalized)
        matched_reference_chars += max(len(ref_text), 1)
        matched_character_errors += edit_distance(ref_text, pred_text)
    # Unmatched references are missed detections: their full text counts as
    # deletion errors so Event Recall == 0 can never yield perfect accuracy.
    unmatched_character_errors = sum(
        max(len(normalize_for_metric(ref_event.text_normalized)), 1)
        for ref_index, ref_event in enumerate(reference)
        if ref_index not in matched_ref_ids
    )
    total_character_errors = matched_character_errors + unmatched_character_errors

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
        matched_text_accuracy=(
            max(0.0, 1.0 - matched_character_errors / max(matched_reference_chars, 1))
            if matches
            else 0.0
        ),
        event_precision=len(matches) / max(len(predicted), 1),
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
