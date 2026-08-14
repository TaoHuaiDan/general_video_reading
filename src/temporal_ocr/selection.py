"""Candidate-frame selection using quality and information complementarity."""

from __future__ import annotations

from temporal_ocr.geometry import signature_distance
from temporal_ocr.types import CanonicalObservation


class ComplementaryCandidateSelector:
    def __init__(self, limit: int = 3, diversity_weight: float = 0.35) -> None:
        if limit <= 0:
            raise ValueError("candidate limit must be positive")
        self.limit = limit
        self.diversity_weight = diversity_weight

    def select(
        self,
        candidates: list[CanonicalObservation],
        incoming: CanonicalObservation,
    ) -> list[CanonicalObservation]:
        pool = [*candidates, incoming]
        if len(pool) <= self.limit:
            return sorted(pool, key=lambda item: item.quality, reverse=True)

        selected = [max(pool, key=lambda item: item.quality)]
        remaining = [item for item in pool if item is not selected[0]]
        while remaining and len(selected) < self.limit:
            def marginal_score(item: CanonicalObservation) -> float:
                diversity = min(
                    signature_distance(item.signature, chosen.signature)
                    for chosen in selected
                )
                information = abs(item.completeness - selected[0].completeness)
                return item.quality + self.diversity_weight * (0.7 * diversity + 0.3 * information)

            chosen = max(remaining, key=marginal_score)
            selected.append(chosen)
            remaining.remove(chosen)
        return selected
