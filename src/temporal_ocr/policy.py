"""A deterministic rule-based controller for accuracy and throughput budgets."""

from __future__ import annotations

from temporal_ocr.config import DetectionConfig, PolicyConfig
from temporal_ocr.types import PolicyDecision, RuntimeSignals


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class RuleBasedPolicyScheduler:
    """Adjust runtime knobs from continuous statistics, not video-type labels."""

    def __init__(
        self,
        policy: PolicyConfig | None = None,
        detection: DetectionConfig | None = None,
    ) -> None:
        self.policy = policy or PolicyConfig()
        self.detection = detection or DetectionConfig()
        self._last: PolicyDecision | None = None

    def decide(self, signals: RuntimeSignals) -> PolicyDecision:
        reasons: list[str] = []
        probe = self.policy.default_probe_interval_sec
        audit = self.detection.audit_interval_sec
        width = self.detection.fast_width
        stable_wait = 0.45
        maximum_wait = 1.8
        batch_size = self.policy.default_batch_size
        batch_wait = self.policy.default_batch_wait_ms

        pressure = max(signals.ocr_queue_length, signals.detection_queue_length)
        if pressure >= self.policy.queue_pressure_threshold:
            probe *= 1.6
            batch_size *= 2
            batch_wait += 15
            reasons.append("queue_pressure")

        if signals.layout_stability >= 0.90 and signals.audit_new_text_yield < 0.02:
            audit *= 1.5
            probe *= 1.25
            reasons.append("stable_layout")
        if signals.audit_new_text_yield >= 0.10:
            # A high-yield audit should temporarily increase scrutiny, but
            # halving the interval on every subsequent audit creates an audit
            # storm on long dialogue scenes. Keep the response bounded.
            audit *= 0.80
            width = max(width, self.detection.local_width)
            reasons.append("audit_found_misses")

        if signals.global_motion_magnitude > 8.0 and signals.motion_confidence < 0.55:
            audit *= 0.5
            width = max(width, self.detection.local_width)
            reasons.append("unreliable_motion_compensation")
        elif signals.global_motion_magnitude > 8.0 and signals.motion_confidence >= 0.75:
            reasons.append("motion_compensated")

        if signals.typewriter_score >= 0.60:
            stable_wait *= 1.7
            maximum_wait *= 1.25
            reasons.append("typewriter_pattern")
        if 0.0 < signals.average_text_lifetime < 1.0:
            probe *= 0.65
            stable_wait *= 0.55
            maximum_wait *= 0.70
            reasons.append("short_lived_text")

        if signals.gpu_utilization < 0.35 and pressure > 0:
            batch_wait -= 10
            reasons.append("gpu_headroom")
        elif signals.gpu_utilization > 0.90:
            batch_wait += 10
            reasons.append("gpu_saturated")

        decision = PolicyDecision(
            probe_interval_sec=_clamp(
                probe,
                self.policy.min_probe_interval_sec,
                self.policy.max_probe_interval_sec,
            ),
            audit_interval_sec=_clamp(
                audit,
                self.detection.min_audit_interval_sec,
                self.detection.max_audit_interval_sec,
            ),
            fast_detection_width=int(max(320, min(self.detection.audit_width, width))),
            stable_wait_sec=_clamp(stable_wait, 0.15, 1.5),
            maximum_wait_sec=_clamp(maximum_wait, 0.5, 4.0),
            batch_size=int(
                _clamp(
                    float(batch_size),
                    float(self.policy.min_batch_size),
                    float(self.policy.max_batch_size),
                )
            ),
            batch_wait_ms=int(
                _clamp(
                    float(batch_wait),
                    float(self.policy.min_batch_wait_ms),
                    float(self.policy.max_batch_wait_ms),
                )
            ),
            reason=tuple(reasons),
        )
        self._last = decision
        return decision
