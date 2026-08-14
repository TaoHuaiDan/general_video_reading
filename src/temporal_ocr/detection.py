"""Hierarchical detection planning independent of concrete OCR models."""

from __future__ import annotations

from temporal_ocr.config import DetectionConfig
from temporal_ocr.types import DetectionRequest, DetectionTier, PolicyDecision, Polygon


class HierarchicalDetectionPlanner:
    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()
        self.last_audit = float("-inf")
        self.last_probe = float("-inf")

    def plan(
        self,
        *,
        timestamp: float,
        scene_cut: bool,
        changed_scopes: tuple[Polygon, ...],
        changed_ratio: float,
        has_active_tracks: bool,
        decision: PolicyDecision,
        motion_reliable: bool,
    ) -> list[DetectionRequest]:
        requests: list[DetectionRequest] = []
        audit_due = timestamp - self.last_audit >= decision.audit_interval_sec
        unreliable_audit_due = (
            not motion_reliable
            and timestamp - self.last_audit >= self.config.min_audit_interval_sec
        )
        if scene_cut or audit_due or unreliable_audit_due:
            reason = (
                "scene_cut"
                if scene_cut
                else "periodic_audit"
                if audit_due
                else "motion_unreliable"
            )
            requests.append(
                DetectionRequest(
                    tier=DetectionTier.AUDIT,
                    reason=reason,
                    target_width=self.config.audit_width,
                )
            )
            self.last_audit = timestamp
            self.last_probe = timestamp
            return requests

        if timestamp - self.last_probe < decision.probe_interval_sec:
            # A typewriter or wrapped-dialogue update can arrive during the
            # normal probe cooldown.  Let the local path inspect it when
            # geometry already exists, while leaving broad changes to the
            # regular FAST/audit schedule.
            if (
                changed_scopes
                and has_active_tracks
                and changed_ratio < self.config.fast_trigger_change_ratio
                and decision.enable_local_detection
            ):
                return [
                    DetectionRequest(
                        tier=DetectionTier.LOCAL,
                        reason="urgent_local_change",
                        target_width=self.config.local_width,
                        scopes=changed_scopes,
                    )
                ]
            return requests
        self.last_probe = timestamp
        # Once geometry tracks exist, a small changed region is cheaper and
        # safer to inspect locally. Re-running a full-frame detector on every
        # sampled frame was the dominant cost on high-resolution VN footage.
        broad_change = changed_ratio >= self.config.fast_trigger_change_ratio
        if not has_active_tracks or broad_change:
            requests.append(
                DetectionRequest(
                    tier=DetectionTier.FAST,
                    reason="scheduled_probe" if not has_active_tracks else "broad_change",
                    target_width=decision.fast_detection_width,
                )
            )
        if changed_scopes and decision.enable_local_detection:
            requests.append(
                DetectionRequest(
                    tier=DetectionTier.LOCAL,
                    reason="motion_compensated_local_change",
                    target_width=self.config.local_width,
                    scopes=changed_scopes,
                )
            )
        return requests
