"""Global camera motion estimation with a confidence-gated fallback."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from temporal_ocr.config import MotionConfig
from temporal_ocr.geometry import polygon_motion_magnitude, to_gray
from temporal_ocr.types import MotionEstimate, Polygon


def identity_motion() -> MotionEstimate:
    return MotionEstimate(
        matrix=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        model="identity",
        confidence=1.0,
        inlier_ratio=1.0,
        residual_error=0.0,
        valid=True,
    )


class GlobalMotionEstimator:
    """Estimate previous-frame to current-frame background motion."""

    def __init__(self, config: MotionConfig | None = None) -> None:
        self.config = config or MotionConfig()

    def estimate(
        self,
        previous: np.ndarray,
        current: np.ndarray,
        *,
        excluded_polygons: Sequence[Polygon] = (),
    ) -> MotionEstimate:
        previous_gray = to_gray(previous)
        current_gray = to_gray(current)
        scale = min(1.0, self.config.max_width / max(previous_gray.shape[1], 1))
        if scale < 1.0:
            size = (
                max(2, round(previous_gray.shape[1] * scale)),
                max(2, round(previous_gray.shape[0] * scale)),
            )
            previous_small = cv2.resize(previous_gray, size, interpolation=cv2.INTER_AREA)
            current_small = cv2.resize(current_gray, size, interpolation=cv2.INTER_AREA)
        else:
            previous_small = previous_gray
            current_small = current_gray

        mask = np.full(previous_small.shape, 255, dtype=np.uint8)
        for polygon in excluded_polygons:
            points = np.asarray(polygon, dtype=np.float32) * scale
            cv2.fillConvexPoly(mask, points.astype(np.int32), 0)

        features = cv2.goodFeaturesToTrack(
            previous_small,
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
            mask=mask,
        )
        if features is None or len(features) < self.config.min_features:
            return MotionEstimate(matrix=identity_motion().matrix, model="identity")

        tracked, status, _error = cv2.calcOpticalFlowPyrLK(
            previous_small,
            current_small,
            features,
            np.empty_like(features),
            winSize=(21, 21),
            maxLevel=3,
        )
        if tracked is None or status is None:
            return MotionEstimate(matrix=identity_motion().matrix, model="identity")
        valid = status.reshape(-1).astype(bool)
        source = features.reshape(-1, 2)[valid]
        destination = tracked.reshape(-1, 2)[valid]
        if len(source) < self.config.min_features:
            return MotionEstimate(matrix=identity_motion().matrix, model="identity")

        matrix, inliers = cv2.estimateAffinePartial2D(
            source,
            destination,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.config.ransac_threshold,
        )
        if matrix is None or inliers is None:
            return MotionEstimate(matrix=identity_motion().matrix, model="identity")

        projected = cv2.transform(source.reshape(1, -1, 2), matrix).reshape(-1, 2)
        residuals = np.linalg.norm(projected - destination, axis=1)
        inlier_mask = inliers.reshape(-1).astype(bool)
        inlier_ratio = float(np.mean(inlier_mask))
        residual = float(np.mean(residuals[inlier_mask])) if np.any(inlier_mask) else float("inf")

        # Matrix was estimated on the reduced frame. Translation must be restored
        # to original coordinates; rotation and scale are dimensionless.
        restored = matrix.astype(np.float32).copy()
        restored[:, 2] /= max(scale, 1e-9)
        valid_estimate = (
            inlier_ratio >= self.config.min_inlier_ratio
            and residual <= self.config.max_residual_error
        )
        confidence = max(
            0.0,
            min(
                1.0,
                inlier_ratio * (1.0 - residual / max(self.config.max_residual_error * 2.0, 1e-9)),
            ),
        )
        return MotionEstimate(
            matrix=restored,
            model="affine",
            confidence=confidence,
            inlier_ratio=inlier_ratio,
            residual_error=residual,
            valid=valid_estimate,
        )

    @staticmethod
    def magnitude(estimate: MotionEstimate) -> float:
        if estimate.matrix is None:
            return 0.0
        return polygon_motion_magnitude(np.asarray(estimate.matrix))


def compensate_previous(
    previous: np.ndarray,
    estimate: MotionEstimate,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Warp a previous luma image into current-frame coordinates."""
    height, width = output_shape
    if not estimate.valid:
        return previous
    matrix = np.asarray(estimate.matrix, dtype=np.float32)
    if matrix.shape == (2, 3):
        return cv2.warpAffine(
            previous,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    return cv2.warpPerspective(
        previous,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
