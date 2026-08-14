from __future__ import annotations

import cv2
import numpy as np

from temporal_ocr.change import TileChangeDetector
from temporal_ocr.config import MotionConfig
from temporal_ocr.geometry import canonicalize_crop, image_signature, signature_distance
from temporal_ocr.motion import GlobalMotionEstimator, identity_motion


def test_perspective_normalization_returns_horizontal_crop() -> None:
    image = np.zeros((160, 240, 3), dtype=np.uint8)
    polygon = ((35.0, 40.0), (205.0, 25.0), (215.0, 100.0), (25.0, 115.0))
    cv2.fillConvexPoly(image, np.asarray(polygon, dtype=np.int32), (255, 255, 255))
    cv2.putText(image, "TEXT", (55, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    crop, transform = canonicalize_crop(image, polygon, target_height=48)

    assert crop.shape[0] == 48
    assert crop.shape[1] > crop.shape[0]
    assert transform.shape == (3, 3)
    assert crop.std() > 10


def test_signature_detects_content_change() -> None:
    left = np.zeros((48, 160), dtype=np.uint8)
    right = left.copy()
    cv2.putText(right, "A", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
    assert signature_distance(image_signature(left), image_signature(left.copy())) == 0.0
    assert signature_distance(image_signature(left), image_signature(right)) > 0.01


def test_global_motion_estimator_recovers_translation() -> None:
    rng = np.random.default_rng(42)
    previous = rng.integers(0, 255, size=(240, 320), dtype=np.uint8)
    matrix = np.asarray([[1.0, 0.0, 8.0], [0.0, 1.0, 5.0]], dtype=np.float32)
    current = cv2.warpAffine(previous, matrix, (320, 240), borderMode=cv2.BORDER_REFLECT)
    estimator = GlobalMotionEstimator(
        MotionConfig(min_features=12, min_inlier_ratio=0.3, max_residual_error=4.0)
    )

    estimate = estimator.estimate(previous, current)

    assert estimate.valid
    assert abs(float(estimate.matrix[0, 2]) - 8.0) < 1.5
    assert abs(float(estimate.matrix[1, 2]) - 5.0) < 1.5


def test_motion_compensation_reduces_changed_tiles() -> None:
    rng = np.random.default_rng(7)
    previous = rng.integers(0, 255, size=(180, 240), dtype=np.uint8)
    matrix = np.asarray([[1.0, 0.0, 6.0], [0.0, 1.0, 4.0]], dtype=np.float32)
    current = cv2.warpAffine(previous, matrix, (240, 180), borderMode=cv2.BORDER_REPLICATE)
    detector = TileChangeDetector(rows=6, cols=8, threshold=0.04)

    uncompensated = detector.compare(previous, current, identity_motion())
    estimate = identity_motion()
    estimate.matrix = matrix
    compensated = detector.compare(previous, current, estimate)

    assert compensated.changed_ratio < uncompensated.changed_ratio
