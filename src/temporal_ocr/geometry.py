"""Geometry, normalization and visual signature helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import TypeAlias

import cv2
import numpy as np

from temporal_ocr.types import Point, Polygon

NormalizedRegion: TypeAlias = tuple[float, float, float, float]


def validate_normalized_regions(
    regions: Iterable[Iterable[float]] | None,
) -> tuple[tuple[float, float, float, float], ...]:
    """Validate normalized ``[left, top, right, bottom]`` ignore regions.

    Keeping the public representation normalized makes an MCP request stable
    across resolutions and across independently decoded chunks.  Internally
    the engine converts these rectangles to pixel-space quadrilaterals for
    the current frame.
    """
    if regions is None:
        return ()
    normalized: list[tuple[float, float, float, float]] = []
    for index, region in enumerate(regions):
        values = tuple(float(value) for value in region)
        if len(values) != 4:
            raise ValueError(
                f"exclude_regions[{index}] must contain [left, top, right, bottom]"
            )
        left, top, right, bottom = values
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"exclude_regions[{index}] must contain finite numbers")
        if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
            raise ValueError(
                f"exclude_regions[{index}] must satisfy 0 <= left < right <= 1 "
                "and 0 <= top < bottom <= 1"
            )
        rectangle = (left, top, right, bottom)
        if rectangle not in normalized:
            normalized.append(rectangle)
    return tuple(normalized)


def normalized_region_polygon(
    region: Sequence[float],
    width: int,
    height: int,
) -> Polygon:
    """Convert a normalized rectangle to a clamped pixel-space polygon."""
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")
    left, top, right, bottom = (float(value) for value in region)
    x1 = max(0.0, min(float(width), left * width))
    y1 = max(0.0, min(float(height), top * height))
    x2 = max(x1, min(float(width), right * width))
    y2 = max(y1, min(float(height), bottom * height))
    return (
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),
    )


def as_polygon(points: Iterable[Iterable[float]]) -> Polygon:
    values = tuple((float(x), float(y)) for x, y in points)
    if len(values) != 4:
        raise ValueError("a text polygon must contain exactly four points")
    return values  # type: ignore[return-value]


def order_quad(polygon: Polygon) -> np.ndarray:
    points = np.asarray(polygon, dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    return np.asarray(
        [
            points[np.argmin(sums)],
            points[np.argmin(diffs)],
            points[np.argmax(sums)],
            points[np.argmax(diffs)],
        ],
        dtype=np.float32,
    )


def polygon_bbox(polygon: Polygon) -> tuple[float, float, float, float]:
    points = np.asarray(polygon, dtype=np.float32)
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def polygon_center(polygon: Polygon) -> Point:
    points = np.asarray(polygon, dtype=np.float32)
    center = points.mean(axis=0)
    return float(center[0]), float(center[1])


def polygon_iou(left: Polygon, right: Polygon) -> float:
    left_points = cv2.convexHull(np.asarray(left, dtype=np.float32))
    right_points = cv2.convexHull(np.asarray(right, dtype=np.float32))
    left_area = abs(float(cv2.contourArea(left_points)))
    right_area = abs(float(cv2.contourArea(right_points)))
    if left_area <= 0 or right_area <= 0:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(left_points, right_points)
    union = left_area + right_area - float(intersection)
    return max(0.0, min(1.0, float(intersection) / max(union, 1e-9)))


def polygon_area(polygon: Polygon) -> float:
    points = cv2.convexHull(np.asarray(polygon, dtype=np.float32))
    return abs(float(cv2.contourArea(points)))


def polygon_intersection_over_smaller(left: Polygon, right: Polygon) -> float:
    """Measure containment without penalizing differently sized detections.

    IoU is deliberately insufficient here: a local high-resolution detector may
    return several character fragments fully inside a coarse detector's text-line
    box. Their IoU is small even though the detections describe the same region.
    """
    left_points = cv2.convexHull(np.asarray(left, dtype=np.float32))
    right_points = cv2.convexHull(np.asarray(right, dtype=np.float32))
    smaller = min(
        abs(float(cv2.contourArea(left_points))),
        abs(float(cv2.contourArea(right_points))),
    )
    if smaller <= 0:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(left_points, right_points)
    return max(0.0, min(1.0, float(intersection) / smaller))


def polygon_coverage(subject: Polygon, cover: Polygon) -> float:
    """Return the fraction of ``subject`` covered by ``cover``."""
    subject_points = cv2.convexHull(np.asarray(subject, dtype=np.float32))
    cover_points = cv2.convexHull(np.asarray(cover, dtype=np.float32))
    subject_area = abs(float(cv2.contourArea(subject_points)))
    if subject_area <= 0:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(subject_points, cover_points)
    return max(0.0, min(1.0, float(intersection) / subject_area))


def transform_polygon(polygon: Polygon, matrix: np.ndarray) -> Polygon:
    points = np.asarray(polygon, dtype=np.float32).reshape(1, 4, 2)
    if matrix.shape == (2, 3):
        transformed = cv2.transform(points, matrix)
    elif matrix.shape == (3, 3):
        transformed = cv2.perspectiveTransform(points, matrix)
    else:
        raise ValueError("motion matrix must be 2x3 affine or 3x3 homography")
    return as_polygon(transformed.reshape(4, 2))


def polygon_motion_magnitude(matrix: np.ndarray) -> float:
    return float(math.hypot(matrix[0, 2], matrix[1, 2]))


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def canonicalize_crop(
    image: np.ndarray,
    polygon: Polygon,
    *,
    target_height: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a quadrilateral text observation into a horizontal canonical crop."""
    source = order_quad(polygon)
    width_top = np.linalg.norm(source[1] - source[0])
    width_bottom = np.linalg.norm(source[2] - source[3])
    height_left = np.linalg.norm(source[3] - source[0])
    height_right = np.linalg.norm(source[2] - source[1])
    source_width = max(2.0, float(max(width_top, width_bottom)))
    source_height = max(2.0, float(max(height_left, height_right)))
    target_width = max(2, round(source_width * target_height / source_height))
    destination = np.asarray(
        [[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    crop = cv2.warpPerspective(
        image,
        transform,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return crop, transform


def image_signature(image: np.ndarray, size: int = 16) -> bytes:
    """Return a compact difference hash after geometric normalization."""
    gray = to_gray(image)
    resized = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    return np.packbits(bits.reshape(-1)).tobytes()


def signature_distance(left: bytes, right: bytes) -> float:
    if not left or not right or len(left) != len(right):
        return 1.0
    different = sum((a ^ b).bit_count() for a, b in zip(left, right))
    return different / (8.0 * len(left))


def candidate_quality(image: np.ndarray) -> tuple[float, float, float, float]:
    """Return normalized sharpness, contrast, completeness and occlusion scores."""
    gray = to_gray(image)
    sharpness_raw = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    contrast_raw = float(gray.std())
    sharpness = sharpness_raw / (sharpness_raw + 200.0)
    contrast = contrast_raw / (contrast_raw + 45.0)
    foreground = cv2.Canny(gray, 60, 160)
    completeness = min(1.0, float(np.count_nonzero(foreground)) / max(gray.size * 0.18, 1.0))
    clipped = np.mean((gray <= 3) | (gray >= 252))
    occlusion = min(1.0, float(clipped) * 2.0)
    return sharpness, contrast, completeness, occlusion
