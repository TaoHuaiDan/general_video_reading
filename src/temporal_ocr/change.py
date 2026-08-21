"""Motion-compensated tile change maps and scene-change statistics."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from temporal_ocr.geometry import to_gray
from temporal_ocr.motion import compensate_previous
from temporal_ocr.types import MotionEstimate, Polygon


@dataclass(slots=True)
class ChangeMapResult:
    score: float
    changed_ratio: float
    changed_tiles: tuple[tuple[int, int], ...]
    scopes: tuple[Polygon, ...]
    pixel_delta: np.ndarray | None = None


class TileChangeDetector:
    def __init__(self, rows: int = 10, cols: int = 16, threshold: float = 0.035) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError("tile grid dimensions must be positive")
        self.rows = rows
        self.cols = cols
        self.threshold = threshold

    def compare(
        self,
        previous: np.ndarray,
        current: np.ndarray,
        motion: MotionEstimate,
        *,
        excluded_polygons: tuple[Polygon, ...] = (),
    ) -> ChangeMapResult:
        current_gray = to_gray(current)
        previous_gray = to_gray(previous)
        if previous_gray.shape != current_gray.shape:
            previous_gray = cv2.resize(
                previous_gray,
                (current_gray.shape[1], current_gray.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        aligned = compensate_previous(previous_gray, motion, current_gray.shape[:2])
        delta = cv2.absdiff(aligned, current_gray).astype(np.float32) / 255.0
        height, width = current_gray.shape[:2]
        # Static watermarks and other caller-declared regions must not create
        # local-change requests.  Compute tile means over the remaining pixels
        # instead of merely zeroing the excluded area, otherwise a large mask
        # would artificially dilute nearby legitimate changes.
        valid_mask = np.ones((height, width), dtype=np.uint8)
        for polygon in excluded_polygons:
            points = np.asarray(polygon, dtype=np.float32).astype(np.int32)
            cv2.fillConvexPoly(valid_mask, points, 0)
        delta[valid_mask == 0] = 0.0
        changed: list[tuple[int, int]] = []
        scores: list[float] = []
        changed_mask = np.zeros((self.rows, self.cols), dtype=np.uint8)
        valid_tile_count = 0
        for row in range(self.rows):
            y1 = row * height // self.rows
            y2 = (row + 1) * height // self.rows
            for col in range(self.cols):
                x1 = col * width // self.cols
                x2 = (col + 1) * width // self.cols
                tile_valid = valid_mask[y1:y2, x1:x2].astype(bool)
                if not np.any(tile_valid):
                    # A fully excluded tile is outside change analysis: it
                    # must neither raise the score nor count as an unchanged
                    # tile that dilutes ``changed_ratio``.
                    continue
                tile_score = float(delta[y1:y2, x1:x2][tile_valid].mean())
                scores.append(tile_score)
                valid_tile_count += 1
                if tile_score >= self.threshold:
                    changed.append((row, col))
                    changed_mask[row, col] = 1

        # Text motion often lights up several neighbouring tiles. Merge them
        # before local detection so one moving line does not launch one model
        # invocation per character-sized tile.
        scopes: list[Polygon] = []
        if changed:
            connected = cv2.dilate(changed_mask, np.ones((3, 3), dtype=np.uint8))
            component_count, labels = cv2.connectedComponents(connected)
            for label in range(1, component_count):
                rows, cols = np.where(labels == label)
                if rows.size == 0:
                    continue
                row_min = max(0, int(rows.min()) - 1)
                row_max = min(self.rows, int(rows.max()) + 2)
                col_min = max(0, int(cols.min()) - 1)
                col_max = min(self.cols, int(cols.max()) + 2)
                x1 = col_min * width // self.cols
                x2 = col_max * width // self.cols
                y1 = row_min * height // self.rows
                y2 = row_max * height // self.rows
                scopes.append(
                    (
                        (float(x1), float(y1)),
                        (float(x2), float(y1)),
                        (float(x2), float(y2)),
                        (float(x1), float(y2)),
                    )
                )
        return ChangeMapResult(
            score=float(np.mean(scores)) if scores else 0.0,
            changed_ratio=(
                len(changed) / float(valid_tile_count) if valid_tile_count else 0.0
            ),
            changed_tiles=tuple(changed),
            scopes=tuple(scopes),
            pixel_delta=delta,
        )
