"""RapidOCR 3 adapters implementing separated detection and batched recognition."""

from __future__ import annotations

import importlib
import time
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from temporal_ocr.geometry import as_polygon, polygon_bbox, polygon_coverage
from temporal_ocr.types import (
    DetectionObservation,
    DetectionRequest,
    DetectionTier,
    FramePacket,
    OCRResult,
    OCRTask,
)


def _rapidocr_class() -> Any:
    try:
        module = importlib.import_module("rapidocr")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("RapidOCR is required: pip install -e .[ocr]") from exc
    return module.RapidOCR


class RapidOCRRuntime:
    """Shared model runtime so detector and recognizer load weights only once."""

    _serial = 0

    def __init__(self, *, params: dict[str, Any] | None = None) -> None:
        RapidOCRRuntime._serial += 1
        # Stable per-process identity: two runtimes may host different model
        # configurations, and recognition results are not guaranteed to match.
        self.serial = RapidOCRRuntime._serial
        self.engine = _rapidocr_class()(params=params)


class RapidOCRDetector:
    """Detection-only adapter; recognition remains a separate batch stage."""

    name = "rapidocr-v3-detector"

    def __init__(
        self,
        *,
        runtime: RapidOCRRuntime | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.runtime = runtime or RapidOCRRuntime(params=params)
        self.engine = self.runtime.engine

    @staticmethod
    def _threshold(tier: DetectionTier) -> float:
        if tier == DetectionTier.FAST:
            return 0.35
        if tier == DetectionTier.LOCAL:
            return 0.28
        return 0.22

    def _detect_crop(
        self,
        crop: np.ndarray,
        frame: FramePacket,
        request: DetectionRequest,
        *,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> list[DetectionObservation]:
        height, width = crop.shape[:2]
        scale = min(1.0, request.target_width / max(width, 1))
        working = crop
        if scale < 1.0:
            working = cv2.resize(
                crop,
                (max(2, round(width * scale)), max(2, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        output = self.engine(
            working,
            use_det=True,
            use_cls=False,
            use_rec=False,
            box_thresh=self._threshold(request.tier),
        )
        boxes = getattr(output, "boxes", None)
        scores = getattr(output, "scores", None)
        if boxes is None:
            return []
        score_values = list(scores) if scores is not None else [1.0] * len(boxes)
        observations: list[DetectionObservation] = []
        for box, score in zip(boxes, score_values):
            points = np.asarray(box, dtype=np.float32)
            points /= max(scale, 1e-9)
            points[:, 0] += offset_x
            points[:, 1] += offset_y
            polygon = as_polygon(points)
            if any(
                polygon_coverage(polygon, excluded) >= 0.35
                for excluded in request.exclude_regions
            ):
                # Keep the exclusion at the backend boundary as well as in
                # the engine.  This prevents watermark boxes from entering
                # geometry/content tracking even with a custom postprocessor.
                continue
            observations.append(
                DetectionObservation(
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    polygon=polygon,
                    confidence=float(score),
                    tier=request.tier,
                )
            )
        return observations

    def detect(
        self,
        frame: FramePacket,
        request: DetectionRequest,
    ) -> list[DetectionObservation]:
        image = np.asarray(frame.image)
        if not request.scopes:
            return self._detect_crop(image, frame, request)
        observations: list[DetectionObservation] = []
        height, width = image.shape[:2]
        for scope in request.scopes:
            x1, y1, x2, y2 = polygon_bbox(scope)
            left = max(0, min(width - 1, int(np.floor(x1))))
            top = max(0, min(height - 1, int(np.floor(y1))))
            right = max(left + 1, min(width, int(np.ceil(x2))))
            bottom = max(top + 1, min(height, int(np.ceil(y2))))
            observations.extend(
                self._detect_crop(
                    image[top:bottom, left:right],
                    frame,
                    request,
                    offset_x=left,
                    offset_y=top,
                )
            )
        return observations


class RapidOCRBatchRecognizer:
    """Use RapidOCR's internal TextRecognizer to batch normalized text crops."""

    name = "rapidocr-v3-recognizer"

    def __init__(
        self,
        *,
        runtime: RapidOCRRuntime | None = None,
        params: dict[str, Any] | None = None,
        fallback_threshold: float = 0.82,
    ) -> None:
        self.runtime = runtime or RapidOCRRuntime(params=params)
        self.engine = self.runtime.engine
        typings = importlib.import_module("rapidocr.ch_ppocr_rec.typings")
        self._input_type = typings.TextRecInput
        self.fallback_threshold = fallback_threshold
        # Semantic fingerprint: model runtime plus the fallback policy that
        # decides whether secondary candidates can change the final result.
        # Performance-only knobs (thread counts) are deliberately excluded.
        self.cache_namespace = (
            f"{self.name}:runtime-{self.runtime.serial}"
            f":fallback-threshold-{self.fallback_threshold}"
        )

    def _recognize_images(self, images: list[np.ndarray]) -> tuple[list[str], list[float], float]:
        if not images:
            return [], [], 0.0
        started = time.perf_counter()
        output = self.engine.text_rec(self._input_type(img=images))
        elapsed = float(getattr(output, "elapse", 0.0) or (time.perf_counter() - started))
        texts = [str(item) for item in (getattr(output, "txts", None) or ())]
        scores = [float(item) for item in (getattr(output, "scores", None) or ())]
        if len(texts) != len(images) or len(scores) != len(images):
            raise RuntimeError(
                f"RapidOCR recognizer returned {len(texts)} texts for {len(images)} images"
            )
        return texts, scores, elapsed

    def recognize_batch(self, tasks: Sequence[OCRTask]) -> list[OCRResult]:
        if not tasks:
            return []
        primary_candidates = [max(task.candidates, key=lambda item: item.quality) for task in tasks]
        texts, scores, elapsed = self._recognize_images(
            [np.asarray(candidate.image) for candidate in primary_candidates]
        )
        alternatives: dict[int, list[tuple[str, float]]] = {}
        fallback_images: list[np.ndarray] = []
        fallback_owners: list[int] = []
        for index, (task, score) in enumerate(zip(tasks, scores)):
            if score >= self.fallback_threshold or len(task.candidates) <= 1:
                continue
            primary = primary_candidates[index]
            for candidate in task.candidates:
                if candidate is primary:
                    continue
                fallback_images.append(np.asarray(candidate.image))
                fallback_owners.append(index)

        fallback_elapsed = 0.0
        if fallback_images:
            fallback_texts, fallback_scores, fallback_elapsed = self._recognize_images(
                fallback_images
            )
            for owner, text, score in zip(fallback_owners, fallback_texts, fallback_scores):
                alternatives.setdefault(owner, []).append((text, score))

        results: list[OCRResult] = []
        total_elapsed = elapsed + fallback_elapsed
        for index, task in enumerate(tasks):
            candidates = [(texts[index], scores[index]), *alternatives.get(index, [])]
            chosen_text, chosen_score = max(candidates, key=lambda item: item[1])
            other_texts = tuple(
                text
                for text, _score in candidates
                if text and text != chosen_text
            )
            results.append(
                OCRResult(
                    content_id=task.content_id,
                    text=chosen_text,
                    confidence=chosen_score,
                    alternatives=tuple(dict.fromkeys(other_texts)),
                    backend=self.name,
                    inference_sec=total_elapsed / max(len(tasks), 1),
                )
            )
        return results
