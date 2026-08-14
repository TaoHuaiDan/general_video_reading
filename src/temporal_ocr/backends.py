"""Pluggable detector and recognizer contracts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from temporal_ocr.types import (
    DetectionObservation,
    DetectionRequest,
    FramePacket,
    OCRResult,
    OCRTask,
)


class TextDetector(Protocol):
    name: str

    def detect(
        self,
        frame: FramePacket,
        request: DetectionRequest,
    ) -> list[DetectionObservation]: ...


class TextRecognizer(Protocol):
    name: str

    def recognize_batch(self, tasks: Sequence[OCRTask]) -> list[OCRResult]: ...


class CallableDetector:
    def __init__(
        self,
        callback: Callable[[FramePacket, DetectionRequest], list[DetectionObservation]],
        name: str = "callable-detector",
    ) -> None:
        self.callback = callback
        self.name = name

    def detect(
        self,
        frame: FramePacket,
        request: DetectionRequest,
    ) -> list[DetectionObservation]:
        return self.callback(frame, request)


class CallableRecognizer:
    def __init__(
        self,
        callback: Callable[[Sequence[OCRTask]], list[OCRResult]],
        name: str = "callable-recognizer",
    ) -> None:
        self.callback = callback
        self.name = name

    def recognize_batch(self, tasks: Sequence[OCRTask]) -> list[OCRResult]:
        return self.callback(tasks)


class NullDetector:
    name = "null-detector"

    def detect(
        self,
        frame: FramePacket,
        request: DetectionRequest,
    ) -> list[DetectionObservation]:
        return []


class NullRecognizer:
    name = "null-recognizer"

    def recognize_batch(self, tasks: Sequence[OCRTask]) -> list[OCRResult]:
        return [
            OCRResult(
                content_id=task.content_id,
                text="",
                confidence=0.0,
                backend=self.name,
            )
            for task in tasks
        ]
