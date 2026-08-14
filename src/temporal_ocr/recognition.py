"""Recognition batching and conservative exact-result caching."""

from __future__ import annotations

import time
from collections.abc import Sequence

from temporal_ocr.backends import TextRecognizer
from temporal_ocr.types import OCRResult, OCRTask


class RecognitionCache:
    """Exact normalized-image cache; perceptual reuse remains opt-in later."""

    def __init__(self) -> None:
        self._items: dict[tuple[str, bytes], OCRResult] = {}

    def get(self, backend: str, signature: bytes) -> OCRResult | None:
        return self._items.get((backend, signature))

    def put(self, backend: str, signature: bytes, result: OCRResult) -> None:
        self._items[(backend, signature)] = result

    def __len__(self) -> int:
        return len(self._items)


class OCRBatchQueue:
    def __init__(self) -> None:
        self._items: list[tuple[float, OCRTask]] = []

    def push(self, task: OCRTask) -> None:
        self._items.append((time.perf_counter(), task))

    def should_flush(self, batch_size: int, batch_wait_ms: int) -> bool:
        if len(self._items) >= batch_size:
            return True
        if not self._items:
            return False
        age_ms = (time.perf_counter() - self._items[0][0]) * 1000.0
        return age_ms >= batch_wait_ms

    def pop_batch(self, batch_size: int) -> list[OCRTask]:
        selected = self._items[:batch_size]
        del self._items[:batch_size]
        return [task for _queued_at, task in selected]

    def __len__(self) -> int:
        return len(self._items)


def recognize_tasks(
    recognizer: TextRecognizer,
    tasks: Sequence[OCRTask],
) -> list[OCRResult]:
    if not tasks:
        return []
    results = recognizer.recognize_batch(tasks)
    expected = {task.content_id for task in tasks}
    actual = {result.content_id for result in results}
    if expected != actual:
        raise RuntimeError(
            f"recognizer returned mismatched content ids: expected={expected}, actual={actual}"
        )
    return results
