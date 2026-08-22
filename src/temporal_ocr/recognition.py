"""Recognition batching and conservative exact-result caching."""

from __future__ import annotations

import hashlib
import time
from collections import Counter, OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np

from temporal_ocr.backends import TextRecognizer
from temporal_ocr.geometry import exact_signature
from temporal_ocr.types import OCRResult, OCRTask

# Entries hold only an OCR result keyed by a 32-byte digest, so a few thousand
# entries stay well under a few megabytes while still covering the distinct
# normalized crops of a feature-length video.
DEFAULT_CACHE_MAX_ENTRIES = 4096


def candidate_set_signature(images: Sequence[Any]) -> bytes:
    """Return an exact key for a task-level OCR result.

    The final OCR result of a task is a function of its complete candidate
    set (primary crop plus fallbacks), not of the primary crop alone.  The
    key covers every candidate's exact pixel digest **in candidate order**:
    primary selection (``max`` by quality) and alternative ordering break
    ties by list position, so the computation is not proven permutation
    invariant and an ordered digest is the conservative exact key.
    """
    images = list(images)
    digest = hashlib.blake2b(digest_size=32)
    digest.update(b"temporal-ocr/candidate-set/v2")
    digest.update(len(images).to_bytes(8, "big"))
    for image in images:
        digest.update(exact_signature(np.asarray(image)))
    return digest.digest()


class RecognitionCache:
    """Exact task-result cache; perceptual reuse remains opt-in later.

    Keys are candidate-set digests (see :func:`candidate_set_signature`), so
    a cached result is bound to the exact crop set that produced it.  The
    cache is a bounded LRU when ``max_entries`` is given.  Passing ``None``
    restores unbounded retention for callers that manage memory themselves.
    """

    def __init__(self, max_entries: int | None = DEFAULT_CACHE_MAX_ENTRIES) -> None:
        if max_entries is not None and max_entries <= 0:
            raise ValueError("max_entries must be positive when provided")
        self._max_entries = max_entries
        self._items: OrderedDict[tuple[str, bytes], OCRResult] = OrderedDict()

    def get(self, backend: str, signature: bytes) -> OCRResult | None:
        key = (backend, signature)
        item = self._items.get(key)
        if item is not None:
            self._items.move_to_end(key)
        return item

    def put(self, backend: str, signature: bytes, result: OCRResult) -> None:
        key = (backend, signature)
        self._items[key] = result
        # Assignment alone does not refresh OrderedDict recency; an updated
        # entry must count as recently used or fresh updates get evicted.
        self._items.move_to_end(key)
        if self._max_entries is not None:
            while len(self._items) > self._max_entries:
                self._items.popitem(last=False)

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
    # Task identity is (content_id, revision): several revisions of one
    # content track may legitimately share a batch.  Multiset comparison
    # rejects missing, duplicated, extra and wrong-revision results.
    expected = Counter((task.content_id, task.revision) for task in tasks)
    actual = Counter((result.content_id, result.revision) for result in results)
    if expected != actual:
        raise RuntimeError(
            "recognizer returned mismatched task identities: "
            f"expected={sorted(expected.items())}, actual={sorted(actual.items())}"
        )
    return results
