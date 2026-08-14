"""Temporal OCR Engine public API."""

from temporal_ocr.chunking import run_video_ocr_chunked
from temporal_ocr.config import EngineConfig
from temporal_ocr.engine import TemporalOCREngine
from temporal_ocr.runner import OCRExecution, run_video_ocr
from temporal_ocr.types import TextEvent

__all__ = [
    "EngineConfig",
    "OCRExecution",
    "TemporalOCREngine",
    "TextEvent",
    "run_video_ocr",
    "run_video_ocr_chunked",
]
__version__ = "0.1.0"
