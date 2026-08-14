"""Temporal OCR Engine public API."""

from temporal_ocr.config import EngineConfig
from temporal_ocr.engine import TemporalOCREngine
from temporal_ocr.types import TextEvent

__all__ = ["EngineConfig", "TemporalOCREngine", "TextEvent"]
__version__ = "0.1.0"
