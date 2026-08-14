from __future__ import annotations

import cv2
import numpy as np

from temporal_ocr.geometry import image_signature
from temporal_ocr.rapidocr_backend import (
    RapidOCRBatchRecognizer,
    RapidOCRDetector,
    RapidOCRRuntime,
)
from temporal_ocr.types import (
    CanonicalObservation,
    DetectionRequest,
    DetectionTier,
    FramePacket,
    OCRTask,
)


def test_rapidocr_detector_and_batch_recognizer_use_separate_interfaces() -> None:
    image = np.full((120, 420, 3), 255, dtype=np.uint8)
    cv2.putText(image, "HELLO 123", (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
    runtime = RapidOCRRuntime(params={"Global.log_level": "error"})
    detector = RapidOCRDetector(runtime=runtime)
    recognizer = RapidOCRBatchRecognizer(runtime=runtime)
    frame = FramePacket(frame_id=0, timestamp=0.0, image=image)

    observations = detector.detect(
        frame,
        DetectionRequest(DetectionTier.AUDIT, "test", 960),
    )
    candidate = CanonicalObservation(
        geometry_id=1,
        frame_id=0,
        timestamp=0.0,
        image=image,
        signature=image_signature(image),
        sharpness=1.0,
        contrast=1.0,
        completeness=1.0,
        occlusion=0.0,
    )
    results = recognizer.recognize_batch([OCRTask(1, 1, (candidate,))])

    assert observations
    assert len(results) == 1
    assert "HELLO" in results[0].text
    assert results[0].confidence > 0.8
