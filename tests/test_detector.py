import cv2
import numpy as np

from assemblyvision.detector import ColorMarkerDetector


def test_detects_blue_component():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.rectangle(frame, (50, 50), (100, 100), (255, 80, 0), -1)
    detector = ColorMarkerDetector({"component": {"lower": [95, 100, 60], "upper": [135, 255, 255]}}, 100)
    detections = detector.detect(frame)
    assert len(detections) == 1
    assert detections[0].label == "component"
    assert detections[0].confidence > 0.5
