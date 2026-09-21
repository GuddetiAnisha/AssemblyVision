from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from .models import Detection


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class ColorMarkerDetector:
    """HSV marker baseline. Replace this class with a learned detector."""

    def __init__(self, classes: dict, min_area: int = 180):
        self.classes = classes
        self.min_area = min_area

    def detect(self, frame: np.ndarray) -> list[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        output: list[Detection] = []
        kernel = np.ones((3, 3), np.uint8)
        for label, ranges in self.classes.items():
            lower = np.array(ranges["lower"], dtype=np.uint8)
            upper = np.array(ranges["upper"], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            if label == "fastener":
                mask |= cv2.inRange(hsv, np.array([170, 120, 70]), np.array([179, 255, 255]))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                confidence = min(0.99, 0.55 + area / max(frame.shape[0] * frame.shape[1] * 0.05, 1))
                output.append(Detection(label, float(confidence), (x, y, x + w, y + h)))
        return sorted(output, key=lambda item: item.confidence, reverse=True)
