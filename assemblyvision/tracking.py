from __future__ import annotations

from collections import defaultdict

from .models import Detection


class CentroidTracker:
    """Small label-aware tracker for movement features and short occlusions."""

    def __init__(self, missing_tolerance: int = 4):
        self.missing_tolerance = missing_tolerance
        self.previous: dict[str, tuple[float, float]] = {}
        self.missing = defaultdict(int)

    def update(self, detections: list[Detection]) -> dict[str, dict]:
        best: dict[str, Detection] = {}
        for detection in detections:
            if detection.label not in best or detection.confidence > best[detection.label].confidence:
                best[detection.label] = detection
        tracks = {}
        for label, detection in best.items():
            center = detection.centroid
            old = self.previous.get(label, center)
            dx, dy = center[0] - old[0], center[1] - old[1]
            tracks[label] = {"center": center, "speed": (dx * dx + dy * dy) ** 0.5,
                             "confidence": detection.confidence, "visible": True}
            self.previous[label] = center
            self.missing[label] = 0
        for label in list(self.previous):
            if label not in best:
                self.missing[label] += 1
                if self.missing[label] <= self.missing_tolerance:
                    tracks[label] = {"center": self.previous[label], "speed": 0.0,
                                     "confidence": 0.25, "visible": False}
                else:
                    del self.previous[label]
                    del self.missing[label]
        return tracks
