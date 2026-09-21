from __future__ import annotations

import cv2
import numpy as np

from .models import Detection


COLORS = {"component": (255, 120, 0), "socket": (0, 210, 0), "fastener": (0, 0, 255), "inspection": (0, 220, 255)}


def annotate(frame: np.ndarray, detections: list[Detection], state: str, confidence: float) -> np.ndarray:
    canvas = frame.copy()
    for item in detections:
        x1, y1, x2, y2 = item.bbox
        color = COLORS.get(item.label, (255, 255, 255))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"{item.label} {item.confidence:.2f}", (x1, max(18, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (20, 24, 32), -1)
    cv2.putText(canvas, f"Current activity: {state.replace('_', ' ').title()} | {confidence:.0%}",
                (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas
