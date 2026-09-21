from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass(frozen=True)
class ActivityEvent:
    step: str
    frame: int
    timestamp_s: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrameResult:
    frame: int
    timestamp_s: float
    state: str
    state_confidence: float
    detections: list[Detection]
    latency_ms: float
