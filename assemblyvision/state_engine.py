from __future__ import annotations

from math import dist

from .models import ActivityEvent


class AssemblyStateEngine:
    """Ordered, dwell-time state machine for a simple manual assembly task."""

    def __init__(self, config: dict, fps: float = 20.0):
        self.steps = config["steps"]
        self.dwell_frames = int(config["dwell_frames"])
        self.proximity = float(config["proximity_px"])
        self.motion_threshold = float(config["motion_threshold_px"])
        self.fps = fps or 20.0
        self.index = 0
        self.candidate_frames = 0
        self.events: list[ActivityEvent] = []
        self.finished = False

    @property
    def state(self) -> str:
        return self.steps[self.index]

    def _condition(self, tracks: dict) -> tuple[bool, float]:
        component = tracks.get("component")
        socket = tracks.get("socket")
        fastener = tracks.get("fastener")
        inspection = tracks.get("inspection")
        near = bool(component and socket and dist(component["center"], socket["center"]) < self.proximity)
        conditions = {
            "pick_component": bool(component and component["speed"] >= self.motion_threshold),
            "position_component": near,
            "insert_component": near and bool(component and component["speed"] < self.motion_threshold),
            "fasten": near and bool(fastener and dist(fastener["center"], socket["center"]) < self.proximity),
            "inspect": bool(inspection and inspection["visible"]),
            "completed": not self.finished,
        }
        active = conditions.get(self.state, False)
        relevant = [v["confidence"] for v in tracks.values() if v["visible"]]
        return active, (sum(relevant) / len(relevant) if relevant else 0.0)

    def update(self, tracks: dict, frame: int) -> tuple[str, float, ActivityEvent | None]:
        if self.finished:
            return self.state, 1.0, None
        active, confidence = self._condition(tracks)
        self.candidate_frames = self.candidate_frames + 1 if active else 0
        event = None
        if self.candidate_frames >= self.dwell_frames:
            completed = self.state
            event = ActivityEvent(completed, frame, frame / self.fps, confidence)
            self.events.append(event)
            if self.index < len(self.steps) - 1:
                self.index += 1
            else:
                self.finished = True
            self.candidate_frames = 0
        progress = min(1.0, self.candidate_frames / max(self.dwell_frames, 1))
        return self.state, float(max(confidence * progress, 0.05)), event
