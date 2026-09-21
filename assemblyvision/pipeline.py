from __future__ import annotations

import time
from pathlib import Path

import cv2

from .detector import ColorMarkerDetector
from .models import FrameResult
from .state_engine import AssemblyStateEngine
from .tracking import CentroidTracker
from .visualization import annotate


class AssemblyPipeline:
    def __init__(self, config: dict):
        self.config = config

    def process(self, video_path: str | Path, output_path: str | Path | None = None,
                keep_frames: bool = False) -> dict:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        detector = ColorMarkerDetector(**self.config["detector"])
        tracker = CentroidTracker(self.config["state_engine"]["missing_tolerance"])
        engine = AssemblyStateEngine(self.config["state_engine"], fps)
        writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        results: list[FrameResult] = []
        preview_frames = []
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            started = time.perf_counter()
            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            observed_state = engine.state
            _, confidence, _ = engine.update(tracks, index)
            latency = (time.perf_counter() - started) * 1000
            result = FrameResult(index, index / fps, observed_state, confidence, detections, latency)
            results.append(result)
            rendered = annotate(frame, detections, observed_state, confidence)
            if writer:
                writer.write(rendered)
            if keep_frames and index % max(1, int(fps)) == 0:
                preview_frames.append(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
            index += 1
        capture.release()
        if writer:
            writer.release()
        mean_latency = sum(r.latency_ms for r in results) / max(len(results), 1)
        return {"frames": results, "events": engine.events, "fps": fps,
                "mean_latency_ms": mean_latency, "processing_fps": 1000 / max(mean_latency, 0.001),
                "completed": engine.finished, "preview_frames": preview_frames}
