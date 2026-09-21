from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


INTERVALS = [
    {"label": "pick_component", "start_s": 0.0, "end_s": 2.0},
    {"label": "position_component", "start_s": 2.0, "end_s": 4.0},
    {"label": "insert_component", "start_s": 4.0, "end_s": 6.0},
    {"label": "fasten", "start_s": 6.0, "end_s": 8.0},
    {"label": "inspect", "start_s": 8.0, "end_s": 10.0},
    {"label": "completed", "start_s": 10.0, "end_s": 12.0},
]


def generate_demo(video_path: str | Path, annotation_path: str | Path, fps: int = 20,
                  brightness: float = 1.0, occlusion: bool = False, shift: int = 0) -> None:
    video_path, annotation_path = Path(video_path), Path(annotation_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    size = (640, 360)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    total = int(INTERVALS[-1]["end_s"] * fps)
    for frame_idx in range(total):
        t = frame_idx / fps
        frame = np.full((size[1], size[0], 3), 38, dtype=np.uint8)
        cv2.rectangle(frame, (60, 80), (580, 310), (65, 65, 65), -1)
        socket = (430 + shift, 195)
        cv2.circle(frame, socket, 36, (0, 205, 0), -1)
        if t < 4:
            progress = min(t / 4, 1)
            component = (int(130 + progress * 300) + shift, int(235 - progress * 40))
        else:
            component = socket
        cv2.rectangle(frame, (component[0] - 23, component[1] - 18), (component[0] + 23, component[1] + 18), (255, 80, 0), -1)
        if 6 <= t < 12:
            cv2.circle(frame, (socket[0] + 12, socket[1] - 8), 14, (0, 0, 255), -1)
        if 8 <= t < 12:
            cv2.rectangle(frame, (500 + shift, 105), (550 + shift, 135), (0, 230, 255), -1)
        if occlusion and 3.0 < t < 3.7:
            cv2.rectangle(frame, (350, 120), (500, 280), (42, 42, 42), -1)
        frame = cv2.convertScaleAbs(frame, alpha=brightness, beta=0)
        cv2.putText(frame, INTERVALS[min(int(t // 2), 5)]["label"].replace("_", " ").title(), (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
        writer.write(frame)
    writer.release()
    annotation_path.write_text(json.dumps({"fps": fps, "frames": total, "intervals": INTERVALS}, indent=2), encoding="utf-8")
