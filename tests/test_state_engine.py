from assemblyvision.state_engine import AssemblyStateEngine


CONFIG = {"steps": ["pick_component", "position_component", "insert_component", "fasten", "inspect", "completed"],
          "dwell_frames": 2, "proximity_px": 50, "motion_threshold_px": 5}


def track(center, speed=0, confidence=0.9):
    return {"center": center, "speed": speed, "confidence": confidence, "visible": True}


def feed(engine, tracks, start):
    event = None
    for frame in range(start, start + 2):
        _, _, event = engine.update(tracks, frame)
    return event


def test_full_ordered_sequence():
    engine = AssemblyStateEngine(CONFIG, fps=10)
    assert feed(engine, {"component": track((10, 10), 8)}, 0).step == "pick_component"
    near = {"component": track((100, 100), 7), "socket": track((105, 105))}
    assert feed(engine, near, 2).step == "position_component"
    near["component"]["speed"] = 0
    assert feed(engine, near, 4).step == "insert_component"
    near["fastener"] = track((102, 102))
    assert feed(engine, near, 6).step == "fasten"
    near["inspection"] = track((150, 80))
    assert feed(engine, near, 8).step == "inspect"
    assert engine.state == "completed"
