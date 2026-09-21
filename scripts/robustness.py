from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from assemblyvision.config import load_config
from assemblyvision.pipeline import AssemblyPipeline
from assemblyvision.synthetic import generate_demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="reports/robustness.csv")
    args = parser.parse_args()
    scenarios = [("baseline", 1.0, False, 0), ("low_light", 0.55, False, 0),
                 ("occlusion", 1.0, True, 0), ("camera_shift", 1.0, False, -40)]
    rows = []
    with tempfile.TemporaryDirectory() as temp:
        for name, brightness, occlusion, shift in scenarios:
            video, labels = Path(temp) / f"{name}.mp4", Path(temp) / f"{name}.json"
            generate_demo(video, labels, brightness=brightness, occlusion=occlusion, shift=shift)
            result = AssemblyPipeline(load_config(args.config)).process(video)
            rows.append({"scenario": name, "completed": result["completed"], "events": len(result["events"]),
                         "latency_ms": result["mean_latency_ms"], "processing_fps": result["processing_fps"]})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
