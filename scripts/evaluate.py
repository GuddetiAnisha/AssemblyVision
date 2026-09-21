from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assemblyvision.config import load_config
from assemblyvision.evaluation import classification_metrics, labels_from_intervals
from assemblyvision.pipeline import AssemblyPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AssemblyVision on an annotated video")
    parser.add_argument("--video", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="reports/metrics.json")
    args = parser.parse_args()
    annotations = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    result = AssemblyPipeline(load_config(args.config)).process(args.video)
    predicted = [item.state for item in result["frames"]]
    truth = labels_from_intervals(annotations["intervals"], len(predicted), result["fps"])
    metrics = classification_metrics(truth, predicted)
    metrics.update({"mean_latency_ms": result["mean_latency_ms"], "processing_fps": result["processing_fps"],
                    "events": [event.to_dict() for event in result["events"]]})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                      "mean_latency_ms": metrics["mean_latency_ms"]}, indent=2))


if __name__ == "__main__":
    main()
