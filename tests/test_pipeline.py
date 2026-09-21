from pathlib import Path

from assemblyvision.config import load_config
from assemblyvision.pipeline import AssemblyPipeline
from assemblyvision.synthetic import generate_demo


def test_demo_pipeline(tmp_path: Path):
    video, labels = tmp_path / "demo.mp4", tmp_path / "demo.json"
    generate_demo(video, labels, fps=10)
    root = Path(__file__).resolve().parents[1]
    result = AssemblyPipeline(load_config(root / "configs/default.yaml")).process(video)
    assert len(result["frames"]) == 120
    assert len(result["events"]) >= 5
    assert result["mean_latency_ms"] > 0
