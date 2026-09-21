import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assemblyvision.synthetic import generate_demo

ROOT = Path(__file__).resolve().parents[1]
generate_demo(ROOT / "data/samples/assembly_demo.mp4", ROOT / "data/samples/assembly_demo.json")
print("Created data/samples/assembly_demo.mp4 and assembly_demo.json")
