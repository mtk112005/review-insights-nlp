"""End-to-end run. Stages can be selected: python run_pipeline.py data baselines transformers topics aspects errors"""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
warnings.filterwarnings("ignore")
import os
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from reviewnlp import aspects, baselines, data, error_analysis, topics, transformers_models  # noqa: E402

STAGES = {
    "data": data.load,
    "baselines": baselines.run,
    "transformers": transformers_models.run,   # ~40 min on a laptop CPU (fine-tuning dominates)
    "topics": topics.run,
    "aspects": aspects.run,
    "errors": error_analysis.run,
}

if __name__ == "__main__":
    for name in sys.argv[1:] or STAGES:
        t = time.time()
        print(f"=== {name}")
        STAGES[name]()
        print(f"    {name} done in {time.time() - t:.0f}s")
