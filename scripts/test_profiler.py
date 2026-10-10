
import sys
from pathlib import Path
from pprint import pprint

# Make the project package importable when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.quality.profiler import profile_csv


if __name__ == "__main__":
    report = profile_csv(PROJECT_ROOT / "data" / "sample.csv")
    pprint(report, sort_dicts=False)
