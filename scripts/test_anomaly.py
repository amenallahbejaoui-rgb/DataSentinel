import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly.detector import detect_anomalies
from src.quality.pipeline import run_quality_pipeline


df = pd.DataFrame({
    "age": [25, 28, 32, 35, 30, 29, 31, 27, 33, 150],
    "income": [2000, 2200, 2500, 2300, 2100, 2400, 2600, 2200, 2500, 50000],
})

result = detect_anomalies(df, contamination=0.1)

print(result)
print("\nFlagged rows:")
print(result[result["anomaly"]])

rules = {
    "age": {"min": 0, "max": 120},
    "income": {"min": 0},
}

df.to_csv("data/anomaly_test.csv", index=False)

report = run_quality_pipeline("data/anomaly_test.csv", rules)

print("\nPipeline anomaly count:", report["anomalous_rows"])
print("Total issues:", report["total_issues"])