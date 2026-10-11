
import json
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.quality.pipeline import run_quality_pipeline


rules = {
    "customer_id": {"required": True, "unique": True},
    "age": {"min": 0, "max": 120},
    "income": {"min": 0},
    "city": {
        "allowed_values": [
            "Tunis",
            "Sousse",
            "Sfax",
            "Monastir",
        ]
    },
}

# Create a test dataset with known problems.
df = pd.DataFrame({
    "customer_id": [1, 2, 2, 4],
    "age": [25, -3, 130, 40],
    "income": [1200, -50, 2500, 3000],
    "city": ["Tunis", "Sousse", "Unknown", "Sfax"],
})

df.to_csv("data/pipeline_test.csv", index=False)

report = run_quality_pipeline(
    "data/pipeline_test.csv",
    rules,
)

print(json.dumps(report, indent=2))

assert report["rows"] == 4
assert report["columns"] == 4
assert report["total_issues"] > 0
assert report["high_severity_issues"] > 0

with open("reports/quality_report.json", "w", encoding="utf-8") as file:
    json.dump(report, file, indent=2, ensure_ascii=False)

print("\nPASS: Unified quality pipeline works")
print("Report saved to reports/quality_report.json")
