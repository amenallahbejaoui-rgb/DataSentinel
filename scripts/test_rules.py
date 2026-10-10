
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.quality.rules import validate_dataset


df = pd.DataFrame({
    "customer_id": [1, 2, 2, 4],
    "age": [25, -3, 130, 40],
    "income": [1200, -50, 2500, 3000],
    "city": ["Tunis", "Sousse", "Unknown", "Sfax"],
})

rules = {
    "customer_id": {"required": True, "unique": True},
    "age": {"min": 0, "max": 120},
    "income": {"min": 0},
    "city": {
        "allowed_values": ["Tunis", "Sousse", "Sfax", "Monastir"]
    },
}

issues = validate_dataset(df, rules)

for issue in issues:
    print(issue)

assert any(i["type"] == "out_of_range" and i["column"] == "age" for i in issues)
assert any(i["type"] == "out_of_range" and i["column"] == "income" for i in issues)
assert any(i["type"] == "invalid_category" for i in issues)
assert any(i["type"] == "non_unique_value" for i in issues)

print("\nPASS: Quality rules engine works")
