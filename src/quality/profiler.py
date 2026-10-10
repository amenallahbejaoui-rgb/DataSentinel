
from pathlib import Path

import pandas as pd


def profile_csv(file_path: str) -> dict:
    """Profile a CSV dataset and report common data quality issues."""

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The dataset contains no rows.")

    if len(df.columns) == 0:
        raise ValueError("The dataset contains no columns.")

    rows, columns = df.shape

    missing_by_column = df.isna().sum()
    duplicate_rows = int(df.duplicated().sum())

    column_details = {}

    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())

        column_details[str(column)] = {
            "data_type": str(series.dtype),
            "missing_count": missing,
            "missing_percentage": round(missing / rows * 100, 2),
            "unique_values": int(series.nunique()),
        }

    missing_cells = int(missing_by_column.sum())
    total_cells = rows * columns

    completeness = (
        round((1 - missing_cells / total_cells) * 100, 2)
        if total_cells
        else 100.0
    )

    duplicate_percentage = round(duplicate_rows / rows * 100, 2)

    issues = []

    for column, details in column_details.items():
        if details["missing_count"] > 0:
            issues.append({
                "type": "missing_values",
                "column": column,
                "count": details["missing_count"],
                "severity": (
                    "high"
                    if details["missing_percentage"] >= 20
                    else "medium"
                ),
            })

    if duplicate_rows > 0:
        issues.append({
            "type": "duplicate_rows",
            "column": None,
            "count": duplicate_rows,
            "severity": "medium",
        })

    return {
        "file_name": path.name,
        "rows": rows,
        "columns": columns,
        "completeness_percentage": completeness,
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": duplicate_percentage,
        "column_details": column_details,
        "issues": issues,
    }
