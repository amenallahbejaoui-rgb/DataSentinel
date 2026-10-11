
from pathlib import Path
import pandas as pd

from src.quality.profiler import profile_csv
from src.quality.rules import validate_dataset
from src.anomaly.detector import detect_anomalies


def run_quality_pipeline(file_path: str, rules: dict) -> dict:
    path = Path(file_path)

    report = profile_csv(str(path))
    df = pd.read_csv(path)

    rule_issues = validate_dataset(df, rules)
    all_issues = report["issues"] + rule_issues

    anomaly_count = 0

    # Run anomaly detection only when enough rows are available.
    if len(df) >= 5:
        result = detect_anomalies(df)
        anomaly_count = int(result["anomaly"].sum())

        if anomaly_count:
            all_issues.append({
                "type": "anomalous_rows",
                "column": None,
                "count": anomaly_count,
                "severity": "medium",
            })

    return {
        "file_name": report["file_name"],
        "rows": report["rows"],
        "columns": report["columns"],
        "completeness_percentage": report["completeness_percentage"],
        "duplicate_rows": report["duplicate_rows"],
        "column_details": report["column_details"],
        "total_issues": len(all_issues),
        "high_severity_issues": sum(
            issue["severity"] == "high" for issue in all_issues
        ),
        "medium_severity_issues": sum(
            issue["severity"] == "medium" for issue in all_issues
        ),
        "anomalous_rows": anomaly_count,
        "issues": all_issues,
    }
