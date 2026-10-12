
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
    anomaly_records = []
    anomaly_status = "skipped"

    numeric_columns = df.select_dtypes(include="number").columns

    # Isolation Forest requires at least five rows and one numeric column.
    if len(df) >= 5 and len(numeric_columns) > 0:
        result = detect_anomalies(df)
        flagged = result[result["anomaly"]].drop(columns=["anomaly"])
        anomaly_count = len(flagged)
        anomaly_records = flagged.to_dict(orient="records")
        anomaly_status = "completed"

        if anomaly_count:
            all_issues.append({
                "type": "anomalous_rows",
                "column": None,
                "count": anomaly_count,
                "severity": "medium",
            })
    elif len(df) >= 5:
        anomaly_status = "skipped"

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
        "anomaly_status": anomaly_status,
        "anomaly_records": anomaly_records,
        "issues": all_issues,
    }
