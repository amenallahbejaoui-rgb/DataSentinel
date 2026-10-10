
import pandas as pd


def validate_dataset(
    df: pd.DataFrame,
    rules: dict,
) -> list[dict]:
    """Validate a DataFrame against configurable quality rules."""

    issues = []

    for column, config in rules.items():
        if column not in df.columns:
            issues.append({
                "type": "missing_column",
                "column": column,
                "count": 1,
                "severity": "high",
            })
            continue

        series = df[column]

        if config.get("required", False):
            count = int(series.isna().sum())

            if count:
                issues.append({
                    "type": "required_value_missing",
                    "column": column,
                    "count": count,
                    "severity": "high",
                })

        if "min" in config or "max" in config:
            numeric = pd.to_numeric(series, errors="coerce")

            invalid = series.notna() & numeric.isna()

            if "min" in config:
                invalid |= numeric.notna() & (numeric < config["min"])

            if "max" in config:
                invalid |= numeric.notna() & (numeric > config["max"])

            count = int(invalid.sum())

            if count:
                issues.append({
                    "type": "out_of_range",
                    "column": column,
                    "count": count,
                    "severity": "high",
                })

        if "allowed_values" in config:
            invalid = (
                series.notna()
                & ~series.isin(config["allowed_values"])
            )

            count = int(invalid.sum())

            if count:
                issues.append({
                    "type": "invalid_category",
                    "column": column,
                    "count": count,
                    "severity": "medium",
                })

        if config.get("unique", False):
            duplicates = int(
                series.notna().sum() - series.dropna().nunique()
            )

            if duplicates:
                issues.append({
                    "type": "non_unique_value",
                    "column": column,
                    "count": duplicates,
                    "severity": "high",
                })

    return issues
