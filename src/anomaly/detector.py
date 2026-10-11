
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.05,
) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        raise ValueError("Dataset has no numeric columns.")

    if len(numeric_df) < 5:
        raise ValueError("At least 5 rows are required.")

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
    )

    predictions = model.fit_predict(numeric_df)

    result = df.copy()
    result["anomaly"] = predictions == -1

    return result
