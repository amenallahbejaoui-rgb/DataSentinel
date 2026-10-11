
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Generate reproducible synthetic data.
normal = pd.DataFrame({
    "age": [25 + (i % 15) for i in range(100)],
    "income": [1800 + (i % 20) * 100 for i in range(100)],
})

# Add five deliberately unusual observations.
anomalies = pd.DataFrame({
    "age": [150, 10, 80, 22, 45],
    "income": [50000, 200, 60000, 90000, 100],
})

df = pd.concat([normal, anomalies], ignore_index=True)
actual_anomaly = [False] * len(normal) + [True] * len(anomalies)

# Model 1: Isolation Forest
isolation = IsolationForest(
    contamination=5 / len(df),
    random_state=42,
)
if_predictions = isolation.fit_predict(df) == -1

# Model 2: Local Outlier Factor
lof = LocalOutlierFactor(
    contamination=5 / len(df),
)
lof_predictions = lof.fit_predict(df) == -1


def show_results(name, predictions):
    actual = pd.Series(actual_anomaly)
    predicted = pd.Series(predictions)

    true_positives = int((actual & predicted).sum())
    false_positives = int((~actual & predicted).sum())
    missed = int((actual & ~predicted).sum())

    print(f"\n{name}")
    print(f"Correctly flagged anomalies: {true_positives}/5")
    print(f"Normal rows incorrectly flagged: {false_positives}")
    print(f"Anomalies missed: {missed}")


show_results("Isolation Forest", if_predictions)
show_results("Local Outlier Factor", lof_predictions)
