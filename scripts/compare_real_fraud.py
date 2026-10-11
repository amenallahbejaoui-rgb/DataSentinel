
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

# Load the real dataset and take a reproducible random sample.
df = pd.read_csv("data/creditcard.csv")
sample = df.sample(n=30000, random_state=42).copy()

# Keep the true labels for evaluation only.
y_true = sample["Class"].to_numpy()

# Never give the fraud labels to the models.
X = sample.drop(columns=["Class"])

# Scale features so Amount does not dominate distance calculations.
X_scaled = StandardScaler().fit_transform(X)

# Use the dataset's overall fraud rate as the assumed anomaly rate.
contamination = df["Class"].mean()

models = {
    "Isolation Forest": IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
        n_jobs=-1,
    ),
    "Local Outlier Factor": LocalOutlierFactor(
        contamination=contamination,
        n_neighbors=20,
        n_jobs=-1,
    ),
}

print("Sample rows:", len(sample))
print("Actual fraud cases:", int(y_true.sum()))
print("Assumed anomaly rate:", round(contamination, 5))

for name, model in models.items():
    print(f"\nRunning {name}...")

    predictions = model.fit_predict(X_scaled) == -1

    # Convert model predictions into 0 = normal, 1 = anomaly.
    y_pred = predictions.astype(int)

    # Higher scores indicate more anomalous observations.
    if name == "Isolation Forest":
        scores = -model.score_samples(X_scaled)
    else:
        scores = -model.negative_outlier_factor_

    print(f"Fraud detected: {int(((y_true == 1) & (y_pred == 1)).sum())}")
    print(f"False alarms: {int(((y_true == 0) & (y_pred == 1)).sum())}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-score: {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Average precision: {average_precision_score(y_true, scores):.4f}")
