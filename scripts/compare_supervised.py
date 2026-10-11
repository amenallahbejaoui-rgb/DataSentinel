
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)

# Load real transaction data.
df = pd.read_csv("data/creditcard.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

# Keep the same test transactions for both approaches.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

print("Training transactions:", len(X_train))
print("Test transactions:", len(X_test))
print("Fraud cases in test set:", int(y_test.sum()))

# Scale features using training data only.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Isolation Forest: learns patterns without using fraud labels.
iso = IsolationForest(
    contamination=float(y_train.mean()),
    random_state=42,
    n_estimators=100,
    n_jobs=-1,
)
iso.fit(X_train_scaled)

iso_predictions = (iso.predict(X_test_scaled) == -1).astype(int)
iso_scores = -iso.score_samples(X_test_scaled)

# 2. Random Forest: learns from labeled training transactions.
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

rf_predictions = rf.predict(X_test)
rf_scores = rf.predict_proba(X_test)[:, 1]


def evaluate(name, predictions, scores):
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(y_test, predictions, zero_division=0)
    average_precision = average_precision_score(y_test, scores)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    print(f"\n{name}")
    print(f"Fraud detected: {tp}/{tp + fn}")
    print(f"False alarms: {fp}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Average precision: {average_precision:.4f}")


evaluate("Isolation Forest", iso_predictions, iso_scores)
evaluate("Random Forest", rf_predictions, rf_scores)

# Save a summary of the evaluation for later comparison.
results = []

for name, predictions, scores in [
    ("Isolation Forest", iso_predictions, iso_scores),
    ("Random Forest", rf_predictions, rf_scores),
]:
    results.append({
        "model": name,
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "average_precision": average_precision_score(y_test, scores),
    })

results_df = pd.DataFrame(results)
results_df.to_csv("reports/model_comparison.csv", index=False)

print("\nSaved evaluation to reports/model_comparison.csv")
print(results_df.round(4).to_string(index=False))
