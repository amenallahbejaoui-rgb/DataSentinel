
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)

# 200 normal business transactions
normal = pd.DataFrame({
    "customer_monthly_income": rng.integers(1000, 6001, 200),
    "transaction_hour": rng.integers(8, 22, 200),
    "transactions_last_24h": rng.integers(1, 5, 200),
})

# Transaction amount depends partly on customer income.
normal["transaction_amount"] = (
    normal["customer_monthly_income"]
    * rng.uniform(0.02, 0.18, 200)
).round(2)

# 10 unusual combinations, not just huge amounts.
unusual = pd.DataFrame({
    "customer_monthly_income": [5000, 1200, 4000, 2500, 5500,
                                1800, 3000, 4500, 1000, 3500],
    "transaction_hour": [2, 3, 23, 1, 0, 23, 2, 4, 1, 0],
    "transactions_last_24h": [4, 4, 4, 3, 4, 4, 3, 4, 4, 3],
    "transaction_amount": [800, 350, 700, 450, 950,
                           400, 600, 850, 250, 650],
})

df = pd.concat([normal, unusual], ignore_index=True)
actual = np.array([0] * len(normal) + [1] * len(unusual))

# Scale features so large-valued columns don't dominate LOF.
scaler = StandardScaler()
X = scaler.fit_transform(df)

contamination = len(unusual) / len(df)

models = {
    "Isolation Forest": IsolationForest(
        contamination=contamination,
        random_state=42,
    ),
    "Local Outlier Factor": LocalOutlierFactor(
        contamination=contamination,
    ),
}

for name, model in models.items():
    predictions = model.fit_predict(X) == -1

    true_positives = int(np.sum(predictions & (actual == 1)))
    false_positives = int(np.sum(predictions & (actual == 0)))
    missed = int(np.sum(~predictions & (actual == 1)))

    print(f"\n{name}")
    print(f"Correctly flagged: {true_positives}/10")
    print(f"Normal rows flagged: {false_positives}")
    print(f"Anomalies missed: {missed}")

    print("Flagged examples:")
    print(df.loc[predictions].tail(5).round(2).to_string(index=False))
