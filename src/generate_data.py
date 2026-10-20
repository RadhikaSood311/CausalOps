import pandas as pd
import numpy as np

np.random.seed(42)

n = 5000

df = pd.DataFrame({
    "store_id": np.random.randint(1, 11, n),
    "product_id": np.random.randint(1, 21, n),
    "inventory": np.random.randint(20, 500, n),
    "demand": np.random.randint(5, 80, n),
    "supplier_delay": np.random.randint(0, 24, n),
    "delivery_delay": np.random.randint(0, 12, n),
    "workload": np.random.randint(20, 100, n)
})

print(df.head())
print(df.shape)

base_demand = np.random.randint(20, 60, n)

df["demand"] = (
    base_demand
    + df["workload"] * 0.2
    + np.random.normal(0, 5, n)
).astype(int)

df["inventory"] = (
    np.random.randint(50, 300, n)
    - df["demand"] * 2
    + np.random.normal(0, 20, n)
).clip(5, 500).astype(int)

risk_score = (
    -0.03 * df["inventory"]
    + 0.05 * df["demand"]
    + 0.04 * df["supplier_delay"]
    + 0.06 * df["delivery_delay"]
    + 0.03 * df["workload"]
    + np.random.normal(0, 2, n)
)

df["failure_risk"] = (risk_score > risk_score.median()).astype(int)


print("\nDataset info:")
print(df.info())

print("\nStatistics:")
print(df.describe())

print("\nFailure risk distribution:")
print(df["failure_risk"].value_counts())

print("\nFailure risk percentage:")
print(df["failure_risk"].value_counts(normalize=True))

print("\nCorrelation with failure risk:")
print(
    df.corr(numeric_only=True)["failure_risk"]
    .sort_values(ascending=False)
)

df.to_csv("data/operations_data.csv", index=False)