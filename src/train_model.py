import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

# Load dataset
df = pd.read_csv("data/operations_data.csv")

# Features
X = df[
    [
        "inventory",
        "demand",
        "supplier_delay",
        "delivery_delay",
        "workload"
    ]
]

# Target
y = df["failure_risk"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model
model = LogisticRegression()

# Train
model.fit(X_train, y_train)

print("\nLearned coefficients:")

coefficients = pd.Series(
    model.coef_[0],
    index=X.columns
).sort_values()

print(coefficients)
print("Intercept:", model.intercept_[0])

# Predict
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))