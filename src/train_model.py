import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.dummy import DummyClassifier


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


probabilities = model.predict_proba(X_test)[:, 1]

for threshold in [0.3, 0.5, 0.7]:
    y_pred = (probabilities >= threshold).astype(int)

    print(f"\n{'=' * 30}")
    print(f"Threshold: {threshold}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))



# ==========================================
# PROBABILITY CALIBRATION
# ==========================================

probabilities = model.predict_proba(X_test)[:, 1]

brier = brier_score_loss(y_test, probabilities)

fraction_positive, mean_predicted = calibration_curve(
    y_test,
    probabilities,
    n_bins=10,
    strategy="uniform"
)

print("\n========== CALIBRATION ==========")
print("Brier Score:", brier)

print("\nMean predicted probability:", mean_predicted)
print("Actual positive fraction:", fraction_positive)




from sklearn.dummy import DummyClassifier

baseline = DummyClassifier(strategy="prior")
baseline.fit(X_train, y_train)

baseline_probabilities = baseline.predict_proba(X_test)[:, 1]

print(
    "Baseline Brier Score:",
    brier_score_loss(y_test, baseline_probabilities)
)

# Simple baseline model
baseline = DummyClassifier(strategy="prior")
baseline.fit(X_train, y_train)

baseline_probabilities = baseline.predict_proba(X_test)[:, 1]

print(
    "Baseline Brier Score:",
    brier_score_loss(y_test, baseline_probabilities)
)







# ==========================================
# MODEL 2: DECISION TREE CLASSIFIER
# ==========================================

# Create the model
# max_depth=5 limits tree complexity to reduce overfitting
tree_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

# Train the Decision Tree on the same training data
tree_model.fit(X_train, y_train)

# Predict classes on the same test data
tree_pred = tree_model.predict(X_test)

# Evaluate Decision Tree
print("\n========== DECISION TREE RESULTS ==========")

print("Accuracy:", accuracy_score(y_test, tree_pred))

print("\nClassification Report:")
print(classification_report(y_test, tree_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, tree_pred))


# Compare training and test performance
tree_train_pred = tree_model.predict(X_train)

print("\nDecision Tree Training Accuracy:",
      accuracy_score(y_train, tree_train_pred))

print("Decision Tree Test Accuracy:",
      accuracy_score(y_test, tree_pred))


# ==========================================
# CROSS-VALIDATION: MODEL COMPARISON
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

logistic_scores = cross_val_score(
    LogisticRegression(),
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)

tree_scores = cross_val_score(
    DecisionTreeClassifier(max_depth=5, random_state=42),
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)

print("\n========== 5-FOLD CROSS-VALIDATION ==========")

print("Logistic Regression fold scores:", logistic_scores)
print("Logistic Regression mean:", logistic_scores.mean())
print("Logistic Regression std:", logistic_scores.std())

print("\nDecision Tree fold scores:", tree_scores)
print("Decision Tree mean:", tree_scores.mean())
print("Decision Tree std:", tree_scores.std())


# ==========================================
# ERROR ANALYSIS
# ==========================================

error_analysis = X_test.copy()

error_analysis["actual"] = y_test
error_analysis["predicted"] = model.predict(X_test)
error_analysis["risk_probability"] = model.predict_proba(X_test)[:, 1]

# Incorrect predictions only
errors = error_analysis[
    error_analysis["actual"] != error_analysis["predicted"]
].copy()

errors["error_type"] = errors.apply(
    lambda row: (
        "False Negative"
        if row["actual"] == 1
        else "False Positive"
    ),
    axis=1
)

print("\n========== ERROR ANALYSIS ==========")
print("Total errors:", len(errors))
print("\nError types:")
print(errors["error_type"].value_counts())

print("\nSample incorrect predictions:")
print(errors.head(10).to_string(index=False))