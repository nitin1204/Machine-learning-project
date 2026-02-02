"""
Cuisine Classification
----------------------
Objective: Classify restaurants by their primary cuisine.

Steps:
1. Handle missing values and encode categorical variables.
2. Split into training/testing data.
3. Train a RandomForestClassifier.
4. Evaluate with accuracy, precision, recall, and F1 score.
5. Analyze feature importance and confusion matrix.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# ----------------------------------------------------------------
# Load and preprocess
# ----------------------------------------------------------------
df = pd.read_csv("dataset.csv")
print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

df["Cuisines"] = df["Cuisines"].fillna("Unknown")
df["Primary_Cuisine"] = df["Cuisines"].apply(lambda x: x.split(",")[0].strip())

drop_cols = [
    "Restaurant ID", "Restaurant Name", "Address", "Locality",
    "Locality Verbose", "Rating color", "Rating text", "Cuisines"
]
df.drop(columns=drop_cols, inplace=True, errors="ignore")

binary_cols = ["Has Table booking", "Has Online delivery", "Is delivering now", "Switch to order menu"]
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0)

categorical_cols = ["City", "Currency"]
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# Remove cuisines with only one record
counts = df["Primary_Cuisine"].value_counts()
df = df[df["Primary_Cuisine"].isin(counts[counts > 1].index)]

# ----------------------------------------------------------------
# Features / target
# ----------------------------------------------------------------
X = df.drop(columns=["Primary_Cuisine", "Aggregate rating"], errors="ignore")
y = df["Primary_Cuisine"]

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ----------------------------------------------------------------
# Train model
# ----------------------------------------------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ----------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\nModel Performance")
print("-----------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ----------------------------------------------------------------
# Detailed report
# ----------------------------------------------------------------
print("\nClassification Report:")
labels_in_test = sorted(list(set(y_test)))
names_in_test = le_target.inverse_transform(labels_in_test)

report = classification_report(
    y_test,
    y_pred,
    labels=labels_in_test,
    target_names=names_in_test,
    zero_division=0
)
print(report)

# ----------------------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, cmap="Blues", xticklabels=le_target.classes_, yticklabels=le_target.classes_)
plt.title("Confusion Matrix - Cuisine Classification")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------
# Feature Importance
# ----------------------------------------------------------------
feature_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importances.head(10))
feature_importances.to_csv("feature_importance_cuisine.csv", index=False)
print("\nFeature importance saved to feature_importance_cuisine.csv")
