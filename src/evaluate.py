# src/evaluate.py

import json
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load Dataset
# -----------------------------
data_path = "../data/processed/cleaned.csv"

print("Loading dataset...")
df = pd.read_csv(data_path)

# -----------------------------
# Data Cleaning
# -----------------------------
target_column = "Churn"

# Remove empty rows
df.dropna(how="all", inplace=True)

# -----------------------------
# Encode Categorical Columns
# -----------------------------
for column in df.columns:
    if df[column].dtype == "object":
        df[column] = df[column].astype("category").cat.codes

# -----------------------------
# Remove Same Columns as Training
# -----------------------------
drop_columns = [target_column]

if "customerID" in df.columns:
    drop_columns.append("customerID")

X = df.drop(columns=drop_columns)
y = df[target_column]

# -----------------------------
# Load Model
# -----------------------------
print("Loading model...")

model = joblib.load("model/churn_model.pkl")

# -----------------------------
# Predictions
# -----------------------------
print("Running evaluation...")

predictions = model.predict(X)

# -----------------------------
# Metrics
# -----------------------------
accuracy = accuracy_score(y, predictions)

report = classification_report(y, predictions, output_dict=True)

metrics = {
    "accuracy": accuracy,
    "precision_class_0": report["0"]["precision"],
    "precision_class_1": report["1"]["precision"],
    "recall_class_0": report["0"]["recall"],
    "recall_class_1": report["1"]["recall"],
    "f1_score_class_0": report["0"]["f1-score"],
    "f1_score_class_1": report["1"]["f1-score"]
}

# -----------------------------
# Save Metrics
# -----------------------------
print("Saving metrics...")

with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Evaluation Complete")
print(metrics)