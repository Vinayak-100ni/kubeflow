# src/train.py

import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import joblib

# -----------------------------
# MLflow Setup
# -----------------------------
mlflow.set_experiment("customer-churn-training")

# -----------------------------
# Load Dataset
# -----------------------------
data_path = "../data/processed/cleaned.csv"

print("Loading dataset...")
df = pd.read_csv(data_path)

print("Dataset Shape:", df.shape)

# -----------------------------
# Data Cleaning
# -----------------------------
print("Cleaning data...")

# Remove fully empty rows
df.dropna(how="all", inplace=True)

# Target column
target_column = "Churn"

# -----------------------------
# Encode Categorical Columns
# -----------------------------
print("Encoding categorical columns...")

label_encoders = {}

for column in df.columns:
    if df[column].dtype == "object":
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column].astype(str))
        label_encoders[column] = encoder

# -----------------------------
# Remove Unwanted Columns
# -----------------------------
print("Removing unnecessary columns...")

drop_columns = [target_column]

# Remove customerID if present
if "customerID" in df.columns:
    drop_columns.append("customerID")

X = df.drop(columns=drop_columns)
y = df[target_column]

print("Features Used:")
print(X.columns.tolist())

# -----------------------------
# Handle Missing Values
# -----------------------------
print("Handling missing values...")

imputer = SimpleImputer(strategy="mean")

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)

# -----------------------------
# Train Test Split
# -----------------------------
print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Start MLflow Run
# -----------------------------
with mlflow.start_run():

    print("Training model...")

    # Model
    model = LogisticRegression(max_iter=1000)

    # Train Model
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy}")

    # -----------------------------
    # MLflow Logging
    # -----------------------------
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("max_iter", 1000)

    mlflow.log_metric("accuracy", accuracy)

    # Log model to MLflow
    mlflow.sklearn.log_model(model, "model")

    # -----------------------------
    # Save Model Locally
    # -----------------------------
    print("Saving model...")

    os.makedirs("model", exist_ok=True)

    model_path = "model/churn_model.pkl"

    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")

    # -----------------------------
    # Classification Report
    # -----------------------------
    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

print("Training Complete")