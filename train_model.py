# train_model.py
# Train an SVM model on the Pima Indians Diabetes dataset and save artifacts for the Streamlit app.

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = Path("diabetes.csv")  # ensure this file is in the same folder
RANDOM_STATE = 42
TEST_SIZE = 0.2

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Could not find diabetes.csv in the current folder. Please place it here.")

    df = pd.read_csv(DATA_PATH)

    expected_cols = [
        "Pregnancies","Glucose","BloodPressure","SkinThickness",
        "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    X = df.drop(columns="Outcome")
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Fit scaler ONLY on training set to avoid data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    print("Train accuracy:", round(accuracy_score(y_train, y_pred_train), 4))
    print("Test accuracy:", round(accuracy_score(y_test, y_pred_test), 4))
    print("\nClassification report (test):\n", classification_report(y_test, y_pred_test))
    print("Confusion matrix (test):\n", confusion_matrix(y_test, y_pred_test))

    # Save artifacts
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    with open("feature_names.json", "w") as f:
        json.dump(list(X.columns), f)

    print("\nSaved: model.pkl, scaler.pkl, feature_names.json")

if __name__ == "__main__":
    main()
