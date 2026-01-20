"""Quick script to train and save the model locally."""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib

DATA_PATH = project_root / "data" / "developer_stress.csv"
MODEL_PATH = project_root / "models" / "stress_model.joblib"
METRICS_PATH = project_root / "models" / "metrics.json"

FEATURE_COLUMNS = [
    "Hours_Worked",
    "Sleep_Hours",
    "Bugs",
    "Deadline_Days",
    "Coffee_Cups",
    "Meetings",
    "Interruptions",
    "Experience_Years",
    "Code_Complexity",
    "Remote_Work",
]

EXPERIENCE_MAP = {"Junior": 0, "Mid": 1, "Senior": 2}
COMPLEXITY_MAP = {"Low": 0, "Medium": 1, "High": 2}
REMOTE_MAP = {"No": 0, "Yes": 1}

MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_leaf": 2,
    "random_state": 42,
}


def main():
    print("=" * 60)
    print("Developer Stress Prediction - Model Training")
    print("=" * 60)

    print(f"\nLoading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} records")

    print("\nEncoding categorical variables...")
    df["Experience_Years"] = df["Experience_Years"].map(EXPERIENCE_MAP)
    df["Code_Complexity"] = df["Code_Complexity"].map(COMPLEXITY_MAP)
    df["Remote_Work"] = df["Remote_Work"].map(REMOTE_MAP)

    X = df[FEATURE_COLUMNS].values
    y = df["Stress_Level"].values

    print("\nSplitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    print("\nTraining RandomForestRegressor...")
    print(f"Parameters: {MODEL_PARAMS}")
    model = RandomForestRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    print("Training complete!")

    print("\nEvaluating model...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        "train_r2": float(r2_score(y_train, y_train_pred)),
        "train_mse": float(mean_squared_error(y_train, y_train_pred)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
        "test_r2": float(r2_score(y_test, y_test_pred)),
        "test_mse": float(mean_squared_error(y_test, y_test_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
    }

    print("\n" + "-" * 40)
    print("Training Metrics:")
    print("-" * 40)
    print(f"  Train R²:   {metrics['train_r2']:.4f}")
    print(f"  Train MSE:  {metrics['train_mse']:.2f}")
    print(f"  Train RMSE: {metrics['train_rmse']:.2f}")
    print("-" * 40)
    print("Test Metrics:")
    print("-" * 40)
    print(f"  Test R²:    {metrics['test_r2']:.4f}")
    print(f"  Test MSE:   {metrics['test_mse']:.2f}")
    print(f"  Test RMSE:  {metrics['test_rmse']:.2f}")
    print("-" * 40)

    print("\nCross-validation (5-fold)...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    metrics["cv_r2_mean"] = float(cv_scores.mean())
    metrics["cv_r2_std"] = float(cv_scores.std())
    metrics["cv_r2_scores"] = cv_scores.tolist()
    print(f"CV R² Mean: {metrics['cv_r2_mean']:.4f} (+/- {metrics['cv_r2_std']:.4f})")

    print("\nFeature Importance:")
    print("-" * 40)
    importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))
    metrics["feature_importance"] = importance

    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(score * 50)
        print(f"  {feature:20s} {score:.4f} {bar}")

    model_data = {
        "model": model,
        "params": MODEL_PARAMS,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving model to: {MODEL_PATH}")
    joblib.dump(model_data, MODEL_PATH)

    print(f"Saving metrics to: {METRICS_PATH}")
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
