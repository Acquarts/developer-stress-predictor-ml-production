"""Model training utilities."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from src.config import FEATURE_COLUMNS, MODEL_PARAMS
from src.data.preprocessor import DataPreprocessor


class ModelTrainer:
    """Trainer for developer stress prediction model."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize trainer with model parameters.

        Args:
            params: Model hyperparameters. Uses defaults from config if not provided.
        """
        self.params = params or MODEL_PARAMS
        self.model: RandomForestRegressor | None = None
        self.preprocessor = DataPreprocessor()
        self.feature_columns = FEATURE_COLUMNS
        self.metrics: dict[str, float] = {}

    def prepare_data(
        self, df: pd.DataFrame, target_col: str = "Stress_Level", test_size: float = 0.2
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Prepare data for training.

        Args:
            df: Input DataFrame.
            target_col: Name of target column.
            test_size: Fraction of data for testing.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        df_encoded = self.preprocessor.transform_dataframe(df)

        X = df_encoded[self.feature_columns].values
        y = df_encoded[target_col].values

        return train_test_split(X, y, test_size=test_size, random_state=42)

    def train(
        self, X_train: NDArray[np.float64], y_train: NDArray[np.float64]
    ) -> RandomForestRegressor:
        """Train the model.

        Args:
            X_train: Training features.
            y_train: Training targets.

        Returns:
            Trained model.
        """
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate(
        self,
        X_train: NDArray[np.float64],
        X_test: NDArray[np.float64],
        y_train: NDArray[np.float64],
        y_test: NDArray[np.float64],
    ) -> dict[str, float]:
        """Evaluate model performance.

        Args:
            X_train: Training features.
            X_test: Test features.
            y_train: Training targets.
            y_test: Test targets.

        Returns:
            Dictionary with evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        self.metrics = {
            "train_r2": float(r2_score(y_train, y_train_pred)),
            "train_mse": float(mean_squared_error(y_train, y_train_pred)),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            "test_r2": float(r2_score(y_test, y_test_pred)),
            "test_mse": float(mean_squared_error(y_test, y_test_pred)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
        }

        return self.metrics

    def cross_validate(
        self, X: NDArray[np.float64], y: NDArray[np.float64], cv: int = 5
    ) -> dict[str, float]:
        """Perform cross-validation.

        Args:
            X: Features.
            y: Targets.
            cv: Number of folds.

        Returns:
            Dictionary with CV metrics.
        """
        if self.model is None:
            self.model = RandomForestRegressor(**self.params)

        scores = cross_val_score(self.model, X, y, cv=cv, scoring="r2")

        return {
            "cv_r2_mean": float(scores.mean()),
            "cv_r2_std": float(scores.std()),
            "cv_r2_scores": scores.tolist(),
        }

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance from trained model.

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        importances = self.model.feature_importances_
        return dict(zip(self.feature_columns, importances.tolist()))

    def save_model(self, path: str | Path) -> None:
        """Save trained model to disk.

        Args:
            path: Path to save the model.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "params": self.params,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
        }

        joblib.dump(model_data, path)

    def load_model(self, path: str | Path) -> RandomForestRegressor:
        """Load model from disk.

        Args:
            path: Path to the saved model.

        Returns:
            Loaded model.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        model_data = joblib.load(path)

        self.model = model_data["model"]
        self.params = model_data["params"]
        self.feature_columns = model_data["feature_columns"]
        self.metrics = model_data.get("metrics", {})

        return self.model
