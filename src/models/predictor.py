"""Model inference and prediction."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor

from src.config import FEATURE_COLUMNS, get_settings
from src.data.preprocessor import DataPreprocessor


class StressPredictor:
    """Predictor for developer stress levels."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        """Initialize predictor.

        Args:
            model_path: Path to the trained model. Uses config default if not provided.
        """
        settings = get_settings()
        self.model_path = Path(model_path) if model_path else settings.model_path
        self.model: RandomForestRegressor | None = None
        self.preprocessor = DataPreprocessor()
        self.feature_columns = FEATURE_COLUMNS
        self.metrics: dict[str, float] = {}
        self._loaded = False

    def load(self) -> None:
        """Load the model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        model_data = joblib.load(self.model_path)

        self.model = model_data["model"]
        self.feature_columns = model_data["feature_columns"]
        self.metrics = model_data.get("metrics", {})
        self._loaded = True

    def ensure_loaded(self) -> None:
        """Ensure model is loaded, loading it if necessary."""
        if not self._loaded:
            self.load()

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Make a single prediction.

        Args:
            data: Dictionary with feature values.

        Returns:
            Dictionary with prediction results.
        """
        self.ensure_loaded()

        warnings = self.preprocessor.validate_input(data)
        features = self.preprocessor.transform_single(data)
        prediction = float(self.model.predict(features)[0])  # type: ignore

        prediction = max(0.0, min(100.0, prediction))

        return {
            "stress_level": round(prediction, 2),
            "warnings": warnings,
        }

    def predict_batch(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Make batch predictions.

        Args:
            data: List of dictionaries with feature values.

        Returns:
            List of prediction results.
        """
        self.ensure_loaded()

        results = []
        for record in data:
            try:
                result = self.predict(record)
                result["success"] = True
            except Exception as e:
                result = {"success": False, "error": str(e)}
            results.append(result)

        return results

    def predict_array(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Make predictions from numpy array.

        Args:
            features: 2D array of features.

        Returns:
            1D array of predictions.
        """
        self.ensure_loaded()

        predictions = self.model.predict(features)  # type: ignore
        return np.clip(predictions, 0.0, 100.0)

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance from the model.

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        self.ensure_loaded()

        importances = self.model.feature_importances_  # type: ignore
        return dict(
            sorted(
                zip(self.feature_columns, importances.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary with model metadata.
        """
        self.ensure_loaded()

        return {
            "model_type": "RandomForestRegressor",
            "n_estimators": self.model.n_estimators,  # type: ignore
            "max_depth": self.model.max_depth,  # type: ignore
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
        }
