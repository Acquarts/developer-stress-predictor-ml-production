"""Tests for model predictor."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.models.predictor import StressPredictor


class TestStressPredictor:
    """Tests for StressPredictor class."""

    def test_predictor_initialization(self) -> None:
        """Test predictor initializes correctly."""
        predictor = StressPredictor()
        assert predictor.model is None
        assert not predictor._loaded

    def test_predictor_load(self, trained_model_path: Path) -> None:
        """Test model loading."""
        predictor = StressPredictor(model_path=trained_model_path)
        predictor.load()

        assert predictor.model is not None
        assert predictor._loaded

    def test_predictor_load_nonexistent(self) -> None:
        """Test loading nonexistent model raises error."""
        predictor = StressPredictor(model_path="nonexistent.joblib")

        with pytest.raises(FileNotFoundError):
            predictor.load()

    def test_predict_single(
        self, trained_model_path: Path, sample_input_data: dict[str, Any]
    ) -> None:
        """Test single prediction."""
        predictor = StressPredictor(model_path=trained_model_path)
        predictor.load()

        result = predictor.predict(sample_input_data)

        assert "stress_level" in result
        assert "warnings" in result
        assert 0 <= result["stress_level"] <= 100
        assert isinstance(result["warnings"], list)

    def test_predict_clamps_output(self, trained_model_path: Path) -> None:
        """Test that predictions are clamped to 0-100 range."""
        predictor = StressPredictor(model_path=trained_model_path)
        predictor.load()

        extreme_data = {
            "Hours_Worked": 15,
            "Sleep_Hours": 3,
            "Bugs": 50,
            "Deadline_Days": 0,
            "Coffee_Cups": 10,
            "Meetings": 20,
            "Interruptions": 10,
            "Experience_Years": "Junior",
            "Code_Complexity": "High",
            "Remote_Work": "No",
        }

        result = predictor.predict(extreme_data)
        assert 0 <= result["stress_level"] <= 100

    def test_predict_batch(
        self, trained_model_path: Path, sample_input_data: dict[str, Any]
    ) -> None:
        """Test batch predictions."""
        predictor = StressPredictor(model_path=trained_model_path)
        predictor.load()

        batch = [sample_input_data, sample_input_data, sample_input_data]
        results = predictor.predict_batch(batch)

        assert len(results) == 3
        for result in results:
            assert result.get("success", True)
            assert "stress_level" in result

    def test_predict_batch_with_errors(self, trained_model_path: Path) -> None:
        """Test batch predictions handle errors gracefully."""
        predictor = StressPredictor(model_path=trained_model_path)
        predictor.load()

        batch = [
            {"Hours_Worked": 10},
            {
                "Hours_Worked": 10,
                "Sleep_Hours": 6,
                "Bugs": 15,
                "Deadline_Days": 7,
                "Coffee_Cups": 4,
                "Meetings": 3,
                "Interruptions": 5,
                "Experience_Years": "Mid",
                "Code_Complexity": "Medium",
                "Remote_Work": "Yes",
            },
        ]

        results = predictor.predict_batch(batch)

        assert len(results) == 2
        assert results[0]["success"] is False
        assert "error" in results[0]
        assert results[1]["success"] is True

    def test_predict_array(
        self, trained_model_path: Path, sample_encoded_input: dict[str, Any]
    ) -> None:
        """Test array predictions."""
        predictor = StressPredictor(model_path=trained_model_path)
        predictor.load()

        from src.config import FEATURE_COLUMNS

        features = np.array([[sample_encoded_input[col] for col in FEATURE_COLUMNS]])
        predictions = predictor.predict_array(features)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 1
        assert 0 <= predictions[0] <= 100

    def test_get_feature_importance(self, trained_model_path: Path) -> None:
        """Test feature importance retrieval."""
        predictor = StressPredictor(model_path=trained_model_path)
        predictor.load()

        importance = predictor.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == 10
        assert all(isinstance(v, float) for v in importance.values())
        assert all(0 <= v <= 1 for v in importance.values())

        keys = list(importance.keys())
        values = [importance[k] for k in keys]
        assert values == sorted(values, reverse=True)

    def test_get_model_info(self, trained_model_path: Path) -> None:
        """Test model info retrieval."""
        predictor = StressPredictor(model_path=trained_model_path)
        predictor.load()

        info = predictor.get_model_info()

        assert "model_type" in info
        assert "n_estimators" in info
        assert "max_depth" in info
        assert "feature_columns" in info
        assert "metrics" in info
        assert info["model_type"] == "RandomForestRegressor"

    def test_ensure_loaded_autoload(self, trained_model_path: Path) -> None:
        """Test that ensure_loaded auto-loads the model."""
        predictor = StressPredictor(model_path=trained_model_path)
        assert not predictor._loaded

        predictor.ensure_loaded()
        assert predictor._loaded
