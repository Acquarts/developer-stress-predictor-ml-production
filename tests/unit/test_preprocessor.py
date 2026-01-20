"""Tests for data preprocessor."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""

    def test_encode_categorical_experience(self, preprocessor: DataPreprocessor) -> None:
        """Test experience encoding."""
        data = {"Experience_Years": "Junior"}
        encoded = preprocessor.encode_categorical(data)
        assert encoded["Experience_Years"] == 0

        data = {"Experience_Years": "Mid"}
        encoded = preprocessor.encode_categorical(data)
        assert encoded["Experience_Years"] == 1

        data = {"Experience_Years": "Senior"}
        encoded = preprocessor.encode_categorical(data)
        assert encoded["Experience_Years"] == 2

    def test_encode_categorical_complexity(self, preprocessor: DataPreprocessor) -> None:
        """Test complexity encoding."""
        data = {"Code_Complexity": "Low"}
        encoded = preprocessor.encode_categorical(data)
        assert encoded["Code_Complexity"] == 0

        data = {"Code_Complexity": "Medium"}
        encoded = preprocessor.encode_categorical(data)
        assert encoded["Code_Complexity"] == 1

        data = {"Code_Complexity": "High"}
        encoded = preprocessor.encode_categorical(data)
        assert encoded["Code_Complexity"] == 2

    def test_encode_categorical_remote(self, preprocessor: DataPreprocessor) -> None:
        """Test remote work encoding."""
        data = {"Remote_Work": "No"}
        encoded = preprocessor.encode_categorical(data)
        assert encoded["Remote_Work"] == 0

        data = {"Remote_Work": "Yes"}
        encoded = preprocessor.encode_categorical(data)
        assert encoded["Remote_Work"] == 1

    def test_encode_already_encoded(self, preprocessor: DataPreprocessor) -> None:
        """Test that already encoded values are preserved."""
        data = {"Experience_Years": 1, "Code_Complexity": 2, "Remote_Work": 0}
        encoded = preprocessor.encode_categorical(data)
        assert encoded == data

    def test_transform_single(
        self, preprocessor: DataPreprocessor, sample_input_data: dict[str, Any]
    ) -> None:
        """Test single record transformation."""
        result = preprocessor.transform_single(sample_input_data)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 10)
        assert result.dtype == np.float64

    def test_transform_single_missing_feature(self, preprocessor: DataPreprocessor) -> None:
        """Test that missing features raise KeyError."""
        incomplete_data = {"Hours_Worked": 10, "Sleep_Hours": 6}

        with pytest.raises(KeyError):
            preprocessor.transform_single(incomplete_data)

    def test_transform_batch(
        self, preprocessor: DataPreprocessor, sample_input_data: dict[str, Any]
    ) -> None:
        """Test batch transformation."""
        batch = [sample_input_data, sample_input_data]
        result = preprocessor.transform_batch(batch)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 10)

    def test_transform_dataframe(
        self, preprocessor: DataPreprocessor, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test DataFrame transformation."""
        result = preprocessor.transform_dataframe(sample_dataframe)

        assert isinstance(result, pd.DataFrame)
        assert result["Experience_Years"].dtype in [np.int64, np.int32]
        assert result["Code_Complexity"].dtype in [np.int64, np.int32]
        assert result["Remote_Work"].dtype in [np.int64, np.int32]

    def test_validate_input_valid(
        self, preprocessor: DataPreprocessor, sample_input_data: dict[str, Any]
    ) -> None:
        """Test validation with valid input."""
        warnings = preprocessor.validate_input(sample_input_data)
        assert len(warnings) == 0

    def test_validate_input_out_of_range(self, preprocessor: DataPreprocessor) -> None:
        """Test validation with out-of-range values."""
        invalid_data = {
            "Hours_Worked": 25,
            "Sleep_Hours": 2,
            "Bugs": 100,
        }

        warnings = preprocessor.validate_input(invalid_data)
        assert len(warnings) > 0
        assert any("Hours_Worked" in w for w in warnings)

    def test_validate_input_invalid_categorical(self, preprocessor: DataPreprocessor) -> None:
        """Test validation with invalid categorical values."""
        invalid_data = {
            "Experience_Years": "Expert",
            "Code_Complexity": "Extreme",
            "Remote_Work": "Sometimes",
        }

        warnings = preprocessor.validate_input(invalid_data)
        assert len(warnings) == 3

    def test_inverse_encode_experience(self, preprocessor: DataPreprocessor) -> None:
        """Test inverse encoding of experience."""
        assert preprocessor.inverse_encode_experience(0) == "Junior"
        assert preprocessor.inverse_encode_experience(1) == "Mid"
        assert preprocessor.inverse_encode_experience(2) == "Senior"
        assert preprocessor.inverse_encode_experience(99) == "Unknown"

    def test_inverse_encode_complexity(self, preprocessor: DataPreprocessor) -> None:
        """Test inverse encoding of complexity."""
        assert preprocessor.inverse_encode_complexity(0) == "Low"
        assert preprocessor.inverse_encode_complexity(1) == "Medium"
        assert preprocessor.inverse_encode_complexity(2) == "High"

    def test_inverse_encode_remote(self, preprocessor: DataPreprocessor) -> None:
        """Test inverse encoding of remote work."""
        assert preprocessor.inverse_encode_remote(0) == "No"
        assert preprocessor.inverse_encode_remote(1) == "Yes"
