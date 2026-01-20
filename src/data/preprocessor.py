"""Data preprocessing pipeline for developer stress prediction."""

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.config import (
    COMPLEXITY_MAP,
    EXPERIENCE_MAP,
    FEATURE_COLUMNS,
    FEATURE_RANGES,
    REMOTE_MAP,
)


class DataPreprocessor:
    """Preprocessor for developer stress prediction data.

    Handles encoding of categorical variables and validation of input data.
    """

    def __init__(self) -> None:
        """Initialize the preprocessor with encoding mappings."""
        self.experience_map = EXPERIENCE_MAP
        self.complexity_map = COMPLEXITY_MAP
        self.remote_map = REMOTE_MAP
        self.feature_columns = FEATURE_COLUMNS

    def validate_input(self, data: dict[str, Any]) -> list[str]:
        """Validate input data and return list of warnings.

        Args:
            data: Dictionary with feature values.

        Returns:
            List of warning messages for out-of-range values.
        """
        warnings = []

        for feature, (min_val, max_val) in FEATURE_RANGES.items():
            if feature in data:
                value = data[feature]
                if not min_val <= value <= max_val:
                    warnings.append(
                        f"{feature} value {value} is outside typical range [{min_val}, {max_val}]"
                    )

        if "Experience_Years" in data and data["Experience_Years"] not in self.experience_map:
            warnings.append(
                f"Invalid Experience_Years: {data['Experience_Years']}. "
                f"Expected one of {list(self.experience_map.keys())}"
            )

        if "Code_Complexity" in data and data["Code_Complexity"] not in self.complexity_map:
            warnings.append(
                f"Invalid Code_Complexity: {data['Code_Complexity']}. "
                f"Expected one of {list(self.complexity_map.keys())}"
            )

        if "Remote_Work" in data and data["Remote_Work"] not in self.remote_map:
            warnings.append(
                f"Invalid Remote_Work: {data['Remote_Work']}. "
                f"Expected one of {list(self.remote_map.keys())}"
            )

        return warnings

    def encode_categorical(self, data: dict[str, Any]) -> dict[str, Any]:
        """Encode categorical variables to numeric values.

        Args:
            data: Dictionary with raw feature values.

        Returns:
            Dictionary with encoded values.
        """
        encoded = data.copy()

        if "Experience_Years" in encoded and isinstance(encoded["Experience_Years"], str):
            encoded["Experience_Years"] = self.experience_map[encoded["Experience_Years"]]

        if "Code_Complexity" in encoded and isinstance(encoded["Code_Complexity"], str):
            encoded["Code_Complexity"] = self.complexity_map[encoded["Code_Complexity"]]

        if "Remote_Work" in encoded and isinstance(encoded["Remote_Work"], str):
            encoded["Remote_Work"] = self.remote_map[encoded["Remote_Work"]]

        return encoded

    def transform_single(self, data: dict[str, Any]) -> NDArray[np.float64]:
        """Transform a single input record to model-ready format.

        Args:
            data: Dictionary with feature values.

        Returns:
            2D numpy array ready for model prediction.

        Raises:
            KeyError: If required features are missing.
        """
        encoded = self.encode_categorical(data)

        missing = set(self.feature_columns) - set(encoded.keys())
        if missing:
            raise KeyError(f"Missing required features: {missing}")

        features = [encoded[col] for col in self.feature_columns]
        return np.array([features], dtype=np.float64)

    def transform_batch(self, data: list[dict[str, Any]]) -> NDArray[np.float64]:
        """Transform a batch of input records to model-ready format.

        Args:
            data: List of dictionaries with feature values.

        Returns:
            2D numpy array ready for model prediction.
        """
        return np.vstack([self.transform_single(record) for record in data])

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a DataFrame for model training.

        Args:
            df: DataFrame with raw data.

        Returns:
            DataFrame with encoded categorical variables.
        """
        df_encoded = df.copy()

        if "Experience_Years" in df_encoded.columns:
            df_encoded["Experience_Years"] = df_encoded["Experience_Years"].map(self.experience_map)

        if "Code_Complexity" in df_encoded.columns:
            df_encoded["Code_Complexity"] = df_encoded["Code_Complexity"].map(self.complexity_map)

        if "Remote_Work" in df_encoded.columns:
            df_encoded["Remote_Work"] = df_encoded["Remote_Work"].map(self.remote_map)

        return df_encoded

    def inverse_encode_experience(self, value: int) -> str:
        """Convert encoded experience value back to string."""
        inverse = {v: k for k, v in self.experience_map.items()}
        return inverse.get(value, "Unknown")

    def inverse_encode_complexity(self, value: int) -> str:
        """Convert encoded complexity value back to string."""
        inverse = {v: k for k, v in self.complexity_map.items()}
        return inverse.get(value, "Unknown")

    def inverse_encode_remote(self, value: int) -> str:
        """Convert encoded remote work value back to string."""
        inverse = {v: k for k, v in self.remote_map.items()}
        return inverse.get(value, "Unknown")
