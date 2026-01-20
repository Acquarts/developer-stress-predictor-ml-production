"""Feature engineering utilities."""

import pandas as pd

from src.config import FEATURE_COLUMNS


class FeatureEngineer:
    """Feature engineering for developer stress prediction."""

    def __init__(self) -> None:
        """Initialize feature engineer."""
        self.feature_columns = FEATURE_COLUMNS

    def get_feature_importance_labels(self) -> dict[str, str]:
        """Get human-readable labels for features."""
        return {
            "Hours_Worked": "Hours Worked per Day",
            "Sleep_Hours": "Hours of Sleep",
            "Bugs": "Number of Bugs",
            "Deadline_Days": "Days Until Deadline",
            "Coffee_Cups": "Coffee Cups per Day",
            "Meetings": "Number of Meetings",
            "Interruptions": "Daily Interruptions",
            "Experience_Years": "Experience Level",
            "Code_Complexity": "Code Complexity",
            "Remote_Work": "Remote Work",
        }

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from dataframe.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with only feature columns.
        """
        return df[self.feature_columns].copy()

    def extract_target(self, df: pd.DataFrame, target_col: str = "Stress_Level") -> pd.Series:
        """Extract target variable from dataframe.

        Args:
            df: Input DataFrame.
            target_col: Name of target column.

        Returns:
            Series with target values.
        """
        return df[target_col].copy()

    def get_stress_level_description(self, stress_level: float) -> str:
        """Get human-readable description for stress level.

        Args:
            stress_level: Predicted stress level (0-100).

        Returns:
            Description of the stress level.
        """
        if stress_level < 20:
            return "Very Low - Relaxed and comfortable"
        elif stress_level < 40:
            return "Low - Manageable workload"
        elif stress_level < 60:
            return "Moderate - Some pressure but coping"
        elif stress_level < 80:
            return "High - Significant stress, consider taking breaks"
        else:
            return "Very High - Critical stress level, immediate action recommended"

    def get_stress_color(self, stress_level: float) -> str:
        """Get color code for stress level visualization.

        Args:
            stress_level: Predicted stress level (0-100).

        Returns:
            Hex color code.
        """
        if stress_level < 20:
            return "#2ECC71"  # Green
        elif stress_level < 40:
            return "#82E0AA"  # Light green
        elif stress_level < 60:
            return "#F4D03F"  # Yellow
        elif stress_level < 80:
            return "#E67E22"  # Orange
        else:
            return "#E74C3C"  # Red
