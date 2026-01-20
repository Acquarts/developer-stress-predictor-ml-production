"""Data loading utilities."""

from pathlib import Path

import pandas as pd


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load the developer stress dataset from CSV.

    Args:
        file_path: Path to the CSV file.

    Returns:
        DataFrame with the loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or has invalid format.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Data file is empty: {path}")

    required_columns = {
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
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df
