"""Pytest fixtures and configuration."""

import os
import tempfile
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

os.environ["API_KEY"] = "test-api-key"
os.environ["LOG_LEVEL"] = "WARNING"

from src.data.preprocessor import DataPreprocessor
from src.models.trainer import ModelTrainer


@pytest.fixture
def sample_input_data() -> dict[str, Any]:
    """Sample input data for predictions."""
    return {
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
    }


@pytest.fixture
def sample_encoded_input() -> dict[str, Any]:
    """Sample encoded input data."""
    return {
        "Hours_Worked": 10,
        "Sleep_Hours": 6,
        "Bugs": 15,
        "Deadline_Days": 7,
        "Coffee_Cups": 4,
        "Meetings": 3,
        "Interruptions": 5,
        "Experience_Years": 1,
        "Code_Complexity": 1,
        "Remote_Work": 1,
    }


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 100

    return pd.DataFrame(
        {
            "Hours_Worked": np.random.randint(4, 16, n_samples),
            "Sleep_Hours": np.random.randint(3, 9, n_samples),
            "Bugs": np.random.randint(0, 51, n_samples),
            "Deadline_Days": np.random.randint(0, 61, n_samples),
            "Coffee_Cups": np.random.randint(0, 11, n_samples),
            "Meetings": np.random.randint(0, 21, n_samples),
            "Interruptions": np.random.randint(0, 11, n_samples),
            "Experience_Years": np.random.choice(["Junior", "Mid", "Senior"], n_samples),
            "Code_Complexity": np.random.choice(["Low", "Medium", "High"], n_samples),
            "Remote_Work": np.random.choice(["Yes", "No"], n_samples),
            "Stress_Level": np.random.uniform(0, 100, n_samples),
        }
    )


@pytest.fixture
def preprocessor() -> DataPreprocessor:
    """Preprocessor instance."""
    return DataPreprocessor()


@pytest.fixture
def trainer() -> ModelTrainer:
    """Trainer instance."""
    return ModelTrainer()


@pytest.fixture(scope="module")
def trained_model_path(request: pytest.FixtureRequest) -> Generator[Path, None, None]:
    """Create a trained model and return its path."""
    np.random.seed(42)
    n_samples = 100

    sample_dataframe = pd.DataFrame(
        {
            "Hours_Worked": np.random.randint(4, 16, n_samples),
            "Sleep_Hours": np.random.randint(3, 9, n_samples),
            "Bugs": np.random.randint(0, 51, n_samples),
            "Deadline_Days": np.random.randint(0, 61, n_samples),
            "Coffee_Cups": np.random.randint(0, 11, n_samples),
            "Meetings": np.random.randint(0, 21, n_samples),
            "Interruptions": np.random.randint(0, 11, n_samples),
            "Experience_Years": np.random.choice(["Junior", "Mid", "Senior"], n_samples),
            "Code_Complexity": np.random.choice(["Low", "Medium", "High"], n_samples),
            "Remote_Work": np.random.choice(["Yes", "No"], n_samples),
            "Stress_Level": np.random.uniform(0, 100, n_samples),
        }
    )

    tmpdir = tempfile.mkdtemp()
    model_path = Path(tmpdir) / "test_model.joblib"

    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_data(sample_dataframe)
    trainer.train(X_train, y_train)
    trainer.evaluate(X_train, X_test, y_train, y_test)
    trainer.save_model(model_path)

    yield model_path

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="module")
def test_client(trained_model_path: Path) -> Generator[TestClient, None, None]:
    """FastAPI test client with trained model."""
    os.environ["MODEL_PATH"] = str(trained_model_path)

    from src.api.dependencies import get_predictor

    get_predictor.cache_clear()

    from src.api.main import app

    with TestClient(app) as client:
        yield client

    get_predictor.cache_clear()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Authentication headers for API requests."""
    return {"X-API-Key": "test-api-key"}
