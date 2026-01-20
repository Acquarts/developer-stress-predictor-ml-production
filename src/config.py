"""Centralized configuration management using pydantic-settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Configuration
    api_key: str = "dev-api-key"
    api_host: str = "0.0.0.0"  # nosec B104 - Required for Docker containers
    api_port: int = 8000

    # Model Configuration
    model_path: Path = Path("models/stress_model.joblib")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "console"] = "json"

    # GCP Configuration
    gcp_project_id: str | None = None
    gcp_region: str = "us-central1"

    # Streamlit
    streamlit_api_url: str = "http://localhost:8000"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.gcp_project_id is not None


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Feature configuration
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

CATEGORICAL_COLUMNS = ["Experience_Years", "Code_Complexity", "Remote_Work"]
NUMERICAL_COLUMNS = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_COLUMNS]

# Encoding mappings
EXPERIENCE_MAP = {"Junior": 0, "Mid": 1, "Senior": 2}
COMPLEXITY_MAP = {"Low": 0, "Medium": 1, "High": 2}
REMOTE_MAP = {"No": 0, "Yes": 1}

# Model hyperparameters (from GridSearchCV)
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_leaf": 2,
    "random_state": 42,
}

# Feature ranges for validation
FEATURE_RANGES = {
    "Hours_Worked": (4, 15),
    "Sleep_Hours": (3, 8),
    "Bugs": (0, 50),
    "Deadline_Days": (0, 60),
    "Coffee_Cups": (0, 10),
    "Meetings": (0, 20),
    "Interruptions": (0, 10),
}
