"""Pydantic schemas for API request/response validation."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    """Input schema for stress prediction."""

    Hours_Worked: int = Field(..., ge=1, le=24, description="Hours worked per day")
    Sleep_Hours: int = Field(..., ge=0, le=12, description="Hours of sleep")
    Bugs: int = Field(..., ge=0, description="Number of bugs encountered")
    Deadline_Days: int = Field(..., ge=0, description="Days until deadline")
    Coffee_Cups: int = Field(..., ge=0, le=20, description="Coffee cups per day")
    Meetings: int = Field(..., ge=0, le=24, description="Number of meetings")
    Interruptions: int = Field(..., ge=0, le=50, description="Number of interruptions")
    Experience_Years: Literal["Junior", "Mid", "Senior"] = Field(
        ..., description="Experience level"
    )
    Code_Complexity: Literal["Low", "Medium", "High"] = Field(
        ..., description="Code complexity"
    )
    Remote_Work: Literal["Yes", "No"] = Field(..., description="Remote work status")

    model_config = {
        "json_schema_extra": {
            "examples": [
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
                }
            ]
        }
    }


class PredictionOutput(BaseModel):
    """Output schema for stress prediction."""

    stress_level: float = Field(..., ge=0, le=100, description="Predicted stress level (0-100)")
    warnings: list[str] = Field(default_factory=list, description="Validation warnings")


class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions."""

    predictions: list[PredictionInput] = Field(
        ..., min_length=1, max_length=100, description="List of prediction inputs"
    )


class BatchPredictionOutput(BaseModel):
    """Output schema for batch predictions."""

    results: list[dict] = Field(..., description="List of prediction results")
    total: int = Field(..., description="Total number of predictions")
    successful: int = Field(..., description="Number of successful predictions")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_type: str = Field(..., description="Type of model")
    n_estimators: int = Field(..., description="Number of estimators")
    max_depth: int | None = Field(..., description="Maximum depth")
    feature_columns: list[str] = Field(..., description="Feature columns used")
    metrics: dict[str, Any] = Field(..., description="Training metrics")


class FeatureImportanceResponse(BaseModel):
    """Feature importance response."""

    importance: dict[str, float] = Field(..., description="Feature importance scores")


class ErrorResponse(BaseModel):
    """Error response schema."""

    detail: str = Field(..., description="Error message")
    error_code: str | None = Field(None, description="Error code")
