"""Utility functions for Streamlit app."""

import os
from typing import Any

import httpx

API_URL = os.getenv("STREAMLIT_API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "dev-api-key")


def get_api_headers() -> dict[str, str]:
    """Get headers for API requests."""
    return {"X-API-Key": API_KEY, "Content-Type": "application/json"}


def predict_stress(input_data: dict[str, Any]) -> dict[str, Any]:
    """Make a prediction via the API.

    Args:
        input_data: Dictionary with feature values.

    Returns:
        Prediction result from API.
    """
    response = httpx.post(
        f"{API_URL}/predict",
        json=input_data,
        headers=get_api_headers(),
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def predict_batch(inputs: list[dict[str, Any]]) -> dict[str, Any]:
    """Make batch predictions via the API.

    Args:
        inputs: List of feature dictionaries.

    Returns:
        Batch prediction results from API.
    """
    response = httpx.post(
        f"{API_URL}/predict/batch",
        json={"predictions": inputs},
        headers=get_api_headers(),
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def get_model_info() -> dict[str, Any]:
    """Get model information from API.

    Returns:
        Model information dictionary.
    """
    response = httpx.get(
        f"{API_URL}/model/info",
        headers=get_api_headers(),
        timeout=10.0,
    )
    response.raise_for_status()
    return response.json()


def get_feature_importance() -> dict[str, float]:
    """Get feature importance from API.

    Returns:
        Dictionary of feature importance scores.
    """
    response = httpx.get(
        f"{API_URL}/model/features",
        headers=get_api_headers(),
        timeout=10.0,
    )
    response.raise_for_status()
    return response.json()["importance"]


def check_api_health() -> bool:
    """Check if API is healthy.

    Returns:
        True if API is healthy, False otherwise.
    """
    try:
        response = httpx.get(f"{API_URL}/health", timeout=5.0)
        return response.status_code == 200 and response.json().get("model_loaded", False)
    except Exception:
        return False


def get_stress_color(stress_level: float) -> str:
    """Get color for stress level visualization.

    Args:
        stress_level: Predicted stress level (0-100).

    Returns:
        Hex color code.
    """
    if stress_level < 20:
        return "#2ECC71"
    elif stress_level < 40:
        return "#82E0AA"
    elif stress_level < 60:
        return "#F4D03F"
    elif stress_level < 80:
        return "#E67E22"
    else:
        return "#E74C3C"


def get_stress_description(stress_level: float) -> str:
    """Get description for stress level.

    Args:
        stress_level: Predicted stress level (0-100).

    Returns:
        Description string.
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
