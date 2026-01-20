"""FastAPI dependencies for dependency injection."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from src.config import get_settings
from src.models.predictor import StressPredictor


@lru_cache
def get_predictor() -> StressPredictor:
    """Get cached predictor instance."""
    predictor = StressPredictor()
    predictor.load()
    return predictor


async def verify_api_key(
    x_api_key: Annotated[str | None, Header()] = None,
) -> str:
    """Verify API key from request header.

    Args:
        x_api_key: API key from X-API-Key header.

    Returns:
        The validated API key.

    Raises:
        HTTPException: If API key is missing or invalid.
    """
    settings = get_settings()

    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
        )

    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    return x_api_key


PredictorDep = Annotated[StressPredictor, Depends(get_predictor)]
ApiKeyDep = Annotated[str, Depends(verify_api_key)]
