"""Structured logging configuration."""

import logging
import sys
from typing import Any

import structlog

from src.config import get_settings


def setup_logging() -> None:
    """Configure structured logging for the application."""
    settings = get_settings()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            (
                structlog.dev.ConsoleRenderer()
                if settings.log_format == "console"
                else structlog.processors.JSONRenderer()
            ),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level),
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name.

    Returns:
        Configured structlog logger.
    """
    return structlog.get_logger(name)


def log_prediction(
    logger: structlog.BoundLogger,
    input_data: dict[str, Any],
    prediction: float,
    request_id: str | None = None,
) -> None:
    """Log a prediction event.

    Args:
        logger: Logger instance.
        input_data: Input features.
        prediction: Predicted stress level.
        request_id: Optional request ID for correlation.
    """
    logger.info(
        "prediction_made",
        request_id=request_id,
        prediction=prediction,
        hours_worked=input_data.get("Hours_Worked"),
        sleep_hours=input_data.get("Sleep_Hours"),
        experience=input_data.get("Experience_Years"),
    )
