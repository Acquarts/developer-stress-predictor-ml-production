"""Prometheus metrics for monitoring."""

from prometheus_client import Counter, Histogram, Info

PREDICTION_COUNTER = Counter(
    "stress_predictions_total",
    "Total number of stress predictions",
    ["status"],
)

PREDICTION_LATENCY = Histogram(
    "stress_prediction_latency_seconds",
    "Latency of stress predictions",
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0],
)

PREDICTION_VALUE = Histogram(
    "stress_prediction_value",
    "Distribution of predicted stress levels",
    buckets=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
)

BATCH_SIZE = Histogram(
    "stress_batch_prediction_size",
    "Size of batch predictions",
    buckets=[1, 5, 10, 25, 50, 100],
)

MODEL_INFO = Info(
    "stress_model",
    "Information about the loaded model",
)

REQUEST_COUNTER = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)


class MetricsCollector:
    """Collector for application metrics."""

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self._model_info_set = False

    def record_prediction(self, stress_level: float, latency: float, success: bool = True) -> None:
        """Record a prediction event.

        Args:
            stress_level: Predicted stress level.
            latency: Prediction latency in seconds.
            success: Whether prediction was successful.
        """
        status = "success" if success else "error"
        PREDICTION_COUNTER.labels(status=status).inc()
        PREDICTION_LATENCY.observe(latency)

        if success:
            PREDICTION_VALUE.observe(stress_level)

    def record_batch_prediction(self, batch_size: int, successful: int, latency: float) -> None:
        """Record a batch prediction event.

        Args:
            batch_size: Total number of predictions in batch.
            successful: Number of successful predictions.
            latency: Total batch latency in seconds.
        """
        BATCH_SIZE.observe(batch_size)
        PREDICTION_COUNTER.labels(status="success").inc(successful)
        PREDICTION_COUNTER.labels(status="error").inc(batch_size - successful)
        PREDICTION_LATENCY.observe(latency)

    def record_request(self, method: str, endpoint: str, status_code: int) -> None:
        """Record an HTTP request.

        Args:
            method: HTTP method.
            endpoint: Request endpoint.
            status_code: Response status code.
        """
        REQUEST_COUNTER.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code),
        ).inc()

    def set_model_info(
        self,
        model_type: str,
        n_estimators: int,
        max_depth: int | None,
        r2_score: float,
    ) -> None:
        """Set model information metric.

        Args:
            model_type: Type of model.
            n_estimators: Number of estimators.
            max_depth: Maximum depth.
            r2_score: Model R² score.
        """
        if not self._model_info_set:
            MODEL_INFO.info(
                {
                    "model_type": model_type,
                    "n_estimators": str(n_estimators),
                    "max_depth": str(max_depth) if max_depth else "None",
                    "r2_score": f"{r2_score:.4f}",
                }
            )
            self._model_info_set = True
