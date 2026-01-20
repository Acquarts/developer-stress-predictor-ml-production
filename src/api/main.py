"""FastAPI application for developer stress prediction."""

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src import __version__
from src.api.dependencies import ApiKeyDep, PredictorDep, get_predictor
from src.api.schemas import (
    BatchPredictionInput,
    BatchPredictionOutput,
    ErrorResponse,
    FeatureImportanceResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionInput,
    PredictionOutput,
)
from src.monitoring import MetricsCollector, get_logger, setup_logging

logger = get_logger(__name__)
metrics = MetricsCollector()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    setup_logging()
    logger.info("starting_application", version=__version__)

    try:
        predictor = get_predictor()
        model_info = predictor.get_model_info()
        metrics.set_model_info(
            model_type=model_info["model_type"],
            n_estimators=model_info["n_estimators"],
            max_depth=model_info["max_depth"],
            r2_score=model_info["metrics"].get("test_r2", 0.0),
        )
        logger.info("model_loaded", model_info=model_info)
    except Exception as e:
        logger.error("model_load_failed", error=str(e))

    yield

    logger.info("shutting_down_application")


app = FastAPI(
    title="Developer Stress Prediction API",
    description="ML-powered API to predict developer stress levels based on work patterns",
    version=__version__,
    lifespan=lifespan,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Log all requests and record metrics."""
    start_time = time.time()

    response = await call_next(request)

    latency = time.time() - start_time
    metrics.record_request(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code,
    )

    logger.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=round(latency * 1000, 2),
    )

    return response


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Check API health status."""
    try:
        predictor = get_predictor()
        model_loaded = predictor._loaded
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        version=__version__,
    )


@app.get("/metrics", tags=["Monitoring"])
async def prometheus_metrics() -> Response:
    """Expose Prometheus metrics."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post(
    "/predict",
    response_model=PredictionOutput,
    tags=["Predictions"],
    summary="Predict stress level",
    description="Predict the stress level for a developer based on work patterns",
)
async def predict(
    input_data: PredictionInput,
    predictor: PredictorDep,
    _api_key: ApiKeyDep,
) -> PredictionOutput:
    """Make a single stress prediction."""
    start_time = time.time()

    try:
        result = predictor.predict(input_data.model_dump())
        latency = time.time() - start_time

        metrics.record_prediction(
            stress_level=result["stress_level"],
            latency=latency,
            success=True,
        )

        logger.info(
            "prediction_success",
            stress_level=result["stress_level"],
            latency_ms=round(latency * 1000, 2),
        )

        return PredictionOutput(**result)

    except Exception as e:
        latency = time.time() - start_time
        metrics.record_prediction(stress_level=0, latency=latency, success=False)
        logger.error("prediction_error", error=str(e))
        raise


@app.post(
    "/predict/batch",
    response_model=BatchPredictionOutput,
    tags=["Predictions"],
    summary="Batch predict stress levels",
    description="Predict stress levels for multiple developers",
)
async def predict_batch(
    input_data: BatchPredictionInput,
    predictor: PredictorDep,
    _api_key: ApiKeyDep,
) -> BatchPredictionOutput:
    """Make batch stress predictions."""
    start_time = time.time()

    records = [item.model_dump() for item in input_data.predictions]
    results = predictor.predict_batch(records)

    latency = time.time() - start_time
    successful = sum(1 for r in results if r.get("success", True))

    metrics.record_batch_prediction(
        batch_size=len(results),
        successful=successful,
        latency=latency,
    )

    logger.info(
        "batch_prediction_complete",
        total=len(results),
        successful=successful,
        latency_ms=round(latency * 1000, 2),
    )

    return BatchPredictionOutput(
        results=results,
        total=len(results),
        successful=successful,
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Get model information",
)
async def model_info(
    predictor: PredictorDep,
    _api_key: ApiKeyDep,
) -> ModelInfoResponse:
    """Get information about the loaded model."""
    info = predictor.get_model_info()
    return ModelInfoResponse(**info)


@app.get(
    "/model/features",
    response_model=FeatureImportanceResponse,
    tags=["Model"],
    summary="Get feature importance",
)
async def feature_importance(
    predictor: PredictorDep,
    _api_key: ApiKeyDep,
) -> FeatureImportanceResponse:
    """Get feature importance from the model."""
    importance = predictor.get_feature_importance()
    return FeatureImportanceResponse(importance=importance)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions."""
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        error=str(exc),
        error_type=type(exc).__name__,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "error_code": "INTERNAL_ERROR"},
    )


if __name__ == "__main__":
    import uvicorn

    from src.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
