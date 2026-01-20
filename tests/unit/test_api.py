"""Tests for FastAPI endpoints."""

from typing import Any

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for health endpoint."""

    def test_health_check(self, test_client: TestClient) -> None:
        """Test health check endpoint."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "version" in data


class TestPredictEndpoint:
    """Tests for prediction endpoint."""

    def test_predict_success(
        self,
        test_client: TestClient,
        auth_headers: dict[str, str],
        sample_input_data: dict[str, Any],
    ) -> None:
        """Test successful prediction."""
        response = test_client.post(
            "/predict",
            json=sample_input_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "stress_level" in data
        assert 0 <= data["stress_level"] <= 100
        assert "warnings" in data

    def test_predict_missing_auth(
        self, test_client: TestClient, sample_input_data: dict[str, Any]
    ) -> None:
        """Test prediction without API key."""
        response = test_client.post("/predict", json=sample_input_data)

        assert response.status_code == 401
        assert "Missing API key" in response.json()["detail"]

    def test_predict_invalid_auth(
        self, test_client: TestClient, sample_input_data: dict[str, Any]
    ) -> None:
        """Test prediction with invalid API key."""
        response = test_client.post(
            "/predict",
            json=sample_input_data,
            headers={"X-API-Key": "invalid-key"},
        )

        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]

    def test_predict_validation_error(
        self, test_client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Test prediction with invalid input."""
        invalid_data = {
            "Hours_Worked": 30,
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

        response = test_client.post(
            "/predict",
            json=invalid_data,
            headers=auth_headers,
        )

        assert response.status_code == 422

    def test_predict_invalid_experience(
        self, test_client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Test prediction with invalid experience level."""
        invalid_data = {
            "Hours_Worked": 10,
            "Sleep_Hours": 6,
            "Bugs": 15,
            "Deadline_Days": 7,
            "Coffee_Cups": 4,
            "Meetings": 3,
            "Interruptions": 5,
            "Experience_Years": "Expert",
            "Code_Complexity": "Medium",
            "Remote_Work": "Yes",
        }

        response = test_client.post(
            "/predict",
            json=invalid_data,
            headers=auth_headers,
        )

        assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Tests for batch prediction endpoint."""

    def test_batch_predict_success(
        self,
        test_client: TestClient,
        auth_headers: dict[str, str],
        sample_input_data: dict[str, Any],
    ) -> None:
        """Test successful batch prediction."""
        response = test_client.post(
            "/predict/batch",
            json={"predictions": [sample_input_data, sample_input_data]},
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total" in data
        assert "successful" in data
        assert data["total"] == 2
        assert data["successful"] == 2

    def test_batch_predict_empty(
        self, test_client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Test batch prediction with empty list."""
        response = test_client.post(
            "/predict/batch",
            json={"predictions": []},
            headers=auth_headers,
        )

        assert response.status_code == 422


class TestModelEndpoints:
    """Tests for model information endpoints."""

    def test_model_info(self, test_client: TestClient, auth_headers: dict[str, str]) -> None:
        """Test model info endpoint."""
        response = test_client.get("/model/info", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "n_estimators" in data
        assert "feature_columns" in data

    def test_feature_importance(
        self, test_client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Test feature importance endpoint."""
        response = test_client.get("/model/features", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "importance" in data
        assert len(data["importance"]) == 10


class TestMetricsEndpoint:
    """Tests for metrics endpoint."""

    def test_metrics(self, test_client: TestClient) -> None:
        """Test prometheus metrics endpoint."""
        response = test_client.get("/metrics")

        assert response.status_code == 200
        assert "stress_predictions_total" in response.text or "http_requests_total" in response.text
