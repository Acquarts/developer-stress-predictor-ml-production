# 🧠 Developer Stress Predictor

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Run-4285F4?logo=google-cloud&logoColor=white)](https://cloud.google.com/run)
[![Docker](https://img.shields.io/badge/Docker-24.0-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Pytest](https://img.shields.io/badge/Pytest-8.0-0A9EDC?logo=pytest&logoColor=white)](https://pytest.org)
[![Ruff](https://img.shields.io/badge/Ruff-0.2-D7FF64?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![CI](https://github.com/Acquarts/developer-stress-predictor-ml-production/actions/workflows/ci.yml/badge.svg)](https://github.com/Acquarts/developer-stress-predictor-ml-production/actions/workflows/ci.yml)
[![CD](https://github.com/Acquarts/developer-stress-predictor-ml-production/actions/workflows/cd.yml/badge.svg)](https://github.com/Acquarts/developer-stress-predictor-ml-production/actions/workflows/cd.yml)

An end-to-end Machine Learning application that predicts developer stress levels based on work patterns, habits, and environmental factors. Built with production-grade practices including CI/CD pipelines, containerization, automated testing, and cloud deployment.

<p align="center">
  <a href="https://stress-streamlit-562289298058.us-central1.run.app">
    <img src="https://img.shields.io/badge/🚀_Try_the_Live_App-FF4B4B?style=for-the-badge" alt="Live App">
  </a>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#️-architecture)
- [Model Details](#-model-details)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [Deployment](#-deployment)
- [Testing](#-testing)
- [Project Structure](#-project-structure)
- [Tech Stack](#️-tech-stack)
- [Contributing](#-contributing)

---

## 🎯 Overview

Developer burnout is a real problem in the tech industry. This project provides a data-driven approach to predict and monitor stress levels, helping developers and teams take proactive measures before burnout occurs.

The application consists of:
- **REST API**: A FastAPI backend serving predictions with OpenAPI documentation
- **Web UI**: An interactive Streamlit dashboard for easy predictions and visualization
- **ML Model**: A Random Forest Regressor trained on developer work patterns

Both services are deployed on Google Cloud Run with automatic scaling and CI/CD pipelines.

---

## ✨ Features

### 🔮 Prediction Engine
- Predicts stress level on a scale of 0-100
- Takes into account 10 different work-related factors
- Provides personalized recommendations based on stress level
- Supports both single and batch predictions

### 🌐 REST API
- Full OpenAPI/Swagger documentation
- API key authentication
- Health checks and monitoring endpoints
- Model introspection (feature importance, metrics)

### 📊 Interactive Dashboard
- User-friendly form for inputting work patterns
- Real-time stress level visualization with gauge charts
- Monitoring dashboard with model metrics
- Prediction history tracking

### 🏭 Production Ready
- Dockerized services with multi-stage builds
- CI pipeline with linting, type checking, and 41+ tests
- CD pipeline with automatic deployment to Cloud Run
- Secret management with Google Secret Manager
- Auto-scaling from 0 to handle variable load

---

## 🏗️ Architecture

```
                                    ┌──────────────────────────────────────┐
                                    │           Google Cloud Run           │
                                    └──────────────────────────────────────┘
                                                      │
                       ┌──────────────────────────────┼──────────────────────────────┐
                       │                              │                              │
                       ▼                              ▼                              ▼
              ┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
              │                 │          │                 │          │                 │
              │   Streamlit     │─────────▶│    FastAPI      │─────────▶│  RandomForest   │
              │   Frontend      │   HTTP   │    Backend      │          │     Model       │
              │                 │          │                 │          │                 │
              └─────────────────┘          └─────────────────┘          └─────────────────┘
                     │                            │                            │
                     │                            │                            │
              ┌──────┴──────┐              ┌──────┴──────┐              ┌──────┴──────┐
              │  Plotly     │              │  Pydantic   │              │ scikit-learn│
              │  Charts     │              │  Validation │              │  R² = 0.89  │
              └─────────────┘              └─────────────┘              └─────────────┘
```

### Data Flow

1. **User Input** → Streamlit collects work pattern data through an interactive form
2. **API Request** → Data is validated and sent to FastAPI backend
3. **Prediction** → Random Forest model processes features and returns stress level
4. **Visualization** → Results displayed with gauge charts and recommendations

---

## 📊 Model Details

### Input Features

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| `Hours_Worked` | Numeric | Hours worked per day | 1-24 |
| `Sleep_Hours` | Numeric | Hours of sleep per night | 1-12 |
| `Bugs` | Numeric | Number of bugs to fix | 0-50+ |
| `Deadline_Days` | Numeric | Days until deadline | 0-60+ |
| `Coffee_Cups` | Numeric | Daily coffee consumption | 0-20 |
| `Meetings` | Numeric | Number of daily meetings | 0-24 |
| `Interruptions` | Numeric | Daily interruptions count | 0-50 |
| `Experience_Years` | Categorical | Developer experience level | Junior / Mid / Senior |
| `Code_Complexity` | Categorical | Project complexity | Low / Medium / High |
| `Remote_Work` | Categorical | Remote work status | Yes / No |

### Performance Metrics

| Metric | Train | Test |
|--------|-------|------|
| R² Score | 0.92 | 0.89 |
| RMSE | 4.1 | 5.2 |
| MAE | 3.2 | 4.1 |

### Feature Importance

The top factors influencing stress prediction:

1. 😴 **Sleep Hours** - Most significant predictor
2. ⏰ **Hours Worked** - Strong correlation with stress
3. 🔔 **Interruptions** - Frequent interruptions increase stress
4. 🐛 **Bugs** - Technical debt impact
5. 📅 **Deadline Days** - Time pressure effects

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)
- Google Cloud account (optional, for cloud deployment)

### Local Installation

```bash
# Clone the repository
git clone https://github.com/Acquarts/developer-stress-predictor-ml-production.git
cd developer-stress-predictor-ml-production

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Running Locally

**Option 1: Run services separately**

```bash
# Terminal 1 - Start the API
uvicorn src.api.main:app --reload --port 8000

# Terminal 2 - Start Streamlit
streamlit run streamlit_app/app.py
```

**Option 2: Use Docker Compose**

```bash
docker-compose up --build

# Services available at:
# API:       http://localhost:8000
# API Docs:  http://localhost:8000/docs
# Streamlit: http://localhost:8501
```

### Training a New Model

```bash
python scripts/train_model.py
```

This will train a new model on the data in `data/developer_stress.csv` and save it to `models/stress_model.joblib`.

---

## 📡 API Reference

### Base URL

- **Local**: `http://localhost:8000`
- **Production**: `https://stress-api-562289298058.us-central1.run.app`

### Authentication

All prediction endpoints require an API key in the header:

```bash
X-API-Key: your-api-key
```

### Endpoints

#### Health Check

```http
GET /health
```

Returns service health status and model loading state.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

#### Single Prediction

```http
POST /predict
```

**Request Body:**
```json
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
  "Remote_Work": "Yes"
}
```

**Response:**
```json
{
  "stress_level": 67.5,
  "warnings": ["Consider taking breaks - stress level is elevated"]
}
```

#### Batch Prediction

```http
POST /predict/batch
```

**Request Body:**
```json
{
  "predictions": [
    { "Hours_Worked": 8, "Sleep_Hours": 7, ... },
    { "Hours_Worked": 12, "Sleep_Hours": 5, ... }
  ]
}
```

#### Model Information

```http
GET /model/info
```

Returns model metadata including type, parameters, and training metrics.

#### Feature Importance

```http
GET /model/features
```

Returns feature importance scores from the trained model.

---

## 🚢 Deployment

### Google Cloud Run

The project includes GitHub Actions workflows for automatic deployment:

1. **CI Pipeline** (`.github/workflows/ci.yml`)
   - Runs on every push and PR
   - Linting with Ruff
   - Type checking with MyPy
   - Unit and integration tests with Pytest
   - Security scanning with Bandit

2. **CD Pipeline** (`.github/workflows/cd.yml`)
   - Triggers on push to `main`
   - Builds Docker images
   - Pushes to Google Artifact Registry
   - Deploys to Cloud Run
   - Runs smoke tests

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `GCP_PROJECT_ID` | Google Cloud project ID |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Workload Identity Federation provider |
| `GCP_SERVICE_ACCOUNT` | Service account email for deployment |

### Manual Deployment

```bash
# Build and push API image
docker build -f infrastructure/Dockerfile -t stress-api .
docker push gcr.io/YOUR_PROJECT/stress-api

# Build and push Streamlit image
docker build -f infrastructure/Dockerfile.streamlit -t stress-streamlit .
docker push gcr.io/YOUR_PROJECT/stress-streamlit

# Deploy to Cloud Run
gcloud run deploy stress-api --image gcr.io/YOUR_PROJECT/stress-api --region us-central1
gcloud run deploy stress-streamlit --image gcr.io/YOUR_PROJECT/stress-streamlit --region us-central1
```

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_api.py

# Run with verbose output
pytest -v
```

### Test Structure

```
tests/
├── conftest.py           # Shared fixtures
├── unit/
│   ├── test_api.py       # API endpoint tests
│   ├── test_predictor.py # Model prediction tests
│   └── test_preprocessor.py # Data preprocessing tests
└── integration/
    └── test_pipeline.py  # End-to-end tests
```

### Coverage

The project maintains 80%+ test coverage across all modules.

---

## 📁 Project Structure

```
developer-stress-predictor/
│
├── 📂 .github/
│   └── workflows/
│       ├── ci.yml              # Continuous Integration
│       └── cd.yml              # Continuous Deployment
│
├── 📂 src/
│   ├── __init__.py
│   ├── config.py               # Application configuration
│   ├── 📂 api/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI application
│   │   ├── schemas.py          # Pydantic models
│   │   └── dependencies.py     # Dependency injection
│   ├── 📂 data/
│   │   ├── __init__.py
│   │   └── preprocessor.py     # Data transformations
│   ├── 📂 models/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Model training
│   │   └── predictor.py        # Model inference
│   └── 📂 monitoring/
│       ├── __init__.py
│       └── metrics.py          # Prometheus metrics
│
├── 📂 streamlit_app/
│   ├── app.py                  # Main Streamlit app
│   ├── utils.py                # Utility functions
│   └── 📂 components/
│       ├── __init__.py
│       ├── prediction_form.py  # Input form component
│       ├── results_display.py  # Results visualization
│       └── monitoring_dashboard.py  # Monitoring UI
│
├── 📂 tests/
│   ├── conftest.py             # Test fixtures
│   ├── 📂 unit/
│   └── 📂 integration/
│
├── 📂 infrastructure/
│   ├── Dockerfile              # API container
│   ├── Dockerfile.streamlit    # Streamlit container
│   └── docker-compose.yml      # Local development
│
├── 📂 models/
│   └── stress_model.joblib     # Trained model
│
├── 📂 data/
│   └── developer_stress.csv    # Training data
│
├── 📂 scripts/
│   └── train_model.py          # Training script
│
├── 📂 notebooks/
│   └── developer_stress.ipynb  # Exploratory analysis
│
├── .env.example                # Environment template
├── .gitignore
├── pyproject.toml              # Project configuration
├── requirements.txt            # Dependencies
└── README.md
```

---

## 🛠️ Tech Stack

### Machine Learning
| Technology | Purpose |
|------------|---------|
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white) | Model training & inference |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white) | Data manipulation |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white) | Numerical computing |
| ![Joblib](https://img.shields.io/badge/Joblib-3776AB?logo=python&logoColor=white) | Model serialization |

### Backend
| Technology | Purpose |
|------------|---------|
| ![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white) | REST API framework |
| ![Pydantic](https://img.shields.io/badge/Pydantic-E92063?logo=pydantic&logoColor=white) | Data validation |
| ![Uvicorn](https://img.shields.io/badge/Uvicorn-499848?logo=gunicorn&logoColor=white) | ASGI server |

### Frontend
| Technology | Purpose |
|------------|---------|
| ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white) | Web application |
| ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white) | Interactive charts |

### Infrastructure
| Technology | Purpose |
|------------|---------|
| ![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white) | Containerization |
| ![Google Cloud](https://img.shields.io/badge/Cloud%20Run-4285F4?logo=google-cloud&logoColor=white) | Serverless deployment |
| ![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?logo=github-actions&logoColor=white) | CI/CD pipelines |

### Quality Assurance
| Technology | Purpose |
|------------|---------|
| ![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?logo=pytest&logoColor=white) | Testing framework |
| ![Ruff](https://img.shields.io/badge/Ruff-D7FF64?logo=ruff&logoColor=black) | Linting |
| ![MyPy](https://img.shields.io/badge/MyPy-3776AB?logo=python&logoColor=white) | Type checking |
| ![Bandit](https://img.shields.io/badge/Bandit-3776AB?logo=python&logoColor=white) | Security scanning |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ☕ and 🧠 by <a href="https://github.com/Acquarts">Acquarts</a>
</p>

<p align="center">
  <a href="https://stress-streamlit-562289298058.us-central1.run.app">
    <img src="https://img.shields.io/badge/🚀_Try_the_Live_App-FF4B4B?style=for-the-badge" alt="Live App">
  </a>
</p>
