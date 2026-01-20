# 🧠 Developer Stress Predictor

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Run-4285F4?logo=google-cloud&logoColor=white)](https://cloud.google.com/run)
[![CI](https://github.com/Acquarts/developer-stress-predictor-ml-production/actions/workflows/ci.yml/badge.svg)](https://github.com/Acquarts/developer-stress-predictor-ml-production/actions/workflows/ci.yml)
[![CD](https://github.com/Acquarts/developer-stress-predictor-ml-production/actions/workflows/cd.yml/badge.svg)](https://github.com/Acquarts/developer-stress-predictor-ml-production/actions/workflows/cd.yml)

ML-powered tool to predict developer stress levels based on work patterns and habits.

🚀 **[Try the Live App](https://stress-streamlit-562289298058.us-central1.run.app)**

---

## ✨ Features

- **Stress Prediction**: Predicts stress level (0-100) based on 10 work-related factors
- **REST API**: FastAPI with OpenAPI docs, batch predictions, and health checks
- **Interactive UI**: Streamlit dashboard with visualizations and recommendations
- **Production Ready**: CI/CD, Docker, monitoring, and auto-scaling on Cloud Run

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Streamlit  │────▶│   FastAPI   │────▶│ RandomForest│
│     UI      │     │     API     │     │    Model    │
└─────────────┘     └─────────────┘     └─────────────┘
     Cloud Run          Cloud Run           R² = 0.89
```

## 📊 Model

| Feature | Description |
|---------|-------------|
| Hours_Worked | Hours worked per day |
| Sleep_Hours | Hours of sleep |
| Bugs | Number of bugs to fix |
| Deadline_Days | Days until deadline |
| Coffee_Cups | Daily coffee intake |
| Meetings | Number of meetings |
| Interruptions | Daily interruptions |
| Experience_Years | Junior / Mid / Senior |
| Code_Complexity | Low / Medium / High |
| Remote_Work | Yes / No |

**Performance**: R² = 0.89 | RMSE = 5.2

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/Acquarts/developer-stress-predictor-ml-production.git
cd developer-stress-predictor-ml-production
pip install -r requirements.txt

# Run API
uvicorn src.api.main:app --reload

# Run Streamlit (in another terminal)
streamlit run streamlit_app/app.py
```

## 🐳 Docker

```bash
docker-compose up
# API: http://localhost:8000/docs
# UI:  http://localhost:8501
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |
| GET | `/health` | Health check |
| GET | `/model/info` | Model metadata |
| GET | `/model/features` | Feature importance |

## 🧪 Testing

```bash
pytest --cov=src
```

## 📁 Project Structure

```
├── src/
│   ├── api/           # FastAPI endpoints
│   ├── data/          # Data preprocessing
│   └── models/        # ML model training & inference
├── streamlit_app/     # Streamlit UI
├── tests/             # Unit & integration tests
├── infrastructure/    # Dockerfiles
└── .github/workflows/ # CI/CD pipelines
```

## 🛠️ Tech Stack

- **ML**: scikit-learn, pandas, numpy
- **API**: FastAPI, Pydantic, uvicorn
- **UI**: Streamlit, Plotly
- **Infra**: Docker, Cloud Run, GitHub Actions
- **Quality**: pytest, ruff, mypy

---

Made with ☕ by [Acquarts](https://github.com/Acquarts)
