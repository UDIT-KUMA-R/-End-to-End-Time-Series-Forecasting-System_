# End-to-End Time Series Forecasting System

## Project Overview

Production-ready Time Series Forecasting Backend System built using:

- Python
- FastAPI
- ARIMA
- Prophet
- XGBoost
- LSTM
- TensorFlow
- Scikit-learn

The system forecasts future weekly sales for each state using historical sales data.

---

# Features

- Multi-model forecasting
- Automatic model comparison
- Best model selection
- State-wise forecasting
- REST API using FastAPI
- Production-ready architecture
- Logging and exception handling
- Model persistence
- Swagger API documentation

---

# Tech Stack

- Python
- Pandas
- NumPy
- Statsmodels
- Prophet
- XGBoost
- TensorFlow/Keras
- FastAPI
- Uvicorn

---

# Project Structure

```text
project/
├── data/
├── models/
├── reports/
├── src/
├── requirements.txt
└── README.md
```

---

# Models Used

1. ARIMA
2. Prophet
3. XGBoost
4. LSTM

---

# Evaluation Metrics

- MAE
- RMSE
- MAPE

---

# API Endpoints

## Health Check

GET /health

---

## Available Models

GET /models

---

## Predict Future Sales

POST /predict

Example Request:

```json
{
  "state": "California",
  "horizon": 8
}
```

---

# Run Project

## Install Requirements

```bash
pip install -r requirements.txt
```

## Run API

```bash
uvicorn src.api:app --reload
```

---

# Swagger Documentation

Open:

```text
http://127.0.0.1:8000/docs
```

---

# Future Improvements

- Docker deployment
- CI/CD pipeline
- Cloud deployment
- Real-time forecasting
- Automated retraining
- Monitoring dashboard

---

# Author

Udit Kumar