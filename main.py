# src/api.py

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

from tensorflow.keras.models import load_model

app = FastAPI()


# =========================
# Load Best Models
# =========================

best_models = pd.read_csv(
    "reports/metrics/best_models.csv"
)


# =========================
# Request Body
# =========================

class PredictionRequest(BaseModel):

    state: str
    horizon: int = 8


# =========================
# Health Endpoint
# =========================

@app.get("/health")
def health():

    return {
        "status": "API is running"
    }


# =========================
# Models Endpoint
# =========================

@app.get("/models")
def models():

    return best_models.to_dict(
        orient="records"
    )


# =========================
# Prediction Endpoint
# =========================

@app.post("/predict")
def predict(request: PredictionRequest):

    state = request.state
    horizon = request.horizon

    # Get best model for state
    row = best_models[
        best_models["state"] == state
    ]

    if row.empty:

        return {
            "error": "State not found"
        }

    model_name = row.iloc[0]["model"]

    predictions = []

    # =====================
    # ARIMA
    # =====================

    if model_name == "ARIMA":

        model = joblib.load(
            f"models/arima/{state}_arima.pkl"
        )

        forecast = model.forecast(
            steps=horizon
        )

        predictions = forecast.tolist()

    # =====================
    # Prophet
    # =====================

    elif model_name == "Prophet":

        model = joblib.load(
            f"models/prophet/{state}_prophet.pkl"
        )

        future = model.make_future_dataframe(
            periods=horizon,
            freq='W'
        )

        forecast = model.predict(future)

        predictions = (
            forecast["yhat"]
            .tail(horizon)
            .tolist()
        )

    # =====================
    # XGBoost
    # =====================

    elif model_name == "XGBoost":

        model = joblib.load(
            f"models/xgboost/{state}_xgboost.pkl"
        )

        df = pd.read_csv(
            "data/processed/feature_engineered_sales.csv"
        )

        state_df = df[
            df["state"] == state
        ]

        feature_columns = [
            'lag_1',
            'lag_7',
            'lag_30',
            'rolling_mean_4',
            'rolling_std_4',
            'rolling_mean_8',
            'rolling_std_8',
            'year',
            'month',
            'quarter',
            'week_of_year',
            'is_holiday'
        ]

        latest_data = (
            state_df[feature_columns]
            .tail(1)
        )

        forecast = model.predict(
            latest_data
        )

        predictions = (
            forecast.tolist() * horizon
        )

    # =====================
    # LSTM
    # =====================

    elif model_name == "LSTM":

        model = load_model(
            f"models/lstm/{state}_lstm.h5"
        )

        predictions = [
            "LSTM prediction generated"
        ]

    return {
        "state": state,
        "best_model": model_name,
        "forecast_horizon": horizon,
        "predictions": predictions
    }