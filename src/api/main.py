from fastapi import FastAPI, HTTPException
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

app = FastAPI(title="Credit Risk Prediction API")

# FIXED: Load tuned v6 model (your run UUID)
mlruns_path = os.path.join(project_root, "mlruns")
model_path = os.path.join(
    mlruns_path,
    "169414275103424388/bba7e4143c5f4ce0a537385c8bb39ba8/artifacts/fine_tuned_rf_model",
)

mlflow.set_tracking_uri(f"file:{mlruns_path}")
model = mlflow.sklearn.load_model(model_path)
print("Tuned model v6 loaded, n_features_in_:", model.n_features_in_)  # 17 with RFM

# Raw cols (17 from v6: 15 + Recency, Frequency)
raw_cols = [
    "Amount",
    "Value",
    "PricingStrategy",
    "FraudResult",
    "TransactionHour",
    "TransactionDay",
    "TransactionMonth",
    "TransactionYear",
    "Value_mean",
    "Recency",
    "Frequency",
    "CurrencyCode",
    "CountryCode",
    "ProviderId",
    "ProductId",
    "ProductCategory",
    "ChannelId",
]

# Import Pydantic
from src.api.pydantic_models import CustomerInput, RiskPrediction


@app.get("/")
def read_root():
    return {"message": "Credit Risk API v6 - Tuned RF Loaded"}


@app.post("/predict", response_model=RiskPrediction)
async def predict_risk(input_data: CustomerInput):
    try:
        input_df = pd.DataFrame([input_data.dict()])
        # Add missing cols with defaults (17 raw for v6)
        defaults = {
            "TransactionDay": 16,
            "TransactionMonth": 7,
            "TransactionYear": 2019,
            "Value_mean": 9900.0,
            "Recency": 0,
            "Frequency": 50,
            "CurrencyCode": "UGX",
            "CountryCode": 256,
            "ProviderId": "ProviderId_4",
            "ProductId": "ProductId_6",
            "ProductCategory": "financial_services",
            "ChannelId": "ChannelId_3",
        }
        for col in raw_cols:
            if col not in input_df.columns:
                input_df[col] = defaults[col]
        input_df = input_df[raw_cols]  # Reorder
        print(f"Raw input shape: {input_df.shape}")  # (1, 17)
        prob = model.predict_proba(input_df)[:, 1][0]
        pred = 1 if prob > 0.5 else 0  # Threshold 0.5 (or 0.45 for more low-risk false)
        score = 100 * (1 - prob)
        return RiskPrediction(
            risk_probability=round(prob, 4),
            is_high_risk=bool(pred),
            credit_score=round(score, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
