import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Try loading the trained model pipeline
try:
    model = joblib.load("model.joblib")
    model_status = "loaded"
except Exception:
    model = None
    model_status = "not loaded"

MODEL_VERSION = "v1.0"

app = FastAPI(title="Churn Prediction API", version="1.0")


class Customer(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


@app.post("/predict")
def predict_customer(data: Customer):
    if model is None:
        return {"error": "Model not loaded"}

    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    return {"prediction": int(prediction)}


@app.get("/health")
def health_check():
    return {
        "status": model_status,
        "model_version": MODEL_VERSION
    }
