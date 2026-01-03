"""
FastAPI application to serve churn prediction model.
"""

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained model pipeline
model = joblib.load("model.joblib")

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
    """
    Predict churn for a given customer.
    """
    # Convert input to DataFrame with correct column names
    input_df = pd.DataFrame(
        [
            {
                "CreditScore": data.CreditScore,
                "Geography": data.Geography,
                "Gender": data.Gender,
                "Age": data.Age,
                "Tenure": data.Tenure,
                "Balance": data.Balance,
                "NumOfProducts": data.NumOfProducts,
                "HasCrCard": data.HasCrCard,
                "IsActiveMember": data.IsActiveMember,
                "EstimatedSalary": data.EstimatedSalary,
            }
        ]
    )

    prediction = model.predict(input_df)[0]
    return {"prediction": int(prediction)}
