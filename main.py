from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import json

model = joblib.load("churn_model.pkl")

with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a telecom customer will churn using Random Forest",
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictionResult(BaseModel):
    churn: bool
    churn_probability: float
    risk_level: str

@app.get("/")
def root():
    return {"message": "API is running"}
    
@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResult)
def predict(customer: CustomerData):
    try:
        input_df = pd.DataFrame([customer.model_dump()])

        input_df = pd.get_dummies(input_df, drop_first=True)

        
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        
        
        prediction = model.predict(input_df)[0]
        probability = float(model.predict_proba(input_df)[0][1])

        
        if probability >= 0.7:
            risk = "High"
        elif probability >= 0.4:
            risk = "Medium"
        else:
            risk = "Low"

        return {
            "churn": bool(prediction),
            "churn_probability": round(probability, 4),
            "risk_level": risk
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
