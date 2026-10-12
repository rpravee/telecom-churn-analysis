"""FastAPI service that scores one telecom customer with the saved churn model."""
from functools import lru_cache
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import load_bundle, score

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    """One customer, using the same columns as the IBM Telco dataset."""

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoInternet
    OnlineBackup: YesNoNoInternet
    DeviceProtection: YesNoNoInternet
    TechSupport: YesNoNoInternet
    StreamingTV: YesNoNoInternet
    StreamingMovies: YesNoNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class Prediction(BaseModel):
    churn_probability: float
    contact: bool
    risk_tier: str
    threshold: float


app = FastAPI(
    title="Telecom Churn API",
    description="Scores a customer's churn probability and says whether to contact them.",
)


@lru_cache(maxsize=1)
def get_bundle() -> dict:
    """Load the saved model once and reuse it for every request."""
    return load_bundle()


@app.get("/health")
def health() -> dict:
    try:
        get_bundle()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model file not found. Run: python -m src.train")
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    try:
        bundle = get_bundle()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model file not found. Run: python -m src.train")
    row = pd.DataFrame([customer.model_dump()])
    result = score(row, bundle).iloc[0]
    return Prediction(
        churn_probability=round(float(result["churn_probability"]), 4),
        contact=bool(result["contact"]),
        risk_tier=str(result["risk_tier"]),
        threshold=bundle["threshold"],
    )