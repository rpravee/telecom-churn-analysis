import pytest
from fastapi.testclient import TestClient

from api import app
from src.predict import DEFAULT_MODEL_PATH

pytestmark = pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.exists(), reason="run `python -m src.train` first"
)

client = TestClient(app)

HIGH_RISK = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0, "TotalCharges": 85.0,
}

LOW_RISK = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
    "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 60.0, "TotalCharges": 3600.0,
}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_returns_valid_response():
    r = client.post("/predict", json=HIGH_RISK)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"churn_probability", "contact", "risk_tier", "threshold"}
    assert 0 <= body["churn_probability"] <= 1


def test_high_risk_customer_scores_above_low_risk_customer():
    high = client.post("/predict", json=HIGH_RISK).json()
    low = client.post("/predict", json=LOW_RISK).json()
    assert high["churn_probability"] > low["churn_probability"]
    assert high["contact"] is True
    assert low["contact"] is False


def test_invalid_contract_is_rejected():
    bad = {**HIGH_RISK, "Contract": "Lifetime"}
    assert client.post("/predict", json=bad).status_code == 422


def test_missing_field_is_rejected():
    bad = {k: v for k, v in HIGH_RISK.items() if k != "tenure"}
    assert client.post("/predict", json=bad).status_code == 422