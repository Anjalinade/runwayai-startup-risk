from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ======================================
# App initialization
# ======================================
app = FastAPI(
    title="RunwayAI",
    description="Startup Failure Risk Prediction API",
    version="1.0"
)

# ======================================
# Safe paths (cloud compatible)
# ======================================
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "logistic_model.pkl"
DATA_PATH = BASE_DIR / "data" / "processed_startup_data.csv"

# ======================================
# Load model and feature list
# ======================================
model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)
FEATURE_COLUMNS = df.drop(columns=["failed"]).columns.tolist()

# ======================================
# Input schema (STRICT)
# ======================================
class StartupInput(BaseModel):
    latitude: float
    longitude: float

    age_first_funding_year: float
    age_last_funding_year: float
    age_first_milestone_year: float
    age_last_milestone_year: float

    relationships: float
    funding_rounds: float
    funding_total_usd: float
    milestones: float
    avg_participants: float

    is_CA: int
    is_NY: int
    is_MA: int
    is_TX: int
    is_otherstate: int

    is_software: int
    is_web: int
    is_mobile: int
    is_enterprise: int
    is_advertising: int
    is_gamesvideo: int
    is_ecommerce: int
    is_biotech: int
    is_consulting: int
    is_othercategory: int

    has_VC: int
    has_angel: int
    has_roundA: int
    has_roundB: int
    has_roundC: int
    has_roundD: int

    is_top500: int

# ======================================
# Health check
# ======================================
@app.get("/")
def health():
    return {"status": "RunwayAI backend is live 🚀"}

# ======================================
# Feature list endpoint
# ======================================
@app.get("/features")
def get_features():
    return {"required_features": FEATURE_COLUMNS}

# ======================================
# Prediction endpoint
# ======================================
@app.post("/predict")
def predict_failure(data: StartupInput):
    input_dict = data.dict()

    # Validate schema
    missing = set(FEATURE_COLUMNS) - set(input_dict.keys())
    extra = set(input_dict.keys()) - set(FEATURE_COLUMNS)

    if missing or extra:
        raise HTTPException(
            status_code=400,
            detail={
                "missing_features": list(missing),
                "extra_features": list(extra)
            }
        )

    # Create dataframe in correct order
    X = pd.DataFrame([input_dict])[FEATURE_COLUMNS]

    # Prediction
    failure_prob = float(model.predict_proba(X)[0][1])

    # Risk level
    if failure_prob < 0.4:
        risk_level = "Low Risk"
    elif failure_prob < 0.7:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    # Explainability (top features)
    coef = model.coef_[0]
    contributions = coef * X.iloc[0].values

    feature_contrib = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "contribution": contributions
    })

    top_risk = (
        feature_contrib
        .sort_values("contribution", ascending=False)
        .head(5)["feature"]
        .tolist()
    )

    positive_signals = (
        feature_contrib
        .sort_values("contribution", ascending=True)
        .head(5)["feature"]
        .tolist()
    )

    return {
        "failure_probability": round(failure_prob, 4),
        "risk_level": risk_level,
        "top_risk_factors": top_risk,
        "positive_signals": positive_signals
    }
