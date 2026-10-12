"""Score new customers with the saved model bundle."""
from pathlib import Path

import joblib
import pandas as pd

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_model.joblib"


def load_bundle(path: Path | str = DEFAULT_MODEL_PATH) -> dict:
    return joblib.load(path)


def score(df: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    """Return churn probability, a contact/no-contact decision and a risk tier per customer."""
    bundle = bundle or load_bundle()
    proba = bundle["model"].predict_proba(df[bundle["features"]])[:, 1]
    thr = bundle["threshold"]
    out = pd.DataFrame({"churn_probability": proba}, index=df.index)
    out["contact"] = proba >= thr
    out["risk_tier"] = pd.cut(proba, [-0.01, thr, max(0.6, thr + 0.05), 1.0],
                              labels=["Low", "Medium", "High"])
    return out
