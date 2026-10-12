import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data import DEFAULT_PATH, add_features, clean, load_raw, split_xy
from src.features import build_preprocessor
from src.models import make_pipeline, one_standard_error_choice
from src.profit import RetentionEconomics, best_threshold, campaign_profit


def _tiny_frame(total_charges, tenure):
    return pd.DataFrame({"tenure": tenure, "TotalCharges": total_charges, "Churn": ["No"] * len(tenure)})


def test_clean_fills_blank_total_charges_for_new_customers():
    out = clean(_tiny_frame([" ", "10.5"], [0, 1]))
    assert out["TotalCharges"].tolist() == [0.0, 10.5]
    assert out["Churn"].tolist() == [0, 0]


def test_clean_rejects_blank_total_charges_for_existing_customers():
    with pytest.raises(ValueError):
        clean(_tiny_frame([" "], [5]))


@pytest.fixture(scope="module")
def data():
    df = clean(load_raw())
    X, y = split_xy(df)
    return train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


def test_dataset_shape_and_target(data):
    X_train, X_test, y_train, y_test = data
    assert len(X_train) + len(X_test) == 7043
    assert "customerID" not in X_train and "Churn" not in X_train
    assert set(y_train.unique()) == {0, 1}


def test_pipeline_outputs_valid_probabilities(data):
    X_train, X_test, y_train, _ = data
    pipe = make_pipeline(LogisticRegression(max_iter=2000), X_train).fit(X_train, y_train)
    p = pipe.predict_proba(X_test)[:, 1]
    assert p.shape == (len(X_test),) and ((p >= 0) & (p <= 1)).all()


def test_preprocessor_is_fit_on_training_data_only(data):
    X_train, X_test, *_ = data
    prep = build_preprocessor(X_train).fit(X_train)
    scaler = prep.named_transformers_["num"]
    num = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    np.testing.assert_allclose(scaler.mean_, X_train[num].mean().values)


def test_unseen_category_does_not_crash(data):
    X_train, X_test, y_train, _ = data
    pipe = make_pipeline(LogisticRegression(max_iter=2000), X_train).fit(X_train, y_train)
    row = X_test.iloc[[0]].copy()
    row["Contract"] = "Five year"            # category never seen in training
    assert 0 <= pipe.predict_proba(row)[0, 1] <= 1


def test_add_features(data):
    X_train, *_ = data
    out = add_features(X_train)
    assert out["n_addons"].between(0, 6).all() and set(out["is_autopay"].unique()) <= {0, 1}


def test_breakeven_probability():
    econ = RetentionEconomics(contact_cost=500, customer_value=5000, success_rate=0.3)
    assert econ.breakeven_probability == pytest.approx(1 / 3)


def test_campaign_profit_arithmetic():
    econ = RetentionEconomics(contact_cost=100, customer_value=1000, success_rate=0.5)
    y, p = np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.2, 0.1])
    # contact first two: churner gives 0.5*1000-100 = 400, non-churner gives -100
    assert campaign_profit(y, p, 0.5, econ) == pytest.approx(300)
    assert campaign_profit(y, p, 1.1, econ) == 0          # contact nobody


def test_best_threshold_on_perfectly_separated_scores():
    econ = RetentionEconomics()
    y = np.array([1] * 20 + [0] * 80)
    p = np.where(y == 1, 0.9, 0.05)
    t = best_threshold(y, p, econ)
    assert 0.05 < t <= 0.9
    assert campaign_profit(y, p, t, econ) == pytest.approx(20 * (0.3 * 5000 - 500))


def test_one_standard_error_rule_prefers_simpler_model_within_noise():
    summary = {"Logistic Regression": (0.846, 0.012), "Random Forest": (0.848, 0.010), "LightGBM": (0.850, 0.012)}
    assert one_standard_error_choice(summary, 5) == "Logistic Regression"
    summary["LightGBM"] = (0.90, 0.005)                    # clearly better -> complexity is justified
    assert one_standard_error_choice(summary, 5) == "LightGBM"


def test_saved_model_scores_new_customers():
    from src.predict import DEFAULT_MODEL_PATH, load_bundle, score
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("run `python -m src.train` first")
    df = clean(load_raw()).head(50)
    X, _ = split_xy(df)
    out = score(X, load_bundle())
    assert list(out.columns) == ["churn_probability", "contact", "risk_tier"] and len(out) == 50
