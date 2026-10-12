"""Candidate models, hyper-parameter search spaces and the model-selection rule."""
import numpy as np
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .features import build_preprocessor

# Number of random-search iterations per model family (kept modest so it runs on a laptop).
DEFAULT_N_ITER = {"Logistic Regression": 20, "Decision Tree": 20, "Random Forest": 12, "LightGBM": 30}

# Simplest -> most complex. Used by the "one-standard-error" selection rule.
COMPLEXITY_ORDER = ["Logistic Regression", "Decision Tree", "Random Forest", "LightGBM"]


def make_pipeline(model, X) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(X)), ("model", model)])


def candidate_models(seed: int = 42) -> dict:
    return {
        "Baseline": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=seed),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=10, class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=seed, n_jobs=1),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=15,
            class_weight="balanced", random_state=seed, n_jobs=1, verbose=-1),
    }


def search_spaces() -> dict:
    return {
        "Logistic Regression": {"model__C": loguniform(1e-3, 1e2)},
        "Decision Tree": {
            "model__max_depth": randint(3, 10),
            "model__min_samples_leaf": randint(5, 60),
        },
        "Random Forest": {
            "model__n_estimators": randint(150, 350),
            "model__max_depth": randint(4, 12),
            "model__min_samples_leaf": randint(3, 30),
            "model__max_features": ["sqrt", 0.3, 0.5],
        },
        "LightGBM": {
            "model__n_estimators": randint(100, 400),
            "model__learning_rate": loguniform(0.01, 0.1),
            "model__num_leaves": randint(4, 32),
            "model__min_child_samples": randint(10, 60),
            "model__subsample": uniform(0.6, 0.4),
            "model__subsample_freq": [1],
            "model__colsample_bytree": uniform(0.5, 0.5),
            "model__reg_lambda": loguniform(1e-2, 10),
        },
    }


def one_standard_error_choice(summary: dict, n_splits: int) -> str:
    """Pick the simplest model whose CV score is within one standard error of the best.

    `summary` maps model name -> (mean_cv_score, std_cv_score_across_folds).
    """
    best = max(summary, key=lambda k: summary[k][0])
    best_mean, best_std = summary[best]
    threshold = best_mean - best_std / np.sqrt(n_splits)
    for name in COMPLEXITY_ORDER:
        if name in summary and summary[name][0] >= threshold:
            return name
    return best
