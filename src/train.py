"""Reproduce the notebook's final model from the command line.

    python -m src.train                       # writes models/churn_model.joblib + reports/metrics.json
    python -m src.train --out-model /tmp/m.joblib --out-metrics /tmp/m.json

Same steps as the notebook: stratified split (test set locked) -> randomized search per model family
(5-fold CV) -> one-standard-error selection -> Platt calibration -> profit-optimal threshold from
out-of-fold predictions -> one-shot test evaluation -> save bundle.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     train_test_split)

from .data import load_and_prepare, split_xy
from .models import (DEFAULT_N_ITER, candidate_models, make_pipeline, one_standard_error_choice,
                     search_spaces)
from .profit import RetentionEconomics, best_threshold

ROOT = Path(__file__).resolve().parents[1]
SEED, N_SPLITS = 42, 5


def train(economics: RetentionEconomics | None = None, n_iter: dict | None = None, verbose: bool = True):
    econ = economics or RetentionEconomics()
    n_iter = n_iter or DEFAULT_N_ITER
    df = load_and_prepare(engineered=False)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    searches, summary = {}, {}
    for name, iters in n_iter.items():
        s = RandomizedSearchCV(
            make_pipeline(candidate_models(SEED)[name], X_train), search_spaces()[name],
            n_iter=iters, scoring="roc_auc", cv=cv, random_state=SEED, n_jobs=1)
        s.fit(X_train, y_train)
        i = s.best_index_
        summary[name] = (s.cv_results_["mean_test_score"][i], s.cv_results_["std_test_score"][i])
        searches[name] = s
        if verbose:
            print(f"{name:20s} CV AUC {summary[name][0]:.4f} ± {summary[name][1]:.4f}")

    chosen = one_standard_error_choice(summary, N_SPLITS)
    calibrated = CalibratedClassifierCV(clone(searches[chosen].best_estimator_), method="sigmoid", cv=5)
    oof = cross_val_predict(calibrated, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = best_threshold(y_train, oof, econ)

    model = clone(calibrated).fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    pred = (p >= threshold).astype(int)
    metrics = {
        "model": chosen, "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "cv_roc_auc_tuned": {k: round(v[0], 4) for k, v in summary.items()},
        "test_roc_auc": round(roc_auc_score(y_test, p), 4),
        "test_pr_auc": round(average_precision_score(y_test, p), 4),
        "test_brier": round(brier_score_loss(y_test, p), 4),
        "threshold": round(threshold, 3), "economics_assumptions": asdict(econ),
        "test_recall_at_threshold": round(recall_score(y_test, pred), 4),
        "test_precision_at_threshold": round(precision_score(y_test, pred), 4),
    }
    bundle = {"model": model, "threshold": float(threshold), "features": list(X_train.columns),
              "economics": asdict(econ), "metrics": metrics}
    return bundle


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-model", type=Path, default=ROOT / "models" / "churn_model.joblib")
    ap.add_argument("--out-metrics", type=Path, default=ROOT / "reports" / "metrics.json")
    ap.add_argument("--contact-cost", type=float, default=500)
    ap.add_argument("--customer-value", type=float, default=5000)
    ap.add_argument("--success-rate", type=float, default=0.30)
    args = ap.parse_args()

    econ = RetentionEconomics(args.contact_cost, args.customer_value, args.success_rate)
    bundle = train(econ)
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out_model)
    args.out_metrics.write_text(json.dumps(bundle["metrics"], indent=2))
    print(json.dumps(bundle["metrics"], indent=2))


if __name__ == "__main__":
    main()
