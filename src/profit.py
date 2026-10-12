"""Turn churn probabilities into a retention-campaign decision.

Economics (all assumptions, change them to your business):
  contact_cost   - cost of one retention offer, paid for every contacted customer
  customer_value - value of keeping a customer who would otherwise churn
  success_rate   - share of true churners the offer actually saves
Expected profit of contacting customer i = success_rate * customer_value * churn_i - contact_cost.
Contacting is worthwhile when P(churn) > contact_cost / (success_rate * customer_value).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RetentionEconomics:
    contact_cost: float = 500.0
    customer_value: float = 5000.0
    success_rate: float = 0.30

    @property
    def breakeven_probability(self) -> float:
        return self.contact_cost / (self.success_rate * self.customer_value)


def campaign_profit(y_true, proba, threshold: float, econ: RetentionEconomics) -> float:
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    contacted = proba >= threshold
    gain = y_true * econ.success_rate * econ.customer_value - econ.contact_cost
    return float((contacted * gain).sum())


def profit_curve(y_true, proba, econ: RetentionEconomics, thresholds=None) -> pd.DataFrame:
    thresholds = np.linspace(0.0, 1.0, 101) if thresholds is None else thresholds
    rows = [
        {
            "threshold": t,
            "profit": campaign_profit(y_true, proba, t, econ),
            "contacted_share": float((np.asarray(proba) >= t).mean()),
        }
        for t in thresholds
    ]
    return pd.DataFrame(rows)


def best_threshold(y_true, proba, econ: RetentionEconomics) -> float:
    curve = profit_curve(y_true, proba, econ)
    return float(curve.loc[curve["profit"].idxmax(), "threshold"])
