# Telecom Customer Churn Analysis

A complete end-to-end data science project analyzing customer churn patterns
for a telecom company using Python, Pandas, Seaborn, and Scikit-learn.

## Dataset
- **Source:** [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,032 customers, 21 features
- **Target variable:** `Churn` (Yes / No)

## Tools & Libraries
- Python 3.11
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn

## Project Steps
1. Data loading & inspection
2. Data cleaning (fixed hidden nulls in `TotalCharges`)
3. Exploratory Data Analysis (EDA)
4. Feature encoding with `pd.get_dummies()`
5. Logistic Regression model training
6. Model evaluation & business insights

## Key Findings
- **26% churn rate** across the dataset
- **Month-to-month contract** customers churn at ~42% — far higher than annual contracts
- **Fiber optic** internet service is the strongest predictor of churn
- **Two-year contracts** reduce churn probability the most (-1.30 coefficient)
- Customers with **Online Security** or **Tech Support** are significantly less likely to churn

## Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | 79% |
| Precision (Churn) | 62% |
| Recall (Churn) | 52% |
| F1-Score (Churn) | 56% |

##  Business Recommendations
1. **Incentivize long-term contracts** — offer discounts for 1 or 2 year plans
2. **Target fiber optic customers** with retention offers
3. **Bundle security & support services** — they strongly correlate with loyalty
4. **Flag electronic check users** — highest churn rate by payment method

## Files
- churn_analysis.ipynb — full analysis notebook
- WA_Fn-UseC_-Telco-Customer-Churn.csv — dataset (download from Kaggle)
