# Telco Customer Churn Prediction

Predicts whether a telecom customer will churn or stay, using their account and service details. Built with Logistic Regression, Random Forest, and XGBoost, compared on F1, accuracy, and PR-AUC.

## Dataset

- **Source:** [Telco Customer Churn (IBM dataset)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- ~7,000 fictional customers with demographic info, subscribed services, account details, and whether they churned.

## Goal

Binary classification to predict the `Churn` column (Yes/No) using the other customer features.


## Models Used

- Logistic Regression (baseline)
- Random Forest Classifier (`max_depth=6`, `min_samples_leaf=20` — tuned to fix overfitting)
- XGBoost Classifier

## Results (Test Set)

| Model | F1 | Accuracy | PR-AUC |
|---|---|---|---|
| Logistic Regression | 0.632 | 0.744 | 0.628 |
| Random Forest | 0.627 | 0.742 | 0.648 |
| XGBoost | 0.628 | 0.744 | 0.645 |

All three models perform similarly, with train/test scores staying close together (no overfitting after tuning). Logistic Regression is the most interpretable of the three and performs on par with the others, making it a strong choice for this dataset.

## Visualizations

- `pr_curves.png` — Precision-Recall curves comparing all three models
- `confusion_matrices.png` — Test set confusion matrix for each model
- `xgb_feature_importance.png` — Top 10 features driving churn predictions (XGBoost)

## Tech Stack

- Python
- pandas
- scikit-learn
- XGBoost
- matplotlib / seaborn