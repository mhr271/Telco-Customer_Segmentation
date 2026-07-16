import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    f1_score,
    average_precision_score
)

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

 
df = df.drop(columns=["customerID"])
 
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
 
df["Churn"] = df["Churn"].map({"Yes":1,"No":0})

y = df["Churn"]
x = df.drop("Churn",axis=1)

categorical_cols = x.select_dtypes(include=["object"]).columns
numerical_cols = x.select_dtypes(include=["number"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

x_train , x_test , y_train , y_test = train_test_split(x,y,
    test_size =0.2 ,
    random_state= 100 ,
    stratify = y )

neg, pos = y_train.value_counts()
scale_pos_weight = neg / pos

lr = Pipeline([
    ("preprocess", preprocessor),
    ('model', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=100))
])

lr.fit(x_train, y_train)

lr_y_train_pred = lr.predict(x_train)
lr_y_test_pred = lr.predict(x_test)

lr_y_train_prob = lr.predict_proba(x_train)[:, 1]
lr_y_test_prob = lr.predict_proba(x_test)[:, 1]

lr_train_f1 = f1_score(y_train, lr_y_train_pred)
lr_train_accuracy = accuracy_score(y_train, lr_y_train_pred)
lr_train_pr_auc = average_precision_score(y_train, lr_y_train_prob)
lr_train_confusionmatrix = confusion_matrix(y_train, lr_y_train_pred)

lr_test_f1 = f1_score(y_test, lr_y_test_pred)
lr_test_accuracy = accuracy_score(y_test, lr_y_test_pred)
lr_test_pr_auc = average_precision_score(y_test, lr_y_test_prob)
lr_test_confusionmatrix = confusion_matrix(y_test, lr_y_test_pred)

rf = Pipeline([
 ('preprocess',preprocessor),
 ('model',RandomForestClassifier(n_estimators=300,
        max_depth=6,          
        min_samples_leaf=20,   
        class_weight='balanced',
        random_state=100))
])

rf.fit(x_train,y_train)

rf_y_train_pred = rf.predict(x_train)
rf_y_test_pred = rf.predict(x_test)

rf_y_train_prob = rf.predict_proba(x_train)[:, 1]
rf_y_test_prob = rf.predict_proba(x_test)[:, 1]

rf_train_f1 = f1_score(y_train,rf_y_train_pred)
rf_train_accuracy = accuracy_score(y_train,rf_y_train_pred)
rf_train_pr_auc = average_precision_score(y_train,rf_y_train_prob)
rf_train_confusionmatrix = confusion_matrix(y_train,rf_y_train_pred)

rf_test_f1 = f1_score(y_test,rf_y_test_pred)
rf_test_accuracy = accuracy_score(y_test,rf_y_test_pred)
rf_test_pr_auc = average_precision_score(y_test,rf_y_test_prob)
rf_test_confusionmatrix = confusion_matrix(y_test,rf_y_test_pred)

xgboost = Pipeline([
    ('preprocess',preprocessor),
    ('model' , XGBClassifier(
        max_depth = 4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=42
    ))
])

xgboost.fit(x_train,y_train)

xgboost_y_train_pred = xgboost.predict(x_train)
xgboost_y_test_pred = xgboost.predict(x_test)

xgboost_y_train_prob = xgboost.predict_proba(x_train)[:, 1]
xgboost_y_test_prob = xgboost.predict_proba(x_test)[:, 1]

xgboost_train_f1 = f1_score(y_train,xgboost_y_train_pred)
xgboost_train_accuracy = accuracy_score(y_train,xgboost_y_train_pred)
xgboost_train_pr_auc = average_precision_score(y_train,xgboost_y_train_prob)
xgboost_train_confusionmatrix = confusion_matrix(y_train,xgboost_y_train_pred)

xgboost_test_f1 = f1_score(y_test,xgboost_y_test_pred)
xgboost_test_accuracy = accuracy_score(y_test,xgboost_y_test_pred)
xgboost_test_pr_auc = average_precision_score(y_test,xgboost_y_test_prob)
xgboost_test_confusionmatrix = confusion_matrix(y_test,xgboost_y_test_pred)

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Train F1': [lr_train_f1, rf_train_f1, xgboost_train_f1],
    'Test F1': [lr_test_f1, rf_test_f1, xgboost_test_f1],
    'Train Accuracy': [lr_train_accuracy, rf_train_accuracy, xgboost_train_accuracy],
    'Test Accuracy': [lr_test_accuracy, rf_test_accuracy, xgboost_test_accuracy],
    'Train PR-AUC': [lr_train_pr_auc, rf_train_pr_auc, xgboost_train_pr_auc],
    'Test PR-AUC': [lr_test_pr_auc, rf_test_pr_auc, xgboost_test_pr_auc],
})

results = results.round(3)
print(results.to_string(index=False))

print("Logistic Regression - Test Confusion Matrix:")
print(lr_test_confusionmatrix)

print("\nRandom Forest - Test Confusion Matrix:")
print(rf_test_confusionmatrix)

print("\nXGBoost - Test Confusion Matrix:")
print(xgboost_test_confusionmatrix)