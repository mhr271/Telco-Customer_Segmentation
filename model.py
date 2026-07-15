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
print(df.head())
print(df.shape)
 
df = df.drop(columns=["customerID"])
 
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
 
df["churn"] = df["churn"].map({"Yes":1,"No":0})

y = df["Churn"]
x = df.drop("Churn",axis=1)

categorical_cols = x.select_dtypes(include=["object"]).columns
numerical_cols = x.select_dtypes(include=["number"]).columns

preprocessor = ColumnTransformer(
    transformer=[
    ('num',StandardScaler(),numerical_cols)
    ('cat',OneHotEncoder(handle_unknown='ignore'),categorical_cols)
    ]
)

x_train , x_test , y_train , y_test = train_test_split(
    test_size =0.2 ,
    random_state= 100 ,
    stratify = y )

neg, pos = y_train.value_counts()
scale_pos_weight = neg / pos

lr = Pipeline([
    ( "preprocess" ,preprocessor),
    ('model',LogisticRegression(max_iter=1000,class_weight='balanced',random_state=100))
])

lr.fit(x_train,y_train)

lr_y_train_pred = lr.predict(x_train)
lr_y_test_pred =  lr.predict(x_test)

lr_train_f1 = f1_score(y_train,lr_y_train_pred)
lr_train_accuracy =(y_train,lr_y_train_pred)
lr_train_pr_auc =(y_train,lr_y_train_pred)
lr_train_confusionmatrix = (y_train,lr_y_train_pred)