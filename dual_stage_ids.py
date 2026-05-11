
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# Load Dataset
data = pd.read_csv("dataset.csv")

X = data.drop('label', axis=1)
y = data['label']

# Stage 1: Isolation Forest
iso = IsolationForest(contamination=0.2, random_state=42)
data['anomaly'] = iso.fit_predict(X)

suspicious = data[data['anomaly'] == -1]
X_sus = suspicious.drop(['label', 'anomaly'], axis=1)
y_sus = suspicious['label']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_sus, y_sus, test_size=0.3, random_state=42
)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("Random Forest Results")
print(classification_report(y_test, rf_pred))

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

print("XGBoost Results")
print(classification_report(y_test, xgb_pred))

print("Dual Stage Detection Completed")
