# SERVER-BASED IMPLEMENTATION (FINAL WORKING VERSION)

import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
import smtplib
from email.mime.text import MIMEText
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# -------------------- EMAIL FUNCTION --------------------
def send_email_alert(message):
    sender_email = "ksakthimeena0@gmail.com"
    receiver_email = "ksakthimeena0@gmail.com"
    password = "bune wrcq icgd kzco"  # ⚠️ demo only

    msg = MIMEText(message)
    msg['Subject'] = "Cyber Attack Alert!"
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        print("Email Sent Successfully!")
    except Exception as e:
        print("❌ Email Error:", e)

# -------------------- MODELS --------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
iso_model = IsolationForest(contamination=0.1, random_state=42)
xgb_model = XGBClassifier(eval_metric='mlogloss')

# -------------------- DATA --------------------
DATA_PATH = "."
data = None

# 👉 KEEP ORIGINAL LABELS
def group_attack(label):
    label = str(label).lower()

    # convert numeric labels
    if label in ['0', 'benign']:
        return "Benign"
    elif label in ['1']:
        return "Malware"

    # handle text labels
    if "benign" in label:
        return "Benign"
    else:
        return "Malware"   

def load_data():
    global data

    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]

    if not files:
        print("❌ No CSV files found!")
        exit()

    df_list = [pd.read_csv(os.path.join(DATA_PATH, f)) for f in files[:5]]
    data = pd.concat(df_list)

    label_col = "Label" if "Label" in data.columns else "label"

    # ✅ ensure all labels are string
    data[label_col] = data[label_col].apply(group_attack)

# -------------------- TRAIN --------------------
def train_models():
    global data, label_encoder

    label_col = "Label" if "Label" in data.columns else "label"

    X = data.select_dtypes(include=np.number).fillna(0)

    # ✅ IMPORTANT FIX (no mixed types error)
    y = data[label_col].astype(str)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("Training Models...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42
    )

    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    iso_model.fit(X_train)

    print("Training Completed!")

    os.makedirs("static", exist_ok=True)
    joblib.dump(X.columns.tolist(), "static/features.pkl")

# -------------------- DETECTION --------------------
def run_detection():
    global data

    print("\nStarting Detection...")

    feature_columns = joblib.load("static/features.pkl")
    X = data[feature_columns]

    anomaly = iso_model.predict(X)

    results = []
    responses = []

    alert_sent = False

    for i in range(len(X)):

        if anomaly[i] == -1:
            row = X.iloc[[i]]

            rf_pred = rf_model.predict(row)[0]
            label = label_encoder.inverse_transform([rf_pred])[0]

            if not alert_sent:
                send_email_alert(f"""
🚨 Cyber Attack Alert!
Attack Type: {label}
System Status: Threat Detected
Time: {pd.Timestamp.now()}
""")
                alert_sent = True
        else:
            label = "Benign"

        results.append(label)
        responses.append("Alert" if label != "Benign" else "No Action")

    data['Prediction'] = results
    data['Response'] = responses

    print("Detection Completed!")

    data.to_csv("output_predictions.csv", index=False)
    print("Results saved!")

    # ---------------- GRAPH ----------------
    attack_counts = data['Prediction'].value_counts()

    plt.figure(figsize=(8,5))
    attack_counts.plot(kind='bar')

    plt.title("Attack Detection Results")
    plt.xlabel("Traffic Type")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig("attack_graph.png", dpi=300)
    plt.show()

    print("Graph saved!")

    # ---------------- EVALUATION ----------------
    print("\n📊 Evaluation Metrics:")

    label_col = "Label" if "Label" in data.columns else "label"

    y_true = data[label_col].astype(str)
    y_pred = data['Prediction']

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    labels = sorted(list(set(y_true)))
    print("\nLabels:", labels)

    if "Benign" in labels:
        idx = labels.index("Benign")
        fp = cm[idx].sum() - cm[idx][idx]
    else:
        fp = 0

    print("\n🚨 False Positives:", fp)

# -------------------- MAIN --------------------
if __name__ == "__main__":
    print("Server Started...")

    load_data()
    train_models()
    run_detection()

    print("Process Completed!")