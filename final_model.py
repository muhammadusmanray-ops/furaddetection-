import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# 1. Load Dataset
print("Dataset load ho raha hai...")
file_path = r'D:\creditcard.csv'
df = pd.read_csv(file_path)

# 2. Preprocessing & Scaling
print("Scaling Time aur Amount...")
scaler = StandardScaler()
# Dono columns ko ek saath scale karte hain taake scaler sahi se save ho
df[['scaled_amount', 'scaled_time']] = scaler.fit_transform(df[['Amount', 'Time']])

# Purane columns drop kar dete hain
df.drop(['Time', 'Amount'], axis=1, inplace=True)

# 3. Split Data (80/20)
print("Data Split kar rahe hain (80/20)...")
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Handle Imbalance (SMOTE)
print("SMOTE apply kar rahe hain (Class Imbalance handle karne ke liye)...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print(f"Original training size: {len(y_train)}")
print(f"Resampled training size: {len(y_train_res)}")

# 5. Train Final Model (Random Forest)
print("Final Random Forest Model train ho raha hai... Isme thora time lagega...")
# Hum parameters ko thora optimize kar rahe hain behtar results ke liye
model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X_train_res, y_train_res)

# 6. Evaluation
print("\n--- Final Model Evaluation (20% Test Data) ---")
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Save Model & Scaler
print("\nModel aur Scaler save ho rahe hain...")
joblib.dump(model, 'fraud_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("\nSuccess! 'fraud_model.joblib' aur 'scaler.joblib' save ho gaye hain.")
print("Ab hum Interface (UI) banane ke liye tayyar hain.")
