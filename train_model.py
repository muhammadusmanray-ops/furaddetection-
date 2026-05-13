import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# 1. Load Dataset
file_path = r'D:\creditcard.csv'
df = pd.read_csv(file_path)

print("Data loaded successfully!")

# 2. Preprocessing
# 'Time' aur 'Amount' ko scale karna zaroori hai kyunke baki V1-V28 pehle se scaled hain
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))

# Purane columns drop kar dete hain
df.drop(['Time', 'Amount'], axis=1, inplace=True)

# 3. Split Data
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Original training shape: {X_train.shape}")

# 4. Handle Imbalance (SMOTE)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print(f"Resampled training shape: {X_train_res.shape}")

# 5. Train Model
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_res, y_train_res)

# 6. Evaluate
y_pred = model.predict(X_test)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))
