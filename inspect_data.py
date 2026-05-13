import pandas as pd

# Step 1: Load the dataset
file_path = r'D:\creditcard.csv'
df = pd.read_csv(file_path)

# Step 2: Basic info
print("Dataset Shape:", df.shape)
print("\nColumn Info:")
print(df.info())

# Step 3: Data Scaling
from sklearn.preprocessing import StandardScaler

print("\nScaling Time and Amount...")
scaler = StandardScaler()

# Amount aur Time ko scale karna
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Purane columns ko drop karna
df.drop(['Time', 'Amount'], axis=1, inplace=True)
print("Scaling Mukammal ho gayi!")

# Step 4: Data Splitting
from sklearn.model_selection import train_test_split

print("\nData Split kar rahe hain...")
# X = Features, y = Target (Class)
X = df.drop('Class', axis=1)
y = df['Class']

# 80% Training, 20% Testing (stratify=y zaroori hai imbalance ke liye)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Data Splitting Mukammal!")
print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Check class distribution in training set
print("\nClass distribution in Training set:")
print(y_train.value_counts(normalize=True))

# Step 5: Model Training (Logistic Regression)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

print("\nModel Train kar rahe hain (Logistic Regression)...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Training Mukammal!")

# Step 6: Model Evaluation
print("\nModel ko Test kar rahe hain...")
y_pred = model.predict(X_test)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---")
# [[True Negatives, False Positives], [False Negatives, True Positives]]
print(confusion_matrix(y_test, y_pred))

# Step 7: More Powerful Model (Random Forest)
from sklearn.ensemble import RandomForestClassifier

print("\nModel Train kar rahe hain (Random Forest)... Isme thora waqt lag sakta hai...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("Random Forest Training Mukammal!")

# Model Evaluation
print("\nRandom Forest ko Test kar rahe hain...")
y_pred_rf = rf_model.predict(X_test)

print("\n--- Random Forest Classification Report ---")
print(classification_report(y_test, y_pred_rf))

print("\n--- Random Forest Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred_rf))
