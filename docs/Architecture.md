# 🏗️ FraudPro System Architecture

## 1. Data Pipeline
- **Input:** Credit card transaction dataset (284,807 rows).
- **Preprocessing:** Robust scaling of 'Amount' and 'Time' features.
- **Handling Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance fraud cases.

## 2. Machine Learning Model
- **Algorithm:** Random Forest Classifier.
- **Evaluation:** Focused on **Recall** (0.85) to minimize false negatives in fraud detection.

## 3. AI Integration
- **Engine:** Groq (Llama-3.1-8b-instant).
- **Function:** Real-time reasoning and security insights for flagged transactions.

## 4. Frontend
- **Framework:** Streamlit (Enterprise Dashboard).
