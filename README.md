# 🛡️ FraudPro Enterprise v3.0

An enterprise-grade Credit Card Fraud Detection system built with **Random Forest**, **SMOTE**, and **Groq AI (Llama-3)**. This system provides real-time transaction monitoring, automated risk assessment, and a professional dashboard for financial security audit.

## 🚀 Key Features
- **Real-time Monitoring:** Audit log of global incoming transactions.
- **AI Security Insights:** Automatic transaction analysis powered by Llama-3.1 via Groq.
- **High Performance:** Optimized with SMOTE to handle highly imbalanced fraud datasets.
- **Interactive Dashboard:** Built with Streamlit for a premium user experience.

## 🛠️ Tech Stack
- **Backend:** Python, Scikit-Learn, XGBoost, Joblib
- **Frontend:** Streamlit, Plotly
- **AI:** Groq Cloud API (Llama-3.1-8b-instant)
- **Data Handling:** Pandas, Numpy, SMOTE

## 📦 Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/muhammadusmanray-ops/furaddetection-.git
   cd furaddetection-
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your API Key:
   - Create a `.streamlit/secrets.toml` file (for local) or add it to Streamlit Cloud secrets.
   - `GROQ_API_KEY = "your_key_here"`
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## 📜 License
Developed for CS506 Project Submission.
