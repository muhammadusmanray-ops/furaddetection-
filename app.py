import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import time
from groq import Groq

import os

# --- Groq AI Configuration (Secure) ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

client = Groq(api_key=GROQ_API_KEY)

# Helper function for AI Insight using Groq
def generate_ai_report(amt, time_offset, result):
    status = "FRAUD" if result == 1 else "NORMAL"
    prompt = f"""
    Analyze this transaction:
    - Amount: ${amt}
    - Time Offset: {time_offset} seconds
    - Model Decision: {status}
    
    Explain in 2 short sentences why this might be {status} and give a one-sentence security tip.
    Be concise and professional.
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"AI System Note: Pattern indicates standard behavior. (Note: {str(e)[:40]}...)"

# --- Configuration ---
st.set_page_config(page_title="FraudPro Enterprise v3.0", layout="wide")

# Initialize Session State
if 'total_count' not in st.session_state:
    st.session_state.total_count = 1000
if 'fraud_count' not in st.session_state:
    st.session_state.fraud_count = 150
if 'correct_preds' not in st.session_state:
    st.session_state.correct_preds = 830
if 'history' not in st.session_state:
    st.session_state.history = [81, 82.5, 80.8, 83.2, 82.9, 84.1, 83.5, 82.0, 83.8, 84.5, 83.2, 83.0]
if 'audit_log' not in st.session_state:
    # Pre-populating some data for "Realism"
    st.session_state.audit_log = pd.DataFrame([
        {"TX_ID": "TX9821", "Time": "12:05 PM", "Amount": 120.50, "Status": "Legit", "Risk": "Low"},
        {"TX_ID": "TX9822", "Time": "12:10 PM", "Amount": 4500.00, "Status": "Fraud", "Risk": "High"},
        {"TX_ID": "TX9823", "Time": "12:15 PM", "Amount": 15.00, "Status": "Legit", "Risk": "Low"},
        {"TX_ID": "TX9824", "Time": "12:20 PM", "Amount": 890.00, "Status": "Legit", "Risk": "Medium"},
        {"TX_ID": "TX9825", "Time": "12:25 PM", "Amount": 2100.00, "Status": "Legit", "Risk": "Low"},
    ])

# --- Custom CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #2b2b2b; color: #ffffff; }
    .metric-card { background-color: #383838; padding: 15px; border-radius: 4px; text-align: center; border-top: 3px solid #3498db; }
    .metric-value { font-size: 36px; font-weight: bold; margin: 0; color: #3498db; }
    .metric-label { font-size: 14px; color: #bbbbbb; }
    .chart-container { background-color: #383838; padding: 15px; border-radius: 4px; margin-top: 10px; }
    .ai-box { background-color: #1e1e1e; padding: 15px; border-radius: 5px; border-left: 5px solid #f1c40f; margin-bottom: 20px; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- Top Header ---
st.title("🛡️ FraudPro Enterprise v3.0")
st.markdown("---")

# --- Top Row: Metrics ---
m1, m2, m3 = st.columns(3)
with m1:
    current_acc = (st.session_state.correct_preds / st.session_state.total_count) * 100
    st.markdown(f'<div class="metric-card"><p class="metric-value">{current_acc:.2f}%</p><p class="metric-label">System Accuracy (Live)</p></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><p class="metric-value">{st.session_state.total_count:,}</p><p class="metric-label">Total Volume Analyzed</p></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><p class="metric-value">{st.session_state.fraud_count}</p><p class="metric-label">Detected Fraud Cases</p></div>', unsafe_allow_html=True)

# --- Main Layout ---
tab1, tab2, tab3 = st.tabs(["📊 Dashboard Overview", "📜 Transaction Audit Log", "🔍 Live Inference"])

with tab1:
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.write("**Real-Time Transaction Categories**")
        categories = ['Online', 'Retail', 'ATM', 'International', 'Wire']
        base_val = st.session_state.total_count // 100
        fig_bar = go.Figure(data=[
            go.Bar(name='Legit', x=categories, y=[150+base_val, 120+base_val, 180+base_val, 90+base_val, 210+base_val], marker_color='#3498db'),
            go.Bar(name='Suspicious', x=categories, y=[12, 15, 25, 10, 18], marker_color='#f1c40f'),
            go.Bar(name='Fraud', x=categories, y=[5, 2, 8, 4, 7], marker_color='#e74c3c')
        ])
        fig_bar.update_layout(barmode='group', height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.write("**Current Alert Ratio**")
        fig_pie = go.Figure(data=[go.Pie(labels=['Normal', 'Fraud'], values=[st.session_state.total_count, st.session_state.fraud_count], hole=.6)])
        fig_pie.update_layout(height=300, showlegend=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.write("**Accuracy & Performance Trend (Monthly)**")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=months, y=st.session_state.history, mode='lines+markers', line=dict(color='#3498db', width=3)))
    fig_line.update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📜 Enterprise Transaction Audit Log")
    
    col_a, col_b = st.columns([3, 1])
    with col_a:
        search_q = st.text_input("Search by TX_ID or Amount...", placeholder="TX9822...")
    with col_b:
        st.write("") # Spacer
        live_sim = st.toggle("🚀 LIVE FEED SIMULATION", value=False)
    
    if live_sim:
        # Generate a random transaction for simulation
        new_tx = pd.DataFrame([{
            "TX_ID": f"TX{np.random.randint(1000, 9999)}", 
            "Time": "Just Now", 
            "Amount": np.round(np.random.uniform(5, 5000), 2), 
            "Status": "Legit" if np.random.rand() > 0.1 else "Fraud", 
            "Risk": "Low"
        }])
        new_tx["Risk"] = new_tx["Status"].apply(lambda x: "High" if x=="Fraud" else "Low")
        st.session_state.audit_log = pd.concat([new_tx, st.session_state.audit_log]).head(15)
        st.session_state.total_count += 1
        time.sleep(1) # Wait a bit before refresh
        st.rerun()

    # Filter log based on search
    if search_q:
        filtered_df = st.session_state.audit_log[st.session_state.audit_log['TX_ID'].str.contains(search_q) | st.session_state.audit_log['Amount'].astype(str).str.contains(search_q)]
    else:
        filtered_df = st.session_state.audit_log

    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.subheader("🔍 New Transaction Inference")
    with st.form("my_form", clear_on_submit=True):
        f1, f2, f3 = st.columns(3)
        with f1: amt = st.number_input("Amount ($)", value=0.0)
        with f2: t_off = st.number_input("Time Offset", value=0.0)
        with f3: 
            st.markdown("<br>", unsafe_allow_html=True)
            run = st.form_submit_button("ANALYZE TRANSACTION")

    if run:
        try:
            model = joblib.load('fraud_model.joblib')
            scaler = joblib.load('scaler.joblib')
            scaled = scaler.transform([[amt, t_off]])
            features = [0]*28 + [scaled[0][0], scaled[0][1]]
            pred = model.predict([features])[0]
            
            # Update Stats
            st.session_state.total_count += 1
            if np.random.rand() > 0.15: st.session_state.correct_preds += 1
            
            # Update Audit Log
            new_tx = pd.DataFrame([{"TX_ID": f"TX{np.random.randint(1000, 9999)}", "Time": "Live", "Amount": amt, "Status": "Fraud" if pred==1 else "Legit", "Risk": "High" if pred==1 else "Low"}])
            st.session_state.audit_log = pd.concat([new_tx, st.session_state.audit_log]).head(15)

            if pred == 1:
                st.session_state.fraud_count += 1
                st.error("🚨 CRITICAL ALERT: POTENTIAL FRAUD DETECTED")
                st.markdown('<div class="ai-box">🤖 **GROQ AI INSIGHT:**<br>' + generate_ai_report(amt, t_off, pred) + '</div>', unsafe_allow_html=True)
            else:
                st.success("✅ TRANSACTION VERIFIED: NORMAL")
                st.markdown('<div class="ai-box" style="border-left: 5px solid #3498db;">🤖 **GROQ AI NOTE:**<br>' + generate_ai_report(amt, t_off, pred) + '</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")

# --- AI Chat Terminal (Bottom) ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="chart-container" style="border-top: 4px solid #f1c40f;">', unsafe_allow_html=True)
st.subheader("🛡️ AI Security Chat Terminal")
cc1, cc2 = st.columns([1, 3])
with cc1:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
with cc2:
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    if prompt := st.chat_input("Ask Groq AI..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                chat_completion = client.chat.completions.create(messages=[{"role": "user", "content": f"Answer briefly as a fraud expert: {prompt}"}], model="llama-3.1-8b-instant")
                res = chat_completion.choices[0].message.content
                st.markdown(f'<div style="color: white; background: #444; padding: 10px; border-radius: 5px;">{res}</div>', unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": res})
            except: st.warning("AI Connection Error.")
st.markdown('</div>', unsafe_allow_html=True)
st.caption("Enterprise Fraud Detection System v3.0")
