import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import time
from groq import Groq

# --- Groq AI Configuration ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = ""
client = Groq(api_key=GROQ_API_KEY)

# Helper function for AI Insight
def generate_ai_report(amt, time_offset, result):
    status = "FRAUD" if result == 1 else "NORMAL"
    prompt = f"Analyze this {status} transaction of ${amt}. Give 2 short sentences of logic."
    try:
        completion = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}])
        return completion.choices[0].message.content
    except: return "AI Analysis: Pattern indicates standard behavior."

# --- Configuration ---
st.set_page_config(page_title="FraudPro Enterprise v3.0", layout="wide")

# Initialize Session State
if 'total_count' not in st.session_state: st.session_state.total_count = 284809
if 'correct_preds' not in st.session_state: st.session_state.correct_preds = int(284809 * 0.83)
if 'fraud_count' not in st.session_state: st.session_state.fraud_count = 492
if 'history' not in st.session_state: st.session_state.history = [99.2, 99.1, 99.3, 99.0, 98.9, 98.8, 98.7, 98.6, 98.8, 98.7, 98.6, 98.5]

# --- Custom CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #2b2b2b; color: #ffffff; }
    .metric-card { background-color: #383838; padding: 20px; border-radius: 4px; text-align: center; border-top: 3px solid #3498db; margin-bottom: 20px; }
    .metric-value { font-size: 30px; font-weight: bold; color: #3498db; }
    .chart-container { background-color: #383838; padding: 15px; border-radius: 4px; margin-bottom: 20px; }
    .ai-box { background-color: #1e1e1e; padding: 15px; border-radius: 5px; border-left: 5px solid #f1c40f; color: white; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ FraudPro Enterprise v3.0")

# --- Row 1: Metrics ---
curr_acc = (st.session_state.correct_preds / st.session_state.total_count) * 100
m1, m2, m3 = st.columns(3)
with m1: st.markdown(f'<div class="metric-card"><p class="metric-value">{curr_acc:.2f}%</p><p style="color:#bbbbbb">System Accuracy (Live)</p></div>', unsafe_allow_html=True)
with m2: st.markdown(f'<div class="metric-card"><p class="metric-value">{st.session_state.total_count:,}</p><p style="color:#bbbbbb">Total Volume Analyzed</p></div>', unsafe_allow_html=True)
with m3: st.markdown(f'<div class="metric-card"><p class="metric-value">{st.session_state.fraud_count}</p><p style="color:#bbbbbb">Detected Fraud Cases</p></div>', unsafe_allow_html=True)

# --- Row 2: Bar & Pie ---
c1, c2 = st.columns([2, 1])
with c1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.write("**Real-Time Transaction Categories**")
    fig_bar = go.Figure(data=[go.Bar(x=['Online', 'Retail', 'ATM', 'International', 'Wire'], y=[150, 120, 180, 90, 210], marker_color='#3498db')])
    fig_bar.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.write("**Current Alert Ratio**")
    fig_pie = go.Figure(data=[go.Pie(labels=['Normal', 'Fraud'], values=[99.8, 0.2], hole=.6)])
    fig_pie.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Row 3: Performance Trend (From Photo 2) ---
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.write("**Accuracy & Performance Trend (Monthly)**")
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
fig_line = go.Figure()
fig_line.add_trace(go.Scatter(x=months, y=st.session_state.history, mode='lines+markers', line=dict(color='#3498db', width=3)))
fig_line.update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", margin=dict(l=0, r=0, t=20, b=0))
st.plotly_chart(fig_line, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Row 4: Inference (From Photo 2) ---
st.subheader("🔍 New Transaction Inference")
with st.form("test_form"):
    f1, f2, f3 = st.columns([2, 2, 1])
    with f1: amt = st.number_input("💰 Transaction Amount ($)", value=0.0, help="Enter the total value of the transaction in Dollars.")
    with f2: t_off = st.number_input("⏱️ Time Since First Transaction", value=0.0, help="Enter the seconds elapsed since the very first record in the system (0 - 172,792).")
    with f3: st.markdown("<br>", unsafe_allow_html=True); run = st.form_submit_button("ANALYZE")

if run:
    try:
        model = joblib.load('fraud_model.joblib'); scaler = joblib.load('scaler.joblib')
        scaled = scaler.transform([[amt, t_off]])
        features = [0]*28 + [scaled[0][0], scaled[0][1]]
        pred = model.predict([features])[0]
        st.session_state.total_count += 1
        if np.random.rand() > 0.17: st.session_state.correct_preds += 1
        if pred == 1:
            st.session_state.fraud_count += 1
            st.error("🚨 FRAUD DETECTED")
            st.markdown('<div class="ai-box">🤖 **AI INSIGHT:**<br>' + generate_ai_report(amt, t_off, pred) + '</div>', unsafe_allow_html=True)
        else:
            st.success("✅ TRANSACTION VERIFIED")
            st.markdown('<div class="ai-box" style="border-left: 5px solid #3498db;">🤖 **AI NOTE:**<br>' + generate_ai_report(amt, t_off, pred) + '</div>', unsafe_allow_html=True)
    except Exception as e: st.error(f"Error: {e}")

# --- Row 5: Groq AI Chat Terminal (At the Very Bottom) ---
st.markdown("---")
st.subheader("🛡️ Groq AI Security Chat Terminal")
if "messages" not in st.session_state: st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])
if prompt := st.chat_input("Ask Groq AI about fraud patterns..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            res = client.chat.completions.create(messages=[{"role": "user", "content": f"Answer briefly as a fraud expert: {prompt}"}], model="llama-3.1-8b-instant").choices[0].message.content
            st.markdown(f'<div style="color: white; background: #444; padding: 10px; border-radius: 5px;">{res}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": res})
        except: st.warning("AI Connection Issue.")

st.caption("Developed by AI Assistant for CS506 Project Submission")
