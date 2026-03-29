import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ─── PAGE CONFIG ─────────────────────────────
st.set_page_config(page_title="SteelSense Pro", layout="wide")

# ─── LOAD MODEL ─────────────────────────────
@st.cache_resource
def load_model():
    classifier    = joblib.load('xgb_classifier.pkl')
    scaler        = joblib.load('scaler.pkl')
    le            = joblib.load('label_encoder.pkl')
    feature_names = joblib.load('feature_names.pkl')

    class Model:
        def __init__(self, scaler, clf):
            self.scaler = scaler
            self.clf = clf
        def predict(self, df):
            return self.clf.predict(self.scaler.transform(df))
        def predict_proba(self, df):
            return self.clf.predict_proba(self.scaler.transform(df))
        def get_feature_importances(self):
            return self.clf.feature_importances_

    return Model(scaler, classifier), le, feature_names

model, le, feature_names = load_model()
CLASS_NAMES = le.classes_

# ─── SAMPLE DATA (ALL CLASSES) ─────────────────
SAMPLES = {
    'Z_Scratch':  [42,270,1,294,267,17,17,24220,76,108,1227,1,0,80,0,0.05,0.71,0,0,0.15,0,2.43,0.37,0.23,0.87,-0.08,0.52],
    'Bumps':      [100,500,50,800,3200,80,200,180000,90,240,1500,0,1,60,0.3,0.02,0.55,0.1,0.2,0.3,0.05,3.51,0.7,0.9,0.6,0.15,0.88],
    'Stains':     [20,150,10,200,500,30,50,40000,80,120,1100,1,0,100,0.1,0.08,0.65,0.02,0.05,0.1,0.01,2.7,0.48,0.35,0.92,0.02,0.6],
    'K_Scatch':   [5,400,1,30,180,40,10,15000,60,90,1300,0,1,70,0.05,0.12,0.95,0,0.1,0.02,0,2.26,0.88,0.2,0.12,0.25,0.45],
    'Dirtiness':  [60,320,80,450,1800,50,90,130000,70,200,1400,1,0,90,0.2,0.04,0.6,0.05,0.12,0.2,0.02,3.26,0.58,0.62,0.75,0.1,0.78],
}

# ─── SIDEBAR ─────────────────────────────
st.sidebar.title("⚙️ INPUT MODE")
mode = st.sidebar.radio("", ["Manual Entry","Load Sample"])

sample_key = None
if mode == "Load Sample":
    sample_key = st.sidebar.selectbox("Select Sample", list(SAMPLES.keys()))

# ─── MAIN TITLE ─────────────────────────────
st.title("🔩 SteelSense — Defect Intelligence System")

# ─── INPUT ─────────────────────────────
user_inputs = {}

if mode == "Load Sample":
    user_inputs = {fn: sv for fn, sv in zip(feature_names, SAMPLES[sample_key])}

cols = st.columns(3)

for i, feat in enumerate(feature_names):
    with cols[i % 3]:
        default_val = user_inputs.get(feat, 0)
        disabled = (mode == "Load Sample")

        if feat in ['TypeOfSteel_A300','TypeOfSteel_A400']:
            if pd.isna(default_val):
                default_val = 0
            safe_index = int(default_val) if int(default_val) in [0,1] else 0

            user_inputs[feat] = float(
                st.selectbox(feat, [0,1], index=safe_index, disabled=disabled)
            )
        else:
            user_inputs[feat] = st.number_input(
                feat,
                value=float(default_val),
                disabled=disabled
            )

# ─── PREDICTION ─────────────────────────────
if st.button("🚀 Analyse") or mode == "Load Sample":

    input_arr = np.array([[user_inputs.get(f, 0.0) for f in feature_names]])
    input_df = pd.DataFrame(input_arr, columns=feature_names)

    proba = model.predict_proba(input_df)[0]
    pred_idx = int(np.argmax(proba))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = proba[pred_idx] * 100

    # ─── KPI ROW ─────────────────────────
    c1, c2, c3 = st.columns(3)

    if confidence > 80:
        risk = "HIGH"
    elif confidence > 50:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    c1.metric("Defect Type", pred_class)
    c2.metric("Confidence", f"{confidence:.2f}%")
    c3.metric("Risk Level", risk)

    st.markdown("---")

    # ─── CHART 1: PROBABILITY ─────────────
    st.subheader("📊 Class Probability Distribution")

    sorted_idx = np.argsort(proba)[::-1]

    fig_bar = go.Figure(go.Bar(
        x=[proba[i]*100 for i in sorted_idx],
        y=[CLASS_NAMES[i] for i in sorted_idx],
        orientation='h',
        text=[f"{proba[i]*100:.1f}%" for i in sorted_idx],
        textposition='outside'
    ))
    fig_bar.update_layout(height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

    # ─── CHART 2: GAUGE ───────────────────
    st.subheader("🎯 Confidence Gauge")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': "Confidence %"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"},
            ],
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ─── CHART 3: FEATURE IMPORTANCE ───────
    st.subheader("📈 Feature Importance")

    try:
        importances = model.get_feature_importances()
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(10)

        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"],
            y=fi_df["Feature"],
            orientation='h'
        ))
        fig_fi.update_layout(height=400)
        st.plotly_chart(fig_fi, use_container_width=True)

    except:
        st.info("Feature importance not available")

    # ─── RISK PANEL ───────────────────────
    if confidence > 80:
        color = "#ff6b35"
        msg = "High confidence defect — immediate action required"
    elif confidence > 50:
        color = "#ffd60a"
        msg = "Moderate confidence — review recommended"
    else:
        color = "#00ff9d"
        msg = "Low confidence — monitor"

    st.markdown(f"""
    <div style="border:1px solid {color};padding:15px;border-radius:8px;margin-top:10px;">
        <b style="color:{color}">⚠️ {risk} RISK</b><br>
        <span>{msg}</span>
    </div>
    """, unsafe_allow_html=True)