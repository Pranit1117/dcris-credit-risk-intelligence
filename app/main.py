import streamlit as st

st.set_page_config(
    page_title="DCRIS | Dynamic Credit Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080C14;
    color: #E8EAF0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1420 0%, #0A1020 100%);
    border-right: 1px solid #1E2A3A;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #C8D0DC; }

/* Main bg */
.main .block-container {
    background-color: #080C14;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0F1825 0%, #131F30 100%);
    border: 1px solid #1E2E44;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.metric-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6B7A99;
    margin-bottom: 0.4rem;
}
.metric-card .value {
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1.1;
    color: #E8EAF0;
}
.metric-card .delta {
    font-size: 0.8rem;
    color: #48BB78;
    margin-top: 0.3rem;
}

/* Risk badges */
.badge-safe     { background:#0D3321; color:#48BB78; border:1px solid #2F6B4A; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.badge-risky    { background:#33260A; color:#F6AD55; border:1px solid #7A5C1A; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.badge-critical { background:#33100A; color:#FC8181; border:1px solid #7A2A1A; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }

/* Section headers */
.section-header {
    font-size: 1.35rem;
    font-weight: 700;
    color: #E8EAF0;
    margin-bottom: 0.2rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #1E3A5F;
}
.section-sub {
    font-size: 0.85rem;
    color: #6B7A99;
    margin-bottom: 1.5rem;
}

/* Probability gauge text */
.prob-number { font-family:'DM Mono',monospace; font-size:3rem; font-weight:500; }
.prob-safe     { color:#48BB78; }
.prob-risky    { color:#F6AD55; }
.prob-critical { color:#FC8181; }

/* Action box */
.action-box {
    border-radius:12px;
    padding:1rem 1.2rem;
    margin-top:1rem;
    font-size:0.9rem;
    line-height:1.6;
}
.action-safe     { background:#0A2318; border-left:4px solid #48BB78; }
.action-risky    { background:#231A08; border-left:4px solid #F6AD55; }
.action-critical { background:#230A08; border-left:4px solid #FC8181; }

/* Divider */
hr { border-color: #1E2A3A !important; }

/* Input fields */
[data-testid="stNumberInput"] input,
[data-testid="stSlider"] { color: #E8EAF0; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #1A4A8A 0%, #0F2D5C 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2355A0 0%, #163568 100%);
    box-shadow: 0 4px 15px rgba(26,74,138,0.4);
}

/* Logo area */
.logo-text {
    font-size:1.5rem;
    font-weight:800;
    letter-spacing:-0.02em;
    color:#E8EAF0;
    line-height:1;
}
.logo-sub {
    font-size:0.65rem;
    font-weight:500;
    letter-spacing:0.15em;
    text-transform:uppercase;
    color:#3A8FD6;
    margin-top:2px;
}
.bank-tag {
    font-size:0.7rem;
    color:#6B7A99;
    margin-top:4px;
}
</style>
""", unsafe_allow_html=True)

import os, sys

# Add the project root (dcris/) to path so all imports work
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


# Page imports
from app.pages import home, predict, analytics, explainability

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='padding:1rem 0 1.5rem 0;'>
            <div class='logo-text'>⬡ DCRIS</div>
            <div class='logo-sub'>Credit Risk Intelligence</div>
            <div class='bank-tag'>Powered for BANKS</div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.selectbox(
        "Navigate",
        ["🏠 Home Dashboard", "🔍 Individual Prediction", "📊 Risk Analytics", "🧠 Explainability"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
        <div style='font-size:0.72rem;color:#3A4A5C;line-height:1.8;padding:0 0.2rem;'>
            <b style='color:#4A6A8A'>DCRIS v2.1</b><br>
            Model: XGBoost Ensemble<br>
            Survival: Weibull AFT<br>
            Segments: KMeans (k=3)<br>
            Dataset: 15,000 customers<br>
            Last trained: Today
        </div>
    """, unsafe_allow_html=True)

# ── Route pages ───────────────────────────────────────────────────────────────
if "🏠" in page:
    home.render()
elif "🔍" in page:
    predict.render()
elif "📊" in page:
    analytics.render()
elif "🧠" in page:
    explainability.render()
