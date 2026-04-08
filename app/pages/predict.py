import streamlit as st
import plotly.graph_objects as go
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.predictor import predict_single
from app.data_loader import get_models

PLOTLY_DARK = dict(paper_bgcolor="#0F1825", plot_bgcolor="#0F1825",
                   font=dict(color="#C8D0DC", family="DM Sans"))


def gauge_chart(prob: float, risk: str) -> go.Figure:
    color = {"Safe": "#48BB78", "Risky": "#F6AD55", "Critical": "#FC8181"}.get(risk, "#888")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number=dict(suffix="%", font=dict(size=36, color=color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#333",
                      tickfont=dict(color="#6B7A99")),
            bar=dict(color=color, thickness=0.25),
            bgcolor="#1A2235",
            borderwidth=0,
            steps=[
                dict(range=[0, 30], color="#0D2318"),
                dict(range=[30, 60], color="#231A08"),
                dict(range=[60, 100], color="#230A08"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.8, value=prob * 100),
        ),
        title=dict(text="Default Probability", font=dict(size=14, color="#6B7A99")),
    ))
    fig.update_layout(**PLOTLY_DARK, height=260, margin=dict(l=20, r=20, t=30, b=10))
    return fig


def survival_timeline(months: float, risk: str) -> go.Figure:
    color = {"Safe": "#48BB78", "Risky": "#F6AD55", "Critical": "#FC8181"}.get(risk, "#888")
    x = list(range(0, 37))
    lam = 1 / max(months, 1)
    surv = [np.exp(-lam * t) * 100 for t in x]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=surv,
        mode="lines", fill="tozeroy",
        line=dict(color=color, width=2),
        fillcolor="rgba(252,129,129,0.12)" if risk == "Critical" else
                  "rgba(246,173,85,0.12)" if risk == "Risky" else
                  "rgba(72,187,120,0.12)",
        name="Survival %",
    ))
    fig.add_vline(x=months, line_dash="dash", line_color=color, line_width=1.5,
                  annotation_text=f"Est. {months:.0f}mo", annotation_font_color=color)
    fig.update_layout(
        **PLOTLY_DARK,
        title="Survival Curve (Probability of Not Defaulting)",
        xaxis_title="Months",
        yaxis_title="Survival Probability (%)",
        xaxis=dict(gridcolor="#1E2A3A"),
        yaxis=dict(gridcolor="#1E2A3A", range=[0, 105]),
        height=260,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    return fig


def render():
    st.markdown("<div class='section-header'>Individual Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Enter customer details to get real-time risk assessment</div>", unsafe_allow_html=True)

    models = get_models()

    with st.form("prediction_form"):
        st.markdown("#### 📋 Customer Financial Profile")

        col1, col2, col3 = st.columns(3)
        with col1:
            income = st.number_input("Monthly Income (₹)", min_value=10000, max_value=500000, value=65000, step=1000)
            loan_amount = st.number_input("Loan Amount (₹)", min_value=10000, max_value=5000000, value=400000, step=5000)
            credit_score = st.slider("Credit Score", 300, 900, 680)
            employment_length = st.number_input("Employment Length (years)", min_value=0.0, max_value=40.0, value=4.5, step=0.5)

        with col2:
            emi = st.number_input("Monthly EMI (₹)", min_value=500, max_value=200000, value=12000, step=500)
            credit_utilization_ratio = st.slider("Credit Utilization Ratio", 0.0, 1.0, 0.45, 0.01)
            missed_payments = st.number_input("Missed Payments (lifetime)", min_value=0, max_value=24, value=2)
            num_loans = st.number_input("Number of Active Loans", min_value=0, max_value=10, value=2)

        with col3:
            avg_monthly_balance = st.number_input("Avg Monthly Balance (₹)", min_value=500, max_value=500000, value=35000, step=500)
            banking_relationship_length = st.number_input("Banking Relationship (years)", min_value=0.0, max_value=35.0, value=5.0, step=0.5)
            balance_trend = st.slider("Balance Trend (monthly %Δ)", -3.0, 3.0, 0.2, 0.1)
            rolling_missed_3m = st.number_input("Missed Payments (last 3 months)", min_value=0, max_value=3, value=0)

        submitted = st.form_submit_button("🔍 Run Risk Assessment", use_container_width=True)

    if submitted:
        income_to_emi_ratio = income / (emi + 1)
        loan_to_income_ratio = loan_amount / (income + 1)
        payment_behavior_score = max(0, min(100,
            100 - missed_payments * 8 - credit_utilization_ratio * 20 + employment_length * 0.5
        ))
        utilization_trend = 0.0

        customer_data = {
            "income": income,
            "employment_length": employment_length,
            "loan_amount": loan_amount,
            "emi": emi,
            "credit_score": credit_score,
            "credit_utilization_ratio": credit_utilization_ratio,
            "missed_payments": missed_payments,
            "avg_monthly_balance": avg_monthly_balance,
            "balance_trend": balance_trend,
            "utilization_trend": utilization_trend,
            "payment_behavior_score": payment_behavior_score,
            "rolling_missed_3m": rolling_missed_3m,
            "num_loans": num_loans,
            "banking_relationship_length": banking_relationship_length,
            "income_to_emi_ratio": income_to_emi_ratio,
            "loan_to_income_ratio": loan_to_income_ratio,
        }

        with st.spinner("Running AI risk assessment..."):
            result = predict_single(customer_data, models)

        risk = result["risk_level"]
        prob = result["default_probability"]
        months = result["time_to_default_months"]

        badge_map = {
            "Safe": "<span class='badge-safe'>✓ SAFE</span>",
            "Risky": "<span class='badge-risky'>⚠ RISKY</span>",
            "Critical": "<span class='badge-critical'>🚨 CRITICAL</span>",
        }
        action_class = {"Safe": "action-safe", "Risky": "action-risky", "Critical": "action-critical"}

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 📊 Risk Assessment Results")

        # Top result row
        r1, r2, r3 = st.columns([1, 1, 1])
        with r1:
            st.plotly_chart(gauge_chart(prob, risk), use_container_width=True)
        with r2:
            st.plotly_chart(survival_timeline(months, risk), use_container_width=True)
        with r3:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='metric-card'>
                <div class='label'>Risk Level</div>
                <div style='font-size:1.5rem;margin:6px 0'>{badge_map[risk]}</div>
            </div>
            <div class='metric-card'>
                <div class='label'>Est. Time to Default</div>
                <div class='value'>{months:.0f} <span style='font-size:1rem;color:#6B7A99'>months</span></div>
            </div>
            <div class='metric-card'>
                <div class='label'>Cluster Segment</div>
                <div style='font-size:1.1rem;font-weight:600;color:#3A8FD6'>{result["cluster_segment"]}</div>
            </div>
            """, unsafe_allow_html=True)

        # Model breakdown
        st.markdown("#### 🤖 Model Ensemble Breakdown")
        mc1, mc2, mc3, mc4 = st.columns(4)
        for col, name, val in zip(
            [mc1, mc2, mc3, mc4],
            ["Ensemble (Primary)", "XGBoost", "Random Forest", "Logistic Reg."],
            [prob, result["xgb_probability"], result["rf_probability"], result["lr_probability"]],
        ):
            with col:
                pct = val * 100
                color = "#FC8181" if pct >= 60 else "#F6AD55" if pct >= 30 else "#48BB78"
                st.markdown(f"""<div class='metric-card' style='text-align:center'>
                    <div class='label'>{name}</div>
                    <div class='value' style='font-size:1.6rem;color:{color}'>{pct:.1f}%</div>
                </div>""", unsafe_allow_html=True)

        # Recommended action
        st.markdown(f"""
        <div class='action-box {action_class[risk]}'>
            <b>📋 Recommended Bank Action</b><br><br>
            {result["recommended_action"]}
        </div>
        """, unsafe_allow_html=True)
