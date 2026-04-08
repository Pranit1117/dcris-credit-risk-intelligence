import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from app.data_loader import get_data

PLOTLY_DARK = dict(
    paper_bgcolor="#0F1825",
    plot_bgcolor="#0F1825",
    font=dict(color="#C8D0DC", family="DM Sans"),
    xaxis=dict(gridcolor="#1E2A3A", zerolinecolor="#1E2A3A"),
    yaxis=dict(gridcolor="#1E2A3A", zerolinecolor="#1E2A3A"),
)


def render():
    st.markdown("<div class='section-header'>Home Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Portfolio-wide credit risk overview across all customers</div>", unsafe_allow_html=True)

    df = get_data()

    # ── KPI row ──────────────────────────────────────────────────────────────
    total = len(df)
    critical_pct = (df["risk_segment"] == "Critical").mean() * 100
    risky_pct = (df["risk_segment"] == "Risky").mean() * 100
    safe_pct = (df["risk_segment"] == "Safe").mean() * 100
    avg_prob = df["default_probability"].mean() * 100
    default_rate = df["defaulted"].mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='label'>Total Customers</div>
            <div class='value'>{total:,}</div>
            <div class='delta'>↑ Active portfolio</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='label'>Critical Risk</div>
            <div class='value' style='color:#FC8181'>{critical_pct:.1f}%</div>
            <div class='delta' style='color:#FC8181'>⚠ Immediate action</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='label'>Risky Customers</div>
            <div class='value' style='color:#F6AD55'>{risky_pct:.1f}%</div>
            <div class='delta' style='color:#F6AD55'>↗ Monitor closely</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card'>
            <div class='label'>Safe Customers</div>
            <div class='value' style='color:#48BB78'>{safe_pct:.1f}%</div>
            <div class='delta' style='color:#48BB78'>✓ Low risk</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class='metric-card'>
            <div class='label'>Avg Default Prob</div>
            <div class='value'>{avg_prob:.1f}%</div>
            <div class='delta'>Portfolio-wide</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Donut + Default prob distribution ──────────────────────────
    col1, col2 = st.columns([1, 1.6])

    with col1:
        seg_counts = df["risk_segment"].value_counts()
        fig_donut = go.Figure(go.Pie(
            labels=seg_counts.index.tolist(),
            values=seg_counts.values.tolist(),
            hole=0.65,
            marker=dict(colors=["#48BB78", "#F6AD55", "#FC8181"],
                        line=dict(color="#080C14", width=2)),
            textinfo="label+percent",
            textfont=dict(size=12, color="#E8EAF0"),
        ))
        fig_donut.update_layout(
            title=dict(text="Risk Segment Distribution", font=dict(color="#E8EAF0", size=14)),
            showlegend=False,
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            **{k: v for k, v in PLOTLY_DARK.items() if k in ["paper_bgcolor", "font"]},
        )
        fig_donut.add_annotation(
            text=f"<b>{total:,}</b><br><span style='font-size:10px'>Customers</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#E8EAF0"),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        fig_hist = px.histogram(
            df, x="default_probability", nbins=50,
            color="risk_segment",
            color_discrete_map={"Safe": "#48BB78", "Risky": "#F6AD55", "Critical": "#FC8181"},
            barmode="overlay", opacity=0.75,
        )
        fig_hist.update_layout(
            title="Default Probability Distribution by Segment",
            **PLOTLY_DARK,
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#C8D0DC")),
            xaxis_title="Default Probability",
            yaxis_title="Count",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Row 2: Credit score vs default prob scatter + missed payments bar ──
    col3, col4 = st.columns(2)

    with col3:
        sample = df.sample(min(2000, len(df)), random_state=42)
        fig_scatter = px.scatter(
            sample, x="credit_score", y="default_probability",
            color="risk_segment",
            color_discrete_map={"Safe": "#48BB78", "Risky": "#F6AD55", "Critical": "#FC8181"},
            opacity=0.5, size_max=4,
        )
        fig_scatter.update_traces(marker=dict(size=3))
        fig_scatter.update_layout(
            title="Credit Score vs Default Probability",
            **PLOTLY_DARK,
            height=320,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#C8D0DC")),
            xaxis_title="Credit Score",
            yaxis_title="Default Probability",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col4:
        mp_seg = df.groupby("risk_segment")["missed_payments"].mean().reset_index()
        color_map = {"Safe": "#48BB78", "Risky": "#F6AD55", "Critical": "#FC8181"}
        fig_bar = px.bar(
            mp_seg, x="risk_segment", y="missed_payments",
            color="risk_segment",
            color_discrete_map=color_map,
            text="missed_payments",
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                              textfont=dict(color="#E8EAF0"))
        fig_bar.update_layout(
            title="Avg Missed Payments by Risk Segment",
            **PLOTLY_DARK,
            height=320,
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
            xaxis_title="Risk Segment",
            yaxis_title="Avg Missed Payments",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Row 3: Income distribution + credit utilization ───────────────────
    col5, col6 = st.columns(2)

    with col5:
        fig_income = px.box(
            df, x="risk_segment", y="income",
            color="risk_segment",
            color_discrete_map={"Safe": "#48BB78", "Risky": "#F6AD55", "Critical": "#FC8181"},
        )
        fig_income.update_layout(
            title="Income Distribution by Risk Segment",
            **PLOTLY_DARK,
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
            xaxis_title="Risk Segment",
            yaxis_title="Monthly Income (₹)",
        )
        st.plotly_chart(fig_income, use_container_width=True)

    with col6:
        fig_util = px.violin(
            df, x="risk_segment", y="credit_utilization_ratio",
            color="risk_segment",
            color_discrete_map={"Safe": "#48BB78", "Risky": "#F6AD55", "Critical": "#FC8181"},
            box=True,
        )
        fig_util.update_layout(
            title="Credit Utilization by Risk Segment",
            **PLOTLY_DARK,
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
            xaxis_title="Risk Segment",
            yaxis_title="Utilization Ratio",
        )
        st.plotly_chart(fig_util, use_container_width=True)

    # ── Top 10 high-risk customers table ─────────────────────────────────
    st.markdown("<br><div class='section-header' style='font-size:1.1rem'>🚨 Top High-Risk Customers</div>", unsafe_allow_html=True)
    top_risk = df.nlargest(10, "default_probability")[
        ["customer_id", "default_probability", "credit_score", "missed_payments",
         "credit_utilization_ratio", "income", "risk_segment"]
    ].copy()
    top_risk["default_probability"] = (top_risk["default_probability"] * 100).round(1).astype(str) + "%"
    top_risk["income"] = top_risk["income"].apply(lambda x: f"₹{x:,.0f}")
    top_risk["credit_utilization_ratio"] = (top_risk["credit_utilization_ratio"] * 100).round(1).astype(str) + "%"
    top_risk.columns = ["Customer ID", "Default Prob", "Credit Score", "Missed Payments",
                        "Credit Util.", "Income", "Segment"]
    st.dataframe(top_risk, use_container_width=True, hide_index=True)
