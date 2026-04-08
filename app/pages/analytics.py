import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.data_loader import get_data, get_models
from src.preprocessor import preprocess, FEATURE_COLUMNS

PLOTLY_DARK = dict(
    paper_bgcolor="#0F1825", plot_bgcolor="#0F1825",
    font=dict(color="#C8D0DC", family="DM Sans"),
    xaxis=dict(gridcolor="#1E2A3A", zerolinecolor="#1E2A3A"),
    yaxis=dict(gridcolor="#1E2A3A", zerolinecolor="#1E2A3A"),
)
COLOR_MAP = {"Safe": "#48BB78", "Risky": "#F6AD55", "Critical": "#FC8181"}


def render():
    st.markdown("<div class='section-header'>Risk Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Cluster intelligence, feature importance & behavioural patterns</div>", unsafe_allow_html=True)

    df = get_data()
    models = get_models()

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🗂 Cluster Analysis", "📈 Feature Importance", "🔬 Behavioural Patterns"])

    # ── Tab 1: Cluster Analysis ──────────────────────────────────────────
    with tab1:
        st.markdown("#### Customer Segments (KMeans k=3)")

        col1, col2 = st.columns([1.5, 1])
        with col1:
            # 2D PCA-like scatter using two most informative features
            sample = df.sample(min(3000, len(df)), random_state=42)
            fig_clust = px.scatter(
                sample,
                x="credit_score", y="credit_utilization_ratio",
                color="risk_segment",
                color_discrete_map=COLOR_MAP,
                opacity=0.6,
                hover_data=["missed_payments", "income"],
                labels={"credit_score": "Credit Score", "credit_utilization_ratio": "Credit Utilization"},
            )
            fig_clust.update_traces(marker=dict(size=4))
            fig_clust.update_layout(
                title="Risk Clusters: Credit Score vs Utilization",
                **PLOTLY_DARK, height=380,
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#C8D0DC")),
            )
            st.plotly_chart(fig_clust, use_container_width=True)

        with col2:
            # Segment stats table
            seg_stats = df.groupby("risk_segment").agg(
                Count=("customer_id", "count"),
                Avg_Default_Prob=("default_probability", "mean"),
                Avg_Credit_Score=("credit_score", "mean"),
                Avg_Missed=("missed_payments", "mean"),
                Avg_Income=("income", "mean"),
            ).round(2).reset_index()
            seg_stats["Avg_Default_Prob"] = (seg_stats["Avg_Default_Prob"] * 100).round(1).astype(str) + "%"
            seg_stats["Avg_Income"] = seg_stats["Avg_Income"].apply(lambda x: f"₹{x:,.0f}")
            seg_stats.columns = ["Segment", "Count", "Default Prob", "Credit Score", "Missed Pmts", "Income"]
            st.markdown("##### Segment Summary")
            st.dataframe(seg_stats, use_container_width=True, hide_index=True)

            # Radar chart for segment profiles
            categories = ["Credit Score", "Income", "Pay Behavior", "Utilization (inv)", "Missed (inv)"]

            def norm(series):
                return (series - series.min()) / (series.max() - series.min() + 1e-9)

            df_radar = df.copy()
            df_radar["cs_n"] = norm(df_radar["credit_score"])
            df_radar["inc_n"] = norm(df_radar["income"])
            df_radar["pb_n"] = norm(df_radar["payment_behavior_score"])
            df_radar["util_inv_n"] = 1 - norm(df_radar["credit_utilization_ratio"])
            df_radar["miss_inv_n"] = 1 - norm(df_radar["missed_payments"])

            fig_radar = go.Figure()
            for seg, color in COLOR_MAP.items():
                sub = df_radar[df_radar["risk_segment"] == seg]
                vals = [
                    sub["cs_n"].mean(), sub["inc_n"].mean(), sub["pb_n"].mean(),
                    sub["util_inv_n"].mean(), sub["miss_inv_n"].mean()
                ]
                vals += [vals[0]]
                cats = categories + [categories[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=cats, fill="toself",
                    name=seg, line_color=color,
                    fillcolor="rgba(72,187,120,0.13)" if seg == "Safe" else "rgba(246,173,85,0.13)" if seg == "Risky" else "rgba(252,129,129,0.13)",
                ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="#0F1825",
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor="#1E2A3A", tickfont=dict(color="#6B7A99")),
                    angularaxis=dict(gridcolor="#1E2A3A", tickfont=dict(color="#C8D0DC")),
                ),
                paper_bgcolor="#0F1825",
                font=dict(color="#C8D0DC"),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#C8D0DC")),
                title="Segment Risk Profile (Radar)",
                height=320,
                margin=dict(l=30, r=30, t=40, b=10),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # 3D scatter
        st.markdown("#### 3D Risk Landscape")
        sample3d = df.sample(min(2000, len(df)), random_state=7)
        fig3d = px.scatter_3d(
            sample3d,
            x="credit_score", y="income", z="default_probability",
            color="risk_segment",
            color_discrete_map=COLOR_MAP,
            opacity=0.65,
            labels={"credit_score": "Credit Score", "income": "Income", "default_probability": "Default Prob"},
        )
        fig3d.update_traces(marker=dict(size=2.5))
        fig3d.update_layout(
            scene=dict(
                bgcolor="#0F1825",
                xaxis=dict(backgroundcolor="#0F1825", gridcolor="#1E2A3A", color="#C8D0DC"),
                yaxis=dict(backgroundcolor="#0F1825", gridcolor="#1E2A3A", color="#C8D0DC"),
                zaxis=dict(backgroundcolor="#0F1825", gridcolor="#1E2A3A", color="#C8D0DC"),
            ),
            paper_bgcolor="#0F1825",
            font=dict(color="#C8D0DC"),
            height=450,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig3d, use_container_width=True)

    # ── Tab 2: Feature Importance ────────────────────────────────────────
    with tab2:
        st.markdown("#### XGBoost Feature Importance")

        xgb_model = models["xgb"]
        importances = xgb_model.feature_importances_
        feat_imp = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": importances,
        }).sort_values("importance", ascending=True)

        fig_fi = go.Figure(go.Bar(
            x=feat_imp["importance"],
            y=feat_imp["feature"],
            orientation="h",
            marker=dict(
                color=feat_imp["importance"],
                colorscale=[[0, "#1A3A5C"], [0.5, "#2A6FBF"], [1.0, "#3A8FD6"]],
                showscale=False,
            ),
            text=[f"{v:.4f}" for v in feat_imp["importance"]],
            textposition="outside",
            textfont=dict(color="#C8D0DC", size=10),
        ))
        fig_fi.update_layout(
            **PLOTLY_DARK, height=500,
            xaxis_title="Feature Importance Score",
            margin=dict(l=10, r=80, t=20, b=10),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        # Correlation heatmap
        st.markdown("#### Feature Correlation Matrix")
        corr_cols = ["credit_score", "income", "credit_utilization_ratio", "missed_payments",
                     "payment_behavior_score", "loan_to_income_ratio", "balance_trend",
                     "rolling_missed_3m", "income_to_emi_ratio", "defaulted"]
        corr = df[corr_cols].corr().round(2)
        fig_heat = px.imshow(
            corr, text_auto=True,
            color_continuous_scale=[[0, "#8B0000"], [0.5, "#0F1825"], [1, "#006400"]],
            zmin=-1, zmax=1,
            aspect="auto",
        )
        fig_heat.update_layout(
            paper_bgcolor="#0F1825", font=dict(color="#C8D0DC"),
            height=450, margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Tab 3: Behavioural Patterns ──────────────────────────────────────
    with tab3:
        st.markdown("#### Defaulter vs Non-Defaulter Behaviour")

        col1, col2 = st.columns(2)

        with col1:
            # Payment behavior score distribution
            fig_pbs = px.histogram(
                df, x="payment_behavior_score", color="defaulted",
                color_discrete_map={0: "#48BB78", 1: "#FC8181"},
                barmode="overlay", opacity=0.7, nbins=40,
                labels={"defaulted": "Defaulted"},
            )
            fig_pbs.update_layout(
                **PLOTLY_DARK, height=280, title="Payment Behavior Score",
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_pbs, use_container_width=True)

        with col2:
            fig_btd = px.histogram(
                df[df["defaulted"] == 1], x="time_to_default", nbins=36,
                color_discrete_sequence=["#FC8181"],
            )
            fig_btd.update_layout(
                **PLOTLY_DARK, height=280, title="Time-to-Default Distribution (Defaulters)",
                xaxis_title="Months to Default",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_btd, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig_uti = px.box(
                df, x=df["defaulted"].map({0: "Non-Defaulter", 1: "Defaulter"}),
                y="credit_utilization_ratio",
                color=df["defaulted"].map({0: "Non-Defaulter", 1: "Defaulter"}),
                color_discrete_map={"Non-Defaulter": "#48BB78", "Defaulter": "#FC8181"},
            )
            fig_uti.update_layout(
                **PLOTLY_DARK, height=280, title="Credit Utilization vs Default Status",
                margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
            )
            st.plotly_chart(fig_uti, use_container_width=True)

        with col4:
            fig_bal = px.scatter(
            df.sample(min(2000, len(df)), random_state=1),
            x="balance_trend", y="default_probability",
            color="risk_segment", color_discrete_map=COLOR_MAP,
            opacity=0.5,
        )
                trendline="lowess",
            )
            fig_bal.update_traces(marker=dict(size=3))
            fig_bal.update_layout(
                **PLOTLY_DARK, height=280, title="Balance Trend vs Default Probability",
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_bal, use_container_width=True)
