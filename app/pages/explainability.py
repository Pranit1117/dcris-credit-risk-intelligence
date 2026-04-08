import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from app.data_loader import get_data, get_models
from src.preprocessor import preprocess, FEATURE_COLUMNS

PLOTLY_DARK = dict(
    paper_bgcolor="#0F1825",
    plot_bgcolor="#0F1825",
    font=dict(color="#C8D0DC", family="DM Sans"),
    xaxis=dict(gridcolor="#1E2A3A", zerolinecolor="#1E2A3A"),
    yaxis=dict(gridcolor="#1E2A3A", zerolinecolor="#1E2A3A"),
)


@st.cache_resource(show_spinner="Computing SHAP values (one-time)...")
def get_explainer_cached(_xgb_model):
    import shap
    explainer = shap.TreeExplainer(_xgb_model)
    return explainer


@st.cache_data(show_spinner="Computing SHAP summary...")
def get_shap_summary(_explainer, _X_bytes):
    X_scaled = np.frombuffer(_X_bytes).reshape(-1, len(FEATURE_COLUMNS))
    idx = np.random.choice(len(X_scaled), size=min(600, len(X_scaled)), replace=False)
    X_sample = X_scaled[idx]
    shap_vals = _explainer.shap_values(X_sample)
    return shap_vals, X_sample


def shap_single(explainer, x_row):
    x_2d = x_row.reshape(1, -1)
    sv = explainer.shap_values(x_2d)[0]
    df_shap = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "shap_value": sv,
        "abs_impact": np.abs(sv),
    }).sort_values("abs_impact", ascending=False)
    return df_shap


def render():
    st.markdown("<div class='section-header'>Explainability (SHAP)</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Understand why each prediction was made</div>", unsafe_allow_html=True)

    df = get_data()
    models = get_models()

    X_scaled, _, df_clean = preprocess(df, scaler=models["scaler"], fit_scaler=False)
    explainer = get_explainer_cached(models["xgb"])

    tab1, tab2 = st.tabs(["🌐 Global Feature Impact", "👤 Individual Explanation"])

    with tab1:
        st.markdown("#### Global SHAP Feature Importance")

        X_bytes = X_scaled.tobytes()
        shap_vals, X_sample = get_shap_summary(explainer, X_bytes)

        mean_abs = np.abs(shap_vals).mean(axis=0)
        feat_imp_df = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=True)

        fig_global = go.Figure(go.Bar(
            x=feat_imp_df["mean_abs_shap"],
            y=feat_imp_df["feature"],
            orientation="h",
            marker=dict(
                color=feat_imp_df["mean_abs_shap"],
                colorscale=[[0, "#1A3358"], [0.5, "#1E5FA8"], [1.0, "#3A9FE6"]],
            ),
            text=[f"{v:.4f}" for v in feat_imp_df["mean_abs_shap"]],
            textposition="outside",
            textfont=dict(color="#C8D0DC", size=10),
        ))
        fig_global.update_layout(
            paper_bgcolor="#0F1825",
            plot_bgcolor="#0F1825",
            font=dict(color="#C8D0DC", family="DM Sans"),
            xaxis=dict(gridcolor="#1E2A3A", zerolinecolor="#1E2A3A", title="Mean |SHAP Value|"),
            yaxis=dict(gridcolor="#1E2A3A", zerolinecolor="#1E2A3A"),
            height=480,
            margin=dict(l=10, r=80, t=10, b=10),
        )
        st.plotly_chart(fig_global, use_container_width=True)

        st.markdown("#### SHAP Beeswarm-style Distribution")
        top_feats = feat_imp_df.tail(8)["feature"].tolist()
        top_idx = [FEATURE_COLUMNS.index(f) for f in top_feats]
        top_feats_rev = top_feats[::-1]
        top_idx_rev = top_idx[::-1]

        fig_bee = go.Figure()
        for i, (fi, fname) in enumerate(zip(top_idx_rev, top_feats_rev)):
            sv = shap_vals[:, fi]
            fv = X_sample[:, fi]
            fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
            jitter = np.random.uniform(-0.3, 0.3, size=len(sv))
            fig_bee.add_trace(go.Scatter(
                x=sv,
                y=[i + j for j in jitter],
                mode="markers",
                marker=dict(
                    size=3,
                    color=fv_norm,
                    colorscale=[[0, "#3A5FBF"], [0.5, "#8888AA"], [1.0, "#FC8181"]],
                    opacity=0.6,
                ),
                name=fname,
                showlegend=False,
                hovertemplate=f"<b>{fname}</b><br>SHAP: %{{x:.4f}}<extra></extra>",
            ))

        fig_bee.update_layout(
            paper_bgcolor="#0F1825",
            plot_bgcolor="#0F1825",
            font=dict(color="#C8D0DC", family="DM Sans"),
            xaxis=dict(
                gridcolor="#1E2A3A",
                zerolinecolor="#1E2A3A",
                title="SHAP Value (impact on model output)",
            ),
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(8)),
                ticktext=top_feats_rev,
                gridcolor="#1E2A3A",
            ),
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig_bee.add_vline(x=0, line_dash="dash", line_color="#444", line_width=1)
        st.plotly_chart(fig_bee, use_container_width=True)

    with tab2:
        st.markdown("#### Explain a Specific Customer's Prediction")

        customer_ids = df["customer_id"].tolist() if "customer_id" in df.columns else list(range(len(df)))
        selected_id = st.selectbox("Select Customer ID", customer_ids[:200], index=0)

        if "customer_id" in df.columns:
            row_idx = df[df["customer_id"] == selected_id].index[0]
        else:
            row_idx = selected_id

        customer_row = df.iloc[row_idx]
        x_row = X_scaled[row_idx]

        prob = float(customer_row.get("default_probability", 0.5))
        risk = str(customer_row.get("risk_segment", "Unknown"))

        badge_map = {
            "Safe": "<span class='badge-safe'>✓ SAFE</span>",
            "Risky": "<span class='badge-risky'>⚠ RISKY</span>",
            "Critical": "<span class='badge-critical'>🚨 CRITICAL</span>",
        }
        color_map = {"Safe": "#48BB78", "Risky": "#F6AD55", "Critical": "#FC8181"}

        info_col, shap_col = st.columns([1, 2])

        with info_col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='label'>Customer</div>
                <div style='font-size:1rem;font-weight:600;color:#3A8FD6;font-family:monospace'>{customer_row.get('customer_id', 'N/A')}</div>
            </div>
            <div class='metric-card'>
                <div class='label'>Default Probability</div>
                <div class='value' style='color:{color_map.get(risk, "#888")}'>{prob*100:.1f}%</div>
            </div>
            <div class='metric-card'>
                <div class='label'>Risk Level</div>
                <div style='font-size:1.2rem;margin-top:4px'>{badge_map.get(risk, risk)}</div>
            </div>
            """, unsafe_allow_html=True)

            stats = {
                "Credit Score": f"{int(customer_row.get('credit_score', 0))}",
                "Income": f"Rs.{customer_row.get('income', 0):,.0f}",
                "Missed Payments": f"{int(customer_row.get('missed_payments', 0))}",
                "Credit Util.": f"{customer_row.get('credit_utilization_ratio', 0)*100:.1f}%",
                "Pay Behavior": f"{customer_row.get('payment_behavior_score', 0):.1f}/100",
            }
            for k, v in stats.items():
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"padding:4px 0;border-bottom:1px solid #1E2A3A;font-size:0.85rem'>"
                    f"<span style='color:#6B7A99'>{k}</span>"
                    f"<span style='color:#E8EAF0;font-family:monospace'>{v}</span></div>",
                    unsafe_allow_html=True
                )

        with shap_col:
            df_shap = shap_single(explainer, x_row)
            shap_vals_top = df_shap["shap_value"].head(12).iloc[::-1].tolist()
            shap_feats_top = df_shap["feature"].head(12).iloc[::-1].tolist()

            fig_shap = go.Figure(go.Bar(
                x=shap_vals_top,
                y=shap_feats_top,
                orientation="h",
                marker=dict(
                    color=["#FC8181" if v > 0 else "#48BB78" for v in shap_vals_top],
                ),
                text=[f"{v:+.4f}" for v in shap_vals_top],
                textposition="outside",
                textfont=dict(color="#C8D0DC", size=10),
            ))
            fig_shap.update_layout(
                paper_bgcolor="#0F1825",
                plot_bgcolor="#0F1825",
                font=dict(color="#C8D0DC", family="DM Sans"),
                xaxis=dict(
                    gridcolor="#1E2A3A",
                    zerolinecolor="#1E2A3A",
                    title="SHAP Value (red=increases risk, green=reduces risk)",
                ),
                yaxis=dict(gridcolor="#1E2A3A", zerolinecolor="#1E2A3A"),
                height=420,
                title=f"SHAP Feature Impact — {customer_row.get('customer_id', 'Customer')}",
                margin=dict(l=10, r=80, t=50, b=10),
            )
            fig_shap.add_vline(x=0, line_dash="dash", line_color="#555", line_width=1)
            st.plotly_chart(fig_shap, use_container_width=True)

        st.markdown("#### SHAP Waterfall")
        df_wf = df_shap.head(10).copy()
        fig_wf = go.Figure(go.Waterfall(
            orientation="h",
            measure=["relative"] * len(df_wf),
            x=df_wf["shap_value"].tolist(),
            y=df_wf["feature"].tolist(),
            connector=dict(line=dict(color="#1E2A3A", width=1)),
            increasing=dict(marker=dict(color="#FC8181")),
            decreasing=dict(marker=dict(color="#48BB78")),
        ))
        fig_wf.update_layout(
            paper_bgcolor="#0F1825",
            plot_bgcolor="#0F1825",
            font=dict(color="#C8D0DC", family="DM Sans"),
            xaxis=dict(gridcolor="#1E2A3A", zerolinecolor="#1E2A3A", title="Cumulative SHAP Impact"),
            yaxis=dict(gridcolor="#1E2A3A", zerolinecolor="#1E2A3A"),
            height=380,
            title="Feature Contribution Waterfall",
            margin=dict(l=10, r=30, t=50, b=10),
        )
        st.plotly_chart(fig_wf, use_container_width=True)