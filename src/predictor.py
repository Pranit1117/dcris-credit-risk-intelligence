import numpy as np
import pandas as pd
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logger import get_logger
from src.preprocessor import preprocess, FEATURE_COLUMNS

logger = get_logger("predictor")

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def load_all_models():
    models = {}
    for name in ["logistic_model", "rf_model", "xgb_model", "survival_model", "kmeans_model"]:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if os.path.exists(path):
            models[name.replace("_model", "")] = joblib.load(path)
        else:
            logger.warning(f"Model not found: {path}")

    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    models["scaler"] = joblib.load(scaler_path)

    mapping_path = os.path.join(MODELS_DIR, "cluster_mapping.pkl")
    if os.path.exists(mapping_path):
        models["cluster_mapping"] = joblib.load(mapping_path)

    logger.info(f"Loaded models: {list(models.keys())}")
    return models


def get_risk_level(prob: float) -> str:
    if prob < 0.30:
        return "Safe"
    elif prob < 0.60:
        return "Risky"
    else:
        return "Critical"


def get_action(risk_level: str, prob: float, months: float) -> str:
    actions = {
        "Safe": "✅ No immediate action required. Continue regular monitoring. Eligible for credit limit increase.",
        "Risky": (
            f"⚠️ Proactive outreach recommended. Consider restructuring EMI schedule. "
            f"Monitor for next {max(1, int(months))} months. Credit counseling advised."
        ),
        "Critical": (
            f"🚨 URGENT: Initiate NPA prevention protocol. Immediate call with relationship manager. "
            f"Offer one-time settlement or moratorium. Escalate to risk committee within 7 days."
        ),
    }
    return actions.get(risk_level, "Monitor closely.")


def predict_single(customer_data: dict, models: dict) -> dict:
    """Run full prediction for one customer."""
    df = pd.DataFrame([customer_data])

    # Preprocess
    X_scaled, _, df_clean = preprocess(df, scaler=models["scaler"], fit_scaler=False)

    # Layer 1: Default probability (use XGBoost as primary)
    xgb_prob = float(models["xgb"].predict_proba(X_scaled)[0, 1])
    rf_prob = float(models["rf"].predict_proba(X_scaled)[0, 1])
    lr_prob = float(models["logistic"].predict_proba(X_scaled)[0, 1])
    ensemble_prob = 0.5 * xgb_prob + 0.3 * rf_prob + 0.2 * lr_prob

    # Layer 2: Time to default
    surv_features = [
        "income", "credit_score", "credit_utilization_ratio",
        "missed_payments", "payment_behavior_score",
        "rolling_missed_3m", "balance_trend", "loan_to_income_ratio",
        "income_to_emi_ratio", "employment_length", "banking_relationship_length",
    ]
    df_surv = df_clean[surv_features].fillna(df_clean[surv_features].median())
    df_surv["defaulted"] = 1  # for prediction
    df_surv["time_to_default"] = 12  # placeholder
    median_survival = float(models["survival"].predict_median(df_surv).iloc[0])
    median_survival = max(1.0, min(36.0, median_survival))

    # Layer 3: Cluster
    cluster_id = int(models["kmeans"].predict(X_scaled)[0])
    mapping = models.get("cluster_mapping", {0: "Safe", 1: "Risky", 2: "Critical"})
    cluster_label = mapping.get(cluster_id, "Unknown")

    risk_level = get_risk_level(ensemble_prob)
    action = get_action(risk_level, ensemble_prob, median_survival)

    result = {
        "default_probability": round(ensemble_prob, 4),
        "xgb_probability": round(xgb_prob, 4),
        "rf_probability": round(rf_prob, 4),
        "lr_probability": round(lr_prob, 4),
        "time_to_default_months": round(median_survival, 1),
        "risk_level": risk_level,
        "cluster_segment": cluster_label,
        "recommended_action": action,
    }
    logger.info(f"Prediction: prob={ensemble_prob:.3f}, risk={risk_level}, months={median_survival:.1f}")
    return result


def predict_batch(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Run predictions for a full dataframe."""
    logger.info(f"Running batch prediction for {len(df)} customers...")
    X_scaled, _, df_clean = preprocess(df, scaler=models["scaler"], fit_scaler=False)

    xgb_probs = models["xgb"].predict_proba(X_scaled)[:, 1]
    rf_probs = models["rf"].predict_proba(X_scaled)[:, 1]
    lr_probs = models["logistic"].predict_proba(X_scaled)[:, 1]
    ensemble_probs = 0.5 * xgb_probs + 0.3 * rf_probs + 0.2 * lr_probs

    cluster_ids = models["kmeans"].predict(X_scaled)
    mapping = models.get("cluster_mapping", {0: "Safe", 1: "Risky", 2: "Critical"})
    cluster_labels = [mapping.get(c, "Unknown") for c in cluster_ids]

    df_out = df_clean.copy()
    df_out["default_probability"] = ensemble_probs.round(4)
    df_out["risk_level"] = [get_risk_level(p) for p in ensemble_probs]
    df_out["cluster_segment"] = cluster_labels
    df_out["xgb_probability"] = xgb_probs.round(4)

    logger.info("Batch prediction complete.")
    return df_out
