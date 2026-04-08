import numpy as np
import pandas as pd
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logger import get_logger
from src.preprocessor import preprocess, save_scaler, FEATURE_COLUMNS

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score
)
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from lifelines import WeibullAFTFitter

logger = get_logger("trainer")

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def train_layer1(X_train, y_train, X_test, y_test):
    """Layer 1: Default prediction - 3 classifiers."""
    logger.info("=== LAYER 1: Default Prediction ===")
    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_prob)
    logger.info(f"Logistic Regression AUC: {lr_auc:.4f}")
    results["logistic"] = {"model": lr, "auc": lr_auc, "probs": lr_prob}

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_prob)
    logger.info(f"Random Forest AUC: {rf_auc:.4f}")
    results["random_forest"] = {"model": rf, "auc": rf_auc, "probs": rf_prob}

    # XGBoost
    scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pos, random_state=42,
        eval_metric="logloss", verbosity=0
    )
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_pred = xgb.predict(X_test)
    xgb_prob = xgb.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_prob)
    logger.info(f"XGBoost AUC: {xgb_auc:.4f}")
    results["xgboost"] = {"model": xgb, "auc": xgb_auc, "probs": xgb_prob}

    for name, r in results.items():
        path = os.path.join(MODELS_DIR, f"{name}_model.pkl")
        joblib.dump(r["model"], path)
        logger.info(f"Saved {name} model to {path}")

    return results


def train_layer2(df_train: pd.DataFrame):
    """Layer 2: Time-to-default via Weibull AFT survival model."""
    logger.info("=== LAYER 2: Time-to-Default (Survival Analysis) ===")

    survival_features = [
        "income", "credit_score", "credit_utilization_ratio",
        "missed_payments", "payment_behavior_score",
        "rolling_missed_3m", "balance_trend", "loan_to_income_ratio",
        "income_to_emi_ratio", "employment_length", "banking_relationship_length",
        "defaulted", "time_to_default"
    ]
    df_surv = df_train[survival_features].dropna().copy()

    # Weibull AFT: event=defaulted, duration=time_to_default
    waf = WeibullAFTFitter(penalizer=0.01)
    waf.fit(df_surv, duration_col="time_to_default", event_col="defaulted")
    logger.info("Weibull AFT model fitted successfully.")
    logger.info(f"Concordance Index: {waf.concordance_index_:.4f}")

    path = os.path.join(MODELS_DIR, "survival_model.pkl")
    joblib.dump(waf, path)
    logger.info(f"Saved survival model to {path}")
    return waf


def train_layer3(X_scaled: np.ndarray):
    """Layer 3: Risk segmentation via KMeans."""
    logger.info("=== LAYER 3: Risk Segmentation (KMeans) ===")
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    km.fit(X_scaled)
    labels = km.labels_
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        logger.info(f"  Cluster {u}: {c} customers ({c/len(labels):.1%})")

    path = os.path.join(MODELS_DIR, "kmeans_model.pkl")
    joblib.dump(km, path)
    logger.info(f"Saved KMeans model to {path}")
    return km, labels


def label_clusters(df: pd.DataFrame, cluster_labels: np.ndarray, default_probs: np.ndarray):
    df = df.copy()
    df["cluster"] = cluster_labels
    df["default_probability"] = default_probs  # FIXED: was "default_prob"

    cluster_risk = df.groupby("cluster")["default_probability"].mean().sort_values()
    mapping = {
        cluster_risk.index[0]: "Safe",
        cluster_risk.index[1]: "Risky",
        cluster_risk.index[2]: "Critical",
    }
    df["risk_segment"] = df["cluster"].map(mapping)
    return df, mapping


def run_training():
    from src.data_generator import generate_credit_dataset
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Generate / load data
    data_path = os.path.join(os.path.dirname(MODELS_DIR), "data", "credit_data.csv")
    if not os.path.exists(data_path):
        df = generate_credit_dataset(15000)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    logger.info(f"Loaded data: {df.shape}, default rate: {df['defaulted'].mean():.2%}")

    # Preprocess
    X_scaled, scaler, df_clean = preprocess(df, fit_scaler=True)
    y = df_clean["defaulted"].values

    # Save scaler and feature list
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    save_scaler(scaler, scaler_path)
    joblib.dump(FEATURE_COLUMNS, os.path.join(MODELS_DIR, "feature_cols.pkl"))

    # Train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    idx_tr, idx_te = train_test_split(range(len(df_clean)), test_size=0.2, random_state=42)
    df_train = df_clean.iloc[idx_tr].reset_index(drop=True)

    # Layer 1
    layer1_results = train_layer1(X_tr, y_tr, X_te, y_te)

    # Layer 2
    survival_model = train_layer2(df_train)

    # Layer 3 (use XGBoost probs for cluster labeling)
    km_model, cluster_labels = train_layer3(X_scaled)
    xgb_probs_all = layer1_results["xgboost"]["model"].predict_proba(X_scaled)[:, 1]
    rf_probs_all = layer1_results["random_forest"]["model"].predict_proba(X_scaled)[:, 1]
    lr_probs_all = layer1_results["logistic"]["model"].predict_proba(X_scaled)[:, 1]
    ensemble_probs_all = 0.5 * xgb_probs_all + 0.3 * rf_probs_all + 0.2 * lr_probs_all
    df_labeled, cluster_mapping = label_clusters(df_clean, cluster_labels, ensemble_probs_all)    # Save processed data
    processed_path = os.path.join(os.path.dirname(MODELS_DIR), "data", "processed_data.csv")
    df_labeled.to_csv(processed_path, index=False)
    logger.info(f"Saved processed data to {processed_path}")

    # Save cluster mapping
    joblib.dump(cluster_mapping, os.path.join(MODELS_DIR, "cluster_mapping.pkl"))

    logger.info("=== ALL MODELS TRAINED SUCCESSFULLY ===")
    return layer1_results, survival_model, km_model


if __name__ == "__main__":
    run_training()
