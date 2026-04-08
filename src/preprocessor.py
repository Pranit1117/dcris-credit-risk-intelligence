import numpy as np
import pandas as pd
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logger import get_logger
from sklearn.preprocessing import StandardScaler

logger = get_logger("preprocessor")

FEATURE_COLUMNS = [
    "income", "employment_length", "loan_amount", "emi", "credit_score",
    "credit_utilization_ratio", "missed_payments", "num_loans",
    "banking_relationship_length", "avg_monthly_balance",
    "balance_trend", "utilization_trend", "payment_behavior_score",
    "rolling_missed_3m", "income_to_emi_ratio", "loan_to_income_ratio",
]


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Median imputation for numeric columns
    for col in ["employment_length", "avg_monthly_balance"]:
        if col in df.columns:
            median_val = df[col].median()
            missing = df[col].isna().sum()
            if missing > 0:
                df[col] = df[col].fillna(median_val)
                logger.info(f"Imputed {missing} missing values in '{col}' with median={median_val:.2f}")
    return df


def handle_outliers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        clipped = df[col].clip(q1, q99)
        n_clipped = (df[col] != clipped).sum()
        if n_clipped > 0:
            logger.info(f"Clipped {n_clipped} outliers in '{col}' to [{q1:.2f}, {q99:.2f}]")
        df[col] = clipped
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """All feature engineering; safe to call on both train and inference data."""
    df = df.copy()

    # Safety: fill derived feature NaNs
    if "income_to_emi_ratio" not in df.columns:
        df["income_to_emi_ratio"] = df["income"] / (df["emi"] + 1)
    if "loan_to_income_ratio" not in df.columns:
        df["loan_to_income_ratio"] = df["loan_amount"] / (df["income"] + 1)
    if "payment_behavior_score" not in df.columns:
        df["payment_behavior_score"] = (
            100
            - df.get("missed_payments", 0) * 8
            - df.get("credit_utilization_ratio", 0) * 20
        ).clip(0, 100)
    if "rolling_missed_3m" not in df.columns:
        df["rolling_missed_3m"] = df.get("missed_payments", 0).clip(0, 3)
    if "balance_trend" not in df.columns:
        df["balance_trend"] = 0.0
    if "utilization_trend" not in df.columns:
        df["utilization_trend"] = 0.0

    return df


def preprocess(df: pd.DataFrame, scaler=None, fit_scaler: bool = False):
    """Full preprocessing pipeline. Returns (X_scaled, scaler)."""
    logger.info("Starting preprocessing pipeline...")

    df = handle_missing_values(df)

    outlier_cols = ["income", "loan_amount", "emi", "avg_monthly_balance", "income_to_emi_ratio", "loan_to_income_ratio"]
    df = handle_outliers(df, outlier_cols)

    df = engineer_features(df)

    # Select features
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    X = df[FEATURE_COLUMNS].copy()

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logger.info("Fitted new StandardScaler.")
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit_scaler=False")
        X_scaled = scaler.transform(X)

    logger.info(f"Preprocessing complete. Feature matrix shape: {X_scaled.shape}")
    return X_scaled, scaler, df


def save_scaler(scaler, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    logger.info(f"Scaler saved to {path}")


def load_scaler(path: str):
    scaler = joblib.load(path)
    logger.info(f"Scaler loaded from {path}")
    return scaler
