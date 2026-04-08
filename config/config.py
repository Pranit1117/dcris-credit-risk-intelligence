import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

RAW_DATA_PATH = os.path.join(DATA_DIR, "credit_data.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_data.csv")

MODEL_PATHS = {
    "logistic": os.path.join(MODELS_DIR, "logistic_model.pkl"),
    "random_forest": os.path.join(MODELS_DIR, "rf_model.pkl"),
    "xgboost": os.path.join(MODELS_DIR, "xgb_model.pkl"),
    "survival": os.path.join(MODELS_DIR, "survival_model.pkl"),
    "kmeans": os.path.join(MODELS_DIR, "kmeans_model.pkl"),
    "scaler": os.path.join(MODELS_DIR, "scaler.pkl"),
    "feature_cols": os.path.join(MODELS_DIR, "feature_cols.pkl"),
}

RANDOM_STATE = 42
N_CLUSTERS = 3
TEST_SIZE = 0.2

RISK_THRESHOLDS = {
    "safe": 0.3,
    "risky": 0.6,
    "critical": 1.0,
}

FEATURE_COLUMNS = [
    "income", "employment_length", "loan_amount", "emi", "credit_score",
    "credit_utilization_ratio", "missed_payments", "num_loans",
    "banking_relationship_length", "avg_monthly_balance",
    "balance_trend", "utilization_trend", "payment_behavior_score",
    "rolling_missed_3m", "income_to_emi_ratio", "loan_to_income_ratio",
]

LOG_FILE = os.path.join(LOGS_DIR, "dcris.log")

for d in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)
