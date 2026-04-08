import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logger import get_logger
from src.preprocessor import FEATURE_COLUMNS

logger = get_logger("explainer")

# Lazy import shap to avoid numba crash at module load time
_shap = None

def _get_shap():
    global _shap
    if _shap is None:
        import shap as _shap_module
        _shap = _shap_module
    return _shap


def get_shap_explainer(xgb_model):
    shap = _get_shap()
    explainer = shap.TreeExplainer(xgb_model)
    logger.info("SHAP TreeExplainer created.")
    return explainer


def get_shap_values(explainer, X_scaled: np.ndarray):
    return explainer.shap_values(X_scaled)


def shap_summary_data(explainer, X_scaled: np.ndarray, max_samples: int = 500):
    idx = np.random.choice(len(X_scaled), size=min(max_samples, len(X_scaled)), replace=False)
    X_sample = X_scaled[idx]
    shap_vals = explainer.shap_values(X_sample)
    return shap_vals, X_sample, FEATURE_COLUMNS


def shap_single_customer(explainer, x_row: np.ndarray):
    x_2d = x_row.reshape(1, -1)
    sv = explainer.shap_values(x_2d)[0]
    df_shap = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "shap_value": sv,
        "abs_impact": np.abs(sv),
    }).sort_values("abs_impact", ascending=False)
    return df_shap