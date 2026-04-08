import streamlit as st
import pandas as pd
import os
import sys
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_PATH = os.path.join(DATA_DIR, "processed_data.csv")


def _ensure_trained():
    """Run training pipeline if models or data are missing."""
    required = [
        os.path.join(MODELS_DIR, "scaler.pkl"),
        os.path.join(MODELS_DIR, "xgb_model.pkl"),
        PROCESSED_PATH,
    ]
    if all(os.path.exists(f) for f in required):
        return  # already trained, skip

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    with st.spinner("⚙️ First-time setup: Training models (~2 min)..."):
        from src.trainer import run_training
        run_training()

        # Create alias pkl files predictor expects
        aliases = [
            ("random_forest_model.pkl", "rf_model.pkl"),
            ("xgboost_model.pkl", "xgb_model.pkl"),
        ]
        for src_name, dst_name in aliases:
            src = os.path.join(MODELS_DIR, src_name)
            dst = os.path.join(MODELS_DIR, dst_name)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy(src, dst)


from src.predictor import load_all_models


@st.cache_resource(show_spinner="Loading DCRIS models...")
def get_models():
    _ensure_trained()
    return load_all_models()


@st.cache_data(show_spinner="Loading dataset...")
def get_data():
    _ensure_trained()
    return pd.read_csv(PROCESSED_PATH)