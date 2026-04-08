import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


def ensure_models_exist():
    """Train models if they don't exist — runs on Streamlit Cloud startup."""
    models_dir = os.path.join(BASE_DIR, "models")
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    required = [
        os.path.join(models_dir, "xgb_model.pkl"),
        os.path.join(models_dir, "scaler.pkl"),
        os.path.join(data_dir, "processed_data.csv"),
    ]

    if not all(os.path.exists(f) for f in required):
        import streamlit as st
        with st.spinner("🔧 First-time setup: Training models (takes ~2 min)..."):
            from src.trainer import run_training
            import shutil
            run_training()
            # Create alias files
            aliases = [
                ("random_forest_model.pkl", "rf_model.pkl"),
                ("xgboost_model.pkl", "xgb_model.pkl"),
            ]
            for src_name, dst_name in aliases:
                src = os.path.join(models_dir, src_name)
                dst = os.path.join(models_dir, dst_name)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy(src, dst)