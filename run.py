#!/usr/bin/env python3
"""
DCRIS — Dynamic Credit Risk Intelligence System
Run this script to train all models and launch the Streamlit dashboard.
Usage:
    python run.py          # train + launch
    python run.py --train  # train only
    python run.py --app    # launch app only (models must exist)
"""

import sys
import os
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.logger import get_logger

logger = get_logger("run")


def check_models_exist() -> bool:
    required = [
        "models/logistic_model.pkl",
        "models/rf_model.pkl",
        "models/xgb_model.pkl",
        "models/survival_model.pkl",
        "models/kmeans_model.pkl",
        "models/scaler.pkl",
    ]
    base = os.path.dirname(os.path.abspath(__file__))
    missing = [r for r in required if not os.path.exists(os.path.join(base, r))]
    if missing:
        logger.warning(f"Missing model files: {missing}")
        return False
    return True


def train():
    logger.info("Starting model training pipeline...")
    from src.trainer import run_training
    run_training()

    # Create alias files if needed
    base = os.path.dirname(os.path.abspath(__file__))
    aliases = [
        ("models/random_forest_model.pkl", "models/rf_model.pkl"),
        ("models/xgboost_model.pkl", "models/xgb_model.pkl"),
    ]
    import shutil
    for src_rel, dst_rel in aliases:
        src = os.path.join(base, src_rel)
        dst = os.path.join(base, dst_rel)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            logger.info(f"Created alias: {dst_rel}")

    logger.info("Training complete.")


def launch_app():
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "main.py")
    logger.info(f"Launching Streamlit app: {app_path}")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.port", "8501",
        "--server.headless", "true",
        "--theme.base", "dark",
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCRIS Runner")
    parser.add_argument("--train", action="store_true", help="Train models only")
    parser.add_argument("--app", action="store_true", help="Launch app only")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.app:
        if not check_models_exist():
            logger.info("Models not found. Training first...")
            train()
        launch_app()
    else:
        # Default: train + launch
        if not check_models_exist():
            train()
        else:
            logger.info("Models already exist. Skipping training. Use --train to retrain.")
        launch_app()
