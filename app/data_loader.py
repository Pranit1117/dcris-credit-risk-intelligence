import streamlit as st
import pandas as pd
import os
import sys

# Go up two levels: app/ -> dcris/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, BASE_DIR)
from src.predictor import load_all_models

PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed_data.csv")


@st.cache_resource(show_spinner="Loading DCRIS models...")
def get_models():
    return load_all_models()


@st.cache_data(show_spinner="Loading dataset...")
def get_data():
    df = pd.read_csv(PROCESSED_PATH)
    return df
