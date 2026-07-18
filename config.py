"""Configuration constants for the AgriDrone application."""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
#Avinash
# NDVI thresholds per crop
NDVI_THRESHOLDS = {
    "Wheat": {"healthy_min": 0.6, "early_min": 0.35, "severe_min": 0.1},
    "Cotton": {"healthy_min": 0.65, "early_min": 0.40, "severe_min": 0.15},
    "Rice": {"healthy_min": 0.55, "early_min": 0.30, "severe_min": 0.05},
    "Sugarcane": {"healthy_min": 0.62, "early_min": 0.38, "severe_min": 0.12},
}

# Disease class labels
DISEASE_CLASSES = {
    0: {"label": "Healthy", "color": "#2ecc71", "hex": "#2ecc71"},
    1: {"label": "Early Disease", "color": "#f1c40f", "hex": "#f1c40f"},
    2: {"label": "Severe Disease", "color": "#e74c3c", "hex": "#e74c3c"},
}

# Color scale for Plotly heatmap
HEATMAP_COLORS = ["#2ecc71", "#f1c40f", "#e74c3c"]

# Grid dimensions
GRID_ROWS = 25
GRID_COLS = 25

# Disease spread parameters
SPREAD_PROB_EARLY = 0.30
SPREAD_PROB_SEVERE = 0.60

# Gemini configuration
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_TEMPERATURE = 0.5
OPENAI_MODEL = "gpt-3.5-turbo"
# Feature names for model training
FEATURE_NAMES = [
    "ndvi",
    "red_intensity", 
    "green_intensity",
    "texture_variance",
    "moisture_index"
]

# Default field configurations
DEFAULT_FIELDS = {
    "flat_farm.json": "Open field with minimal disease",
    "pond_farm.json": "Field with water obstacles",
    "dense_field.json": "High-density disease stress test",
}