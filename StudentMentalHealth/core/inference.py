"""
core/inference.py
-----------------
Single-student prediction using the trained Random Forest model.

The model and feature template are loaded lazily on first call so that
Django startup does not crash when the artefact files are absent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path resolution — works regardless of where Django is started from
# ---------------------------------------------------------------------------

_DJANGO_APP_DIR = Path(__file__).resolve().parent        # .../StudentMentalHealth/core/
_DJANGO_ROOT    = _DJANGO_APP_DIR.parent                 # .../StudentMentalHealth/
_PROJECT_ROOT   = _DJANGO_ROOT.parent                    # project root (contains pipeline.py)
_SRC_DIR        = _PROJECT_ROOT / "src"
_OUTPUTS_DIR    = _PROJECT_ROOT / "outputs"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from preprocessing import clean, build_model_matrix


# ---------------------------------------------------------------------------
# Lazy artefact loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_artefacts() -> tuple:
    """
    Load model and feature template on first call and cache them.
    Raises RuntimeError with a clear message if files are missing.
    """
    model_path    = _OUTPUTS_DIR / "rf.pkl"
    template_path = _OUTPUTS_DIR / "feature_template.csv"

    if not model_path.exists():
        raise RuntimeError(
            f"Model file not found: {model_path}\n"
            "Run the pipeline first:  python pipeline.py --data data/raw/train.csv --output outputs/"
        )
    if not template_path.exists():
        raise RuntimeError(
            f"Feature template not found: {template_path}\n"
            "Run the pipeline first:  python pipeline.py --data data/raw/train.csv --output outputs/"
        )

    model    = joblib.load(model_path)
    template = pd.read_csv(template_path)
    return model, template


# ---------------------------------------------------------------------------
# Public inference function
# ---------------------------------------------------------------------------

def predict_student(student_dict: dict) -> dict:
    """
    Run the depression risk model on a single student record.

    Parameters
    ----------
    student_dict : dict
        Raw survey field values matching the CSV column names.

    Returns
    -------
    dict with keys:
        risk_score : float   — model probability (0–1)
        prediction : int     — 0 (low) or 1 (high)
        risk_level : str     — "Low", "Moderate", or "High"
    """
    model, feature_template = _load_artefacts()

    df = pd.DataFrame([student_dict])

    df = clean(df)
    if "Depression" not in df.columns:
        df["Depression"] = 0

    X, _ = build_model_matrix(df)

    # Align columns to the exact schema the model was trained on
    X_aligned = pd.DataFrame(0, index=[0], columns=feature_template.columns)
    for col in feature_template.columns:
        if col in X.columns:
            X_aligned[col] = X[col].values

    X_aligned = X_aligned.fillna(0).astype(float)

    prob = float(model.predict_proba(X_aligned)[0][1])
    pred = int(prob >= 0.5)

    if prob < 0.40:
        level = "Low"
    elif prob < 0.70:
        level = "Moderate"
    else:
        level = "High"

    return {
        "risk_score": round(prob, 4),
        "prediction": pred,
        "risk_level": level,
    }