"""
inference.py
------------
Single-student prediction pipeline (Django-ready)
"""

import joblib
import pandas as pd
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_PATH = os.path.join(BASE_DIR, "src")

sys.path.append(SRC_PATH)
from preprocessing import clean, build_model_matrix


# ---------------------------------------------------------------------------
# Load artefacts (load once in Django, not per request)
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "..", "outputs", "rf.pkl")
TEMPLATE_PATH = os.path.join(BASE_DIR, "..", "outputs", "feature_template.csv")
MODEL_PATH = os.path.abspath(MODEL_PATH)
TEMPLATE_PATH = os.path.abspath(TEMPLATE_PATH)
model = joblib.load(MODEL_PATH)
feature_template = pd.read_csv(TEMPLATE_PATH)


# ---------------------------------------------------------------------------
# Core Inference Function
# ---------------------------------------------------------------------------

def predict_student(student_dict: dict) -> dict:
    """
    Input:
        student_dict → raw survey response (same structure as CSV row)

    Output:
        {
            "risk_score": float,
            "prediction": int
        }
    """

    # Step 1 — Convert to DataFrame
    df = pd.DataFrame([student_dict])

    # Step 2 — Clean
    df = clean(df)
    if "Depression" not in df.columns:
        df["Depression"] = 0  # dummy placeholder

    # Step 3 — Feature matrix
    X, _ = build_model_matrix(df)

    # ------------------------------------------------------------------
    # CRITICAL: Align with training schema
    # ------------------------------------------------------------------

    X_aligned = pd.DataFrame(columns=feature_template.columns)

    for col in feature_template.columns:
        if col in X.columns:
            X_aligned[col] = X[col]
        else:
            X_aligned[col] = 0  # missing features default to 0

    X_aligned = X_aligned.fillna(0)

    # Step 4 — Predict
    prob = model.predict_proba(X_aligned)[0][1]
    pred = int(prob >= 0.5)

    return {
        "risk_score": float(prob),
        "prediction": pred
    }