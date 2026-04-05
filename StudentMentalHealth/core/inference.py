"""
core/inference.py
-----------------
Single-student depression risk prediction using the trained Random Forest.

Bugs fixed (v2)
---------------
Bug 1 — LabelEncoder re-fit on a single row
    The old build_model_matrix() loop reused one LabelEncoder instance.
    Only the last bin column's classes were retained; earlier columns were
    silently encoded as 0.  Fixed: pipeline now saves one encoder per
    column in bin_encoders.pkl; we load and apply them here.

Bug 2 — Rule features all zeroed out
    Rule_XXX columns require item-set matching, not column alignment.
    The old code zero-filled all missing columns including rule features,
    pulling every prediction toward the population baseline (~0.58).
    Fixed: we rebuild the student's transaction item set and check each
    rule antecedent explicitly against depression_rules.csv.

Bug 3 — Missing bin_encoders.pkl
    The old pipeline never saved this file; inference raised RuntimeError.
    Fixed: pipeline.py now explicitly saves it.
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
_DJANGO_APP_DIR = Path(__file__).resolve().parent          # core/
_DJANGO_ROOT    = _DJANGO_APP_DIR.parent                   # StudentMentalHealth/
_PROJECT_ROOT   = _DJANGO_ROOT.parent                      # repo root
_SRC_DIR        = _PROJECT_ROOT / "src"
_OUTPUTS_DIR    = _DJANGO_ROOT / "outputs"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from preprocessing import engineer_features, BIN_COLS      # noqa: E402


# ---------------------------------------------------------------------------
# Artefact loading (cached for the lifetime of the process)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_artefacts() -> tuple:
    required = {
        "model":     _OUTPUTS_DIR / "rf.pkl",
        "template":  _OUTPUTS_DIR / "feature_template.csv",
        "encoders":  _OUTPUTS_DIR / "bin_encoders.pkl",
        "dep_rules": _OUTPUTS_DIR / "depression_rules.csv",
    }

    HF_REPO = os.environ.get("HF_REPO", "YOUR_USERNAME/student-mental-health-model")
    HF_FILES = ["rf.pkl", "bin_encoders.pkl"]  # only large files

    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    for filename in HF_FILES:
        dest = _OUTPUTS_DIR / filename
        if not dest.exists():
            try:
                from huggingface_hub import hf_hub_download
                logger.info(f"Downloading {filename} from Hugging Face...")
                cached = hf_hub_download(
                    repo_id=HF_REPO,
                    filename=filename,
                    repo_type="model",
                )
                import shutil
                shutil.copy(cached, dest)
                logger.info(f"{filename} ready.")
            except Exception as e:
                raise RuntimeError(f"Failed to download {filename}: {e}")

    for name, path in required.items():
        if not path.exists():
            raise RuntimeError(
                f"Required artefact '{name}' not found at: {path}"
            )

    model     = joblib.load(required["model"])
    template  = pd.read_csv(required["template"])
    encoders  = joblib.load(required["encoders"])
    dep_rules = pd.read_csv(required["dep_rules"])
    return model, template, encoders, dep_rules
# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_BASE_FEATURES = [
    "Age", "CGPA", "Academic Pressure",
    "Study Satisfaction", "Work/Study Hours",
    "Suicidal_Thoughts", "Family_History",
]


def _clean_single(student_dict: dict) -> pd.DataFrame:
    """Convert a raw survey dict to a single-row DataFrame ready for
    engineer_features()."""
    df = pd.DataFrame([student_dict])

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.replace(r"^'+|'+$", "", regex=True).str.strip()

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].str.title()
    if "Dietary Habits" in df.columns:
        df["Dietary Habits"] = df["Dietary Habits"].str.title()
    if "Sleep Duration" in df.columns:
        df["Sleep Duration"] = df["Sleep Duration"].str.lower().str.strip()

    df["Suicidal_Thoughts"] = (
        df.get("Have you ever had suicidal thoughts ?", pd.Series(["No"]))
        .astype(str).str.strip().str.lower().eq("yes").astype(int)
    )
    df["Family_History"] = (
        df.get("Family History of Mental Illness", pd.Series(["No"]))
        .astype(str).str.strip().str.lower().eq("yes").astype(int)
    )

    # Depression column is required by engineer_features but not used for inference
    df["Depression"] = 0

    # Work Pressure and Job Satisfaction default to 0 for student-only respondents
    for col, default in [("Work Pressure", 0), ("Job Satisfaction", 0)]:
        if col not in df.columns:
            df[col] = default

    return df


def _build_feature_row(df_eng: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    row: dict = {}

    for col in _BASE_FEATURES:
        row[col] = float(df_eng[col].iloc[0]) if col in df_eng.columns else 0.0

    enc_features = []
    for col in BIN_COLS:
        enc_col = col + "_Enc"
        enc_features.append(enc_col)
        if col in encoders and col in df_eng.columns:
            le = encoders[col]
            val = str(df_eng[col].iloc[0])
            # FIX: match the sentinel used during training
            val = "Unknown" if val == "nan" else val
            row[enc_col] = int(le.transform([val])[0]) if val in le.classes_ else 0
        else:
            row[enc_col] = 0

    row["Gender_Male"] = int(
        df_eng["Gender"].iloc[0].lower() == "male"
        if "Gender" in df_eng.columns else 0
    )

    feature_cols = _BASE_FEATURES + enc_features + ["Gender_Male"]

    return pd.DataFrame([row])[feature_cols]


def _build_item_set(df_eng: pd.DataFrame) -> set[str]:
    items: set[str] = set()
    r = df_eng.iloc[0]

    if "Gender" in df_eng.columns:
        items.add(f"Gender_{r['Gender']}")

    for col in BIN_COLS:
        if col in df_eng.columns:
            val = str(r[col])
            if val != "nan":          # FIX: don't add nan to item set
                items.add(val)

    items.add("Suicidal_Yes" if r.get("Suicidal_Thoughts", 0) else "Suicidal_No")
    items.add("FamilyHistory_Yes" if r.get("Family_History", 0) else "FamilyHistory_No")

    return items
    
    


def _compute_rule_features(items: set[str], dep_rules: pd.DataFrame) -> dict[str, int]:
    """For each depression rule, check whether all antecedent items are present."""
    result: dict[str, int] = {}
    for i, rule_row in dep_rules.iterrows():
        ant_items = {a.strip() for a in rule_row["antecedents"].split(",")}
        result[f"Rule_{i:03d}"] = int(ant_items.issubset(items))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_student(student_dict: dict) -> dict:
    """
    Run the depression risk model on a single student record.

    Parameters
    ----------
    student_dict : dict — raw survey field values (CSV column names as keys)

    Returns
    -------
    dict:
        risk_score : float  — model probability (0–1)
        prediction : int    — 0 or 1
        risk_level : str    — "Low", "Moderate", or "High"
    """
    model, template, encoders, dep_rules = _load_artefacts()

    df = _clean_single(student_dict)
    df_eng = engineer_features(df)
    X = _build_feature_row(df_eng, encoders)
    items = _build_item_set(df_eng)
    rule_feats = _compute_rule_features(items, dep_rules)

    # Align to training feature template
    X_aligned = pd.DataFrame(0, index=[0], columns=template.columns)
    for col in X.columns:
        if col in X_aligned.columns:
            X_aligned[col] = X[col].values
    for col, val in rule_feats.items():
        if col in X_aligned.columns:
            X_aligned[col] = val

    X_aligned = X_aligned.fillna(0).astype(float)

    prob = float(model.predict_proba(X_aligned)[0][1])
    pred = int(prob >= 0.5)

    if prob < 0.40:
        level = "Low"
    elif prob < 0.70:
        level = "Moderate"
    else:
        level = "High"

    logger.debug(
        "predict_student | items=%s | rule_hits=%d | prob=%.4f",
        items, sum(rule_feats.values()), prob,
    )

    return {
        "risk_score": round(prob, 4),
        "prediction": pred,
        "risk_level": level,
    }