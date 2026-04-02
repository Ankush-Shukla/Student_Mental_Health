"""
core/inference.py
-----------------
Single-student prediction using the trained Random Forest model.

Two bugs fixed vs previous version:

  Bug 1 — LabelEncoder re-fit on a single row
  --------------------------------------------
  During training, build_model_matrix() calls LabelEncoder.fit_transform()
  on the full dataset for each bin column. For example, AcadPressure_Bin
  has classes ['AcadPressure_High', 'AcadPressure_Low', 'AcadPressure_Moderate']
  and alphabetical encoding:
      High -> 0,  Low -> 1,  Moderate -> 2

  The old inference code called fit_transform() on a single-row DataFrame,
  which always produces [0] regardless of the actual value — every categorical
  bin was silently encoded as 0 on every prediction.

  Fix: encoders are fitted once on the full training set during pipeline.py
  and saved to bin_encoders.pkl. We load and apply them here.

  Bug 2 — Rule features all zeroed out
  -------------------------------------
  The enriched feature matrix has 19 binary Rule_XXX columns that encode
  whether a student satisfies Apriori rule antecedents. The old code aligned
  columns against the template but set every missing column to 0, including
  all rule features (which require transaction-item matching, not just
  column alignment). This systematically pulled all predictions toward the
  population baseline (~0.58).

  Fix: we rebuild the student's transaction item set and check each rule
  antecedent explicitly using the saved depression_rules.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

_DJANGO_APP_DIR = Path(__file__).resolve().parent
_DJANGO_ROOT    = _DJANGO_APP_DIR.parent
_PROJECT_ROOT   = _DJANGO_ROOT.parent
_SRC_DIR        = _PROJECT_ROOT / "src"
_OUTPUTS_DIR    = _PROJECT_ROOT / "outputs"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from preprocessing import engineer_features


@lru_cache(maxsize=1)
def _load_artefacts() -> tuple:
    required = {
        "model":     _OUTPUTS_DIR / "rf.pkl",
        "template":  _OUTPUTS_DIR / "feature_template.csv",
        "encoders":  _OUTPUTS_DIR / "bin_encoders.pkl",
        "dep_rules": _OUTPUTS_DIR / "depression_rules.csv",
    }
    for name, path in required.items():
        if not path.exists():
            raise RuntimeError(
                f"Artefact '{name}' not found: {path}\n"
                "Run the pipeline first: python pipeline.py "
                "--data data/raw/train.csv --output outputs/"
            )
    model     = joblib.load(required["model"])
    template  = pd.read_csv(required["template"])
    encoders  = joblib.load(required["encoders"])
    dep_rules = pd.read_csv(required["dep_rules"])
    return model, template, encoders, dep_rules


_BIN_COLS = [
    "Age_Bin", "CGPA_Bin", "AcadPressure_Bin", "WorkPressure_Bin",
    "StudyHrs_Bin", "StudySat_Bin", "JobSat_Bin", "Sleep_Cat", "FinStress_Bin",
]

_BASE_FEATURES = [
    "Age", "CGPA", "Academic Pressure",
    "Study Satisfaction", "Work/Study Hours",
    "Suicidal_Thoughts", "Family_History",
]


def _clean_single(student_dict: dict) -> pd.DataFrame:
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
    df["Depression"] = 0
    return df


def _build_feature_row(df_eng: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    feature_cols = list(_BASE_FEATURES)
    row          = {}
    for col in _BASE_FEATURES:
        row[col] = float(df_eng[col].iloc[0]) if col in df_eng.columns else 0.0
    for col in _BIN_COLS:
        le      = encoders[col]
        val     = str(df_eng[col].iloc[0]) if col in df_eng.columns else "nan"
        enc_col = col + "_Enc"
        feature_cols.append(enc_col)
        row[enc_col] = int(le.transform([val])[0]) if val in le.classes_ else 0
    feature_cols.append("Gender_Male")
    row["Gender_Male"] = int(
        df_eng["Gender"].iloc[0].lower() == "male" if "Gender" in df_eng.columns else 0
    )
    return pd.DataFrame([row])[feature_cols]


def _build_item_set(df_eng: pd.DataFrame) -> set:
    items = set()
    r = df_eng.iloc[0]
    if "Gender" in df_eng.columns:
        items.add(f"Gender_{r['Gender']}")
    for col in _BIN_COLS:
        if col in df_eng.columns:
            items.add(str(r[col]))
    items.add("Suicidal_Yes" if r.get("Suicidal_Thoughts", 0) else "Suicidal_No")
    items.add("FamilyHistory_Yes" if r.get("Family_History", 0) else "FamilyHistory_No")
    return items


def _compute_rule_features(items: set, dep_rules: pd.DataFrame) -> dict:
    result = {}
    for i, rule_row in dep_rules.iterrows():
        ant_items = {a.strip() for a in rule_row["antecedents"].split(",")}
        result[f"Rule_{i:03d}"] = int(ant_items.issubset(items))
    return result


def predict_student(student_dict: dict) -> dict:
    """
    Run the depression risk model on a single student record.

    Parameters
    ----------
    student_dict : dict  — raw survey field values (CSV column names as keys)

    Returns
    -------
    dict:
        risk_score : float  — model probability (0-1)
        prediction : int    — 0 or 1
        risk_level : str    — "Low", "Moderate", or "High"
    """
    model, template, encoders, dep_rules = _load_artefacts()

    df         = _clean_single(student_dict)
    df_eng     = engineer_features(df)
    X          = _build_feature_row(df_eng, encoders)
    items      = _build_item_set(df_eng)
    rule_feats = _compute_rule_features(items, dep_rules)

    X_aligned = pd.DataFrame(0, index=[0], columns=template.columns)
    for col in X.columns:
        if col in template.columns:
            X_aligned[col] = X[col].values
    for col, val in rule_feats.items():
        if col in template.columns:
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

    return {
        "risk_score": round(prob, 4),
        "prediction": pred,
        "risk_level": level,
    }