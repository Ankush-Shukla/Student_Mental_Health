"""
preprocessing.py
----------------
Loads raw survey data, cleans string artefacts, engineers categorical
bins, and produces both a model-ready feature matrix and a boolean
transaction DataFrame for Apriori mining.

Fix (v2): build_model_matrix now creates one LabelEncoder per bin column
and returns them alongside X and y so the pipeline can persist them
correctly.

Fix (v3): build_transactions now skips NaN/None bin values instead of
adding the literal string "nan" to the transaction set. Previously Apriori
mined rules involving the item "nan", contaminating all association rules.
"""

import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_quotes(series: pd.Series) -> pd.Series:
    return series.str.replace(r"^'+|'+$", "", regex=True).str.strip()


def _bin_age(series: pd.Series) -> pd.Series:
    bins   = [0, 21, 25, 30, 100]
    labels = ["Age_<=21", "Age_22-25", "Age_26-30", "Age_>30"]
    return pd.cut(series, bins=bins, labels=labels, right=True)


def _bin_cgpa(series: pd.Series) -> pd.Series:
    bins   = [-np.inf, 6.5, 8.5, np.inf]
    labels = ["CGPA_Low", "CGPA_Mid", "CGPA_High"]
    return pd.cut(series, bins=bins, labels=labels, right=True)


def _bin_pressure(series: pd.Series, prefix: str) -> pd.Series:
    bins   = [-1, 2, 3, 5]
    labels = [f"{prefix}_Low", f"{prefix}_Moderate", f"{prefix}_High"]
    return pd.cut(series, bins=bins, labels=labels, right=True)


def _bin_study_hours(series: pd.Series) -> pd.Series:
    bins   = [-1, 4, 7, 24]
    labels = ["StudyHrs_Low", "StudyHrs_Moderate", "StudyHrs_High"]
    return pd.cut(series, bins=bins, labels=labels, right=True)


def _bin_satisfaction(series: pd.Series, prefix: str) -> pd.Series:
    bins   = [0, 2, 3, 5]
    labels = [f"{prefix}_Low", f"{prefix}_Moderate", f"{prefix}_High"]
    return pd.cut(series, bins=bins, labels=labels, right=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    str_cols = df.select_dtypes(include=["object", "str"]).columns
    for col in str_cols:
        df[col] = _strip_quotes(df[col].astype(str))

    df["Gender"]         = df["Gender"].str.title()
    df["Dietary Habits"] = df["Dietary Habits"].str.title()
    df["Sleep Duration"] = df["Sleep Duration"].str.lower().str.strip()

    df["Suicidal_Thoughts"] = (
        df["Have you ever had suicidal thoughts ?"]
        .str.strip().str.lower().eq("yes").astype(int)
    )

    df["Family_History"] = (
        df["Family History of Mental Illness"]
        .str.strip().str.lower().eq("yes").astype(int)
    )

    if "id" in df.columns:
        df = df.drop_duplicates(subset="id")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Age_Bin"]          = _bin_age(df["Age"])
    df["CGPA_Bin"]         = _bin_cgpa(df["CGPA"])
    df["AcadPressure_Bin"] = _bin_pressure(df["Academic Pressure"], "AcadPressure")
    df["WorkPressure_Bin"] = _bin_pressure(df["Work Pressure"],     "WorkPressure")
    df["StudyHrs_Bin"]     = _bin_study_hours(df["Work/Study Hours"])
    df["StudySat_Bin"]     = _bin_satisfaction(df["Study Satisfaction"], "StudySat")
    df["JobSat_Bin"]       = _bin_satisfaction(df["Job Satisfaction"],   "JobSat")

    sleep_map = {
        "less than 5 hours": "Sleep_<5h",
        "5-6 hours":         "Sleep_5-6h",
        "7-8 hours":         "Sleep_7-8h",
        "more than 8 hours": "Sleep_>8h",
        "others":            "Sleep_Other",
    }
    df["Sleep_Cat"] = df["Sleep Duration"].map(sleep_map).fillna("Sleep_Other")

    fs = pd.to_numeric(df["Financial Stress"].replace("?", np.nan), errors="coerce")
    df["FinStress_Bin"] = pd.cut(
        fs, bins=[-1, 2, 3, 5],
        labels=["FinStress_Low", "FinStress_Moderate", "FinStress_High"]
    )

    return df


# Canonical list of bin columns — shared by pipeline and inference
BIN_COLS: list[str] = [
    "Age_Bin", "CGPA_Bin", "AcadPressure_Bin", "WorkPressure_Bin",
    "StudyHrs_Bin", "StudySat_Bin", "JobSat_Bin", "Sleep_Cat", "FinStress_Bin",
]


def build_model_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, dict[str, LabelEncoder]]:
    df = engineer_features(df)

    base_features = [
        "Age", "CGPA", "Academic Pressure",
        "Study Satisfaction", "Work/Study Hours",
        "Suicidal_Thoughts", "Family_History",
    ]

    encoders: dict[str, LabelEncoder] = {}
    enc_features: list[str] = []

    for col in BIN_COLS:
        le = LabelEncoder()
        df[col] = df[col].astype(object).fillna("Unknown").astype(str)
        df[col + "_Enc"] = le.fit_transform(df[col])
        encoders[col] = le
        enc_features.append(col + "_Enc")

    df["Gender_Male"] = (df["Gender"].str.lower() == "male").astype(int)

    feature_cols = base_features + enc_features + ["Gender_Male"]
    X = df[feature_cols].copy()
    y = df["Depression"].astype(int)

    return X, y, encoders


def build_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = engineer_features(df)

    rows = []
    for _, row in df.iterrows():
        items: dict[str, bool] = {}

        # Gender
        items[f"Gender_{row['Gender']}"] = True

        # FIX: skip NaN bin values — previously str(NaN) = "nan" was added
        # to the transaction set, causing Apriori to mine spurious rules
        # involving the literal item "nan".
        for col in BIN_COLS:
            val = row[col]
            if pd.notna(val) and str(val) != "nan":
                items[str(val)] = True

        items["Suicidal_Yes" if row["Suicidal_Thoughts"] else "Suicidal_No"] = True
        items["FamilyHistory_Yes" if row["Family_History"] else "FamilyHistory_No"] = True
        items["Depression_Yes" if row["Depression"] == 1 else "Depression_No"] = True

        rows.append(items)

    tx_df = pd.DataFrame(rows).fillna(False).astype(bool)
    return tx_df