"""
preprocessing.py
----------------
Loads raw survey data, cleans string artefacts, engineers categorical
bins, and produces both a model-ready feature matrix and a boolean
transaction DataFrame for Apriori mining.
"""

import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_quotes(series: pd.Series) -> pd.Series:
    """Remove leading/trailing single-quotes and whitespace that appear in
    some fields (e.g. "'5-6 hours'" -> "5-6 hours")."""
    return series.str.replace(r"^'+|'+$", "", regex=True).str.strip()


def _bin_age(series: pd.Series) -> pd.Series:
    bins   = [0, 21, 25, 30, 100]
    labels = ["Age_<=21", "Age_22-25", "Age_26-30", "Age_>30"]
    return pd.cut(series, bins=bins, labels=labels, right=True)


def _bin_cgpa(series: pd.Series) -> pd.Series:
    bins   = [0, 5.0, 7.5, 10.0]
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
    """Read the raw CSV and return an unmodified DataFrame."""
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw survey DataFrame in-place (copy returned):
      - Strip quote artefacts from string columns
      - Normalize known categorical values to title-case
      - Replace sentinel strings ('Others', blank) with NaN where appropriate
      - Drop duplicated id rows
    """
    df = df.copy()

    # Strip quote artefacts
    str_cols = df.select_dtypes(include=["object", "str"]).columns
    for col in str_cols:
        df[col] = _strip_quotes(df[col].astype(str))

    # Normalize
    df["Gender"]  = df["Gender"].str.title()
    df["Dietary Habits"] = df["Dietary Habits"].str.title()
    df["Sleep Duration"] = df["Sleep Duration"].str.lower().str.strip()

    # Suicidal thoughts -> binary
    df["Suicidal_Thoughts"] = (
        df["Have you ever had suicidal thoughts ?"]
        .str.strip().str.lower().eq("yes").astype(int)
    )

    # Family history -> binary
    df["Family_History"] = (
        df["Family History of Mental Illness"]
        .str.strip().str.lower().eq("yes").astype(int)
    )

    # Deduplicate on id
    df = df.drop_duplicates(subset="id")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive binned / encoded columns used for both modelling and mining.
    Returns a new DataFrame with appended columns; original columns retained.
    """
    df = df.copy()

    df["Age_Bin"]          = _bin_age(df["Age"])
    df["CGPA_Bin"]         = _bin_cgpa(df["CGPA"])
    df["AcadPressure_Bin"] = _bin_pressure(df["Academic Pressure"], "AcadPressure")
    df["WorkPressure_Bin"] = _bin_pressure(df["Work Pressure"],     "WorkPressure")
    df["StudyHrs_Bin"]     = _bin_study_hours(df["Work/Study Hours"])
    df["StudySat_Bin"]     = _bin_satisfaction(df["Study Satisfaction"], "StudySat")
    df["JobSat_Bin"]       = _bin_satisfaction(df["Job Satisfaction"],   "JobSat")

    # Sleep -> ordinal category
    sleep_map = {
        "less than 5 hours": "Sleep_<5h",
        "5-6 hours":         "Sleep_5-6h",
        "7-8 hours":         "Sleep_7-8h",
        "more than 8 hours": "Sleep_>8h",
        "others":            "Sleep_Other",
    }
    df["Sleep_Cat"] = df["Sleep Duration"].map(sleep_map).fillna("Sleep_Other")

    # Financial stress (mixed: some numeric, some string)
    fs = pd.to_numeric(df["Financial Stress"], errors="coerce")
    df["FinStress_Bin"] = pd.cut(
        fs, bins=[-1, 2, 3, 5],
        labels=["FinStress_Low", "FinStress_Moderate", "FinStress_High"]
    )

    return df


def build_model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Construct a numeric feature matrix (X) and label vector (y) for
    supervised learning.  Only deterministic, non-leaking features are used.

    Returns
    -------
    X : pd.DataFrame  — numeric feature matrix
    y : pd.Series     — binary depression label (0 / 1)
    """
    df = engineer_features(df)

    feature_cols = [
        "Age", "CGPA", "Academic Pressure", "Work Pressure",
        "Study Satisfaction", "Job Satisfaction", "Work/Study Hours",
        "Suicidal_Thoughts", "Family_History",
    ]

    # Encode binned categoricals numerically
    bin_cols = [
        "Age_Bin", "CGPA_Bin", "AcadPressure_Bin", "WorkPressure_Bin",
        "StudyHrs_Bin", "StudySat_Bin", "JobSat_Bin", "Sleep_Cat",
        "FinStress_Bin",
    ]
    le = LabelEncoder()
    for col in bin_cols:
        df[col + "_Enc"] = le.fit_transform(df[col].astype(str))
        feature_cols.append(col + "_Enc")

    # Gender one-hot
    df["Gender_Male"] = (df["Gender"].str.lower() == "male").astype(int)
    feature_cols.append("Gender_Male")

    X = df[feature_cols].copy()
    y = df["Depression"].astype(int)

    return X, y


def build_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform each student record into a boolean (one-hot) item DataFrame
    suitable for Apriori mining.

    Each column represents one item (e.g. "CGPA_High", "Sleep_5-6h").
    A True value means the student possesses that item.
    """
    df = engineer_features(df)

    rows = []
    for _, row in df.iterrows():
        items: dict[str, bool] = {}

        items[f"Gender_{row['Gender']}"]          = True
        items[str(row["Age_Bin"])]                = True
        items[str(row["CGPA_Bin"])]               = True
        items[str(row["AcadPressure_Bin"])]       = True
        items[str(row["WorkPressure_Bin"])]       = True
        items[str(row["StudyHrs_Bin"])]           = True
        items[str(row["StudySat_Bin"])]           = True
        items[str(row["JobSat_Bin"])]             = True
        items[str(row["Sleep_Cat"])]              = True
        items[str(row["FinStress_Bin"])]          = True
        items["Suicidal_Yes" if row["Suicidal_Thoughts"] else "Suicidal_No"] = True
        items["FamilyHistory_Yes" if row["Family_History"] else "FamilyHistory_No"] = True
        items["Depression_Yes" if row["Depression"] == 1 else "Depression_No"] = True

        rows.append(items)

    tx_df = pd.DataFrame(rows).fillna(False).astype(bool)
    return tx_df
