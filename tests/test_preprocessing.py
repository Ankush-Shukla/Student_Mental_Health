"""
tests/test_preprocessing.py
----------------------------
Unit tests for the preprocessing module.
Run with: python -m unittest discover tests/ -v
"""

import sys
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from preprocessing import (
    clean,
    engineer_features,
    build_model_matrix,
    build_transactions,
)


def _make_sample() -> pd.DataFrame:
    return pd.DataFrame({
        "id":       [1, 2, 3, 4],
        "Gender":   ["Male", "Female", "male", "'Female'"],
        "Age":      [20.0, 23.0, 19.0, 30.0],
        "City":     ["Delhi", "Mumbai", "Pune", "Chennai"],
        "Profession": ["Student", "Student", "Student", "Teacher"],
        "Academic Pressure": [4.0, 2.0, 3.0, 1.0],
        "Work Pressure":     [0.0, 0.0, 0.0, 3.0],
        "CGPA":              [8.5, 6.2, 9.1, 7.0],
        "Study Satisfaction":[3.0, 5.0, 2.0, 4.0],
        "Job Satisfaction":  [0.0, 0.0, 0.0, 3.0],
        "Sleep Duration":    ["'5-6 hours'", "'7-8 hours'", "'Less than 5 hours'", "'7-8 hours'"],
        "Dietary Habits":    ["Healthy", "Moderate", "Unhealthy", "Healthy"],
        "Degree":            ["BSc", "B.Tech", "BA", "MBA"],
        "Have you ever had suicidal thoughts ?": ["Yes", "No", "No", "No"],
        "Work/Study Hours":  [8.0, 4.0, 10.0, 6.0],
        "Financial Stress":  ["3.0", "1.0", "5.0", "2.0"],
        "Family History of Mental Illness": ["Yes", "No", "Yes", "No"],
        "Depression": [1, 0, 1, 0],
    })


class TestClean(unittest.TestCase):

    def setUp(self):
        self.df = _make_sample()

    def test_strips_quotes_from_gender(self):
        df = clean(self.df)
        self.assertNotIn("'", df["Gender"].iloc[3])

    def test_suicidal_thoughts_yes_is_1(self):
        df = clean(self.df)
        self.assertEqual(df["Suicidal_Thoughts"].iloc[0], 1)

    def test_suicidal_thoughts_no_is_0(self):
        df = clean(self.df)
        self.assertEqual(df["Suicidal_Thoughts"].iloc[1], 0)

    def test_family_history_yes_is_1(self):
        df = clean(self.df)
        self.assertEqual(df["Family_History"].iloc[0], 1)

    def test_family_history_no_is_0(self):
        df = clean(self.df)
        self.assertEqual(df["Family_History"].iloc[1], 0)

    def test_no_duplicate_ids_after_clean(self):
        dup = pd.concat([self.df, self.df.iloc[:1]], ignore_index=True)
        df  = clean(dup)
        self.assertTrue(df["id"].is_unique)

    def test_row_count_preserved_without_duplicates(self):
        df = clean(self.df)
        self.assertEqual(len(df), len(self.df))


class TestEngineerFeatures(unittest.TestCase):

    def setUp(self):
        self.df = engineer_features(clean(_make_sample()))

    def test_cgpa_bin_column_exists(self):
        self.assertIn("CGPA_Bin", self.df.columns)

    def test_age_bin_column_exists(self):
        self.assertIn("Age_Bin", self.df.columns)

    def test_sleep_cat_column_exists(self):
        self.assertIn("Sleep_Cat", self.df.columns)

    def test_finstress_bin_column_exists(self):
        self.assertIn("FinStress_Bin", self.df.columns)

    def test_sleep_cat_values_are_valid(self):
        valid = {"Sleep_<5h", "Sleep_5-6h", "Sleep_7-8h", "Sleep_>8h", "Sleep_Other"}
        actual = set(self.df["Sleep_Cat"].unique())
        self.assertTrue(actual.issubset(valid), f"Unexpected values: {actual - valid}")

    def test_academic_pressure_bin_three_levels(self):
        valid = {"AcadPressure_Low", "AcadPressure_Moderate", "AcadPressure_High"}
        actual = set(self.df["AcadPressure_Bin"].dropna().unique())
        self.assertTrue(actual.issubset(valid))


class TestBuildModelMatrix(unittest.TestCase):

    def setUp(self):
        self.X, self.y = build_model_matrix(clean(_make_sample()))

    def test_row_count_matches_input(self):
        self.assertEqual(self.X.shape[0], 4)

    def test_label_length_matches_features(self):
        self.assertEqual(len(self.y), self.X.shape[0])

    def test_no_nans_in_feature_matrix(self):
        self.assertFalse(self.X.isnull().any().any())

    def test_labels_are_binary(self):
        self.assertTrue(set(self.y.unique()).issubset({0, 1}))

    def test_feature_matrix_is_numeric(self):
        for col in self.X.columns:
            self.assertTrue(
                np.issubdtype(self.X[col].dtype, np.number),
                f"Column {col} is not numeric"
            )


class TestBuildTransactions(unittest.TestCase):

    def setUp(self):
        self.tx = build_transactions(clean(_make_sample()))

    def test_all_columns_are_boolean(self):
        for col in self.tx.columns:
            self.assertEqual(self.tx[col].dtype, bool, f"{col} is not bool")

    def test_depression_yes_column_present(self):
        self.assertIn("Depression_Yes", self.tx.columns)

    def test_depression_no_column_present(self):
        self.assertIn("Depression_No", self.tx.columns)

    def test_row_count_matches_input(self):
        self.assertEqual(len(self.tx), 4)

    def test_depression_yes_and_no_mutually_exclusive(self):
        both = (self.tx["Depression_Yes"] & self.tx["Depression_No"]).any()
        self.assertFalse(both, "A record cannot be both Depression_Yes and Depression_No")

    def test_gender_items_present(self):
        gender_cols = [c for c in self.tx.columns if c.startswith("Gender_")]
        self.assertGreater(len(gender_cols), 0)


if __name__ == "__main__":
    unittest.main()
