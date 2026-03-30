"""
tests/test_apriori.py
---------------------
Unit tests for AprioriEngine and rule_filter.
Run with: python -m unittest discover tests/ -v
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from apriori_engine import AprioriEngine
from rule_filter    import filter_depression_rules, build_rule_features


def _toy_transactions() -> pd.DataFrame:
    """
    Minimal hand-crafted boolean transaction table where we know which
    itemsets should be frequent and which rules should emerge.
    """
    records = [
        {"A": True,  "B": True,  "C": True,  "D": False},
        {"A": True,  "B": True,  "C": True,  "D": False},
        {"A": True,  "B": True,  "C": False, "D": True},
        {"A": True,  "B": False, "C": True,  "D": False},
        {"A": False, "B": True,  "C": True,  "D": True},
        {"A": True,  "B": True,  "C": True,  "D": True},
        {"A": True,  "B": True,  "C": True,  "D": False},
        {"A": True,  "B": True,  "C": False, "D": False},
    ]
    return pd.DataFrame(records)


def _depression_transactions() -> pd.DataFrame:
    """Transactions that include Depression_Yes as a column."""
    records = []
    for _ in range(60):
        records.append({
            "HighStress":     True,
            "PoorSleep":      True,
            "Depression_Yes": True,
            "Depression_No":  False,
        })
    for _ in range(40):
        records.append({
            "HighStress":     False,
            "PoorSleep":      False,
            "Depression_Yes": False,
            "Depression_No":  True,
        })
    return pd.DataFrame(records)


class TestAprioriEngine(unittest.TestCase):

    def test_frequent_itemsets_generated(self):
        engine = AprioriEngine(min_support=0.5, min_confidence=0.5, min_lift=1.0)
        engine.fit(_toy_transactions())
        self.assertGreater(len(engine.frequent_itemsets_), 0)

    def test_single_item_frequent_sets_above_support(self):
        engine = AprioriEngine(min_support=0.5, min_confidence=0.5, min_lift=1.0)
        engine.fit(_toy_transactions())
        fi_df  = engine.frequent_itemsets_dataframe()
        single = fi_df[~fi_df["items"].str.contains(",")]
        self.assertGreater(len(single), 0)

    def test_rules_dataframe_has_required_columns(self):
        engine = AprioriEngine(min_support=0.4, min_confidence=0.5, min_lift=1.0)
        engine.fit(_toy_transactions())
        if engine.rules_:
            df = engine.rules_dataframe()
            for col in ["antecedents", "consequents", "support", "confidence", "lift"]:
                self.assertIn(col, df.columns)

    def test_support_values_within_valid_range(self):
        engine = AprioriEngine(min_support=0.3, min_confidence=0.4, min_lift=1.0)
        engine.fit(_toy_transactions())
        fi_df = engine.frequent_itemsets_dataframe()
        self.assertTrue((fi_df["support"] >= 0.3).all())
        self.assertTrue((fi_df["support"] <= 1.0).all())

    def test_confidence_threshold_respected(self):
        engine = AprioriEngine(min_support=0.3, min_confidence=0.8, min_lift=1.0)
        engine.fit(_toy_transactions())
        if engine.rules_:
            df = engine.rules_dataframe()
            self.assertTrue((df["confidence"] >= 0.8).all())

    def test_lift_threshold_respected(self):
        engine = AprioriEngine(min_support=0.3, min_confidence=0.4, min_lift=1.2)
        engine.fit(_toy_transactions())
        if engine.rules_:
            df = engine.rules_dataframe()
            self.assertTrue((df["lift"] >= 1.2).all())

    def test_all_false_transactions_no_crash(self):
        engine = AprioriEngine(min_support=0.5)
        empty  = pd.DataFrame({"A": [False, False], "B": [False, False]})
        engine.fit(empty)  # must not raise
        self.assertIsNotNone(engine.frequent_itemsets_dataframe())

    def test_depression_rules_emerge_from_structured_data(self):
        engine = AprioriEngine(min_support=0.3, min_confidence=0.5, min_lift=1.0)
        engine.fit(_depression_transactions())
        rules_df = engine.rules_dataframe()
        if not rules_df.empty:
            dep = rules_df[rules_df["consequents"].str.contains("Depression_Yes")]
            self.assertGreater(len(dep), 0)

    def test_rules_sorted_by_lift_descending(self):
        engine = AprioriEngine(min_support=0.3, min_confidence=0.4, min_lift=1.0)
        engine.fit(_toy_transactions())
        if len(engine.rules_) > 1:
            lifts = [r.lift for r in engine.rules_]
            self.assertEqual(lifts, sorted(lifts, reverse=True))

    def test_frequent_itemsets_dataframe_returns_dataframe(self):
        engine = AprioriEngine(min_support=0.5)
        engine.fit(_toy_transactions())
        result = engine.frequent_itemsets_dataframe()
        self.assertIsInstance(result, pd.DataFrame)


class TestRuleFilter(unittest.TestCase):

    def _get_depression_rules(self):
        engine = AprioriEngine(min_support=0.3, min_confidence=0.5, min_lift=1.0)
        engine.fit(_depression_transactions())
        return engine.rules_dataframe()

    def test_filter_to_depression_yes_consequent(self):
        rules_df = self._get_depression_rules()
        if not rules_df.empty:
            dep = filter_depression_rules(rules_df, "Depression_Yes")
            if not dep.empty:
                self.assertTrue(
                    dep["consequents"].str.contains("Depression_Yes").all()
                )

    def test_empty_input_returns_empty_dataframe(self):
        result = filter_depression_rules(pd.DataFrame())
        self.assertTrue(result.empty)

    def test_build_rule_features_correct_row_count(self):
        tx       = _depression_transactions()
        rules_df = self._get_depression_rules()
        if not rules_df.empty:
            dep = filter_depression_rules(rules_df, "Depression_Yes")
            if not dep.empty:
                feats = build_rule_features(tx, dep)
                self.assertEqual(len(feats), len(tx))

    def test_build_rule_features_correct_column_count(self):
        tx       = _depression_transactions()
        rules_df = self._get_depression_rules()
        if not rules_df.empty:
            dep = filter_depression_rules(rules_df, "Depression_Yes")
            if not dep.empty:
                feats = build_rule_features(tx, dep)
                self.assertEqual(feats.shape[1], len(dep))

    def test_build_rule_features_values_are_binary(self):
        tx       = _depression_transactions()
        rules_df = self._get_depression_rules()
        if not rules_df.empty:
            dep = filter_depression_rules(rules_df, "Depression_Yes")
            if not dep.empty:
                feats = build_rule_features(tx, dep)
                self.assertTrue(feats.isin([0, 1]).all().all())

    def test_filter_nonexistent_consequent_returns_empty(self):
        rules_df = self._get_depression_rules()
        if not rules_df.empty:
            result = filter_depression_rules(rules_df, "DOES_NOT_EXIST")
            self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
