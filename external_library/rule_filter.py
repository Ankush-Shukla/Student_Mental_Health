"""
rule_filter.py
--------------
Post-processes raw association rules:
  - Removes redundant supersets
  - Focuses on rules whose consequent is Depression_Yes or Depression_No
  - Constructs binary rule-match features to append to the model matrix
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_depression_rules(
    rules_df: pd.DataFrame,
    target_consequent: str = "Depression_Yes",
) -> pd.DataFrame:
    """
    Keep only rules where the consequent is the target label, then remove
    dominated rules (same or larger antecedent with equal or worse lift).

    Parameters
    ----------
    rules_df          : DataFrame from AprioriEngine.rules_dataframe()
    target_consequent : item string to filter consequents on

    Returns
    -------
    Filtered and deduplicated DataFrame sorted by lift descending.
    """
    if rules_df.empty:
        return rules_df

    mask = rules_df["consequents"].str.contains(target_consequent, regex=False)
    df   = rules_df[mask].copy().reset_index(drop=True)

    if df.empty:
        return df

    # Remove rules whose antecedent is a superset of another rule with >= lift
    keep  = []
    items = df["antecedents"].str.split(", ").tolist()
    lifts = df["lift"].tolist()

    for i in range(len(df)):
        ant_i = set(items[i])
        dominated = False
        for j in range(len(df)):
            if i == j:
                continue
            ant_j = set(items[j])
            # i is dominated if j is a strict subset with at least as high lift
            if ant_j < ant_i and lifts[j] >= lifts[i]:
                dominated = True
                break
        if not dominated:
            keep.append(i)

    return df.iloc[keep].sort_values("lift", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_rule_features(
    tx_df: pd.DataFrame,
    rules_df: pd.DataFrame,
    prefix: str = "Rule",
) -> pd.DataFrame:
    """
    For each rule in rules_df, create a binary column indicating whether
    a transaction satisfies the rule's antecedent items.

    Parameters
    ----------
    tx_df    : boolean transaction DataFrame (from preprocessing.build_transactions)
    rules_df : filtered rules DataFrame
    prefix   : column name prefix

    Returns
    -------
    DataFrame of shape (len(tx_df), len(rules_df)) with bool/int columns.
    """
    if rules_df.empty:
        return pd.DataFrame(index=tx_df.index)

    feature_dict: dict[str, np.ndarray] = {}

    for i, row in rules_df.iterrows():
        antecedent_items = [a.strip() for a in row["antecedents"].split(",")]
        col_name = f"{prefix}_{i:03d}"

        # Check which items exist as columns in tx_df
        present = [item for item in antecedent_items if item in tx_df.columns]
        missing = [item for item in antecedent_items if item not in tx_df.columns]

        if missing:
            # Rule cannot be evaluated — mark all as False
            feature_dict[col_name] = np.zeros(len(tx_df), dtype=int)
            continue

        # A transaction satisfies the antecedent if all antecedent items are True
        mask = tx_df[present].all(axis=1).astype(int)
        feature_dict[col_name] = mask.values

    return pd.DataFrame(feature_dict, index=tx_df.index)


def summarise_rules(rules_df: pd.DataFrame, top_n: int = 20) -> str:
    """Return a formatted string summary of the top N rules."""
    if rules_df.empty:
        return "No rules found."

    lines = [
        f"{'#':<4} {'Antecedents':<55} {'Consequents':<20} "
        f"{'Supp':>6} {'Conf':>6} {'Lift':>6}",
        "-" * 105,
    ]

    for i, row in rules_df.head(top_n).iterrows():
        ant = row["antecedents"][:53]
        con = row["consequents"][:18]
        lines.append(
            f"{i:<4} {ant:<55} {con:<20} "
            f"{row['support']:>6.3f} {row['confidence']:>6.3f} {row['lift']:>6.3f}"
        )

    return "\n".join(lines)
