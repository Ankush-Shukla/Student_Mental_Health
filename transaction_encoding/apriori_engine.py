"""
apriori_engine.py
-----------------
Pure-Python / NumPy implementation of the Apriori algorithm.
No external Apriori library required.

Public classes
--------------
AprioriEngine   — mines frequent itemsets and generates association rules
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FrequentItemset:
    items:   frozenset
    support: float


@dataclass
class AssociationRule:
    antecedent:  frozenset
    consequent:  frozenset
    support:     float
    confidence:  float
    lift:        float
    conviction:  float = field(default=0.0)

    def to_dict(self) -> dict:
        return {
            "antecedents": ", ".join(sorted(self.antecedent)),
            "consequents":  ", ".join(sorted(self.consequent)),
            "support":      round(self.support,    4),
            "confidence":   round(self.confidence, 4),
            "lift":         round(self.lift,        4),
            "conviction":   round(self.conviction,  4),
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AprioriEngine:
    """
    Apriori association rule miner.

    Parameters
    ----------
    min_support    : float  — minimum itemset support (fraction, 0–1)
    min_confidence : float  — minimum rule confidence
    min_lift       : float  — minimum rule lift
    max_itemset_len: int    — cap on itemset size (prevents combinatorial blow-up)
    """

    def __init__(
        self,
        min_support:     float = 0.05,
        min_confidence:  float = 0.60,
        min_lift:        float = 1.10,
        max_itemset_len: int   = 4,
    ) -> None:
        self.min_support     = min_support
        self.min_confidence  = min_confidence
        self.min_lift        = min_lift
        self.max_itemset_len = max_itemset_len

        self.frequent_itemsets_: list[FrequentItemset] = []
        self.rules_:             list[AssociationRule] = []

    # ------------------------------------------------------------------
    # Mining
    # ------------------------------------------------------------------

    def fit(self, tx_df: pd.DataFrame) -> "AprioriEngine":
        """
        Mine frequent itemsets and derive association rules from a boolean
        transaction DataFrame (rows = records, columns = items).
        """
        # Convert to a list-of-frozensets representation
        columns = list(tx_df.columns)
        col_idx = {c: i for i, c in enumerate(columns)}
        n       = len(tx_df)

        # Boolean matrix for fast counting
        matrix = tx_df.values.astype(bool)

        # --- Step 1: frequent 1-itemsets ---
        item_support = matrix.mean(axis=0)
        freq_items = [
            frozenset([columns[i]])
            for i, s in enumerate(item_support)
            if s >= self.min_support
        ]

        all_freq: list[FrequentItemset] = [
            FrequentItemset(items=fs, support=item_support[col_idx[next(iter(fs))]])
            for fs in freq_items
        ]

        support_cache: dict[frozenset, float] = {
            fs.items: fs.support for fs in all_freq
        }

        current_level = freq_items

        # --- Step 2: generate larger itemsets ---
        for k in range(2, self.max_itemset_len + 1):
            candidates = self._candidate_gen(current_level, k)
            if not candidates:
                break

            next_level = []
            for cand in candidates:
                col_indices = [col_idx[item] for item in cand]
                # All rows where every item in candidate is True
                sup = matrix[:, col_indices].all(axis=1).mean()
                if sup >= self.min_support:
                    fs  = FrequentItemset(items=cand, support=sup)
                    all_freq.append(fs)
                    support_cache[cand] = sup
                    next_level.append(cand)

            current_level = next_level

        self.frequent_itemsets_ = all_freq

        # --- Step 3: generate rules ---
        self.rules_ = self._generate_rules(all_freq, support_cache)

        return self

    # ------------------------------------------------------------------
    # Rule generation
    # ------------------------------------------------------------------

    def _generate_rules(
        self,
        freq_itemsets: list[FrequentItemset],
        support_cache: dict[frozenset, float],
    ) -> list[AssociationRule]:
        rules = []

        for fi in freq_itemsets:
            if len(fi.items) < 2:
                continue

            items = list(fi.items)
            for r in range(1, len(items)):
                for consequent_items in itertools.combinations(items, r):
                    consequent  = frozenset(consequent_items)
                    antecedent  = fi.items - consequent

                    if antecedent not in support_cache:
                        continue

                    ant_sup = support_cache[antecedent]
                    con_sup = support_cache.get(consequent, None)
                    if con_sup is None:
                        con_sup = support_cache.get(
                            frozenset(consequent_items), None
                        )
                    if con_sup is None:
                        continue

                    confidence = fi.support / ant_sup if ant_sup > 0 else 0.0
                    lift       = confidence / con_sup  if con_sup > 0 else 0.0

                    if confidence < self.min_confidence:
                        continue
                    if lift < self.min_lift:
                        continue

                    # conviction = (1 - con_sup) / (1 - confidence)
                    if confidence < 1.0:
                        conviction = (1 - con_sup) / (1 - confidence)
                    else:
                        conviction = float("inf")

                    rules.append(AssociationRule(
                        antecedent=antecedent,
                        consequent=consequent,
                        support=fi.support,
                        confidence=confidence,
                        lift=lift,
                        conviction=conviction,
                    ))

        rules.sort(key=lambda r: (r.lift, r.confidence), reverse=True)
        return rules

    # ------------------------------------------------------------------
    # Candidate generation (Apriori join step)
    # ------------------------------------------------------------------

    @staticmethod
    def _candidate_gen(prev_level: list[frozenset], k: int) -> list[frozenset]:
        """
        Generate k-itemset candidates from (k-1)-frequent itemsets via the
        Apriori join + pruning strategy.
        """
        prev_set = set(prev_level)
        seen     = set()
        candidates = []

        sorted_prev = [sorted(fs) for fs in prev_level]

        for i, a in enumerate(sorted_prev):
            for j in range(i + 1, len(sorted_prev)):
                b = sorted_prev[j]
                # Join condition: all but last item must be identical
                if a[:-1] == b[:-1]:
                    cand = frozenset(a) | frozenset(b)
                    if cand not in seen:
                        seen.add(cand)
                        # Prune: all (k-1)-subsets must be frequent
                        subsets = [
                            frozenset(s)
                            for s in itertools.combinations(cand, k - 1)
                        ]
                        if all(s in prev_set for s in subsets):
                            candidates.append(cand)
                else:
                    # Items are sorted; once prefix differs, inner loop can stop
                    break

        return candidates

    # ------------------------------------------------------------------
    # Convenience output
    # ------------------------------------------------------------------

    def rules_dataframe(self) -> pd.DataFrame:
        """Return all mined rules as a tidy DataFrame."""
        if not self.rules_:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.rules_])

    def frequent_itemsets_dataframe(self) -> pd.DataFrame:
        """Return frequent itemsets as a tidy DataFrame."""
        if not self.frequent_itemsets_:
            return pd.DataFrame()
        return pd.DataFrame([
            {"items": ", ".join(sorted(fi.items)), "support": round(fi.support, 4)}
            for fi in self.frequent_itemsets_
        ]).sort_values("support", ascending=False)
