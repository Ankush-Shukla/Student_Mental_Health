"""
pipeline.py
-----------
End-to-end runner for the depression risk detection pipeline.

Usage
-----
    python pipeline.py --data data/raw/train.csv --output outputs/

Steps executed
--------------
  1. Load and clean raw data
  2. Build transaction matrix for Apriori
  3. Build model feature matrix
  4. Mine frequent itemsets and association rules
  5. Filter rules to depression-relevant ones
  6. Build rule-based binary features
  7. Generate and save all visualisations
  8. Export artefacts (rules CSV, processed data CSV)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Module discovery
# ---------------------------------------------------------------------------
# Locate each required module by scanning all subdirectories under the project
# root. This makes the pipeline resilient to any folder layout the user adopts
# (src/, transaction_encoding/, external_library/, construction/, root-level, etc.)

_ROOT    = Path(__file__).parent
_MODULES = {
    "preprocessing",
    "apriori_engine",
    "rule_filter",
    "visualize",
}

_found: dict[str, Path] = {}
for _py in _ROOT.rglob("*.py"):
    if _py.stem in _MODULES and _py.stem not in _found:
        _found[_py.stem] = _py.parent

_missing = _MODULES - set(_found)
if _missing:
    print(
        f"ERROR: Could not locate the following modules anywhere under {_ROOT}:\n"
        f"  {', '.join(sorted(_missing))}\n"
        f"Expected files: {', '.join(m + '.py' for m in sorted(_missing))}",
        file=sys.stderr,
    )
    sys.exit(1)

for _dir in set(_found.values()):
    _dir_str = str(_dir)
    if _dir_str not in sys.path:
        sys.path.insert(0, _dir_str)

from preprocessing  import load_raw, clean, engineer_features, build_model_matrix, build_transactions
from apriori_engine import AprioriEngine
from rule_filter    import filter_depression_rules, build_rule_features, summarise_rules
from visualize      import (
    plot_correlation_heatmap,
    plot_rule_scatter,
    plot_rule_heatmap,
    plot_rule_network,
    plot_target_distribution,
    plot_top_rules_bar,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _section(title: str) -> None:
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(data_path: str, output_dir: str, min_support: float, min_confidence: float,
        min_lift: float, max_itemset_len: int, sample_size: int | None) -> None:

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load & clean
    # ------------------------------------------------------------------
    _section("Step 1 — Data Loading & Cleaning")
    _log(f"Loading: {data_path}")
    raw = load_raw(data_path)
    _log(f"Raw shape: {raw.shape}")

    df = clean(raw)
    _log(f"After cleaning: {df.shape}")

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        _log(f"Sampled to: {df.shape} (for faster Apriori mining)")

    # ------------------------------------------------------------------
    # 2. Transaction matrix
    # ------------------------------------------------------------------
    _section("Step 2 — Building Transaction Matrix")
    _log("Encoding records as boolean item transactions...")
    tx_df = build_transactions(df)
    _log(f"Transaction matrix: {tx_df.shape}  ({tx_df.columns.tolist()[:5]} ...)")

    # ------------------------------------------------------------------
    # 3. Model feature matrix
    # ------------------------------------------------------------------
    _section("Step 3 — Building Model Feature Matrix")
    _log("Engineering features for predictive models...")
    X, y = build_model_matrix(df)
    _log(f"Feature matrix: {X.shape}  |  Positive class rate: {y.mean():.3f}")

    # ------------------------------------------------------------------
    # 4. Apriori mining
    # ------------------------------------------------------------------
    _section("Step 4 — Apriori Association Rule Mining")
    _log(f"Parameters: support>={min_support}  confidence>={min_confidence}  "
         f"lift>={min_lift}  max_len={max_itemset_len}")

    engine = AprioriEngine(
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        max_itemset_len=max_itemset_len,
    )

    t0 = time.time()
    engine.fit(tx_df)
    elapsed = time.time() - t0

    fi_df    = engine.frequent_itemsets_dataframe()
    rules_df = engine.rules_dataframe()

    _log(f"Mining complete in {elapsed:.1f}s")
    _log(f"Frequent itemsets : {len(fi_df)}")
    _log(f"Association rules  : {len(rules_df)}")

    if rules_df.empty:
        _log("WARNING: No rules generated. Consider lowering min_support or min_confidence.")

    # ------------------------------------------------------------------
    # 5. Rule filtering
    # ------------------------------------------------------------------
    _section("Step 5 — Rule Filtering")
    dep_rules = pd.DataFrame()
    if not rules_df.empty:
        dep_rules = filter_depression_rules(rules_df, target_consequent="Depression_Yes")
        _log(f"Depression-targeting rules after filtering: {len(dep_rules)}")
        if not dep_rules.empty:
            print()
            print(summarise_rules(dep_rules, top_n=15))

    # ------------------------------------------------------------------
    # 6. Rule-based features
    # ------------------------------------------------------------------
    _section("Step 6 — Rule-Based Feature Construction")
    if not dep_rules.empty:
        rule_features = build_rule_features(tx_df, dep_rules)
        X_enriched    = pd.concat([X.reset_index(drop=True),
                                   rule_features.reset_index(drop=True)], axis=1)
        _log(f"Enriched feature matrix: {X_enriched.shape}")
    else:
        X_enriched = X.copy()
        _log("No rule features added (no rules found).")

    # ------------------------------------------------------------------
    # 7. Visualisations
    # ------------------------------------------------------------------
    _section("Step 7 — Generating Visualisations")

    _log("Target distribution...")
    p = plot_target_distribution(y, output_dir)
    _log(f"  Saved -> {p}")

    _log("Feature correlation heatmap...")
    p = plot_correlation_heatmap(X, output_dir)
    _log(f"  Saved -> {p}")

    if not rules_df.empty:
        _log("Rule scatter (support vs confidence)...")
        p = plot_rule_scatter(rules_df, output_dir)
        _log(f"  Saved -> {p}")

        _log("Top rules bar chart...")
        p = plot_top_rules_bar(rules_df, top_n=15, output_dir=output_dir)
        _log(f"  Saved -> {p}")

    if not dep_rules.empty:
        _log("Rule confidence heatmap...")
        p = plot_rule_heatmap(dep_rules, top_n=20, output_dir=output_dir)
        _log(f"  Saved -> {p}")

        _log("Rule network graph...")
        p = plot_rule_network(dep_rules, top_n=30, output_dir=output_dir)
        _log(f"  Saved -> {p}")

    # ------------------------------------------------------------------
    # 8. Export artefacts
    # ------------------------------------------------------------------
    _section("Step 8 — Exporting Artefacts")

    rules_path = os.path.join(output_dir, "association_rules.csv")
    if not rules_df.empty:
        rules_df.to_csv(rules_path, index=False)
        _log(f"All rules       -> {rules_path}")

    dep_rules_path = os.path.join(output_dir, "depression_rules.csv")
    if not dep_rules.empty:
        dep_rules.to_csv(dep_rules_path, index=False)
        _log(f"Depression rules -> {dep_rules_path}")

    fi_path = os.path.join(output_dir, "frequent_itemsets.csv")
    if not fi_df.empty:
        fi_df.to_csv(fi_path, index=False)
        _log(f"Frequent itemsets -> {fi_path}")

    features_path = os.path.join(output_dir, "features_enriched.csv")
    X_enriched["Depression"] = y.values
    X_enriched.to_csv(features_path, index=False)
    _log(f"Enriched features -> {features_path}")

    _section("Pipeline Complete")
    _log(f"All outputs written to: {os.path.abspath(output_dir)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Depression Risk Detection — Apriori Pipeline"
    )
    parser.add_argument(
        "--data",    required=True, help="Path to raw CSV (e.g. data/raw/train.csv)"
    )
    parser.add_argument(
        "--output",  default="outputs", help="Output directory (default: outputs/)"
    )
    parser.add_argument(
        "--support",    type=float, default=0.05, help="Minimum support (default 0.05)"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.60, help="Minimum confidence (default 0.60)"
    )
    parser.add_argument(
        "--lift",       type=float, default=1.10, help="Minimum lift (default 1.10)"
    )
    parser.add_argument(
        "--max-len",    type=int,   default=3,    help="Max itemset length (default 3)"
    )
    parser.add_argument(
        "--sample",     type=int,   default=None,
        help="Row limit for faster dev runs (default: use full dataset)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        data_path=args.data,
        output_dir=args.output,
        min_support=args.support,
        min_confidence=args.confidence,
        min_lift=args.lift,
        max_itemset_len=args.max_len,
        sample_size=args.sample,
    )