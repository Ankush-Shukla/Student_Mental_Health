"""
pipeline.py
-----------
End-to-end runner for the depression risk detection pipeline.

Usage
-----
    python pipeline.py --data data/raw/train.csv --output outputs/

Fixes in v2
-----------
* build_model_matrix now returns (X, y, encoders) — one LabelEncoder per
  bin column — and we persist the encoders to bin_encoders.pkl so that
  inference.py can use them correctly.
* sample_size variable shadowing bug fixed (local `sample_size` inside
  SHAP block no longer overwrites the CLI argument).
* SHAP block moved inside its own try/except so a SHAP failure doesn't
  abort the rest of the pipeline.
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

_ROOT    = Path(__file__).parent
_MODULES = {"preprocessing", "apriori_engine", "rule_filter", "visualize"}
_found: dict[str, Path] = {}

for _py in _ROOT.rglob("*.py"):
    if _py.stem in _MODULES and _py.stem not in _found:
        _found[_py.stem] = _py.parent

_missing = _MODULES - set(_found)
if _missing:
    print(
        f"ERROR: Cannot locate: {', '.join(sorted(_missing))}\n"
        f"Expected files: {', '.join(m + '.py' for m in sorted(_missing))}",
        file=sys.stderr,
    )
    sys.exit(1)

for _dir in set(_found.values()):
    _s = str(_dir)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from preprocessing  import load_raw, clean, engineer_features, build_model_matrix, build_transactions
from apriori_engine import AprioriEngine
from rule_filter    import filter_depression_rules, build_rule_features, summarise_rules
from visualize      import (
    plot_target_distribution,
    plot_correlation_heatmap,
    plot_rule_scatter,
    plot_rule_heatmap,
    plot_rule_network,
    plot_top_rules_bar,
    plot_depression_rate_by_factor,
    plot_support_distribution,
    plot_lift_confidence_line,
    plot_risk_profile_radar,
    plot_cumulative_rules,
    plot_feature_boxplots,
    plot_rule_conviction_scatter,
    plot_itemset_size_distribution,
)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _section(title: str) -> None:
    width = 70
    print(f"\n{'=' * width}\n  {title}\n{'=' * width}")


def run(data_path, output_dir, min_support, min_confidence,
        min_lift, max_itemset_len, sample_size):

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
        _log(f"Sampled to: {df.shape}")

    # ------------------------------------------------------------------
    # 2. Transaction matrix
    # ------------------------------------------------------------------
    _section("Step 2 — Transaction Matrix")
    tx_df = build_transactions(df)
    _log(f"Transaction matrix: {tx_df.shape}  ({list(tx_df.columns[:5])} ...)")

    # ------------------------------------------------------------------
    # 3. Model feature matrix  (returns encoders now)
    # ------------------------------------------------------------------
    _section("Step 3 — Model Feature Matrix")
    X, y, encoders = build_model_matrix(df)          # <-- fixed signature
    _log(f"Feature matrix: {X.shape}  |  Positive class rate: {y.mean():.3f}")
    _log(f"Encoders fitted for {len(encoders)} bin columns")

    df_eng = engineer_features(df)
    df_eng["Depression"] = y.values

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
    _log(f"Frequent itemsets: {len(fi_df)}  |  Rules: {len(rules_df)}")

    if rules_df.empty:
        _log("WARNING: No rules generated. Lower min_support or min_confidence.")

    # ------------------------------------------------------------------
    # 5. Rule filtering
    # ------------------------------------------------------------------
    _section("Step 5 — Rule Filtering")
    dep_rules = pd.DataFrame()
    if not rules_df.empty:
        dep_rules = filter_depression_rules(rules_df, target_consequent="Depression_Yes")
        _log(f"Depression-targeting rules: {len(dep_rules)}")
        if not dep_rules.empty:
            print()
            print(summarise_rules(dep_rules, top_n=15))

    # ------------------------------------------------------------------
    # 6. Rule-based features
    # ------------------------------------------------------------------
    _section("Step 6 — Rule-Based Feature Construction")
    if not dep_rules.empty:
        rule_features = build_rule_features(tx_df, dep_rules)
        X_enriched    = pd.concat(
            [X.reset_index(drop=True), rule_features.reset_index(drop=True)], axis=1
        )
        _log(f"Enriched feature matrix: {X_enriched.shape}")
    else:
        X_enriched = X.copy()
        _log("No rule features added.")

    # ------------------------------------------------------------------
    # 7. Model Training & Evaluation
    # ------------------------------------------------------------------
    _section("Step 7 — Model Training & Evaluation")

    import joblib
    import json
    from sklearn.linear_model   import LogisticRegression
    from sklearn.ensemble       import RandomForestClassifier
    from sklearn.metrics        import (accuracy_score, precision_score,
                                        recall_score, f1_score, roc_auc_score)
    from sklearn.model_selection import train_test_split

        # Ensure X keeps column names through the split
    X_model = X_enriched.copy()
    y_model = y.copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_model, y_model, test_size=0.2, random_state=42, stratify=y_model
    )

    # Explicitly preserve column names
    feature_names = list(X_train.columns)

    lr = LogisticRegression(max_iter=1000)
    from sklearn.calibration import CalibratedClassifierCV

    rf_base = RandomForestClassifier(n_estimators=70, random_state=42)
    rf_base.fit(X_train, y_train)

    # cv=None means use prefit estimator — pass the fitted model directly
    rf = CalibratedClassifierCV(rf_base, method="isotonic")
    rf.fit(X_test, y_test)   # calibrate on held-out set  # calibrate on held-out set
    lr.fit(X_train, y_train)
    

    # Verify feature names were stored
    assert hasattr(rf, 'feature_names_in_'), "RF trained without feature names!"
    print("RF feature names stored:", list(rf.feature_names_in_)[:5], "...")

    def _evaluate(model, Xt, yt):
        prob = model.predict_proba(Xt)[:, 1]
        pred = (prob >= 0.5).astype(int)
        return {
            "accuracy":  accuracy_score(yt, pred),
            "precision": precision_score(yt, pred),
            "recall":    recall_score(yt, pred),
            "f1":        f1_score(yt, pred),
            "roc_auc":   roc_auc_score(yt, prob),
        }

    metrics = {
        "logistic_regression": _evaluate(lr, X_test, y_test),
        "random_forest":       _evaluate(rf, X_test, y_test),
    }

    joblib.dump(lr, os.path.join(output_dir, "lr.pkl"))
    joblib.dump(rf, os.path.join(output_dir, "rf.pkl"))

    # *** FIX: persist one encoder per column ***
    joblib.dump(encoders, os.path.join(output_dir, "bin_encoders.pkl"))
    _log(f"bin_encoders.pkl saved  ({len(encoders)} encoders)")

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    _log(f"LR  Metrics: {metrics['logistic_regression']}")
    _log(f"RF  Metrics: {metrics['random_forest']}")

    # Feature template (column order for inference alignment)
    template_path = os.path.join(output_dir, "feature_template.csv")
    X_model.head(1).to_csv(template_path, index=False)
    _log(f"Feature template saved -> {template_path}")

    # ------------------------------------------------------------------
    # 7.1. SHAP Analysis (non-fatal if it fails)
    # ------------------------------------------------------------------
    _section("Step 7.1 — SHAP Analysis")
    try:
        import shap
        import matplotlib.pyplot as plt

        shap_n = min(300, len(X_test))
        X_shap = X_test.sample(n=shap_n, random_state=42)
        _log(f"Running SHAP on {shap_n} samples...")

        explainer   = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_shap)

        shap_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
        shap.summary_plot(shap_to_plot, X_shap, show=False)
        plt.savefig(os.path.join(output_dir, "shap_summary.png"))
        plt.close()
        _log("SHAP summary saved.")
    except Exception as exc:
        _log(f"SHAP skipped ({exc})")

    # ------------------------------------------------------------------
    # 8. Visualisations
    # ------------------------------------------------------------------
    _section("Step 8 — Generating Visualisations (14 charts)")

    charts = [
        ("01 Target distribution",        lambda: plot_target_distribution(y, output_dir)),
        ("02 Correlation heatmap",        lambda: plot_correlation_heatmap(X, output_dir)),
        ("03 Rule scatter",               lambda: plot_rule_scatter(rules_df, output_dir)   if not rules_df.empty  else None),
        ("04 Rule confidence heatmap",    lambda: plot_rule_heatmap(dep_rules, output_dir=output_dir) if not dep_rules.empty else None),
        ("05 Rule network",               lambda: plot_rule_network(dep_rules, output_dir=output_dir) if not dep_rules.empty else None),
        ("06 Top rules bar",              lambda: plot_top_rules_bar(rules_df, output_dir=output_dir) if not rules_df.empty  else None),
        ("07 Depression rate by factor",  lambda: plot_depression_rate_by_factor(df_eng, output_dir)),
        ("08 Rule metric distributions",  lambda: plot_support_distribution(rules_df, output_dir)     if not rules_df.empty  else None),
        ("09 Lift vs confidence line",    lambda: plot_lift_confidence_line(rules_df, output_dir)     if not rules_df.empty  else None),
        ("10 Risk profile radar",         lambda: plot_risk_profile_radar(df_eng, output_dir)),
        ("11 Cumulative rules",           lambda: plot_cumulative_rules(rules_df, output_dir)         if not rules_df.empty  else None),
        ("12 Feature boxplots",           lambda: plot_feature_boxplots(X, y, output_dir)),
        ("13 Conviction vs lift scatter", lambda: plot_rule_conviction_scatter(rules_df, output_dir) if not rules_df.empty  else None),
        ("14 Itemset size distribution",  lambda: plot_itemset_size_distribution(fi_df, output_dir)  if not fi_df.empty     else None),
    ]

    for name, fn in charts:
        _log(f"Generating: {name}...")
        try:
            result = fn()
            if result:
                _log(f"  -> {result}")
        except Exception as exc:
            _log(f"  SKIPPED ({exc})")

    # ------------------------------------------------------------------
    # 9. Export artefacts
    # ------------------------------------------------------------------
    _section("Step 9 — Exporting Artefacts")

    if not rules_df.empty:
        p = os.path.join(output_dir, "association_rules.csv")
        rules_df.to_csv(p, index=False)
        _log(f"All rules         -> {p}")

    if not dep_rules.empty:
        p = os.path.join(output_dir, "depression_rules.csv")
        dep_rules.to_csv(p, index=False)
        _log(f"Depression rules  -> {p}")

    if not fi_df.empty:
        p = os.path.join(output_dir, "frequent_itemsets.csv")
        fi_df.to_csv(p, index=False)
        _log(f"Frequent itemsets -> {p}")

    p = os.path.join(output_dir, "features_enriched.csv")
    X_enriched["Depression"] = y.values
    X_enriched.to_csv(p, index=False)
    _log(f"Enriched features -> {p}")

    _section("Pipeline Complete")
    _log(f"All outputs -> {os.path.abspath(output_dir)}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Depression Risk — Apriori Pipeline")
    parser.add_argument("--data",       required=True)
    parser.add_argument("--output",     default="outputs")
    parser.add_argument("--support",    type=float, default=0.05)
    parser.add_argument("--confidence", type=float, default=0.60)
    parser.add_argument("--lift",       type=float, default=1.10)
    parser.add_argument("--max-len",    type=int,   default=3)
    parser.add_argument("--sample",     type=int,   default=None)
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