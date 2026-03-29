"""
visualize.py
------------
All plotting functions for the depression risk detection pipeline.
Each function saves a PNG to disk and returns the file path.

Charts produced
---------------
  1.  plot_target_distribution        - label balance bar + pie
  2.  plot_correlation_heatmap        - feature correlation matrix
  3.  plot_rule_scatter               - support vs confidence (colour = lift)
  4.  plot_rule_heatmap               - antecedent x Depression_Yes confidence grid
  5.  plot_rule_network               - directed association-rule graph
  6.  plot_top_rules_bar              - top-N rules ranked by lift
  7.  plot_depression_rate_by_factor  - grouped bar: depression rate per category
  8.  plot_support_distribution       - histogram of rule metrics
  9.  plot_lift_confidence_line       - line chart: lift curve over confidence threshold
  10. plot_risk_profile_radar         - radar: high-risk vs low-risk student profile
  11. plot_cumulative_rules           - cumulative rule count as support threshold drops
  12. plot_feature_boxplots           - boxplots of numeric features by label
  13. plot_rule_conviction_scatter    - conviction vs lift for all rules
  14. plot_itemset_size_distribution  - frequent itemset count + mean support by size
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

P = {
    "primary":   "#12121f",
    "secondary": "#1b1b2f",
    "card":      "#22223b",
    "accent":    "#e63946",
    "accent2":   "#f4a261",
    "accent3":   "#2a9d8f",
    "accent4":   "#457b9d",
    "light":     "#edf2f4",
    "muted":     "#8d99ae",
    "positive":  "#2a9d8f",
    "negative":  "#e63946",
    "grid":      "#2a2a4a",
}

plt.rcParams.update({
    "figure.facecolor":  P["primary"],
    "axes.facecolor":    P["secondary"],
    "axes.edgecolor":    P["muted"],
    "axes.labelcolor":   P["light"],
    "text.color":        P["light"],
    "xtick.color":       P["muted"],
    "ytick.color":       P["muted"],
    "grid.color":        P["grid"],
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
    "font.family":       "monospace",
    "axes.titlesize":    11,
    "axes.labelsize":    9,
    "legend.facecolor":  P["card"],
    "legend.edgecolor":  P["muted"],
    "legend.fontsize":   8,
})


def _save(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _only_single_consequent(rules_df):
    """Drop rules with multi-item consequents."""
    mask = rules_df["consequents"].apply(
        lambda v: len([s for s in v.split(",") if s.strip()]) == 1
    )
    return rules_df[mask].copy()


# ---------------------------------------------------------------------------
# 1. Target distribution
# ---------------------------------------------------------------------------

def plot_target_distribution(y, output_dir="outputs", filename="01_target_distribution.png"):
    counts = y.value_counts().sort_index()
    labels = ["No Depression", "Depression"]
    colors = [P["positive"], P["negative"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor(P["primary"])

    ax = axes[0]
    bars = ax.bar(labels, counts.values, color=colors, width=0.45, edgecolor="none")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + counts.max() * 0.02,
                f"{val:,}\n({val/counts.sum()*100:.1f}%)",
                ha="center", va="bottom", fontsize=9, color=P["light"])
    ax.set_title("Label Distribution — Count", pad=12)
    ax.set_ylabel("Records")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(
        counts.values, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": P["primary"], "linewidth": 2},
        textprops={"color": P["light"], "fontsize": 9},
    )
    for at in autotexts:
        at.set_color(P["primary"])
        at.set_fontweight("bold")
    ax2.set_title("Label Distribution — Share", pad=12)

    fig.suptitle(f"Depression Label Balance  |  n = {len(y):,}",
                 fontsize=13, color=P["light"], y=1.01)
    fig.tight_layout()
    return _save(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# 2. Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(X, output_dir="outputs", filename="02_correlation_heatmap.png"):
    corr = X.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor(P["primary"])

    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.3, linecolor="#0a0a1a",
                ax=ax, cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Matrix (lower triangle)", pad=14,
                 fontsize=12, color=P["light"])
    ax.tick_params(axis="x", rotation=45, labelsize=7.5)
    ax.tick_params(axis="y", rotation=0,  labelsize=7.5)
    return _save(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# 3. Rule scatter
# ---------------------------------------------------------------------------

def plot_rule_scatter(rules_df, output_dir="outputs", filename="03_rule_scatter.png"):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(P["primary"])

    sc = ax.scatter(rules_df["support"], rules_df["confidence"],
                    c=rules_df["lift"], cmap="plasma",
                    s=55, alpha=0.85, edgecolors="none")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Lift", color=P["light"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=P["muted"])

    top5 = rules_df.nlargest(5, "lift")
    for _, row in top5.iterrows():
        lbl = row["antecedents"][:28] + "..." if len(row["antecedents"]) > 28 else row["antecedents"]
        ax.annotate(lbl, xy=(row["support"], row["confidence"]),
                    xytext=(8, 4), textcoords="offset points",
                    fontsize=6, color=P["accent2"],
                    arrowprops={"arrowstyle": "->", "color": P["muted"], "lw": 0.6})

    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.set_title("Association Rules — Support vs Confidence  (colour = Lift)", pad=12)
    ax.grid(True)
    return _save(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# 4. Rule confidence heatmap
# ---------------------------------------------------------------------------

def plot_rule_heatmap(rules_df, top_n=20, output_dir="outputs",
                      filename="04_rule_confidence_heatmap.png"):
    df = _only_single_consequent(rules_df.head(top_n))

    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(P["primary"])
        ax.text(0.5, 0.5, "No single-item consequent rules to display",
                ha="center", va="center", transform=ax.transAxes, color=P["light"])
        return _save(fig, output_dir, filename)

    df["ant_short"] = df["antecedents"].str[:42]
    df["con_short"] = df["consequents"].str.strip()

    pivot = df.pivot_table(index="ant_short", columns="con_short",
                           values="confidence", aggfunc="max").fillna(0)
    pivot = pivot.loc[:, (pivot > 0).any(axis=0)]

    h = max(6, len(pivot) * 0.44)
    fig, ax = plt.subplots(figsize=(max(7, pivot.shape[1] * 3), h))
    fig.patch.set_facecolor(P["primary"])

    sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".2f",
                annot_kws={"size": 8.5}, linewidths=0.4,
                linecolor="#0a0a1a", ax=ax,
                cbar_kws={"shrink": 0.7}, vmin=0, vmax=1)
    ax.set_title(f"Rule Confidence — Top {top_n} Depression Rules", pad=12, fontsize=11)
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    return _save(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# 5. Rule network
# ---------------------------------------------------------------------------

def plot_rule_network(rules_df, top_n=30, output_dir="outputs",
                      filename="05_rule_network.png"):
    df = rules_df.head(top_n)
    G  = nx.DiGraph()

    for _, row in df.iterrows():
        ants = [a.strip() for a in row["antecedents"].split(",")]
        cons = [c.strip() for c in row["consequents"].split(",")]
        for a in ants:
            for c in cons:
                if G.has_edge(a, c):
                    G[a][c]["weight"] = max(G[a][c]["weight"], row["lift"])
                else:
                    G.add_edge(a, c, weight=row["lift"])

    fig, ax = plt.subplots(figsize=(17, 12))
    fig.patch.set_facecolor(P["primary"])
    ax.set_facecolor(P["secondary"])

    pos = nx.spring_layout(G, seed=42, k=2.5)

    node_colors, node_sizes = [], []
    for node in G.nodes():
        if "Depression" in node:
            node_colors.append(P["negative"]); node_sizes.append(1800)
        elif any(k in node for k in ("Stress", "Pressure", "Suicidal", "Fin")):
            node_colors.append(P["accent2"]); node_sizes.append(1300)
        elif any(k in node for k in ("Sleep", "Study", "CGPA", "Age")):
            node_colors.append(P["accent4"]); node_sizes.append(1100)
        else:
            node_colors.append(P["muted"]); node_sizes.append(900)

    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.92)
    nx.draw_networkx_edges(G, pos, ax=ax,
                           width=[0.8 + 2.2 * (w / max_w) for w in edge_weights],
                           alpha=0.55, edge_color=P["muted"], arrows=True, arrowsize=16,
                           connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6.5, font_color=P["light"])

    legend_elements = [
        mpatches.Patch(color=P["negative"], label="Depression outcome"),
        mpatches.Patch(color=P["accent2"],  label="Stress / pressure factor"),
        mpatches.Patch(color=P["accent4"],  label="Lifestyle / academic factor"),
        mpatches.Patch(color=P["muted"],    label="Other"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)
    ax.set_title(f"Association Rule Network — top {top_n} rules by lift", pad=14)
    ax.axis("off")
    return _save(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# 6. Top rules bar chart
# ---------------------------------------------------------------------------

def plot_top_rules_bar(rules_df, top_n=15, output_dir="outputs",
                       filename="06_top_rules_lift.png"):
    df = rules_df.head(top_n).copy()
    df["label"] = df["antecedents"].str[:38] + "  ->  " + df["consequents"].str[:22]
    df = df[::-1]

    colors = [
        P["negative"] if "Depression_Yes" in row["consequents"] else P["accent4"]
        for _, row in df.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(13, max(5, top_n * 0.48)))
    fig.patch.set_facecolor(P["primary"])

    bars = ax.barh(df["label"], df["lift"], color=colors, edgecolor="none", height=0.65)
    for bar, val in zip(bars, df["lift"]):
        ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7.5, color=P["light"])

    ax.set_xlabel("Lift")
    ax.set_title(f"Top {top_n} Association Rules by Lift", pad=12)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(axis="x")
    ax.set_axisbelow(True)

    legend_elements = [
        mpatches.Patch(color=P["negative"], label="-> Depression_Yes"),
        mpatches.Patch(color=P["accent4"],  label="Other consequent"),
    ]
    ax.legend(handles=legend_elements)
    return _save(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# 7. Depression rate by categorical factor
# ---------------------------------------------------------------------------

def plot_depression_rate_by_factor(df_engineered, output_dir="outputs",
                                   filename="07_depression_rate_by_factor.png"):
    factors = {
        "Academic Pressure":  "AcadPressure_Bin",
        "Financial Stress":   "FinStress_Bin",
        "Sleep Duration":     "Sleep_Cat",
        "Age Group":          "Age_Bin",
        "CGPA Range":         "CGPA_Bin",
        "Study Hours":        "StudyHrs_Bin",
        "Study Satisfaction": "StudySat_Bin",
    }

    ncols = 3
    nrows = (len(factors) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4.2))
    fig.patch.set_facecolor(P["primary"])
    axes_flat = axes.flatten()

    for idx, (title, col) in enumerate(factors.items()):
        ax = axes_flat[idx]
        grp = (df_engineered.groupby(col, observed=True)["Depression"]
               .mean().reset_index()
               .rename(columns={"Depression": "rate"})
               .sort_values("rate"))
        grp["rate_pct"] = grp["rate"] * 100

        bar_colors = [
            P["negative"] if r > 60 else P["accent2"] if r > 40 else P["positive"]
            for r in grp["rate_pct"]
        ]
        bars = ax.barh(grp[col].astype(str), grp["rate_pct"],
                       color=bar_colors, edgecolor="none", height=0.55)

        for bar, val in zip(bars, grp["rate_pct"]):
            ax.text(val + 0.8, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=7.5, color=P["light"])

        ax.set_title(title, pad=8, fontsize=9.5)
        ax.set_xlabel("Depression Rate (%)")
        ax.set_xlim(0, 108)
        ax.axvline(50, color=P["muted"], linewidth=0.8, linestyle="--", alpha=0.6)
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)

    for idx in range(len(factors), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Depression Rate (%) by Risk Factor Category",
                 fontsize=13, color=P["light"], y=1.01)
    fig.tight_layout()
    return _save(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# 8. Rule metric distributions
# ---------------------------------------------------------------------------

def plot_support_distribution(rules_df, output_dir="outputs",
                              filename="08_rule_metric_distributions.png"):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor(P["primary"])

    metrics = [
        ("support",    "Support",    P["accent4"]),
        ("confidence", "Confidence", P["accent2"]),
        ("lift",       "Lift",       P["negative"]),
    ]
    for ax, (col, label, color) in zip(axes, metrics):
        vals = rules_df[col].dropna()
        ax.hist(vals, bins=25, color=color, alpha=0.85, edgecolor="none")
        ax.axvline(vals.mean(),   color=P["light"],  linewidth=1.2,
                   linestyle="--", label=f"Mean={vals.mean():.3f}")
        ax.axvline(vals.median(), color=P["accent2"], linewidth=1.2,
                   linestyle=":",  label=f"Median={vals.median():.3f}")
        ax.set_title(f"{label} Distribution", pad=10)
        ax.set_xlabel(label)
        ax.set_ylabel("Rule Count")
        ax.legend(fontsize=7)
        ax.grid(True)
        ax.set_axisbelow(True)

    fig.suptitle("Distribution of Rule Metrics  |  All Rules",
                 fontsize=13, color=P["light"], y=1.01)
    fig.tight_layout()
    return _save(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# 9. Lift / rule-count line chart over confidence threshold
# ---------------------------------------------------------------------------

def plot_lift_confidence_line(rules_df, output_dir="outputs",
                              filename="09_lift_confidence_line.png"):
    thresholds = np.arange(0.50, 0.96, 0.01)
    counts, mean_lifts = [], []
    for t in thresholds:
        sub = rules_df[rules_df["confidence"] >= t]
        counts.append(len(sub))
        mean_lifts.append(sub["lift"].mean() if len(sub) > 0 else np.nan)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(P["primary"])
    ax1.set_facecolor(P["secondary"])

    ax1.fill_between(thresholds, counts, alpha=0.18, color=P["accent4"])
    ax1.plot(thresholds, counts, color=P["accent4"], linewidth=2, label="Rule count")
    ax1.set_xlabel("Minimum Confidence Threshold")
    ax1.set_ylabel("Number of Rules", color=P["accent4"])
    ax1.tick_params(axis="y", colors=P["accent4"])
    ax1.grid(True)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    ax2.set_facecolor(P["secondary"])
    ax2.plot(thresholds, mean_lifts, color=P["accent2"], linewidth=2,
             linestyle="--", label="Mean lift")
    ax2.set_ylabel("Mean Lift of Surviving Rules", color=P["accent2"])
    ax2.tick_params(axis="y", colors=P["accent2"])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Rule Count and Mean Lift vs Confidence Threshold", pad=12)
    fig.tight_layout()
    return _save(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# 10. Radar chart — risk profile
# ---------------------------------------------------------------------------

def plot_risk_profile_radar(df_engineered, output_dir="outputs",
                            filename="10_risk_profile_radar.png"):
    numeric_factors = {
        "Academic\nPressure": "Academic Pressure",
        "Financial\nStress":  "Financial Stress",
        "Study\nHours":       "Work/Study Hours",
        "CGPA":               "CGPA",
        "Age":                "Age",
    }

    df = df_engineered.copy()
    df["Financial Stress"] = pd.to_numeric(
        df["Financial Stress"].replace("?", np.nan), errors="coerce"
    )

    labels = list(numeric_factors.keys())
    cols   = list(numeric_factors.values())
    n      = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    dep_1 = df[df["Depression"] == 1][cols].mean()
    dep_0 = df[df["Depression"] == 0][cols].mean()

    col_min = df[cols].min()
    col_max = df[cols].max()
    rng     = (col_max - col_min).replace(0, 1)

    vals_1 = ((dep_1 - col_min) / rng).tolist()
    vals_0 = ((dep_0 - col_min) / rng).tolist()
    vals_1 += vals_1[:1]
    vals_0 += vals_0[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    fig.patch.set_facecolor(P["primary"])
    ax.set_facecolor(P["secondary"])

    ax.plot(angles, vals_1, color=P["negative"],  linewidth=2.2, label="Depressed")
    ax.fill(angles, vals_1, color=P["negative"],  alpha=0.20)
    ax.plot(angles, vals_0, color=P["positive"],  linewidth=2.2, label="Not Depressed")
    ax.fill(angles, vals_0, color=P["positive"],  alpha=0.20)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9, color=P["light"])
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=7, color=P["muted"])
    ax.set_ylim(0, 1)
    ax.grid(color=P["grid"], linestyle="--", alpha=0.5)
    ax.spines["polar"].set_color(P["muted"])
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12))
    ax.set_title("Student Risk Profile Comparison\n(normalised to global range)",
                 pad=22, fontsize=11, color=P["light"])
    return _save(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# 11. Cumulative rule count as support threshold decreases
# ---------------------------------------------------------------------------

def plot_cumulative_rules(rules_df, output_dir="outputs",
                          filename="11_cumulative_rules.png"):
    thresholds = np.arange(0.30, 0.04, -0.01)
    all_counts, dep_counts = [], []

    for t in thresholds:
        sub = rules_df[rules_df["support"] >= t]
        all_counts.append(len(sub))

        def _exact(v):
            return {s.strip() for s in v.split(",")} == {"Depression_Yes"}

        dep_counts.append(int(sub["consequents"].apply(_exact).sum()))

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(P["primary"])

    ax.fill_between(thresholds, all_counts, alpha=0.14, color=P["accent4"])
    ax.plot(thresholds, all_counts, color=P["accent4"], linewidth=2, label="All rules")
    ax.fill_between(thresholds, dep_counts, alpha=0.22, color=P["negative"])
    ax.plot(thresholds, dep_counts, color=P["negative"], linewidth=2,
            linestyle="--", label="Depression_Yes rules")

    ax.invert_xaxis()
    ax.set_xlabel("Minimum Support Threshold (decreasing ->)")
    ax.set_ylabel("Rule Count")
    ax.set_title("Cumulative Rules as Support Threshold Decreases", pad=12)
    ax.legend()
    ax.grid(True)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _save(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# 12. Feature boxplots by label
# ---------------------------------------------------------------------------

def plot_feature_boxplots(X, y, output_dir="outputs",
                          filename="12_feature_boxplots.png"):
    numeric_cols = ["Age", "CGPA", "Academic Pressure",
                    "Study Satisfaction", "Work/Study Hours"]
    existing = [c for c in numeric_cols if c in X.columns]

    df_plot = X[existing].copy()
    df_plot["Depression"] = y.values
    df_plot["Label"] = df_plot["Depression"].map({0: "No Depression", 1: "Depression"})

    ncols = 3
    nrows = (len(existing) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4.5))
    fig.patch.set_facecolor(P["primary"])
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    color_map = {"No Depression": P["positive"], "Depression": P["negative"]}

    for idx, col in enumerate(existing):
        ax = axes_flat[idx]
        groups = [
            df_plot.loc[df_plot["Label"] == lbl, col].dropna()
            for lbl in ["No Depression", "Depression"]
        ]
        bp = ax.boxplot(
            groups, patch_artist=True, widths=0.4,
            medianprops={"color": P["light"], "linewidth": 2},
            whiskerprops={"color": P["muted"]},
            capprops={"color": P["muted"]},
            flierprops={"marker": "o", "markersize": 2.5,
                        "markerfacecolor": P["muted"], "alpha": 0.4},
            boxprops={"linewidth": 0},
        )
        for patch, lbl in zip(bp["boxes"], ["No Depression", "Depression"]):
            patch.set_facecolor(color_map[lbl])
            patch.set_alpha(0.75)

        ax.set_xticklabels(["No Depression", "Depression"], fontsize=8)
        ax.set_title(col, pad=8, fontsize=9.5)
        ax.set_ylabel("Value")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        for i, grp in enumerate(groups, start=1):
            ax.plot(i, grp.mean(), marker="D", color=P["accent2"],
                    markersize=6, zorder=5, label="Mean" if i == 1 else "")
        if idx == 0:
            ax.legend(fontsize=7)

    for idx in range(len(existing), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Feature Distributions by Depression Label",
                 fontsize=13, color=P["light"], y=1.01)
    fig.tight_layout()
    return _save(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# 13. Conviction vs Lift scatter
# ---------------------------------------------------------------------------

def plot_rule_conviction_scatter(rules_df, output_dir="outputs",
                                 filename="13_conviction_lift_scatter.png"):
    df = rules_df[rules_df["conviction"] < 50].copy()

    is_dep = df["consequents"].apply(
        lambda v: {s.strip() for s in v.split(",")} == {"Depression_Yes"}
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(P["primary"])

    sc = ax.scatter(df.loc[~is_dep, "lift"], df.loc[~is_dep, "conviction"],
                    c=df.loc[~is_dep, "confidence"], cmap="plasma",
                    s=45, alpha=0.65, edgecolors="none", label="Other rules")
    ax.scatter(df.loc[is_dep, "lift"], df.loc[is_dep, "conviction"],
               c=df.loc[is_dep, "confidence"], cmap="plasma",
               s=120, alpha=0.9, edgecolors=P["negative"], linewidths=1.4,
               marker="*", label="Depression_Yes rules")

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Confidence", color=P["light"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=P["muted"])

    ax.axhline(1, color=P["muted"], linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(1, color=P["muted"], linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_xlabel("Lift")
    ax.set_ylabel("Conviction")
    ax.set_title("Rule Quality — Conviction vs Lift  (colour = Confidence)", pad=12)
    ax.legend()
    ax.grid(True)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _save(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# 14. Frequent itemset size distribution
# ---------------------------------------------------------------------------

def plot_itemset_size_distribution(fi_df, output_dir="outputs",
                                   filename="14_itemset_size_distribution.png"):
    fi = fi_df.copy()
    fi["size"] = fi["items"].str.split(", ").apply(len)

    size_counts  = fi.groupby("size").size()
    size_support = fi.groupby("size")["support"].mean()

    sizes     = size_counts.index.tolist()
    bar_colors = [P["accent4"], P["accent2"], P["negative"]][:len(sizes)]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(P["primary"])

    bars = ax1.bar(
        [f"{s}-item" for s in sizes], size_counts.values,
        color=bar_colors, width=0.45, edgecolor="none",
    )
    for bar, val in zip(bars, size_counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + size_counts.max() * 0.02,
                 str(val), ha="center", va="bottom", fontsize=9, color=P["light"])

    ax1.set_ylabel("Frequent Itemset Count")
    ax1.set_xlabel("Itemset Size")
    ax1.set_title("Frequent Itemset Count and Mean Support by Size", pad=12)
    ax1.grid(axis="y")
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    ax2.plot([f"{s}-item" for s in sizes], size_support.values,
             color=P["accent2"], linewidth=2, marker="o", markersize=7,
             linestyle="--", label="Mean support")
    ax2.set_ylabel("Mean Support", color=P["accent2"])
    ax2.tick_params(axis="y", colors=P["accent2"])
    ax2.legend(loc="upper right")

    fig.tight_layout()
    return _save(fig, output_dir, filename)