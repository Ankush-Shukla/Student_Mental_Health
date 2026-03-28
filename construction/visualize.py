"""
visualize.py
------------
All plotting functions for the pipeline.
Each function saves to disk and returns the Axes / Figure object.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

PALETTE = {
    "primary":   "#1a1a2e",
    "secondary": "#16213e",
    "accent":    "#e94560",
    "light":     "#f5f5f5",
    "muted":     "#8892b0",
    "positive":  "#43b88c",
    "negative":  "#e94560",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["primary"],
    "axes.facecolor":    PALETTE["secondary"],
    "axes.edgecolor":    PALETTE["muted"],
    "axes.labelcolor":   PALETTE["light"],
    "text.color":        PALETTE["light"],
    "xtick.color":       PALETTE["muted"],
    "ytick.color":       PALETTE["muted"],
    "grid.color":        "#2a2a4a",
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
    "font.family":       "monospace",
    "axes.titlesize":    11,
    "axes.labelsize":    9,
})


def _save(fig: plt.Figure, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 1. Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    X: pd.DataFrame,
    output_dir: str = "outputs",
    filename:   str = "correlation_heatmap.png",
) -> str:
    corr = X.corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor(PALETTE["primary"])

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        center=0,
        annot=False,
        linewidths=0.3,
        linecolor="#111133",
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix", pad=14, fontsize=12, color=PALETTE["light"])
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)

    return _save(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# 2. Support / Confidence scatter for rules
# ---------------------------------------------------------------------------

def plot_rule_scatter(
    rules_df:   pd.DataFrame,
    output_dir: str = "outputs",
    filename:   str = "rule_scatter.png",
) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(PALETTE["primary"])

    sc = ax.scatter(
        rules_df["support"],
        rules_df["confidence"],
        c=rules_df["lift"],
        cmap="plasma",
        s=60,
        alpha=0.8,
        edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Lift", color=PALETTE["light"])
    cbar.ax.yaxis.set_tick_params(color=PALETTE["muted"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["muted"])

    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.set_title("Association Rules — Support vs Confidence (colour = Lift)", pad=12)
    ax.grid(True)

    return _save(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# 3. Rule confidence heatmap (antecedent x consequent)
# ---------------------------------------------------------------------------

def plot_rule_heatmap(
    rules_df:   pd.DataFrame,
    top_n:      int = 20,
    output_dir: str = "outputs",
    filename:   str = "rule_confidence_heatmap.png",
) -> str:
    df = rules_df.head(top_n).copy()

    # Shorten labels for readability
    df["ant_short"] = df["antecedents"].str[:40]
    df["con_short"] = df["consequents"].str[:25]

    pivot = df.pivot_table(
        index="ant_short", columns="con_short", values="confidence", aggfunc="max"
    ).fillna(0)

    h = max(6, len(pivot) * 0.4)
    fig, ax = plt.subplots(figsize=(10, h))
    fig.patch.set_facecolor(PALETTE["primary"])

    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        linecolor="#111133",
        ax=ax,
        cbar_kws={"shrink": 0.7},
        annot_kws={"size": 7},
    )
    ax.set_title(f"Rule Confidence Heatmap (top {top_n} rules)", pad=12, fontsize=11)
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)
    ax.set_xlabel("")
    ax.set_ylabel("")

    return _save(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# 4. Network graph of association rules
# ---------------------------------------------------------------------------

def plot_rule_network(
    rules_df:   pd.DataFrame,
    top_n:      int = 30,
    output_dir: str = "outputs",
    filename:   str = "rule_network.png",
) -> str:
    df = rules_df.head(top_n)

    G = nx.DiGraph()
    for _, row in df.iterrows():
        ants = [a.strip() for a in row["antecedents"].split(",")]
        cons = [c.strip() for c in row["consequents"].split(",")]
        for a in ants:
            for c in cons:
                if G.has_edge(a, c):
                    G[a][c]["weight"] = max(G[a][c]["weight"], row["lift"])
                else:
                    G.add_edge(a, c, weight=row["lift"], confidence=row["confidence"])

    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor(PALETTE["primary"])
    ax.set_facecolor(PALETTE["secondary"])

    pos = nx.spring_layout(G, seed=42, k=2.2)

    # Node colour: depression nodes highlighted
    node_colors = []
    for node in G.nodes():
        if "Depression" in node:
            node_colors.append(PALETTE["accent"])
        elif "Stress" in node or "Pressure" in node or "Suicidal" in node:
            node_colors.append("#f0a500")
        else:
            node_colors.append(PALETTE["muted"])

    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=900,
        alpha=0.9,
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=[1 + 2 * (w / max_w) for w in edge_weights],
        alpha=0.6,
        edge_color=PALETTE["muted"],
        arrows=True,
        arrowsize=14,
        connectionstyle="arc3,rad=0.1",
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=6.5,
        font_color=PALETTE["light"],
    )

    legend_elements = [
        mpatches.Patch(color=PALETTE["accent"],  label="Depression node"),
        mpatches.Patch(color="#f0a500",           label="Risk factor node"),
        mpatches.Patch(color=PALETTE["muted"],    label="Other node"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8,
              facecolor=PALETTE["secondary"], edgecolor=PALETTE["muted"])

    ax.set_title(f"Association Rule Network (top {top_n} rules by lift)", pad=14)
    ax.axis("off")

    return _save(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# 5. Target distribution
# ---------------------------------------------------------------------------

def plot_target_distribution(
    y: pd.Series,
    output_dir: str = "outputs",
    filename:   str = "target_distribution.png",
) -> str:
    counts = y.value_counts().sort_index()
    labels = ["No Depression", "Depression"]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(PALETTE["primary"])

    bars = ax.bar(
        labels,
        counts.values,
        color=[PALETTE["positive"], PALETTE["negative"]],
        width=0.5,
        edgecolor="none",
    )
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 80,
            f"{val:,}\n({val/counts.sum()*100:.1f}%)",
            ha="center", va="bottom", fontsize=9, color=PALETTE["light"],
        )

    ax.set_title("Depression Label Distribution", pad=12)
    ax.set_ylabel("Record Count")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.grid(axis="y")

    return _save(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# 6. Top-N rules bar chart (lift)
# ---------------------------------------------------------------------------

def plot_top_rules_bar(
    rules_df:   pd.DataFrame,
    top_n:      int = 15,
    output_dir: str = "outputs",
    filename:   str = "top_rules_lift.png",
) -> str:
    df = rules_df.head(top_n).copy()
    df["label"] = (
        df["antecedents"].str[:35] + "  ->  " + df["consequents"].str[:20]
    )
    df = df[::-1]  # reverse so highest lift is at top of horizontal bar

    fig, ax = plt.subplots(figsize=(12, max(5, top_n * 0.45)))
    fig.patch.set_facecolor(PALETTE["primary"])

    colors = [
        PALETTE["accent"] if "Depression_Yes" in row["consequents"] else PALETTE["muted"]
        for _, row in df.iterrows()
    ]

    bars = ax.barh(df["label"], df["lift"], color=colors, edgecolor="none", height=0.65)

    for bar, val in zip(bars, df["lift"]):
        ax.text(
            bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=7.5, color=PALETTE["light"],
        )

    ax.set_xlabel("Lift")
    ax.set_title(f"Top {top_n} Rules by Lift", pad=12)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(axis="x")

    legend_elements = [
        mpatches.Patch(color=PALETTE["accent"], label="Depression_Yes consequent"),
        mpatches.Patch(color=PALETTE["muted"],  label="Other consequent"),
    ]
    ax.legend(handles=legend_elements, fontsize=8,
              facecolor=PALETTE["secondary"], edgecolor=PALETTE["muted"])

    return _save(fig, output_dir, filename)
