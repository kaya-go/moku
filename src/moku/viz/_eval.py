"""Evaluation visualization helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_model_comparison(
    summary_df: pd.DataFrame,
    metrics: list[str] | None = None,
    title: str = "Model Comparison",
    figsize: tuple[float, float] = (8, 4),
) -> None:
    """Grouped bar chart comparing models on key metrics.

    Args:
        summary_df: DataFrame with a ``model`` column and one column per metric.
        metrics: Column names to plot. Defaults to all numeric columns.
        title: Chart title.
        figsize: Figure size.
    """
    if metrics is None:
        metrics = [c for c in summary_df.columns if c != "model" and pd.api.types.is_numeric_dtype(summary_df[c])]

    n_models = len(summary_df)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=figsize)
    for i, (_, row) in enumerate(summary_df.iterrows()):
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=row["model"])
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    plt.show()


def plot_threshold_sweep(
    sweep: dict,
    title: str = "Confidence Threshold Sweep",
    figsize: tuple[float, float] | None = None,
) -> None:
    """Plot P/R/F1 and mAP vs confidence threshold from ``sweep_confidence_threshold``.

    Top plot: macro-averaged P/R/F1 with optimal threshold marked.
    Middle plot: per-class F1 curves.
    Bottom plot: mAP@50:95 and mAP@50 vs threshold (if present in sweep).
    """
    thresholds = sweep["score_thresholds"]
    dt = sweep["distance_threshold"]
    macro = sweep["macro"]
    per_class = sweep["per_class"]
    class_names = list(per_class.keys())
    has_map = "map" in sweep and "map_50" in sweep

    nrows = 3 if has_map else 2
    if figsize is None:
        figsize = (10, 4 * nrows)
    fig, axes = plt.subplots(nrows, 1, figsize=figsize)

    # --- Left: Macro P/R/F1 ---
    axes[0].plot(thresholds, macro["precision"], label="Precision", linewidth=2)
    axes[0].plot(thresholds, macro["recall"], label="Recall", linewidth=2)
    axes[0].plot(thresholds, macro["f1"], label="F1", linewidth=2, color="black")

    best_idx = int(np.argmax(macro["f1"]))
    best_t = thresholds[best_idx]
    best_f1 = macro["f1"][best_idx]
    axes[0].axvline(best_t, color="red", linestyle="--", alpha=0.7)
    axes[0].plot(best_t, best_f1, "ro", markersize=8)
    axes[0].annotate(
        f"best={best_t:.2f}\nF1={best_f1:.3f}",
        xy=(best_t, best_f1),
        xytext=(10, -20),
        textcoords="offset points",
        fontsize=9,
        color="red",
    )

    axes[0].set_xlabel("Confidence Threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlim(0, 0.5)
    axes[0].set_title(f"{title} — Macro (dist≤{dt:.0%})")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # --- Center: Per-class F1 ---
    colors_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for j, name in enumerate(class_names):
        color = colors_cycle[j % len(colors_cycle)]
        f1_vals = per_class[name]["f1"]
        axes[1].plot(thresholds, f1_vals, label=name, linewidth=2, color=color)
        ci = int(np.argmax(f1_vals))
        best_t_cls = thresholds[ci]
        best_f1_cls = f1_vals[ci]
        axes[1].plot(best_t_cls, best_f1_cls, "o", markersize=6, color=color)
        axes[1].axvline(best_t_cls, color=color, linestyle="--", alpha=0.5)
        axes[1].annotate(
            f"{best_t_cls:.2f}",
            xy=(best_t_cls, best_f1_cls),
            xytext=(5, 5 - 15 * j),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )

    axes[1].set_xlabel("Confidence Threshold")
    axes[1].set_ylabel("F1")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlim(0, 0.5)
    axes[1].set_title(f"{title} — Per-Class F1 (dist≤{dt:.0%})")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    # --- Right: mAP vs threshold ---
    if has_map:
        axes[2].plot(thresholds, sweep["map"], label="mAP@50:95", linewidth=2, color="#1b9e77")
        axes[2].plot(thresholds, sweep["map_50"], label="mAP@50", linewidth=2, color="#d95f02")

        best_map_idx = int(np.argmax(sweep["map"]))
        best_map_t = thresholds[best_map_idx]
        best_map_v = sweep["map"][best_map_idx]
        axes[2].axvline(best_map_t, color="red", linestyle="--", alpha=0.7)
        axes[2].plot(best_map_t, best_map_v, "ro", markersize=8)
        axes[2].annotate(
            f"best={best_map_t:.2f}\nmAP={best_map_v:.3f}",
            xy=(best_map_t, best_map_v),
            xytext=(10, -20),
            textcoords="offset points",
            fontsize=9,
            color="red",
        )

        axes[2].set_xlabel("Confidence Threshold")
        axes[2].set_ylabel("mAP")
        axes[2].set_ylim(0, 1.05)
        axes[2].set_xlim(0, 0.5)
        axes[2].set_title(f"{title} — mAP")
        axes[2].legend(fontsize=9)
        axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
